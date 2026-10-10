"""
KD 交叉：文章與 reference 全變體掃描（單股回測段）
==================================================
KD 系列文章（blog/site/content/posts/kd-cross-*.md）與 reference（blog/reference/single_kd/）的
對照表合計上百組進場 × 出場 × 流動性組合，原本是一批 scratchpad driver 跑的、已遺失。這裡把
它們全部登錄成 VARIANTS（條件式見 kd_variants.py），以標準區間 2002～2025、2,258 檔重產。

做法比照 ma_strategy/single_ma_sweep.py：每檔讀一次全史、`add_columns` 只算一次，所有變體共用
同一份備妥的 DataFrame（子類的 add_columns 不再重算）；指標吃全史暖身、成交只在區間內。
平行方式：把變體分成 --workers 組，每組一個子進程各自掃全市場、自己落地結果——每個子進程
只抱自己那組變體的逐筆交易，記憶體不會隨變體數疊加；代價是每檔的讀檔＋備欄會做 workers 次
（相對於上百次 vbt 回測可忽略）。

變體命名：<流動性>__<進場>__<出場>[__z<超賣>_<超買>]
  流動性 raw（無門檻）／lot（1000 張）／amt（成交金額 1,000 萬）／amt30m（舊 3,000 萬版）
  進場   golden＝純黃金交叉；其餘為疊在黃金交叉上的濾網（多個以 + 串），純檔位等基礎見 kd_variants.ENTRY_BASES
  出場   多個以 -or- 串（任一觸發即出）
  例：amt__breakout60__high_death、amt__golden__k_down50-or-k_down80、amt__low_zone__high_death__z30_70

輸出（result/ 不進版控；--out 可改根目錄，冒煙用）：
  result/single_kd/<變體>/single_kd_per_stock.csv、single_kd_aggregate.csv   與策略檔 CLI 同格式
  result/single_kd/<變體>/single_kd_trades.parquet                          逐筆交易（MC 等分析段用）

執行：
  python _02_strategy/kd_strategy/kd_sweep.py [--set matrix entries choch_calib ...] [--variants 名稱 ...]
                                              [--limit N --out <暫存目錄>] [--workers 8] [--list]
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd  # noqa: E402

from _02_strategy.base.vbt import batch, common  # noqa: E402
from _02_strategy.base.vbt.common import DEFAULT_END, DEFAULT_START, GLITCH  # noqa: E402
from _02_strategy.kd_strategy.kd_variants import KdVariant  # noqa: E402
from _02_strategy.kd_strategy.single_kd_strategy import RESULT_DIR  # noqa: E402

LABEL = "single_kd"
# 流動性前綴 → (LIQUIDITY, TURNOVER)
_LIQ = {"raw": ("none", None), "lot": ("lot", None),
        "amt": ("amt", 10_000_000), "amt30m": ("amt", 30_000_000)}

VARIANTS = {}       # 名稱 → {"liq", "entry", "exit", "zone"}
SETS = {}           # 集合名 → [變體名稱]


def _v(liq: str, entry=(), exit_="death", zone=None) -> str:
    """登錄一個變體（同名只登一次），回傳名稱。zone＝(超賣, 超買)，None＝預設 20/80。"""
    # AND／OR 都可交換，名稱一律排序，避免同一組合因列出順序不同被登錄兩次
    entry = (entry,) if isinstance(entry, str) else tuple(sorted(entry))
    exit_ = (exit_,) if isinstance(exit_, str) else tuple(sorted(exit_))
    name = f"{liq}__{'+'.join(entry) or 'golden'}__{'-or-'.join(exit_)}"
    if zone is not None:
        name += f"__z{zone[0]}_{zone[1]}"
    VARIANTS.setdefault(name, {"liq": liq, "entry": entry, "exit": exit_, "zone": zone})
    return name


def _set(key: str, names: list) -> None:
    SETS[key] = list(dict.fromkeys(names))


# ── 2026-07-05 KD 交叉 baseline vs opt1 × 兩種流動性（舊金額門檻 3,000 萬）＋ 文章（一）──
_set("d0705", [
    _v("raw"), _v("lot"), _v("amt30m"),                                   # baseline／+1000張／+3000萬
    _v("raw", "low_zone", "high_death"),                                  # opt1
    _v("lot", "low_zone", "high_death"), _v("amt30m", "low_zone", "high_death"),
])
# ── 2026-07-08 三個進場濾網（裸版）opt3/4/5 ──
_set("d0708", [_v("raw", e) for e in ("cmf_pos", "divergence", "ma120_gt_ma200")])
# ── 2026-07-09 出場優化（裸版）opt6~9 ──
_set("d0709", [_v("raw", (), x) for x in ("high_death", "k_down50", "climax", "top_div")])
# ── 2026-07-10 可成交 1,000 萬：baseline／opt1／opt3~9（＝策略檔 baseline_amt、opt1_amt …）──
_set("d0710", [
    _v("amt"), _v("amt", "low_zone", "high_death"),
    *[_v("amt", e) for e in ("cmf_pos", "divergence", "ma120_gt_ma200")],
    *[_v("amt", (), x) for x in ("high_death", "k_down50", "climax", "top_div")],
])
# ── 2026-07-14 門檻變形 ＋ 進場濾網 × 出場組合（可成交）＝ 文章（二）超買超賣的表 ──
_set("d0714", [
    _v("amt"), _v("amt", "low_zone", "high_death"), _v("amt", (), "high_death"),
    _v("amt", "low_zone", "death"),                                       # 只有進場那半
    _v("amt", "low_zone", "high_death", zone=(30, 70)),                   # 30/70
    _v("amt", "low_zone", "high_death", zone=(10, 90)),                   # 10/90
    _v("amt", "low_zone", "climax"),
    _v("amt", "cmf_pos", "high_death"), _v("amt", "divergence", "high_death"),
    _v("amt", "ma120_gt_ma200", "high_death"),
    _v("amt", "cmf_pos", "climax"), _v("amt", "ma120_gt_ma200", "climax"),
])
# ── 2026-07-20 §1 進場獨立條件 × 死叉（可成交）＝ 文章（三）進場優化全部表 ──
_ENTRIES_0720 = (
    "breakout250", "breakout120", "gap", "low_redk", "breakout60", "divergence", "low_zone",
    "bull_align_5_20_60", "breakout20", "bull_align_20_60_120", "above_ma20", "obv_up_1d",
    "ma120_gt_ma200", "candle_not_black", "candle_short_upper", "candle_combo",
    "above_ma60", "above_ma120", "above_ma200", "vol_above_ma5", "vol_x2", "vol_x1_5",
    "confirm2", "cmf_pos", "kd_spread5",
)
_set("d0720_entries", [_v("amt")] + [_v("amt", e) for e in _ENTRIES_0720])
# ── 2026-07-20 §2 純檔位策略（不判斷交叉）──
_set("d0720_zone", [
    _v("amt", "zone_k_up20", ("k_down50", "k_down80")),                  # K 完整
    _v("amt", "zone_k_up20", "k_down50"),                                 # K 只 ↓50
    _v("amt", "zone_k_up20", "k_down80"),                                 # K 只 ↓80
    _v("amt", "zone_kd_up20", ("kd_down50", "kd_down80")),                # K、D 兩條都（完整）
    _v("amt", "zone_d_up20", ("d_down50", "d_down80")),                   # D 線（完整）
])
# ── 2026-07-20 §3~§5 檔位出場、MA20、低檔＋MA60 ──
_set("d0720_exits", [
    _v("amt", (), ("k_down50", "k_down80")), _v("amt", (), "k_down50"), _v("amt", (), "k_down80"),
    _v("amt", "above_ma20", ("below_ma20", "death")), _v("amt", "above_ma20", "death"),
    _v("amt", (), ("death", "below_ma20")),
    *[_v("amt", ("low_zone", "above_ma60"), x) for x in (
        "k_down80", "below_ma20", ("below_ma20", "k_down80"), "death", ("death", "below_ma20"))],
])
# ── 2026-07-26 MA20/OBV 合併、50 中線上下、交叉強度、多頭排列（× 死叉，可成交）──
_set("d0726_ma20obv", [_v("amt")] + [_v("amt", e) for e in (
    "above_ma20", "obv_up_ma20", ("above_ma20", "obv_up_ma20"),
    "high_zone50", "d_above50", "low_zone50", "d_below50",
    "kd_spread2", "kd_spread5", "kd_spread10",
    "bull_align_5_20_60", ("bull_align_5_20_60", "above_ma20"))])
# ── 2026-07-26 舊整合矩陣 6 進 × 5 出 ＋ 同日 CHoCH 校準（無狀態頂頂低欄、高檔死叉對照欄）──
_ENTRIES_0726 = ((), "low_zone", "high_zone50", "bull_align_5_20_60", "breakout250", "cmf_pos")
_set("d0726_matrix", [_v("amt", e, x) for e in _ENTRIES_0726
                      for x in ("death", "high_death", "k_down80", "climax", "k_down50")])
_set("d0726_choch", [_v("amt", e, x) for e in _ENTRIES_0726 for x in ("lower_high", "high_death")])
# 同日 CHoCH 校準另兩欄（依進場、路徑相依，定義見 kd_variants.PATH_EXITS）：
# 文章版 reference 只有黃金交叉一列（無 look-ahead 版＋重現舊數字的 lab ZigZag 版）；公開稀疏版 6 進場全列
_set("choch_calib", [_v("amt", (), "choch_article"), _v("amt", (), "choch_article_lab")]
     + [_v("amt", e, "choch_sparse") for e in _ENTRIES_0726])
# ── 2026-07-28 整合矩陣 8 進（PF 前 6 ＋ 錨點低檔 opt1、黃金交叉 opt6）× 6 出 ＝ 文章（五）──
_ENTRIES_0728 = ("breakout250", "breakout120", "gap", "low_redk", "breakout60", "divergence",
                 "low_zone", ())
_EXITS_0728 = ("death", "high_death", "k_down80", "climax", "k_down50", "lower_high")
_set("d0728", [_v("amt", e, x) for e in _ENTRIES_0728 for x in _EXITS_0728])
# ── 文章（一）黃金交叉、（四）出場的優化 的表 ──
_set("a_golden", [_v("raw"), _v("amt")])
_set("a_exit", [_v("amt")] + [_v("amt", (), x) for x in (
    "lower_high", "k_down80", "climax", "top_div", "high_death", ("k_down50", "k_down80"))])

# ── 對外集合（--set）──
_set("matrix", SETS["d0728"] + SETS["d0726_matrix"])
_set("entries", SETS["d0708"] + SETS["d0720_entries"] + SETS["d0726_ma20obv"])
_set("exits", SETS["d0709"] + SETS["a_exit"] + SETS["d0720_exits"] + SETS["d0726_choch"])
_set("liquidity", SETS["d0705"] + SETS["d0710"] + SETS["a_golden"])
_set("zone", SETS["d0714"] + SETS["d0720_zone"])
_set("all", list(VARIANTS))


class KdVariantShared(KdVariant):
    """df 由 worker 備妥（已 KdVariant.add_columns），這裡不再重算。"""

    def add_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


def make_variant(name: str) -> KdVariantShared:
    spec = VARIANTS[name]
    v = KdVariantShared()
    v.ENTRY, v.EXIT = spec["entry"], spec["exit"]
    v.LIQUIDITY, turnover = _LIQ[spec["liq"]]
    if turnover is not None:
        v.TURNOVER = turnover
    if spec["zone"] is not None:
        v.OVERSOLD, v.OVERBOUGHT = spec["zone"]
    return v


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """單檔全史備欄（所有變體共用）。"""
    common.ensure_columns(df)
    return KdVariant().add_columns(df.copy())


def run_group(names: list, folder: str, limit: int, out_root: str) -> str:
    """一組變體：逐檔讀全史、備欄一次，組內變體共用；結果依變體落地。"""
    t0 = time.time()
    variants = {n: make_variant(n) for n in names}
    rows = {n: [] for n in names}
    trades = {n: [] for n in names}
    n_stock = 0
    for sid, df in common.iter_market(folder, limit=limit, exclude=GLITCH):
        if len(df.loc[DEFAULT_START:DEFAULT_END]) < 2:
            continue
        df = prepare(df)
        n_stock += 1
        for n, v in variants.items():
            res = v.run(df, sid, start=DEFAULT_START, end=DEFAULT_END)
            rows[n].append({"stock_id": sid, **res["summary"]})
            if not res["trades"].empty:
                trades[n].append(res["trades"])
    for n in names:
        tr = pd.concat(trades[n], ignore_index=True) if trades[n] else pd.DataFrame()
        result = {"per_stock": pd.DataFrame(rows[n]), "trades": tr,
                  "aggregate": batch._aggregate(tr, n_stocks=len(rows[n]), n_failed=0)}
        out_dir = os.path.join(out_root, n)
        batch.write_results(result, out_dir, LABEL)
        tr.to_parquet(os.path.join(out_dir, f"{LABEL}_trades.parquet"), index=False)
    return f"{len(names)} 變體 × {n_stock} 檔｜{time.time() - t0:.0f} 秒"


def _split(names: list, k: int) -> list:
    """輪流分組（登錄順序相近的變體成本相近，輪流分可攤平各組負擔）。"""
    groups = [names[i::k] for i in range(k)]
    return [g for g in groups if g]


def main() -> int:
    ap = argparse.ArgumentParser(description="KD 交叉：文章與 reference 全變體掃描")
    ap.add_argument("--set", nargs="+", default=["all"], choices=sorted(SETS),
                    help="要跑的變體集合（可多個，取聯集）")
    ap.add_argument("--variants", nargs="+", default=None, choices=list(VARIANTS),
                    help="直接指定變體名稱（給了就忽略 --set）")
    ap.add_argument("--folder", default=common.DATA_DIR)
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 檔（冒煙用，請搭配 --out）")
    ap.add_argument("--out", default=os.path.join(RESULT_DIR, "single_kd"),
                    help="輸出根目錄（預設正式結果 result/single_kd/；冒煙請改暫存目錄）")
    ap.add_argument("--workers", type=int, default=8, help="平行子進程數（變體分組）")
    ap.add_argument("--list", action="store_true", help="只列出選到的變體，不執行")
    a = ap.parse_args()

    names = a.variants or list(dict.fromkeys(n for s in a.set for n in SETS[s]))
    if a.list:
        for n in names:
            print(n)
        print(f"共 {len(names)} 變體")
        return 0

    print(f"{len(names)} 變體｜{DEFAULT_START}~{DEFAULT_END}｜輸出 {a.out}", flush=True)
    t0 = time.time()
    groups = _split(names, max(1, a.workers))
    with ProcessPoolExecutor(max_workers=len(groups)) as pool:
        futs = [pool.submit(run_group, g, a.folder, a.limit, a.out) for g in groups]
        for f in futs:
            print(f.result(), flush=True)
    print(f"全部完成｜{time.time() - t0:.0f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
