"""
雙均線交叉：全變體 × 21 組均線掃描（單股回測段）
=================================================
文章（均線交叉系列）與 reference 的對照表都是「同一批股票、換不同旗標跑 21 組」。
`ma_cross_strategy.py` 的 CLI 一次只跑一組一個變體，這裡把「讀檔 ＋ 算指標」對每一組
均線只做一次，所有變體共用同一份備妥的 DataFrame。

變體＝`MACrossStrategy` 的旗標組合（策略檔是唯一定義，這裡只設旗標、不重寫條件）。
交易區間＝標準區間 2002～2025，指標在全史上算（起日前當暖身，見 single.py）。

輸出（result/ 不進版控）：
  result/ma_cross/<變體>/ma_cross_<短>_<長>_per_stock.csv、_aggregate.csv   與 CLI 同格式
  result/ma_cross/<變體>/ma_cross_<短>_<長>_trades.parquet                    逐筆交易（蒙地卡羅等分析段用）

執行：
  python _02_strategy/ma_strategy/ma_cross_sweep.py [--pairs 50/200 60/200] [--variants baseline liq1000]
                                                    [--set article|regen|all] [--limit N --out <暫存目錄>]
  重產區（REGEN_VARIANTS）：2026-10 補回只剩舊數字的變體（舊 CSV 10 個、無狀態 CHoCH、強K 跳空過長均），
  `--set regen` 只跑這批；部分變體只跑舊 CSV 有的組合（VARIANT_PAIRS）。
"""
import argparse
import itertools
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
from _02_strategy.kd_strategy.kd_variants import _lower_high_stateless  # noqa: E402
from _02_strategy.ma_strategy.ma_cross_strategy import RESULT_DIR, MACrossStrategy  # noqa: E402

PERIODS = (5, 10, 20, 50, 60, 120, 200)
PAIRS = [f"{s}/{l}" for s, l in itertools.combinations(PERIODS, 2)]   # 21 組

_LIQ = {"MIN_VOL_ZHANG": 1000}
# 變體名 → 旗標。名稱沿用 result/ma_cross/ 既有資料夾名，reference 與文章對照表才找得到
VARIANTS = {
    # 文章（一）（二）（三）＋ reference 2026-06-17 系列
    "baseline": {},
    "liq1000": _LIQ,
    "liq300": {"MIN_VOL_ZHANG": 300},
    "strongk_liq1000": {"STRONGK": True, **_LIQ},
    "strongk_liq300": {"STRONGK": True, "MIN_VOL_ZHANG": 300},
    "confirm_liq1000": {"CONFIRM": True, **_LIQ},
    "vol_adx25_liq1000": {"VOL_ADX": 25.0, **_LIQ},
    "align_liq1000": {"ALIGN": True, **_LIQ},
    "angle20_liq1000": {"ANGLE_DEG": 20.0, **_LIQ},
    "exit_below_short_liq1000": {"EXIT_BELOW_SHORT": True, **_LIQ},
    # 文章（四）挑代表配置用的 6 種配置（皆 +1000 張）：liq／strongk／angle20／adx25 已在上面
    "strongk_adx25_liq1000": {"STRONGK": True, "VOL_ADX": 25.0, **_LIQ},
    "angle20_adx25_liq1000": {"ANGLE_DEG": 20.0, "VOL_ADX": 25.0, **_LIQ},
    "strongk_angle20_adx25_liq1000": {"STRONGK": True, "ANGLE_DEG": 20.0, "VOL_ADX": 25.0, **_LIQ},
    # reference 2026-06-17／06-18 的無門檻（raw）變體
    "vol_adx25": {"VOL_ADX": 25.0},
    "vol_adx20": {"VOL_ADX": 20.0},
    "confirm": {"CONFIRM": True},
    "angle3": {"ANGLE_DAYS": 3},
    "angle2": {"ANGLE_DAYS": 2},
    "strongk": {"STRONGK": True},
    "vol_both": {"VOL_BOTH": True},
    "choch": {"CHOCH": True},
}
ARTICLE_VARIANTS = list(VARIANTS)

# ── 重產區（2026-10-09）：reference 只剩 2001 起點舊數字、原 driver 已遺失的變體 ─────────
# 旗標定義沿用策略檔（#8 CHOCH、#9 EXIT_BELOW_SHORT、#10 DIVERGE、#11 ANGLE_DEG 自加入後條件式未改，
# 見 git 443f9f3／2a3c7de／9240320 對照現行版）；DIVERGE_MARGIN／SLOPE_WIN 舊 CSV 沒記，取 CLI 預設 0／5。
_LIQ300 = {"MIN_VOL_ZHANG": 300}
REGEN_VARIANTS = {
    # blog/reference/ma_cross/_full_aggregate_all_variants.csv 的 10 個舊變體
    "angle20": {"ANGLE_DEG": 20.0},
    "angle20_liq300": {"ANGLE_DEG": 20.0, **_LIQ300},
    "angle3_liq1000": {"ANGLE_DAYS": 3, **_LIQ},
    "choch_liq1000": {"CHOCH": True, **_LIQ},
    "diverge": {"DIVERGE": True},                                   # 夾角擴大進場＋收斂或死叉出場
    "diverge_entry": {"DIVERGE": True, "DIVERGE_EXIT": False},      # --diverge-entry-only
    "diverge_entry_liq1000": {"DIVERGE": True, "DIVERGE_EXIT": False, **_LIQ},
    "exit_below_short": {"EXIT_BELOW_SHORT": True},
    "vol_adx20_liq1000": {"VOL_ADX": 20.0, **_LIQ},
    "vol_both_liq1000": {"VOL_BOTH": True, **_LIQ},
    # 2026-07-27_ma_cross_stateless_choch.md：黃金交叉進場、只用無狀態盤勢 CHoCH 出場（raw）
    "choch_stateless": {"CHOCH_STATELESS": True},
    # 2026-06-17 task3-4「跳空基準變更」表的 old 欄：強 K 跳空過「長均」（2026-06-26 前定義，raw）
    "strongk_gaplong": {"STRONGK": True, "GAP_BASE": "long"},
}
VARIANTS.update(REGEN_VARIANTS)
# 舊 CSV 只跑了部分組合的變體（其餘全 21 組）
VARIANT_PAIRS = {
    "angle20": ["50/120", "50/200", "60/120", "60/200", "120/200"],
    "angle20_liq300": ["50/120", "50/200", "60/120", "60/200", "120/200"],
    "diverge": ["50/200", "60/200", "120/200"],
}
SETS = {"article": ARTICLE_VARIANTS, "regen": list(REGEN_VARIANTS), "all": list(VARIANTS)}


class MACrossVariant(MACrossStrategy):
    """df 由 run_pair 備妥（已 add_columns，含 ZigZag），這裡不再重算。"""

    # 只在重產變體用的旗標（策略檔沒有，留在 driver 端，不動策略檔）
    CHOCH_STATELESS = False   # 出場只用無狀態盤勢 CHoCH（市場 ZigZag 出現 lower-high 就賣，不依進場、不疊死叉）
    GAP_BASE = "short"        # 強 K 跳空基準：short＝現行（過短均）／long＝2026-06-26 前舊定義（過長均）

    def add_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def buy_signal(self, df: pd.DataFrame) -> pd.Series:
        if self.STRONGK and self.GAP_BASE == "long":
            # 舊定義原文（git 189e4d8 ma_cross_strategy.py）：
            #   df["gap_over_long"] = (df["open"] > df["ma_long"]) & (df["open"] > df["close"].shift(1))
            # 策略檔 buy_signal 讀 gap_over_short，這裡換成舊欄再交給策略檔判斷（其餘條件不重寫）
            gap_long = (df["open"] > df["ma_long"]) & (df["open"] > df["close"].shift(1))
            df = df.assign(gap_over_short=gap_long)
        return super().buy_signal(df)

    def sell_signal(self, df: pd.DataFrame) -> pd.Series:
        if self.CHOCH_STATELESS:
            # 與 KD 頂頂低同一份定義（kd_variants._lower_high_stateless，照抄文章 KDExitLowerHigh）。
            # reference 表的平均持有天數各組都在 34~43 天、不隨均線週期變（5/10 baseline 只 13 天），
            # 判定為「只用 CHoCH 出場、不疊死叉」。
            return _lower_high_stateless(df["zigzag_turn_high"])
        return super().sell_signal(df)


def make_variant(name: str, short: int, long: int) -> MACrossVariant:
    v = MACrossVariant()
    v.SHORT_MA, v.LONG_MA = short, long
    for k, val in VARIANTS[name].items():
        if not hasattr(MACrossVariant, k):
            raise AttributeError(f"MACrossVariant 沒有旗標 {k}")
        setattr(v, k, val)
    return v


def run_pair(pair: str, names: list, folder: str, limit: int,
             out_root: str = os.path.join(RESULT_DIR, "ma_cross")) -> str:
    """一組均線：逐檔讀全史、算一次指標，所有變體共用；結果依變體落地。"""
    t0 = time.time()
    short, long = (int(x) for x in pair.split("/"))
    names = [n for n in names if pair in VARIANT_PAIRS.get(n, PAIRS)]
    if not names:
        return f"{pair}｜無變體"
    prep = MACrossStrategy()
    prep.SHORT_MA, prep.LONG_MA, prep.CHOCH = short, long, True   # CHOCH=True 才會補算 ZigZag
    variants = {n: make_variant(n, short, long) for n in names}
    rows = {n: [] for n in names}
    trades = {n: [] for n in names}
    for sid, df in common.iter_market(folder, limit=limit, exclude=GLITCH):
        if len(df.loc[DEFAULT_START:DEFAULT_END]) < 2:
            continue
        common.ensure_columns(df)
        df = prep.add_columns(df.copy())
        for n, v in variants.items():
            res = v.run(df, sid, start=DEFAULT_START, end=DEFAULT_END)
            rows[n].append({"stock_id": sid, **res["summary"]})
            if not res["trades"].empty:
                trades[n].append(res["trades"])
    label = f"ma_cross_{short}_{long}"
    for n in names:
        tr = pd.concat(trades[n], ignore_index=True) if trades[n] else pd.DataFrame()
        result = {"per_stock": pd.DataFrame(rows[n]), "trades": tr,
                  "aggregate": batch._aggregate(tr, n_stocks=len(rows[n]), n_failed=0)}
        out_dir = os.path.join(out_root, n)
        batch.write_results(result, out_dir, label)
        tr.to_parquet(os.path.join(out_dir, f"{label}_trades.parquet"), index=False)
    return f"{pair}｜{len(rows[names[0]])} 檔 × {len(names)} 變體｜{time.time() - t0:.0f} 秒"


def main() -> int:
    ap = argparse.ArgumentParser(description="雙均線交叉：全變體 × 21 組掃描")
    ap.add_argument("--pairs", nargs="+", default=PAIRS)
    ap.add_argument("--set", default="article", choices=list(SETS),
                    help="變體集合：article＝文章／reference 現行（預設）、regen＝2026-10 重產的舊變體、all")
    ap.add_argument("--variants", nargs="+", default=None, choices=list(VARIANTS),
                    help="直接指定變體（給了就忽略 --set）")
    ap.add_argument("--folder", default=common.DATA_DIR)
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 檔（冒煙用，請搭配 --out）")
    ap.add_argument("--out", default=os.path.join(RESULT_DIR, "ma_cross"),
                    help="輸出根目錄（預設正式結果 result/ma_cross/；冒煙請改暫存目錄）")
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()

    names = a.variants or SETS[a.set]
    print(f"{len(names)} 變體 × {len(a.pairs)} 組｜{DEFAULT_START}~{DEFAULT_END}｜輸出 {a.out}", flush=True)
    with ProcessPoolExecutor(max_workers=min(a.workers, len(a.pairs))) as pool:
        futs = [pool.submit(run_pair, p, names, a.folder, a.limit, a.out) for p in a.pairs]
        for f in futs:
            print(f.result(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
