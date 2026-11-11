"""
雙底 進場 × 出場 組合掃描（vbt 單股）：進場 11（基本版＋#2～#11）× 出場 14（基本版＋附加 #12～#21＋替換 3 條）
× ZigZag 3%／6%／9% × 裸版／可成交＝924 次全量回測（2026-11-06 用戶定：10 × 13 直接配對都先跑一次再挑；
原訂 10 × 15，替換版標 ⚠️ 的頸線停利、自高點回落 20% 不跑）。

出場分兩類：
  add_<rule>：附加，① 高點下移 ② 跌破波段低點保留，新規則疊上去、先觸發者算（同 exit_filter_sweep）。
  rep_<rule>：替換，只用新規則、①② 不用（同 exit_filter_sweep --replace）。只收 atr／ma10／mtop
              （替換版未平倉率 ≤ 10% 的五條去掉 ⚠️ 兩條，見 blog/reference/double_bottom/2026-11-03 紀錄）。
  base：①②，不加任何規則。
組合寫法：DoubleBottomEntryVariant 設 ENTRY（進場），實例再設 EXIT_RULE／REPLACE（出場），
不動策略檔註解狀態，條件式與 entry_filter_sweep／exit_filter_sweep 用的是同一份。

分批與續跑（用戶 2026-11-06：記憶體不要占滿、慢慢跑、分批執行）：
  --entries 只跑指定的進場（一批一條）；已有 row.json 的組合直接跳過，所以中斷後重下同一行即可續跑。
  每組跑完寫 <tag>/row.json；--summary 只把現有 row.json 彙總成 _sweep_summary.csv，不跑回測。

錨點：base 進場 × add_/base 出場 應與 single_double_bottom_exit/ 一致；各進場 × base 出場 應與
single_double_bottom_entry/ 一致；base 進場 × rep_ 出場 應與 single_double_bottom_exit_replace/ 一致
（對帳由 --anchor 執行）。

執行：python _02_strategy/double_bottom_strategy/combo_sweep.py --entries base [--workers 4] [--limit N] [--out 目錄]
      python _02_strategy/double_bottom_strategy/combo_sweep.py --summary
      python _02_strategy/double_bottom_strategy/combo_sweep.py --anchor
輸出：result/single_double_bottom_combo/p<門檻>_<raw|liq1000>__<進場>__<出場>/、_sweep_summary.csv
"""
import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd

from _02_strategy.base.vbt import batch, common
from _02_strategy.base.vbt.common import DEFAULT_END, DEFAULT_START, GLITCH
from _02_strategy.double_bottom_strategy.double_bottom_variants import ENTRY_FILTERS, EXIT_RULES, make_variant
from _02_strategy.double_bottom_strategy.single_double_bottom_strategy import RESULT_DIR

PCTS = (0.03, 0.06, 0.09)
SUB = "single_double_bottom_combo"
REPLACE_KEPT = ("atr", "ma10", "mtop")   # 替換版未平倉率 ≤ 10% 的五條，再去掉 ⚠️ 頸線停利、自高點回落 20%（用戶 2026-11-06：10 × 13，有疑慮的兩條不跑）
EXITS = ("base",) + tuple(f"add_{r}" for r in EXIT_RULES if r != "base") + tuple(f"rep_{r}" for r in REPLACE_KEPT)
# 對帳比對的欄位（通用 10 欄的來源欄＋診斷）
ANCHOR_COLS = ("交易次數", "勝率(%)", "平均持有天數", "平均獲利報酬率(%)", "平均虧損報酬率(%)", "中位數報酬率(%)",
               "期望報酬值(EV)", "獲利因子(PF)", "總獲利", "entries", "exits", "open_at_end")


def make_combo(entry: str, exit_key: str, pct: float, liq: bool):
    """進場＝ENTRY_FILTERS[entry]；出場＝exit_key（base／add_<rule>／rep_<rule>）。"""
    v = make_variant(entry, pct, liq)
    if exit_key == "base":
        v.EXIT_RULE, v.REPLACE = None, False
    else:
        kind, rule = exit_key.split("_", 1)
        if kind not in ("add", "rep") or rule not in EXIT_RULES:
            raise ValueError(f"未知的出場 {exit_key}")
        v.EXIT_RULE, v.REPLACE = rule, kind == "rep"
    return v


def _tag(pct, liq, entry, exit_key) -> str:
    return f"p{int(round(pct * 100))}_{'liq1000' if liq else 'raw'}__{entry}__{exit_key}"


def _job(args):
    pct, liq, entry, exit_key, limit, out_root = args
    t0 = time.time()
    strat = make_combo(entry, exit_key, pct, liq)
    result = batch.run_folder(strat, common.DATA_DIR, start=DEFAULT_START, end=DEFAULT_END,
                              limit=limit, exclude=GLITCH)
    tag = _tag(pct, liq, entry, exit_key)
    # 沒設 STOCK_DATA_DIR 時 DATA_DIR 指向空資料夾，會安靜地產出 0 檔 0 筆，直接中止、不寫 row.json
    if result["aggregate"].get("參與股票數", 0) == 0:
        raise RuntimeError(f"{tag}：0 檔股票參與，檢查 STOCK_DATA_DIR（目前 {common.DATA_DIR}）")
    out_dir = os.path.join(out_root, SUB, tag)
    batch.write_results(result, out_dir, f"single_double_bottom_{tag}", write_trades=True)
    with open(os.path.join(out_dir, "diag.json"), "w", encoding="utf-8") as f:
        json.dump(strat.diag, f, ensure_ascii=False, indent=1)
    row = {"tag": tag, "門檻%": int(round(pct * 100)), "可成交": liq, "進場": entry, "出場": exit_key,
           **result["aggregate"], **strat.diag, "failed": len(result["failed"]), "秒": round(time.time() - t0)}
    # row.json 最後寫：有它＝這組完整跑完，續跑時據此跳過
    with open(os.path.join(out_dir, "row.json"), "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, default=float)
    return row


def write_summary(out_root: str) -> str:
    rows = []
    for p in glob.glob(os.path.join(out_root, SUB, "*", "row.json")):
        with open(p, encoding="utf-8") as f:
            rows.append(json.load(f))
    if not rows:
        raise SystemExit("沒有任何 row.json，先跑回測")
    df = pd.DataFrame(rows)
    df["生效率%"] = (df["exit_new"] / df["exits"] * 100).round(2)
    df["搶先率%"] = (df["exit_new_only"] / df["exits"] * 100).round(2)
    df["未平倉率%"] = (df["open_at_end"] / df["entries"] * 100).round(2)
    out = os.path.join(out_root, SUB, "_sweep_summary.csv")
    df.sort_values(["可成交", "門檻%", "進場", "出場"]).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"→ {out}（{len(df)} 組）")
    return out


def _load_old(path: str, key_col: str, key_map) -> pd.DataFrame:
    """讀舊批次 _sweep_summary，轉成 (門檻%, 可成交, 進場, 出場) 索引。"""
    old = pd.read_csv(path)
    old["進場"], old["出場"] = zip(*old[key_col].map(key_map))
    return old.set_index(["門檻%", "可成交", "進場", "出場"])


def anchor(out_root: str) -> int:
    """跟舊批次逐欄比對；舊檔沒有的欄位略過。回傳不一致數。"""
    new = pd.read_csv(os.path.join(out_root, SUB, "_sweep_summary.csv")).set_index(["門檻%", "可成交", "進場", "出場"])
    sd = os.path.join(RESULT_DIR)
    olds = [
        _load_old(os.path.join(sd, "single_double_bottom_exit", "_sweep_summary.csv"), "變體",
                  lambda r: ("base", "base" if r == "base" else f"add_{r}")),
        _load_old(os.path.join(sd, "single_double_bottom_exit_replace", "_sweep_summary.csv"), "變體",
                  lambda r: ("base", "base" if r == "base" else f"rep_{r}")),
    ]
    for name in ("_sweep_summary.csv", "_sweep_summary_add4.csv", "_sweep_summary_align_long.csv"):
        old = _load_old(os.path.join(sd, "single_double_bottom_entry", name), "變體", lambda r: (r, "base"))
        if name == "_sweep_summary.csv":
            # 這份的 align 是 2026-10-22 改定前的 MA5>MA20>MA60 舊定義，現行長線版在 _align_long
            old = old[old.index.get_level_values("進場") != "align"]
        olds.append(old)
    bad = checked = 0
    for old in olds:
        common_idx = new.index.intersection(old.index)
        for idx in common_idx:
            for c in ANCHOR_COLS:
                if c not in old.columns:
                    continue
                checked += 1
                a, b = new.at[idx, c], old.at[idx, c]
                if not (pd.isna(a) and pd.isna(b)) and abs(float(a) - float(b)) > 1e-9:
                    bad += 1
                    print(f"不一致 {idx} {c}: 新 {a} 舊 {b}")
    print(f"錨點比對 {checked} 格，不一致 {bad}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=RESULT_DIR)
    ap.add_argument("--entries", nargs="+", choices=list(ENTRY_FILTERS), help="這一批要跑的進場")
    ap.add_argument("--summary", action="store_true", help="只彙總現有 row.json")
    ap.add_argument("--anchor", action="store_true", help="彙總後與舊批次對帳")
    a = ap.parse_args()
    if a.summary or a.anchor:
        write_summary(a.out)
        return 1 if (a.anchor and anchor(a.out)) else 0
    if not a.entries:
        raise SystemExit("請用 --entries 指定這一批的進場（分批執行）")
    jobs = [(p, liq, e, x, a.limit, a.out) for e in a.entries for x in EXITS for p in PCTS for liq in (False, True)
            if not os.path.exists(os.path.join(a.out, SUB, _tag(p, liq, e, x), "row.json"))]
    print(f"本批 {len(jobs)} 組（已完成的跳過）", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(_job, j) for j in jobs]):
            r = f.result()
            print(f"{r['tag']}｜{r['交易次數']:,} 筆｜PF {r['獲利因子(PF)']}｜{r['秒']} 秒", flush=True)
    write_summary(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
