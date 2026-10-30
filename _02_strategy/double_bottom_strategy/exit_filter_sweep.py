"""
雙底出場優化掃描（vbt 單股）：ZigZag 3%／6%／9% × (基本版＋10 條附加出場) × 裸版／可成交 1,000 萬＝66 組
==================================================================================================
進場固定基本版（第二個底確認且高於第一個底）。每條出場單獨「附加」在基本版出場（① 高點下移 或
② 收盤跌破最近確認波段低點）上，先觸發者算；規則寫在策略檔 ExtraExit（# 優化 #12～#21），
變體由 double_bottom_variants.make_exit_variant 設 EXIT_RULE。基本版同批次重跑，當同一批的比較基準。

診斷（diag.json）：exits＝區間內出場數；exit_new＝出場那根新規則成立（含與 ①② 同根）；
exit_new_only＝只有新規則成立（搶在 ①② 之前）。
  生效率% ＝ exit_new ÷ exits；搶先率% ＝ exit_new_only ÷ exits；未平倉率% ＝ open_at_end ÷ entries。

執行：python _02_strategy/double_bottom_strategy/exit_filter_sweep.py [--workers 4] [--limit N] [--out <根目錄>]
      [--exits base target ...] [--summary-name _sweep_summary.csv]
輸出：result/single_double_bottom_exit/p<門檻>_<raw|liq1000>__<變體>/、_sweep_summary.csv
"""
import argparse
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
from _02_strategy.double_bottom_strategy.double_bottom_variants import EXIT_RULES, make_exit_variant
from _02_strategy.double_bottom_strategy.single_double_bottom_strategy import RESULT_DIR

PCTS = (0.03, 0.06, 0.09)
SUB = "single_double_bottom_exit"


def _job(args):
    pct, liq, rule, limit, out_root = args
    t0 = time.time()
    strat = make_exit_variant(rule, pct, liq)
    result = batch.run_folder(strat, common.DATA_DIR, start=DEFAULT_START, end=DEFAULT_END,
                              limit=limit, exclude=GLITCH)
    tag = f"p{int(round(pct * 100))}_{'liq1000' if liq else 'raw'}__{rule}"
    out_dir = os.path.join(out_root, SUB, tag)
    batch.write_results(result, out_dir, f"single_double_bottom_{tag}", write_trades=True)
    with open(os.path.join(out_dir, "diag.json"), "w", encoding="utf-8") as f:
        json.dump(strat.diag, f, ensure_ascii=False, indent=1)
    return {"tag": tag, "門檻%": int(round(pct * 100)), "可成交": liq, "變體": rule,
            **result["aggregate"], **strat.diag, "failed": len(result["failed"]),
            "秒": round(time.time() - t0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=RESULT_DIR)
    ap.add_argument("--exits", nargs="+", default=list(EXIT_RULES), choices=list(EXIT_RULES),
                    help="只跑這幾個變體（預設全部；base 一定要在，當同批次基準）")
    ap.add_argument("--summary-name", default="_sweep_summary.csv")
    a = ap.parse_args()
    if "base" not in a.exits:
        raise SystemExit("--exits 必須包含 base（同批次基準）")
    jobs = [(p, liq, r, a.limit, a.out) for p in PCTS for liq in (False, True) for r in a.exits]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(_job, j) for j in jobs]):
            r = f.result()
            rows.append(r)
            print(f"{r['tag']}｜{r['交易次數']:,} 筆｜PF {r['獲利因子(PF)']}｜新規則出場 {r['exit_new']:,}"
                  f"／{r['exits']:,}｜{r['秒']} 秒", flush=True)
    df = pd.DataFrame(rows)
    df["生效率%"] = (df["exit_new"] / df["exits"] * 100).round(2)
    df["搶先率%"] = (df["exit_new_only"] / df["exits"] * 100).round(2)
    df["未平倉率%"] = (df["open_at_end"] / df["entries"] * 100).round(2)
    out = os.path.join(a.out, SUB, a.summary_name)
    df.sort_values(["可成交", "門檻%", "變體"]).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
