"""
匯出期末未平倉部位：雙底出場（附錄：替換）在區間結束時還沒賣出的部位
==================================================================
逐筆 trades.csv 只收已平倉的交易，期末還抱著的部位不在裡面；替換型出場沒有停損時這類部位很多
（未平倉率見 exit_filter_sweep 的 diag）。這裡逐檔重跑同一套訊號，把「區間內最後一筆進場之後
區間內沒有出場」的部位列出來：買進成交日、買進價（成交日開盤）、區間最後一天收盤、帳面報酬、持有日曆天。
帳面報酬只看價格、不含費稅（部位還沒賣出）。

執行：python _02_strategy/double_bottom_strategy/exit_open_positions.py [--pct 9] [--liq] [--exits target ...] [--workers 4]
輸出：result/single_double_bottom_exit_replace/p<門檻>_<raw|liq1000>__<變體>/open_positions.csv
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd

from _02_strategy.base.vbt import common
from _02_strategy.base.vbt.common import DEFAULT_END, DEFAULT_START, GLITCH
from _02_strategy.double_bottom_strategy.double_bottom_variants import EXIT_RULES, make_exit_variant
from _02_strategy.double_bottom_strategy.exit_filter_sweep import SUB_REPLACE
from _02_strategy.double_bottom_strategy.single_double_bottom_strategy import RESULT_DIR


def _open_positions(args) -> str:
    rule, pct, liq = args
    v = make_exit_variant(rule, pct / 100, liq, replace=True)
    rows = []
    for sid, raw in common.iter_market(common.DATA_DIR, exclude=GLITCH):
        common.ensure_columns(raw)
        df, entries, exits = v.window_signals(v.add_columns(raw.copy()), DEFAULT_START, DEFAULT_END)
        if not entries.any():
            continue
        last = entries[entries].index[-1]
        if exits.loc[last:].any():            # 最後一筆進場之後有出場＝已平倉
            continue
        buy_px = float(df.loc[last, "open"])
        end_close = float(df["close"].iloc[-1])
        rows.append({"stock_id": sid, "buy_date": last.date(), "buy_price": buy_px,
                     "end_date": df.index[-1].date(), "end_close": end_close,
                     "unrealized_pct": (end_close / buy_px - 1) * 100,
                     "held_days": (df.index[-1] - last).days})
    tag = f"p{pct}_{'liq1000' if liq else 'raw'}__{rule}"
    out = os.path.join(RESULT_DIR, SUB_REPLACE, tag, "open_positions.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    # 與 diag.json 的 open_at_end 同一定義，數目必須相同
    if len(rows) != v.diag["open_at_end"]:
        raise RuntimeError(f"{tag}：匯出 {len(rows)} 筆 ≠ diag open_at_end {v.diag['open_at_end']}")
    return f"{tag}：{len(rows):,} 筆未平倉 → {out}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=int, default=9, choices=(3, 6, 9))
    ap.add_argument("--liq", action="store_true", help="可成交 1,000 萬版")
    ap.add_argument("--exits", nargs="+", default=["target"], choices=[e for e in EXIT_RULES if e != "base"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for msg in pool.map(_open_positions, [(e, a.pct, a.liq) for e in a.exits]):
            print(msg, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
