"""
截斷測試：雙底出場優化（#12～#21）不得用到判定日之後的資料
==========================================================
每檔股票、每個門檻、每條出場：同一套程式分別吃「全史」與「截到某日的資料」，跑完整的逐根掃描
（build_signals，進出場都是路徑相依），兩邊在截斷日以前的進場與出場成交日必須逐日相同；
任何一天不同＝用到了截斷日之後的資訊。進場一律基本版＋可成交門檻。

執行：python _02_strategy/double_bottom_strategy/verify_exit_no_lookahead.py [--limit 200] [--exits base target ...]
"""
import argparse
import os
import sys
import time

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from _02_strategy.base.vbt import common
from _02_strategy.base.vbt.common import GLITCH
from _02_strategy.double_bottom_strategy.double_bottom_variants import EXIT_RULES, make_exit_variant

CUTOFFS = ("2008-06-30", "2015-12-31", "2020-03-31")
PCTS = (0.03, 0.06, 0.09)


def _signals(v, df):
    """回傳 (進場成交日, 出場成交日) 兩條 bool Series；交易區間不設限（全段都可進場）。"""
    return v.build_signals(v.add_columns(df.copy()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--exits", nargs="+", default=list(EXIT_RULES), choices=list(EXIT_RULES))
    a = ap.parse_args()
    t0 = time.time()
    bad, checks, exit_days, n_stock = 0, 0, 0, 0
    for sid, raw in common.iter_market(common.DATA_DIR, limit=a.limit, exclude=GLITCH):
        if len(raw) < 300:
            continue
        n_stock += 1
        common.ensure_columns(raw)
        for pct in PCTS:
            for rule in a.exits:
                v = make_exit_variant(rule, pct, liquidity=True)
                full_e, full_x = _signals(v, raw)
                for cut in CUTOFFS:
                    part = raw.loc[:cut]
                    if len(part) < 50:
                        continue
                    sub_e, sub_x = _signals(v, part)
                    checks += 1
                    exit_days += int(sub_x.sum())
                    if not (sub_e.equals(full_e.loc[:cut]) and sub_x.equals(full_x.loc[:cut])):
                        bad += 1
                        diff = sub_x.index[(sub_x != full_x.loc[:cut]) | (sub_e != full_e.loc[:cut])]
                        print(f"NG {sid} {pct} {rule} 截到 {cut}：{len(diff)} 天不同，首日 {diff[0].date()}")
    print(f"{n_stock} 檔 × {len(PCTS)} 門檻 × {len(a.exits)} 出場 × {len(CUTOFFS)} 截斷日："
          f"比對 {checks:,} 次（截斷段內出場 {exit_days:,} 筆）｜不一致 {bad}｜{time.time() - t0:.0f} 秒")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
