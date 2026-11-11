# -*- coding: utf-8 -*-
"""
產生〈雙底（四）：出場的優化（上）〉的表格 HTML（基本版總覽、5 條型態類出場各一張）。
收 #12 測幅滿足點、#13 打折測幅 0.65、#14 頸線停利、#15 突破後跌回頸線、#16 跌破突破 K 棒低點；
#17～#21 在〈雙底（五）〉（build_article5_tables.py，共用本檔 stats）。

資料來源＝SPA 單股回測輸出（`_02_strategy/double_bottom_strategy/exit_filter_sweep.py`）：
  result/single_double_bottom_exit/p<3|6|9>_<raw|liq1000>__<base|target|...>/
每組讀逐筆 trades 重算規格 9 欄精確值；位數、底色、表格 HTML 全部沿用 build_article2_tables。
出場的診斷跟進場不同（讀 diag.json）：
  生效率% ＝ exit_new ÷ exits（出場那根新規則有成立，含與原本出場同根）
  搶先率% ＝ exit_new_only ÷ exits（出場那根只有新規則成立）
  未平倉率% ＝ open_at_end ÷ entries

執行：PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python _04_analysis/double_bottom/article/build_article4_tables.py
輸出：終端印各表 HTML；result/single_double_bottom_exit/_article4_stats.csv（verify 用）
"""
import json
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
for p in (_root, _here):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd  # noqa: E402

from _02_strategy.double_bottom_strategy.single_double_bottom_strategy import RESULT_DIR  # noqa: E402
from _03_multi_strategy.base.fast_multi import EXACT, exact_stats  # noqa: E402
from build_article2_tables import COLS, LIQS, PCTS, table  # noqa: E402

SRC = os.path.join(RESULT_DIR, "single_double_bottom_exit")
STATS_FILE = os.path.join(SRC, "_article4_stats.csv")
EXITS = ("base", "target", "target65", "neck", "neckfail", "bolow")


def stats(exits=EXITS, src: str = SRC) -> pd.DataFrame:
    """exits 讓〈雙底（五）〉的 build_article5_tables 共用同一套算法（必含 base，底色要對它比）；
    src 讓〈雙底（六）〉改讀替換版的 single_double_bottom_exit_replace/。"""
    rows = []
    for liq in LIQS:
        for p in PCTS:
            for e in exits:
                tag = f"p{p}_{'liq1000' if liq else 'raw'}__{e}"
                folder = os.path.join(src, tag)
                tr = pd.read_csv(os.path.join(folder, f"single_double_bottom_{tag}_trades.csv"))
                ex = exact_stats(tr)
                with open(os.path.join(folder, "diag.json"), encoding="utf-8") as f:
                    dg = json.load(f)
                n = int((pd.to_numeric(tr["real_pnl"], errors="coerce").fillna(0) != 0).sum())
                rows.append({"門檻%": p, "可成交": liq, "變體": e, "交易次數": n,
                             **{c: ex[c + EXACT] for c in COLS if c != "交易次數"},
                             "exits": dg["exits"], "entries": dg["entries"],
                             "生效率%": dg["exit_new"] / dg["exits"] * 100,
                             "搶先率%": dg["exit_new_only"] / dg["exits"] * 100,
                             "未平倉率%": dg["open_at_end"] / dg["entries"] * 100})
    return pd.DataFrame(rows)


LABELS = {"base": "基本版", "target": "測幅滿足點", "target65": "打折測幅 0.65", "neck": "頸線停利",
          "neckfail": "突破後跌回頸線", "bolow": "跌破突破 K 棒低點",
          # 〈雙底（五）〉用
          "breakeven": "保本停損", "atr": "ATR 移動停損", "trail20": "自高點回落 20%",
          "ma10": "跌破 MA10", "mtop": "高檔 M 頭"}


def main() -> int:
    df = stats()
    df.to_csv(STATS_FILE, index=False, encoding="utf-8-sig")
    for e in EXITS:
        print(f"=== {LABELS[e]} ===")
        print(table(df, e))
    print(f"→ {STATS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
