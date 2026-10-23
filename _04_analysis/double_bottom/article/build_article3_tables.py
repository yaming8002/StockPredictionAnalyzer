# -*- coding: utf-8 -*-
"""
產生〈雙底（三）：進場的優化（下）〉的表格 HTML（基本版總覽、5 條技術分析類條件各一張、最後的獲利因子總表）。
收 #5 站上年線、#6 均線多頭排列、#7 放量 1.5 倍、#10 第二底量縮、#11 確認日紅 K／長下影線（2026-10-22 用戶定）。

算法、位數、底色規則全部沿用 build_article2_tables（同一批 result/single_double_bottom_entry/ 輸出），
這裡只換條件清單。

執行：PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python _04_analysis/double_bottom/article/build_article3_tables.py
輸出：終端印各表 HTML；result/single_double_bottom_entry/_article3_stats.csv（verify 用）
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
for p in (_root, _here):
    if p not in sys.path:
        sys.path.insert(0, p)

from build_article2_tables import LABELS, SRC, stats, summary_table, table  # noqa: E402

ENTRIES = ("base", "ma200", "align", "volume", "volshrink", "bullbar")
STATS_FILE = os.path.join(SRC, "_article3_stats.csv")


def main() -> int:
    df = stats(ENTRIES)
    df.to_csv(STATS_FILE, index=False, encoding="utf-8-sig")
    for e in ENTRIES:
        print(f"=== {LABELS[e]} ===")
        print(table(df, e))
    print("=== 總表 ===")
    print(summary_table(df, ENTRIES))
    print(f"→ {STATS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
