"""
底部島型出場優化：讀 island_exit_sweep 的彙總 CSV，印成通用回測表 10 欄（markdown），給 reference 用。

每列附未平倉率、停損／主出場生效率，以及「剔除（未平倉 > 10%）」標記（判準見 island_exit_sweep.py）。
排序：基準 → 附加（持有 10／20 根）→ 替換；同組內依規則編號。

執行：python _04_analysis/island/exit_summary.py [CSV 路徑] [--raw]（--raw 只出裸版）
"""
import os
import sys

import pandas as pd

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CSV = os.path.join(_root, "_02_strategy", "island_strategy", "result",
                           "single_island_exit", "_sweep_summary.csv")

NAMES = {"none": "", "lowclose": "#1 跌破島內最低收盤", "islandlow": "#2 跌破島底", "gapfill": "#3 回補缺口",
         "time5": "#4 5 根沒漲過買價", "trail5": "#5 回落 5%", "trail10": "#6 回落 10%",
         "trail15": "#7 回落 15%", "trail20": "#8 回落 20%", "atr": "#9 ATR 3 倍", "ma10": "#10 跌破 MA10",
         "ma10switch": "#11 站上 MA10 後跌破出", "gapdown": "#12 向下跳空", "topisland": "#13 頂部島型"}
ORDER = list(NAMES)


def label(row) -> str:
    parts = [p for p in (NAMES[row["stop"]], NAMES[row["exit"]]) if p]
    if row["mode"] == "基準":
        return f"持有 {row['hold']} 根（基準）"
    if row["mode"] == "附加":
        return f"持有 {row['hold']} 根 ＋ {' ＋ '.join(parts)}"
    return "替換：" + " ＋ ".join(parts)


def table(df: pd.DataFrame) -> str:
    head = ("| 版本 | 交易次數 | 勝率% | 平均持有天 | 獲利平均% | 虧損平均% | 中位數% | 期望值 | PF | 總獲利(萬) "
            "| 未平倉% | 停損生效% | 主出場生效% | 剔除 |\n|" + "---|" * 14)
    lines = [head]
    for _, r in df.iterrows():
        lines.append(
            f"| {label(r)} | {r['交易次數']:,} | {r['勝率(%)']:.2f} | {r['平均持有天數']:.1f} "
            f"| {r['平均獲利報酬率(%)']:.2f} | {r['平均虧損報酬率(%)']:.2f} | {r['中位數報酬率(%)']:.2f} "
            f"| {r['期望報酬值(EV)']:.1f} | {r['獲利因子(PF)']:.3f} | {r['總獲利'] / 1e4:,.1f} "
            f"| {r['未平倉率(%)']:.2f} | {r['停損生效率(%)']:.1f} | {r['主出場生效率(%)']:.1f} "
            f"| {'✗' if r['剔除(未平倉>10%)'] else ''} |")
    return "\n".join(lines)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    raw_only = "--raw" in sys.argv                 # 只出裸版（用戶 2026-11-17：先不做可成交的判斷）
    df = pd.read_csv(args[0] if args else DEFAULT_CSV)
    if raw_only:                                    # 剔除只看裸版自己的未平倉率
        df["剔除(未平倉>10%)"] = df["未平倉率(%)"] > 10
    df["_m"] = df["mode"].map({"基準": 0, "附加": 1, "替換": 2})
    df["_h"] = pd.to_numeric(df["hold"], errors="coerce").fillna(0)
    df["_s"] = df["stop"].map(ORDER.index)
    df["_x"] = df["exit"].map(ORDER.index)
    df = df.sort_values(["_m", "_h", "_s", "_x"])
    groups = ((False, "裸版"),) if raw_only else ((False, "裸版"), (True, "可成交（成交金額 > 1,000 萬）"))
    for liq, name in groups:
        print(f"\n### {name}\n")
        print(table(df[df["liq"] == liq]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
