"""
底部島型進場優化：讀 island_entry_sweep 的彙總 CSV，依出場分組印成通用回測表 10 欄（markdown），給 reference 用。

每組先列基準，再列 20 條進場條件（依優化編號）；每列附生效率（被擋掉的進場判定比例）、未平倉率，
以及「vs 基準 PF」差值。最後印跨出場摘要：每條條件在 5 種出場中 PF 高於基準的次數（裸版／可成交）。

執行：python _04_analysis/island/entry_summary.py [CSV 路徑] [--raw]（--raw 只出裸版）
"""
import os
import sys

import pandas as pd

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CSV = os.path.join(_root, "_02_strategy", "island_strategy", "result",
                           "single_island_entry", "_sweep_summary.csv")

ENTRY_NAMES = {"base": "基準（島型成立就買）", "wk_up": "#14 週 MA5 > 週 MA20", "wk_down": "#15 週 MA5 < 週 MA20",
               "wk_above20": "#16 週收盤 > 週 MA20", "wk_ma20_up": "#17 週 MA20 上揚",
               "wk_cross4": "#18 週 MA5 4 週內上穿週 MA20", "above200": "#19 收盤 > MA200",
               "below200": "#20 收盤 < MA200", "align20": "#21 MA20 > MA60 > MA120", "ma60_120": "#22 MA60 > MA120",
               "drop15": "#23 前段跌幅 ≥ 15%", "drop25": "#24 前段跌幅 ≥ 25%", "vol_break": "#25 判定日量 > 均量 1.5 倍",
               "vol_shrink": "#26 島內量縮", "gap1": "#27 向上缺口 ≥ 1%", "gap3": "#28 向上缺口 ≥ 3%",
               "k1": "#29 單日島", "k3plus": "#30 島寬 ≥ 3 根", "rsi30": "#31 島內 RSI < 30",
               "kd20": "#32 島內 K 值 < 20", "redk_upper": "#33 確認日紅 K 收上半部"}
EXIT_NAMES = {"h20": "持有 20 根", "trail10": "替換：#6 回落 10%", "atr": "替換：#9 ATR 3 倍",
              "gapdown": "替換：#12 向下跳空", "trail5": "替換：#5 回落 5%"}


def table(df: pd.DataFrame, base_pf: float) -> str:
    head = ("| 版本 | 交易次數 | 勝率% | 平均持有天 | 獲利平均% | 虧損平均% | 中位數% | 期望值 | PF | 總獲利(萬) "
            "| PF 差 | 生效率% | 未平倉% |\n|" + "---|" * 13)
    lines = [head]
    for _, r in df.iterrows():
        lines.append(
            f"| {ENTRY_NAMES[r['entry']]} | {r['交易次數']:,} | {r['勝率(%)']:.2f} | {r['平均持有天數']:.1f} "
            f"| {r['平均獲利報酬率(%)']:.2f} | {r['平均虧損報酬率(%)']:.2f} | {r['中位數報酬率(%)']:.2f} "
            f"| {r['期望報酬值(EV)']:.1f} | {r['獲利因子(PF)']:.3f} | {r['總獲利'] / 1e4:,.1f} "
            f"| {r['獲利因子(PF)'] - base_pf:+.3f} | {r['生效率(%)']:.1f} | {r['未平倉率(%)']:.2f} |")
    return "\n".join(lines)


def win_count(df: pd.DataFrame) -> pd.DataFrame:
    """每條條件在 5 種出場中 PF 高於同出場同口徑基準的次數。"""
    base = df[df["entry"] == "base"].set_index(["exit_name", "liq"])["獲利因子(PF)"]
    d = df[df["entry"] != "base"].copy()
    d["beat"] = d["獲利因子(PF)"].to_numpy() > base.reindex(
        pd.MultiIndex.from_arrays([d["exit_name"], d["liq"]])).to_numpy()
    out = d.pivot_table(index="entry", columns="liq", values="beat", aggfunc="sum")
    return out.reindex([e for e in ENTRY_NAMES if e != "base"])


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    raw_only = "--raw" in sys.argv
    df = pd.read_csv(args[0] if args else DEFAULT_CSV)
    order = list(ENTRY_NAMES)
    df["_e"] = df["entry"].map(order.index)
    df = df.sort_values("_e")
    groups = ((False, "裸版"),) if raw_only else ((False, "裸版"), (True, "可成交（成交金額 > 1,000 萬）"))
    for liq, name in groups:
        for ex, ex_name in EXIT_NAMES.items():
            sub = df[(df["liq"] == liq) & (df["exit_name"] == ex)]
            base_pf = float(sub.loc[sub["entry"] == "base", "獲利因子(PF)"].iloc[0])
            print(f"\n### {name}｜出場：{ex_name}\n")
            print(table(sub, base_pf))
    print("\n### PF 高於基準的出場數（共 5 種）\n")
    wc = win_count(df)
    print("| 進場條件 | 裸版 | 可成交 |\n|---|---|---|")
    for e, r in wc.iterrows():
        print(f"| {ENTRY_NAMES[e]} | {int(r.get(False, 0))} | {int(r.get(True, 0))} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
