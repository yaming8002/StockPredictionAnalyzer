# -*- coding: utf-8 -*-
"""
產生（十一）（十二）兩篇的結果表 HTML 列。

資料來源＝SPA 的多股回測輸出（**回測程式全部在 SPA / GitHub，這支只負責排版**）：
  _02_strategy/macd_strategy/result/macd_multi/macd_multi_result.csv    低價／高價／流動性
  _02_strategy/macd_strategy/result/macd_multi/macd_multi_random.csv    隨機（1,000 次取中位）

排法比照 KD 系列多股兩篇：**策略區塊依「低價」總獲利高到低排、區塊內依總獲利排**。
標色也比照：拿同一組的**隨機列當基準**，贏它＝粉紅(up)、輸它＝淺綠(down)，隨機列本身不標色。
這樣讀者一眼看得出「這個排序是真本事，還是還不如亂買」。

執行：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 F:/stock-analyzer/.venv/Scripts/python.exe \
        _04_analysis/macd/article/build_article11_tables.py [--mode 定額|比例]
"""
import argparse
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd

from _02_strategy.base.vbt import common  # noqa: E402

SPA = common.result_dir("macd_strategy", "macd_multi")
DET = os.path.join(SPA, "macd_multi_result.csv")
RND = os.path.join(SPA, "macd_multi_random.csv")

# 表格欄位（順序即輸出順序）；前兩欄是標籤欄
COLS = ["交易次數", "擋單", "勝率%", "平均持有天", "獲利平均%", "虧損平均%",
        "中位數%", "期望值/筆", "獲利因子", "總獲利(萬)"]
# 只有這兩欄標色：一個看效率、一個看結果，其餘欄位標了會變成整片顏色、反而讀不出重點
PAINT = ["獲利因子", "總獲利(萬)"]


def fmt(col: str, v) -> str:
    if col in ("交易次數", "擋單"):
        return f"{int(round(v)):,}"
    if col == "平均持有天":
        return f"{v:.0f}"
    if col == "期望值/筆":
        return f"{v:,.0f}".replace("-", "−")
    if col == "中位數%":
        return f"{v:.2f}".replace("-", "−")      # 中位數比照系列其他表留兩位
    if col == "獲利因子":
        return f"{v:.2f}"
    if col == "總獲利(萬)":
        # 負值也要用全形減號，跟其他欄位一致（背離×高價是負的）
        return f"{v:,.0f}".replace("-", "−")
    return f"{v:.1f}".replace("-", "−")


def load(mode: str) -> pd.DataFrame:
    det = pd.read_csv(DET)
    rnd = pd.read_csv(RND)
    df = pd.concat([det, rnd], ignore_index=True)
    return df[df["投法"] == mode].copy()


def rows_for(df: pd.DataFrame) -> str:
    """回傳整張表的 <tr>；策略區塊依低價總獲利排、區塊內依總獲利排。"""
    order = (df[df["排序"] == "低價"]
             .sort_values("總獲利(萬)", ascending=False)["交易策略"].tolist())
    out = []
    for name in order:
        blk = df[df["交易策略"] == name].sort_values("總獲利(萬)", ascending=False)
        base = float(blk[blk["排序"] == "隨機"]["總獲利(萬)"].iloc[0])
        base_pf = float(blk[blk["排序"] == "隨機"]["獲利因子"].iloc[0])
        first = True
        for _, r in blk.iterrows():
            head = f"<b>{name}</b>" if first else ""
            kind = f"<b>{r['排序']}</b>" if r["排序"] == "隨機" else r["排序"]
            cells = []
            for c in COLS:
                cls = ""
                if c in PAINT and r["排序"] != "隨機":
                    ref = base_pf if c == "獲利因子" else base
                    cls = ' class="up"' if r[c] > ref else ' class="down"'
                cells.append(f"<td{cls}>{fmt(c, r[c])}</td>")
            out.append(f"<tr><td>{head}</td><td>{kind}</td>" + "".join(cells) + "</tr>")
            first = False
    return "\n".join(out)


def units_table(df: pd.DataFrame, mode: str) -> str:
    """份數表：S 取自（十）篇蒙地卡羅的最大連敗 P95。"""
    s_map = {"交叉 × 均線多頭排列 × 跌破年線": 32, "交叉 × ADX>25 × 跌破年線": 21,
             "交叉 × 無濾網 × 跌破年線": 25, "零軸 × 收盤>MA200 × 跌破年線": 43,
             "背離 × RSI<50且上升 × 跌破年線": 19}
    out = []
    for name, s in s_map.items():
        n = int(df[df["交易策略"] == name]["份數"].iloc[0])
        amt = (f"{1_000_000 / n:,.0f}" if mode == "定額"
               else f"每筆＝已實現權益 ÷ {n}")
        out.append(f"<tr><td>{name}</td><td>{s}</td><td>{n}</td><td>{amt}</td></tr>")
    return "\n".join(out)


def random_note(df: pd.DataFrame) -> str:
    """隨機列的資金倍數中位（P5／P95）：定額累加與比例都是本金 100 萬 ＋ 已實現損益。"""
    parts = []
    for _, r in df[df["排序"] == "隨機"].iterrows():
        m = 1 + r["總獲利(萬)"] / 100
        lo = 1 + r["總獲利(萬)P5"] / 100
        hi = 1 + r["總獲利(萬)P95"] / 100
        parts.append(f"{r['交易策略']} {m:.2f}（{lo:.2f}／{hi:.2f}）")
    return "、".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="定額", choices=["定額", "比例"])
    a = ap.parse_args()
    df = load(a.mode)
    print(f"=== 份數表（{a.mode}）===")
    print(units_table(df, a.mode))
    print(f"\n=== 結果表（{a.mode}）===")
    print(rows_for(df))
    print(f"\n=== 隨機列資金倍數中位（P5／P95）===")
    print(random_note(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
