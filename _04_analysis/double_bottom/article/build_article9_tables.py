# -*- coding: utf-8 -*-
"""
產生〈雙底（九）：多股共用資金 · 固定金額投入〉〈雙底（十）：多股共用資金 · 固定比例投入〉的表格 HTML。

資料來源＝_03_multi_strategy/double_bottom/db_multi_driver.py 的 db_multi_result.csv
（五組 × 兩投法 × 四排序；隨機＝1,000 次逐欄取中位，精確欄＝逐次精確值的中位）。
排法與上色比照〈MACD（十一）〉build_article11_tables：策略區塊依「低價」總獲利高到低排、區塊內依總獲利排；
拿同一組的隨機列當基準，獲利因子與總獲利兩欄贏它＝粉紅、輸它＝淺綠，隨機列本身不上色。
首欄寫出門檻（ZigZag 9%）：一列代表一格（門檻 × 進場 × 出場），不寫門檻表格說明就不完整（用戶 2026-11-11）。
另出一張資金曲線表（期末權益、年化報酬、最大回撤、已實現權益最低），多股回測規格要附。

執行：PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python _04_analysis/double_bottom/article/build_article9_tables.py [--mode 定額|比例]
"""
import argparse
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
for p in (_root, _here):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd  # noqa: E402

from _02_strategy.base.vbt import common  # noqa: E402
from _03_multi_strategy.double_bottom.db_multi_driver import INIT_CASH, PATH, PICKS  # noqa: E402
from build_article2_tables import half_up  # noqa: E402

EXACT = "_精確"
COLS = ["交易次數", "擋單", "勝率%", "平均持有天", "獲利平均%", "虧損平均%",
        "中位數%", "期望值", "獲利因子(PF)", "總獲利(萬)"]
HEAD = ["交易次數", "擋單", "勝率%", "平均<br>持有天", "獲利<br>平均%", "虧損<br>平均%",
        "中位數%", "期望值<br>/筆", "獲利因子", "總獲利<br>(萬)"]
PAINT = ["獲利因子(PF)", "總獲利(萬)"]
S_BY = {p[0]: p[4] for p in PICKS}
PCT_BY = {p[0]: int(round(p[1] * 100)) for p in PICKS}


def minus(s: str) -> str:
    return s.replace("-", "−")


def fmt(col: str, v) -> str:
    if col in ("交易次數", "擋單", "期望值", "總獲利(萬)"):
        return minus(f"{half_up(v, 0):,}")
    if col == "平均持有天":
        return f"{half_up(v, 0)}"
    if col == "中位數%":
        return minus(f"{half_up(v, 2)}")
    if col == "獲利因子(PF)":
        return f"{half_up(v, 4)}"
    return minus(f"{half_up(v, 1)}")


def label(name: str) -> str:
    return f"ZigZag {PCT_BY[name]}%<br>{name}"


def load(mode: str) -> pd.DataFrame:
    """精確欄覆蓋同名存檔欄；只留指定投法。"""
    df = common.read_result_csv(PATH)
    for c in [c for c in df.columns if c.endswith(EXACT)]:
        base = c[:-len(EXACT)]
        df[base] = df[c].where(df[c].notna(), df[base] if base in df.columns else df[c])
    return df[df["投法"] == mode].copy()


def block_order(df: pd.DataFrame) -> list:
    return df[df["排序"] == "低價"].sort_values("總獲利(萬)", ascending=False)["交易策略"].tolist()


def result_table(df: pd.DataFrame) -> str:
    head = "<thead><tr><th>交易策略</th><th>買入<br>排序</th>" + "".join(f"<th>{h}</th>" for h in HEAD) + "</tr></thead>"
    out = ['<div class="table-wrap"><table class="bt-table">', head, "<tbody>"]
    for name in block_order(df):
        blk = df[df["交易策略"] == name].sort_values("總獲利(萬)", ascending=False)
        ref = blk[blk["排序"] == "隨機"].iloc[0]
        for k, (_, r) in enumerate(blk.iterrows()):
            first = f"<b>{label(name)}</b>" if k == 0 else ""
            kind = f"<b>{r['排序']}</b>" if r["排序"] == "隨機" else r["排序"]
            cells = []
            for c in COLS:
                cls = ""
                if c in PAINT and r["排序"] != "隨機":
                    cls = ' class="up"' if r[c] > ref[c] else ' class="down"'
                cells.append(f"<td{cls}>{fmt(c, r[c])}</td>")
            out.append(f"<tr><td>{first}</td><td>{kind}</td>{''.join(cells)}</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def units_table(df: pd.DataFrame, mode: str) -> str:
    """份數表：S＝〈雙底（八）〉蒙地卡羅的最大連敗 P95。"""
    rule = "S÷0.2" if mode == "定額" else "1÷(1−0.8<sup>1/S</sup>)"
    amt = "每筆金額(元)" if mode == "定額" else "每筆金額"
    head = (f"<thead><tr><th>交易策略</th><th>最大連敗 S</th><th>切成份數<br>({rule})</th>"
            f"<th>{amt}</th></tr></thead>")
    out = ['<div class="table-wrap"><table class="bt-table">', head, "<tbody>"]
    for name in S_BY:
        n = int(df[df["交易策略"] == name]["份數"].iloc[0])
        a = f"{half_up(INIT_CASH / n, 0):,}" if mode == "定額" else f"已實現權益 ÷ {n}"
        out.append(f"<tr><td>{label(name)}</td><td>{S_BY[name]}</td><td>{n}</td><td>{a}</td></tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def equity_table(df: pd.DataFrame) -> str:
    """資金曲線：期末權益（萬）、年化報酬%、最大回撤%、已實現權益最低%（相對本金）。不上色。"""
    head = ("<thead><tr><th>交易策略</th><th>買入<br>排序</th><th>期末權益<br>(萬)</th><th>年化<br>報酬%</th>"
            "<th>最大<br>回撤%</th><th>已實現權益<br>最低%</th></tr></thead>")
    out = ['<div class="table-wrap"><table class="bt-table">', head, "<tbody>"]
    for name in block_order(df):
        blk = df[df["交易策略"] == name].sort_values("總獲利(萬)", ascending=False)
        for k, (_, r) in enumerate(blk.iterrows()):
            first = f"<b>{label(name)}</b>" if k == 0 else ""
            kind = f"<b>{r['排序']}</b>" if r["排序"] == "隨機" else r["排序"]
            vals = [f"{half_up(r['最終權益'] / 10_000, 0):,}", f"{half_up(r['年化報酬(%)'], 2)}",
                    f"{half_up(r['最大回撤(%)'], 1)}", f"{half_up(r['已實現權益最低(%)'], 1)}"]
            out.append(f"<tr><td>{first}</td><td>{kind}</td>{''.join(f'<td>{v}</td>' for v in vals)}</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def random_note(df: pd.DataFrame) -> str:
    """隨機列的資金倍數中位（P5／P95）＝1＋總獲利÷本金（同〈MACD（十一）〉）。"""
    parts = []
    for name in block_order(df):
        r = df[(df["交易策略"] == name) & (df["排序"] == "隨機")].iloc[0]
        m = [1 + r[k] / 100 for k in ("總獲利(萬)" + EXACT, "總獲利(萬)P5" + EXACT, "總獲利(萬)P95" + EXACT)]
        parts.append(f"{name} {half_up(m[0], 2)}（{half_up(m[1], 2)}／{half_up(m[2], 2)}）")
    return "、".join(parts)


def all_tables(mode: str) -> dict:
    df = load(mode)
    return {"份數": units_table(df, mode), "結果": result_table(df), "資金曲線": equity_table(df)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="定額", choices=["定額", "比例"])
    a = ap.parse_args()
    for k, html in all_tables(a.mode).items():
        print(f"=== {k}（{a.mode}）===\n{html}\n")
    print("=== 隨機列資金倍數中位（P5／P95）===\n" + random_note(load(a.mode)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
