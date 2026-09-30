# -*- coding: utf-8 -*-
"""
MACD（十）拼裝整合篇的前置：從已跑完的 CSV 拉出四個母體的候選表。

目的是挑「拼裝矩陣要用哪幾個母體」——四個全做會讓組合數變成四倍，
文章規模會失控（五~九篇平均 24k 字元，成因就是母體數是乘數）。
本腳本只讀現有 CSV、不重跑回測。

資料來源（SPA result/single_macd/，gitignore 不進版控）：
  _matrix_3x3.csv    四母體的 baseline（（四）篇的 3×3 排列組合）
  _entry_sweep_v2.csv 九個進場濾網 × 四母體（五／六篇）
  _exit_parallel.csv  A 組五條的附加版（七篇，2026-09-06 定案的權威口徑）
  _exit_sweep.csv     十條 × 四母體，取其中 B 疊加組（八篇的風控五條）

出場一律只取「附加」版：原出場留著、新規則疊上去、先觸發者算。
取代版是附錄（九篇），不納入候選比較。

執行：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 F:/stock-analyzer/.venv/Scripts/python.exe \
        _04_analysis/macd/population_candidates.py
"""

import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
from _02_strategy.base.vbt import common  # noqa: E402
import os

import pandas as pd

RESULT = (_root + "/_02_strategy/"
          "macd_strategy/result/single_macd")

# 母體 → （四）篇矩陣裡對應的那一格（與 verify_article9.py 的對應一致）
BASE_COMBO = {
    "交叉": "交叉進 × 交叉出",
    "零軸": "零軸進 × 零軸出",
    "背離": "背離進 × 交叉出",
    "交叉進×零軸出": "交叉進 × 零軸出",
}
# 主表固定欄位（見 memory backtest-result-table-spec，中位數為主尺）
COLS = ["交易次數", "勝率%", "平均持有天", "獲利平均%", "虧損平均%",
        "中位數%", "期望值/筆", "獲利因子", "總獲利(萬)"]


def load():
    """讀四份 CSV，出場分成附加版與取代版兩套（口徑不同，絕不可混在一起比）。"""
    matrix = pd.read_csv(f"{RESULT}/_matrix_3x3.csv")
    entry = pd.read_csv(f"{RESULT}/_entry_sweep_v2.csv")
    par = pd.read_csv(f"{RESULT}/_exit_parallel.csv")
    swp = pd.read_csv(f"{RESULT}/_exit_sweep.csv")
    # 附加（主軸）：A 組五條的並行版 ＋ B 組風控五條的疊加版
    exit_add = pd.concat([par[par["組"] == "A並行"],
                          swp[swp["組"] == "B疊加"]], ignore_index=True)
    # 取代（附錄）：A 組五條整條換掉原出場
    exit_rep = swp[swp["組"] == "A取代"].copy()
    return matrix, entry, exit_add, exit_rep


def md_table(rows, header):
    """輸出 markdown 表格，方便直接貼進 reference。"""
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def fmt(v, digits=2):
    return f"{v:,.{digits}f}" if isinstance(v, float) else f"{v:,}"


def best(df, pop, key):
    """取某母體裡獲利因子最高的一列，回傳 (規則名, 該列)。"""
    d = df[df["母體"] == pop].sort_values("獲利因子", ascending=False)
    return (d.iloc[0][key], d.iloc[0]) if len(d) else (None, None)


def main():
    matrix, entry, exit_add, exit_rep = load()
    base = {}

    # ── 表一：四母體的 baseline ──
    rows = []
    for pop, combo in BASE_COMBO.items():
        r = matrix[matrix["組合"] == combo].iloc[0]
        base[pop] = r
        rows.append([pop] + [
            fmt(r[c], 4 if c == "獲利因子" else (0 if c in ("交易次數", "平均持有天") else 2))
            for c in COLS])
    print("## 表一：四個母體的 baseline（（四）篇矩陣，可成交門檻）\n")
    print(md_table(rows, ["母體"] + COLS))

    # ── 表二：每個母體的優化天花板（以獲利因子排序；交易次數差太多，總獲利不可直接比）──
    rows = []
    for pop in BASE_COMBO:
        b = base[pop]
        en, e = best(entry, pop, "濾網")
        an, a = best(exit_add, pop, "出場")
        rn, r = best(exit_rep, pop, "出場")
        rows.append([
            pop, f'{b["獲利因子"]:.4f}',
            en, f'{e["獲利因子"]:.4f}', f'{(e["獲利因子"] / b["獲利因子"] - 1) * 100:+.1f}%',
            an, f'{a["獲利因子"]:.4f}', f'{(a["獲利因子"] / b["獲利因子"] - 1) * 100:+.1f}%',
            rn, f'{r["獲利因子"]:.4f}', f'{(r["獲利因子"] / b["獲利因子"] - 1) * 100:+.1f}%',
        ])
    print("\n## 表二：每個母體的優化天花板（各取獲利因子最高的一條）\n")
    print(md_table(rows, ["母體", "baseline", "最佳進場濾網", "獲利因子", "相對",
                          "最佳出場·附加", "獲利因子", "相對",
                          "最佳出場·取代", "獲利因子", "相對"]))
    print("\n⚠️ 附加與取代是兩種口徑，同一張表只為對照，**絕不可混在一起排名**。"
          "附加＝原出場留著先觸發者算（七／八篇主軸）；取代＝整條換掉（九篇附錄）。")

    # ── 表三：規則敏感度與廣度（值域寬＝拼裝時有搬動空間）──
    rows = []
    for pop in BASE_COMBO:
        b = base[pop]
        e, a, r = (df[df["母體"] == pop] for df in (entry, exit_add, exit_rep))
        rows.append([
            pop, fmt(b["交易次數"], 0),
            f'{e["獲利因子"].min():.3f}~{e["獲利因子"].max():.3f}',
            f'{(e["獲利因子"] > b["獲利因子"]).sum()}/{len(e)}',
            f'{int(e["交易次數"].min()):,}~{int(e["交易次數"].max()):,}',
            f'{a["獲利因子"].min():.3f}~{a["獲利因子"].max():.3f}',
            f'{(a["獲利因子"] > b["獲利因子"]).sum()}/{len(a)}',
            f'{r["獲利因子"].min():.3f}~{r["獲利因子"].max():.3f}',
            f'{(r["獲利因子"] > b["獲利因子"]).sum()}/{len(r)}',
        ])
    print("\n## 表三：規則敏感度與廣度\n")
    print(md_table(rows, ["母體", "baseline 交易次數", "進場獲利因子值域", "進場勝出",
                          "進場後交易次數值域", "附加出場值域", "附加勝出",
                          "取代出場值域", "取代勝出"]))

    # ── 表四：取代版下四母體會不會塌掉（出場換掉後，只剩進場在區分母體）──
    print("\n## 表四：取代版出場下的母體塌縮\n")
    rows = []
    for rule in sorted(exit_rep["出場"].unique()):
        d = exit_rep[exit_rep["出場"] == rule]
        vals = {p: f'{d[d["母體"] == p]["獲利因子"].iloc[0]:.4f}' for p in BASE_COMBO
                if len(d[d["母體"] == p])}
        rows.append([rule] + list(vals.values())
                    + ["是" if vals.get("交叉") == vals.get("交叉進×零軸出") else "否"])
    print(md_table(rows, ["取代型出場"] + list(BASE_COMBO) + ["交叉與 mix 同值"]))
    print("\n出場整條換掉後，「交叉」與「交叉進×零軸出」的進場同樣是黃金交叉，"
          "兩個母體會變成同一組數字——取代版實際上只有三個母體。")

    print(f"\n資料來源：{os.path.basename(RESULT)}/ 的 4 份 CSV，未重跑回測。")


if __name__ == "__main__":
    main()
