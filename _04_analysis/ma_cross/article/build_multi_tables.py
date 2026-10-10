# -*- coding: utf-8 -*-
"""
產生均線交叉（五）（六）多股兩篇與（七）結論篇的表格 HTML 列（只排版，不跑回測）。

資料＝_03_multi_strategy/ma_cross/ma_cross_multi_driver.py --task angle_adx 的輸出
（result/angle_adx/orderings.csv＝流動性／低價／高價三種固定排序；random_dist.csv＝隨機 1,000 次，
代表列是「最終權益中位數那一次」，另附總獲利的 P5／中位／P95）。

  --mode 定額 → （五）：份數表、分批對照表、排序對照表、附錄四張完整表
  --mode 比例 → （六）：同上
  --conclusion → （七）：比例｜公式 隨機代表列的期末（100 萬＋總獲利）對 0050

多股兩篇沿用原文的 ASCII 減號（-）；（七）沿用原文的全形減號（−）。
上色規則：
  分批對照表：本金<80% 大於 0 標 risk；中位差（公式−等分，用顯示值相減）正＝up、負＝down。
  排序對照表：流動性／低價／高價以同列隨機為基準，比「顯示值」——顯示一樣不上色，
             高於＝up、低於＝down（讀者看到兩格數字相同卻有顏色會困惑）。

執行：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python \
        _04_analysis/ma_cross/article/build_multi_tables.py --mode 定額|比例 [--conclusion]
"""
import argparse
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from article_common import PAIRS, fmt, half_up, mc, mrow, multi  # noqa: E402

INIT_CASH = 1_000_000.0
FLOOR = 0.80
ORDERS = ["流動性", "低價", "高價", "隨機"]


def afmt(v, nd: int = 0, comma: bool = True, sign: bool = False) -> str:
    """多股兩篇的格式：ASCII 減號。"""
    return fmt(v, nd, comma, sign).replace("−", "-")


def raw(v) -> str:
    """
    附錄表的比率欄：兩位小數、去掉多餘的 0（與原文格式一致）。multi() 已改用精確值，
    這裡從精確值 half-up 進位到兩位；舊 CSV 的存檔值本來就是兩位，進位後不變。
    """
    return str(float(half_up(v, 2))).replace("−", "-")


def pct_txt(v) -> str:
    """本金<80% 的比例：一位小數，整數就不帶 .0（100、0、92.8）。"""
    s = afmt(v, 1)
    return s[:-2] if s.endswith(".0") else s


def tot_wan(r) -> float:
    """
    總獲利（萬）：有精確值就用它；其次用元為單位的欄位（文章顯示到萬，落在 .5 時才判得準）。
    multi() 已把精確值覆蓋進「總獲利(萬)」，這裡再明確以精確欄優先，避免被兩位小數的元欄蓋掉。
    """
    ex = "總獲利(萬)_精確"
    if ex in r.index and r[ex] == r[ex]:
        return float(r[ex])
    if "總獲利(元)" in r.index and r["總獲利(元)"] == r["總獲利(元)"]:
        return float(r["總獲利(元)"]) / 10_000
    return float(r["總獲利(萬)"])


def sizing_names(mode: str):
    return f"{mode}｜公式", f"{mode}｜等分20"


def units_rows(df, mode: str) -> str:
    """份數表：S＝單股蒙地卡羅的最大連敗 P95；份數與每筆讀 driver 實際用的值。"""
    s = mc()["最大連敗_P95"]
    out = []
    for p in PAIRS:
        r = mrow(df, p, sizing_names(mode)[0], "流動性")
        out.append(f"<tr><td>{p}</td><td>{int(s[p])}</td><td>{int(r['份數'])}</td>"
                   f"<td>{afmt(INIT_CASH / r['份數'])}</td></tr>")
    return "".join(out)


def sizing_rows(df, mode: str) -> str:
    """分批對照：等分 20 份 vs 公式，各自隨機 ×1,000 的擋單、本金<80%、總獲利 P5／中位／P95。"""
    formula, even = sizing_names(mode)
    n_all = mc()["交易數"]
    out = []
    for p in PAIRS:
        e, f = mrow(df, p, even, "隨機"), mrow(df, p, formula, "隨機")
        tds = [f"<td>{p}</td>", f"<td>{afmt(n_all[p])}</td>"]
        meds = []
        for k, r in enumerate((e, f)):
            grp = "grp"
            if k == 1:
                tds.append(f"<td class='grp'>{int(r['份數'])}</td>")
                grp = ""
            tds.append(f"<td class='{grp}'>{afmt(r['擋單'])}</td>" if grp else f"<td>{afmt(r['擋單'])}</td>")
            risk = r["本金曾<80%_比例%"]
            tds.append(f"<td class='risk'>{pct_txt(risk)}</td>" if risk > 0 else f"<td>{pct_txt(risk)}</td>")
            q = [r.get("總獲利(萬)_P5"), r.get("總獲利(萬)_中位"), r.get("總獲利(萬)_P95")]
            tds += [f"<td>{afmt(v) if v == v and v is not None else '?'}</td>" for v in q]
            meds.append(half_up(q[1], 0) if q[1] == q[1] and q[1] is not None else None)
        if None not in meds:
            diff = meds[1] - meds[0]
            tds.append(f"<td class='grp {'up' if diff > 0 else 'down'}'>{afmt(diff, sign=True)}</td>")
        else:
            tds.append("<td class='grp'>?</td>")
        out.append("<tr>" + "".join(tds) + "</tr>")
    return "".join(out)


def order_rows(df, mode: str) -> str:
    """排序對照：擋單（隨機代表列）＋四排序的總獲利（萬）與期望值/筆（元）。"""
    formula = sizing_names(mode)[0]
    out = []
    for p in PAIRS:
        rows = {o: mrow(df, p, formula, o) for o in ORDERS}
        tds = [f"<td>{p}</td>", f"<td>{afmt(rows['隨機']['擋單'])}</td>"]
        for getter in (tot_wan, lambda r: float(r["期望值/筆"])):
            base = half_up(getter(rows["隨機"]), 0)
            for o in ORDERS:
                v = half_up(getter(rows[o]), 0)
                cls = "" if o == "隨機" or v == base else (" class='up'" if v > base else " class='down'")
                tds.append(f"<td{cls}>{afmt(v)}</td>")
        out.append("<tr>" + "".join(tds) + "</tr>")
    return "".join(out)


def appendix_rows(df, mode: str, order: str) -> str:
    """附錄完整 10 欄＋擋單（公式分批、指定排序）。"""
    formula = sizing_names(mode)[0]
    out = []
    for p in PAIRS:
        r = mrow(df, p, formula, order)
        cells = [p, afmt(r["交易次數"]), raw(r["勝率%"]), raw(r["平均持有天"]), raw(r["獲利平均%"]),
                 raw(r["虧損平均%"]), raw(r["中位數%"]), afmt(r["期望值/筆"]), afmt(r["獲利因子"], 2),
                 afmt(tot_wan(r)), afmt(r["擋單"])]
        out.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return "".join(out)


def bench_0050() -> dict:
    """
    0050 買進持有（2015 首日收盤買進 → 2025 年底）：含息與只計價格兩種的期末（萬）、倍數、年化%。
    算法全用 _04_analysis/benchmark/benchmark_0050.py（含息再投入、分割已還原），不另寫一份。
    """
    from _04_analysis.benchmark.benchmark_0050 import (BENCH, BENCHMARK_END, BENCHMARK_START,
                                                       curve_stats, equity_0050)
    from article_common import common
    import pandas as pd
    px = pd.read_parquet(os.path.join(common.DATA_DIR, f"{BENCH}.parquet")).sort_index()
    cal = px.loc[BENCHMARK_START:BENCHMARK_END].index
    years = (cal[-1] - cal[0]).days / 365.25
    curve = equity_0050(cal, INIT_CASH)
    eq = curve["市值"].to_numpy(float)
    price = curve["收盤價"].to_numpy(float)
    price_eq = price / price[0] * INIT_CASH
    return {"起日": cal[0], "迄日": cal[-1], "年數": years, "首日收盤": price[0],
            "含息期末萬": eq[-1] / 1e4, "含息倍數": eq[-1] / INIT_CASH,
            "含息年化%": ((eq[-1] / INIT_CASH) ** (1 / years) - 1) * 100,
            "價格期末萬": price_eq[-1] / 1e4, "價格倍數": price_eq[-1] / INIT_CASH,
            "價格年化%": ((price_eq[-1] / INIT_CASH) ** (1 / years) - 1) * 100,
            "含息統計": curve_stats(eq, years), "px": px}


def conclusion_end(df, pair: str) -> float:
    """（七）的期末（萬）＝100 萬＋比例｜公式 隨機 1,000 次總獲利的中位數（與（六）分批表「中位」同一個數）。"""
    return 100.0 + float(mrow(df, pair, "比例｜公式", "隨機")["總獲利(萬)_中位"])


def conclusion_rows(df, bench_wan: float) -> str:
    """（七）：期末（萬）對 0050 的差；期末為負標（大賠）、低於本金標（倒賠本金）。"""
    out = []
    for p in PAIRS:
        end = half_up(conclusion_end(df, p), 0)
        diff = end - half_up(bench_wan, 0)
        if diff > 0:
            vs = f"**贏 {fmt(diff, sign=True)}** ✅"
        else:
            vs = f"輸 {fmt(diff)}" + ("（大賠）" if end < 0 else "（倒賠本金）" if end < 100 else "")
        out.append(f"| {p} | {fmt(end)} | {vs} |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="定額", choices=["定額", "比例"])
    ap.add_argument("--conclusion", action="store_true")
    ap.add_argument("--bench", type=float, default=None,
                    help="0050 期末（萬）；不給就由 _04_analysis/benchmark/benchmark_0050.py 現算")
    a = ap.parse_args()
    df = multi("angle_adx")
    if a.conclusion:
        print(conclusion_rows(df, a.bench if a.bench is not None else bench_0050()["含息期末萬"]))
        return 0
    print("=== 份數表 ===")
    print(units_rows(df, a.mode))
    print("\n=== 分批對照（等分 20 vs 公式，隨機×1000）===")
    print(sizing_rows(df, a.mode))
    print("\n=== 排序對照（公式）===")
    print(order_rows(df, a.mode))
    for o in ORDERS:
        print(f"\n=== 附錄：{o} ===")
        print(appendix_rows(df, a.mode, o))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
