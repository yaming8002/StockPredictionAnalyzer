# -*- coding: utf-8 -*-
"""
驗〈均線交叉（七）：一路走來的總結評估〉（ma-cross-conclusion）：0050 對照表、21 組期末表、正文統計句。

  0050：_04_analysis/benchmark/benchmark_0050.py（2015-01-05 首日收盤買進、配息當日再投入、抱到 2025 年底）
  均線：（六）的「比例｜公式」隨機 1,000 次總獲利中位數＋本金 100 萬（與（六）分批表「中位」同一個數）
  重點整理引用的單股數字：baseline／liq1000／夾角＋ADX 的 50/200（逐筆重算）

執行（BLOG_DIR 指向 blog；DIVIDEND_FILE 指向 dividend_actions.parquet）：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python _04_analysis/ma_cross/article/verify_article7.py
"""
import math
import os
import re
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from article_common import (MC_VARIANT, PAIRS, Checker, all_metrics, fmt, half_up,  # noqa: E402
                            metrics, multi, num, post_text, same)
from build_multi_tables import bench_0050, conclusion_end  # noqa: E402

SLUG = "ma-cross-conclusion"
YEARS_MA = 24                                  # 2002–2025


def verify_bench(ck: Checker, text: str, b: dict) -> None:
    ck.phrase(text, f"**{b['起日']:%Y-%m-%d} 到 {b['迄日']:%Y-%m-%d}，約 {fmt(b['年數'])} 年**")
    first = b["首日收盤"] * 4                  # parquet 價已還原 2025 的 1 拆 4，當時實際價＝×4
    ck.check(round(first) == 67, f"2015 首日實際價 {first:.2f}（文：約 67 元）")
    ck.check(round(1e6 / first / 1000, 1) == 15.0, "100 萬買到的股數不是約 1.5 萬股")
    px = b["px"]
    hi24 = float(px.loc["2024", "close"].max()) * 4
    ck.check(round(hi24, -1) == 200, f"2024 年最高收盤（還原前）{hi24:.1f}（文：約 200 元）")
    jun = px.loc["2025-06-18":"2025-06-30", "close"]
    ck.check(round(float(jun.mean()), -1) == 50, f"2025-06 分割後收盤均值 {jun.mean():.2f}（文：約 50 元）")
    rows = re.findall(r"^\| (只計價格漲跌|\*\*加計配股配息（全數再投入）\*\*) \|(.*)\|$", text, flags=re.M)
    ck.check(len(rows) == 2, "0050 表找不到兩列")
    for (name, body), key in zip(rows, ("價格", "含息")):
        cells = [c.strip().strip("*") for c in body.split("|")]
        end, mult, cagr = b[f"{key}期末萬"], b[f"{key}倍數"], b[f"{key}年化%"]
        ck.check(cells[0] == "100 萬", f"0050 {key} 投入 {cells[0]}")
        ck.check(same(cells[1].replace("約", "").replace("萬", ""), end), f"0050 {key} 期末 {cells[1]} vs {end:.2f}")
        ck.check(same(cells[2].replace("萬", ""), end - 100), f"0050 {key} 總獲利 {cells[2]}")
        ck.check(same(cells[3].replace("×", ""), mult), f"0050 {key} 倍數 {cells[3]} vs {mult:.4f}")
        ck.check(same(cells[4], cagr), f"0050 {key} 年化 {cells[4]} vs {cagr:.3f}")
    ck.phrase(text, f"配息額外貢獻約 {fmt(b['含息期末萬'] - b['價格期末萬'])} 萬")
    ck.phrase(text, f"0050 含息把 100 萬滾到約 {fmt(b['含息期末萬'])} 萬")
    ck.check(abs(b["含息統計"][0] - 5.51) < 1e-9 and abs(b["含息統計"][1] - 16.81) < 1e-9, "0050 錨點對不上 ×5.51／16.81%")


def verify_table(ck: Checker, text: str, df, bench: int) -> dict:
    rows = re.findall(r"^\| (\d+/\d+) \| (.*?) \| (.*?) \|$", text, flags=re.M)
    ck.check([r[0] for r in rows] == PAIRS, f"期末表列順序／組數不符（{len(rows)} 列）")
    ends = {}
    for p, end_txt, vs in rows:
        end = conclusion_end(df, p)
        ends[p] = half_up(end, 0)
        ck.check(same(end_txt, end), f"期末 {p}：文 {end_txt} vs {end:.3f}")
        diff = ends[p] - bench
        want = (f"**贏 {fmt(diff, sign=True)}** ✅" if diff > 0 else
                f"輸 {fmt(diff)}" + ("（大賠）" if ends[p] < 0 else "（倒賠本金）" if ends[p] < 100 else ""))
        ck.check(vs == want, f"對比 {p}：文 {vs} vs 應 {want}")
    return ends


def verify_claims(ck: Checker, text: str, b: dict, ends: dict) -> None:
    bench = half_up(b["含息期末萬"], 0)
    win = [p for p in PAIRS if ends[p] > bench]
    lose_principal = [p for p in PAIRS if ends[p] < 100]
    ck.check(not win, f"贏過 0050 的組：{win}")
    ck.phrase(text, f"累積金額全部低於這個數，其中 {len(lose_principal)} 組倒賠")
    top3 = sorted(PAIRS, key=lambda p: -ends[p])[:3]
    ck.phrase(text, "最接近的三組是 " + "、".join(f"{p}（{fmt(ends[p])} 萬）" for p in top3))
    ck.phrase(text, f"**{len(lose_principal)} 組（{'、'.join(lose_principal)}）甚至倒賠本金**")
    best = top3[0]
    ck.phrase(text, f"- 0050：**11 年**，100 萬 → {fmt(b['含息期末萬'])} 萬。")
    ck.phrase(text, f"- 均線交叉最好的 {best}：**{YEARS_MA} 年**，100 萬 → {fmt(ends[best])} 萬。")
    ck.check(round(b["年數"]) == 11, "0050 期間不是約 11 年")
    c_b = b["含息年化%"] / 100
    c_m = (conclusion_end(multi("angle_adx"), best) / 100) ** (1 / YEARS_MA) - 1
    d_b, d_m = math.log(2) / math.log(1 + c_b), math.log(2) / math.log(1 + c_m)
    ck.phrase(text, f"0050 這段約 {fmt(c_b * 100, 1)}%／年，大約 **{fmt(d_b, 1)} 年翻一倍**")
    ck.phrase(text, f"均線交叉最好那組約 {fmt(c_m * 100)}%／年，要 **{int(d_m)} 年以上才翻一倍**")
    ck.check(2 < d_m / d_b < 3, f"翻倍年數比 {d_m / d_b:.2f}（文：兩倍多）")
    ck.check(ends[best] < bench and YEARS_MA - round(b["年數"]) == 13, "「金額還少一些、卻多花了十三年」不成立")
    # 重點整理：單股數字
    base, liq = all_metrics("baseline"), all_metrics("liq1000")
    wr = [base[p]["wr"] for p in PAIRS]
    ck.check(20 <= min(wr) and max(wr) <= 40.05, f"baseline 勝率範圍 {min(wr):.1f}～{max(wr):.1f}（文：二到四成）")
    b5, l5 = base["50/200"], liq["50/200"]
    ck.phrase(text, f"**不加任何濾網，24 年有 {fmt(b5['n'])} 筆**")
    ck.phrase(text, f"**單單加一道流動性門檻，就從 {fmt(b5['n'])} 砍到 {fmt(l5['n'])} 筆（少了約六成）**")
    ck.check(0.55 <= 1 - l5["n"] / b5["n"] < 0.65, "流動性砍掉比例不是約六成")
    n_full = metrics(MC_VARIANT, "50/200")["n"]
    ck.check(500 <= n_full < 600, f"完整濾網 50/200 筆數 {n_full}（文：五百多筆）")
    ck.phrase(text, f"**勝率幾乎沒變（{fmt(b5['wr'], 1)}%→{fmt(l5['wr'], 1)}%）**")


def main() -> int:
    ck = Checker("均線交叉（七）")
    text = post_text(SLUG)
    b = bench_0050()
    verify_bench(ck, text, b)
    ends = verify_table(ck, text, multi("angle_adx"), half_up(b["含息期末萬"], 0))
    verify_claims(ck, text, b, ends)
    return ck.done()


if __name__ == "__main__":
    raise SystemExit(main())
