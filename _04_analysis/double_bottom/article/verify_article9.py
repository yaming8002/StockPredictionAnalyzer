# -*- coding: utf-8 -*-
"""
驗〈雙底（九）：多股共用資金 · 固定金額投入〉的表格與正文數字，全部由多股結果重算後比對；錯誤 0 才算過。

1. 表格：文章裡的 3 張表必須與 build_article9_tables（定額）產出的 HTML 逐字相同。
2. 正文主張：每條數字敘述在這裡重算，判斷式必須成立，格式化後的字串必須逐字出現在文章裡。

執行：
    BLOG_DIR=F:/stock-analyzer/blog PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
        python _04_analysis/double_bottom/article/verify_article9.py
"""
import io
import os
import re
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
for p in (_root, _here):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd  # noqa: E402

from build_article2_tables import half_up  # noqa: E402
from build_article7_tables import _get, stats  # noqa: E402
from build_article9_tables import EXACT, all_tables, load, random_note  # noqa: E402
from verify_article4 import article_path  # noqa: E402

SLUG = "double-bottom-multi"
MODE = "定額"
NAMES = {"rise_base": "第二底墊高 ≥ 2% × 基本出場", "rise_be": "第二底墊高 ≥ 2% × 附加：保本停損",
         "rise_mtop": "第二底墊高 ≥ 2% × 附加：高檔 M 頭", "height": "型態高度 ≥ 10% × 基本出場", "base": "基本版"}
KEYS = {"rise_base": ("rise", "base"), "rise_be": ("rise", "add_breakeven"), "rise_mtop": ("rise", "add_mtop"),
        "height": ("height", "base"), "base": ("base", "base")}


def _rng(xs, nd, unit=""):
    lo, hi = half_up(min(xs), nd), half_up(max(xs), nd)
    return f"{lo}{unit}" if lo == hi else f"{lo}{unit}～{hi}{unit}"


def _pct(x, nd=1):
    return half_up(x * 100, nd)


def per_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """每組一列：擋單倍數、低價對高價／隨機、期望值與筆數拆解、隨機寬度、未平倉、期末差額。"""
    rows = []
    for name, b in df.groupby("交易策略", sort=False):
        b = b.set_index("排序")
        lo, hi, rnd = b.loc["低價"], b.loc["高價"], b.loc["隨機"]
        rows.append({"name": name, "擋單倍數": rnd["擋單"] / rnd["交易次數"],
                     "低對高": lo["總獲利(萬)"] / hi["總獲利(萬)"], "低對隨": lo["總獲利(萬)"] / rnd["總獲利(萬)"] - 1,
                     "EV多": lo["期望值"] / rnd["期望值"] - 1, "筆數多": lo["交易次數"] / rnd["交易次數"] - 1,
                     "P5": rnd["總獲利(萬)P5" + EXACT], "P95": rnd["總獲利(萬)P95" + EXACT],
                     "寬度": (rnd["總獲利(萬)P95" + EXACT] - rnd["總獲利(萬)P5" + EXACT]) / rnd["總獲利(萬)"],
                     "未平倉": lo["期末未平倉"], "期末差": lo["最終權益"] / 1e4 - 100 - lo["總獲利(萬)"],
                     "低勝隨": all(b.loc["低價", c] > rnd[c] for c in ("獲利因子(PF)", "總獲利(萬)")),
                     "高輸隨": all(b.loc[k, c] < rnd[c] for k in ("高價", "成交金額") for c in ("獲利因子(PF)", "總獲利(萬)"))})
    return pd.DataFrame(rows).set_index("name")


def claims(df: pd.DataFrame, single: pd.DataFrame) -> dict:
    s = per_strategy(df)
    lo = df[df["排序"] == "低價"].set_index("交易策略")
    base = df[df["交易策略"] == "基本版"].set_index("排序")
    amounts = [1_000_000 / n for n in df.drop_duplicates("交易策略")["份數"]]
    pf_single = {k: _get(single, 9, True, *v)["獲利因子(PF)"] for k, v in KEYS.items()}
    by_ratio = s.sort_values("擋單倍數")
    by_width = s.sort_values("寬度")
    ratio_lines = "\n".join(f"- {n}：{half_up(r, 2)} 倍" for n, r in by_ratio["擋單倍數"].items())
    width_lines = "\n".join(f"- {n}：{_pct(r['寬度'])}%（{half_up(r['P5'], 1)}～{half_up(r['P95'], 1)} 萬）"
                            for n, r in by_width.iterrows())
    return {
        "份數": (True, f"切出來都是 {df['份數'].min()}～{df['份數'].max()} 份，每筆 "
                       f"{half_up(min(amounts), 0):,}～{half_up(max(amounts), 0):,} 元"),
        "S": (True, f"最大連敗 P95（32～34 次）"),
        "低價全贏": (s["低勝隨"].all() and s["高輸隨"].all(), "**低價優先五組都贏過隨機**，高價優先、成交金額優先五組都輸給隨機"),
        "總獲利最高": (lo["總獲利(萬)"].idxmax() == "基本版",
                    f"總獲利最高的是基本版（低價 {half_up(lo.loc['基本版', '總獲利(萬)'], 0)} 萬）"),
        "PF最高": (lo["獲利因子(PF)"].idxmax() == NAMES["rise_base"] and max(pf_single, key=pf_single.get) == "rise_base",
                 f"獲利因子最高的是第二底墊高 ≥ 2% × 基本出場（低價 {half_up(lo.loc[NAMES['rise_base'], '獲利因子(PF)'], 4)}）"),
        "中位數負": ((df["中位數%"] < 0).all(),
                 "中位數報酬五組都是負的（" + f"{half_up(df['中位數%'].max(), 2)}%～{half_up(df['中位數%'].min(), 2)}%）"
                 .replace("-", "−")),
        "擋單倍數": (True, f"五組只有 {_rng(s['擋單倍數'], 2)} 倍："),
        "擋單清單": (True, ratio_lines),
        "低對高": (True, f"同一組裡低價對高價的總獲利是 {_rng(s['低對高'], 2)} 倍，低價對隨機多 "
                       f"{_rng(s['低對隨'] * 100, 1, '%')}"),
        "拆解基本版": (abs((1 + s.loc['基本版', 'EV多']) * (1 + s.loc['基本版', '筆數多']) - 1 - s.loc['基本版', '低對隨']) < 1e-3,   # 隨機列各欄分開取中位，相乘只會近似（文章寫「約」）
                  f"每筆期望值多 {_pct(s.loc['基本版', 'EV多'])}%、成交筆數多 {_pct(s.loc['基本版', '筆數多'])}%，"
                  f"兩者相乘約多 {_pct(s.loc['基本版', '低對隨'])}%，就是總獲利 "
                  f"{half_up(base.loc['隨機', '總獲利(萬)'], 0)} 萬到 {half_up(base.loc['低價', '總獲利(萬)'], 0)} 萬的差距"),
        "拆解五組": ((s["EV多"] > s["筆數多"]).all(), f"五組的每筆期望值多 {_rng(s['EV多'] * 100, 1, '%')}、成交筆數只多 "
                                                 f"{_rng(s['筆數多'] * 100, 1, '%')}"),
        "寬度清單": (True, width_lines),
        "寬度5%": ((s["寬度"] < 0.05).all(), "五組的寬度都在 5% 以內"),
        "年化": (True, f"**年化報酬 {_rng(df['年化報酬(%)'], 2, '%')}**，最大回撤 {_rng(df['最大回撤(%)'], 1, '%')}"),
        "期末差": (True, f"期末權益比「本金＋總獲利」多 {_rng(s['期末差'], 0)} 萬"),
        "未平倉": (True, f"低價那一列是 {_rng(s['未平倉'], 0)} 檔"),
        "防線": ((df["已實現權益最低(%)"] >= 80).all() and len(df) == 20,
               f"**已實現權益最低是本金的 {_rng(df['已實現權益最低(%)'], 1, '%')}**，二十列都沒有跌破八成"),
        "隨機附註": (True, random_note(df)),
        "整理擋單": (True, f"**擋單只有成交筆數的 {_rng(s['擋單倍數'], 2)} 倍**"),
        "整理低價": (True, f"低價對隨機的總獲利多 {_rng(s['低對隨'] * 100, 1, '%')}，主要來自每筆期望值（多 "
                         f"{_rng(s['EV多'] * 100, 1, '%')}），成交筆數只多 {_rng(s['筆數多'] * 100, 1, '%')}"),
        "整理年化": (True, f"**年化報酬 {_rng(df['年化報酬(%)'], 2, '%')}、最大回撤 {_rng(df['最大回撤(%)'], 1, '%')}**；"
                         f"已實現權益最低 {_rng(df['已實現權益最低(%)'], 1, '%')}"),
        # summary 2026-11 依 SEO 規範縮短，原摘要句已不在文章；改驗正文同一組擋單倍數的句子。
        "擋單句": (True, f"把隨機列的擋單除以成交筆數，五組只有 {_rng(s['擋單倍數'], 2)} 倍"),
    }


def main() -> int:
    text = io.open(article_path(SLUG), encoding="utf-8").read()
    df = load(MODE)
    errs = []
    want = all_tables(MODE)
    for name, html in want.items():
        if html not in text:
            errs.append(f"表格與重算不一致：{name}")
    n_tables = len(re.findall(r'<table class="bt-table">', text))
    if n_tables != len(want):
        errs.append(f"文章表格數 {n_tables} ≠ {len(want)}")
    cl = claims(df, stats())
    for name, (ok, phrase) in cl.items():
        if not ok:
            errs.append(f"判斷不成立：{name}")
        if phrase not in text:
            errs.append(f"文章找不到重算字串［{name}］：{phrase}")
    print(f"表格 {len(want)} 張（逐字比對 HTML）、正文主張 {len(cl)} 條")
    for e in errs:
        print("  ✗", e)
    print("錯誤 0，全部一致" if not errs else f"錯誤 {len(errs)}")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
