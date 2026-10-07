# -*- coding: utf-8 -*-
"""
驗第九篇的兩張表與正文統計。

第九篇的表 2 有兩種來源，這裡分開對：
- 「替換出場規則」「未平倉」「持有天變化」的終點 → 本篇自己跑的
  `macd_exit_replace_all.csv`。
- 「原本的出場」「附加出場規則」「持有天變化」的起點 → **直接去讀（四）（七）（八）
  三篇已發佈的 md**，而不是讀 CSV。這樣驗的是「第九篇有沒有正確引用前面的文章」，
  讀者翻回去看到的就是同一個數字。
表 1（被排除的兩條）仍對 SPA 那一輪的 `_exit_pure_risk.csv`，末欄取自（八）篇、不在比對範圍。

執行：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python \
        _04_analysis/macd/article/verify_article9.py
"""

import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
from _02_strategy.base.vbt import common  # noqa: E402
import io
import re
import sys

import pandas as pd

POSTS = os.path.join(common.require_blog_dir(), "site", "content", "posts")
R = common.result_dir("macd_strategy", "single_macd")
p = pd.read_csv(R + "/_exit_pure_risk.csv")
rep = pd.read_csv(os.path.join(common.require_blog_dir(), "reference", "macd",
                               "data", "macd_exit_replace_all.csv"))
LAB1 = {"固定停利 +20%": "停利+20%", "固定停損 2×ATR": "停損2ATR"}
# 第九篇顯示名 → (replace CSV 的出場名, 附加數字來自哪一篇, 該篇的節標題關鍵字)
RULES = {
    "跌破年線": ("跌破MA200", "macd-exit-trend", "跌破年線"),
    "超級趨勢": ("Supertrend翻空", "macd-exit-trend", "超級趨勢"),
    "拋物線 SAR": ("SAR翻空", "macd-exit-trend", "拋物線 SAR"),
    "波段高點走低": ("頂頂低", "macd-exit-trend", "波段高點走低"),
    "跌破二十日低": ("跌破20日低", "macd-exit-trend", "跌破二十日低"),
    "吊燈 3×ATR": ("吊燈3ATR", "macd-exit-risk", "吊燈出場"),
    "自最高點回落 10%": ("自最高點回落10%", "macd-exit-risk", "自最高點回落一成"),
    "抱滿 60 天": ("抱滿60天", "macd-exit-risk", "抱滿六十天"),
}
checked = errs = 0


def read(slug):
    return io.open(f"{POSTS}/{slug}.md", encoding="utf-8").read()


def fail(msg):
    global errs
    print("  X " + msg)
    errs += 1


def cells(rest):
    return [re.sub(r"<[^>]+>", "", c) for c in re.findall(r"<td[^>]*>(.*?)</td>", rest)]


def add_pf(slug, keyword):
    """抓（七）（八）篇某一節表格的三個母體獲利因子（第 8 欄）。"""
    for sec in re.split(r"^## ", read(slug), flags=re.M)[1:]:
        if keyword not in sec.split("\n", 1)[0]:
            continue
        out = {}
        for pop, rest in re.findall(r"<tr><td>(交叉|零軸|背離)</td>(.*?)</tr>", sec):
            out[pop] = float(cells(rest)[7])
        if len(out) == 3:
            return out
    sys.exit(f"（{slug}）找不到節「{keyword}」的三母體表")


def baselines():
    """（四）篇矩陣表的三條對角線：獲利因子與持有天。"""
    t = read("macd-matrix")
    want = {"交叉": "交叉進 × 交叉出", "零軸": "零軸進 × 零軸出", "背離": "背離進 × 交叉出"}
    out = {}
    for pop, label in want.items():
        m = re.search(r"<tr><td>" + re.escape(label) + r"(?:<br>[^<]*)?</td>(.*?)</tr>", t)
        if not m:
            sys.exit(f"（四）篇找不到 {label}")
        c = cells(m.group(1))
        out[pop] = (float(c[7]), float(c[2]))      # 獲利因子, 平均持有天
    return out


text = read("macd-exit-pure")
BASE = baselines()
ADD = {name: add_pf(slug, kw) for name, (_, slug, kw) in RULES.items()}

# ── 表 1：被排除的兩條（末欄不比對）──
T1 = re.compile(r'<tr><td>(固定[^<]+)</td><td>(交叉|零軸|背離)</td>'
                r'<td>([\d,]+)</td><td[^>]*>([\d.]+)%</td>'
                r'<td>(\d+)</td><td>([\d.]+)</td><td>[\d.]+</td></tr>')
rows1 = T1.findall(text)
print(f"  表1（排除）{len(rows1)} 列（預期 6）")
if len(rows1) != 6:
    errs += 1
for x, b, n_in, unc, days, wr in rows1:
    r = p[(p["基礎"] == b) & (p["出場"] == LAB1[x.strip()])].iloc[0]
    for got, want, name, tol in ((float(n_in.replace(",", "")), r["進場筆數"], "進場筆數", 0),
                                 (float(unc), r["未平倉%"], "未平倉%", 0.011),
                                 (float(days), r["平均持有天"], "持有天", 0.5),
                                 (float(wr), r["勝率%"], "勝率%", 0.011)):
        checked += 1
        if abs(got - want) > tol:
            fail(f"表1 {b}×{x} {name}: 文 {got} vs 來源 {want}")

# ── 表 2：8 條 × 3 母體 ──
T2 = re.compile(r'<tr><td>(交叉|零軸|背離)</td><td>([^<]+)</td>'
                r'<td>([\d.]+)%</td><td>([\d.]+)</td>'
                r'<td>([\d.]+)</td><td[^>]*>([\d.]+)</td>'
                r'<td>(\d+) → (\d+)</td></tr>')
rows2 = T2.findall(text)
print(f"  表2（保留）{len(rows2)} 列（預期 24）")
if len(rows2) != 24:
    errs += 1
n_add = n_base = 0
for b, x, unc, pb, addv, repv, d0, d1 in rows2:
    key = x.strip()
    if key not in RULES:
        fail(f"表2 出現未知出場：{key}")
        continue
    r = rep[(rep["母體"] == b) & (rep["出場"] == RULES[key][0])].iloc[0]
    base_pf, base_days = BASE[b]
    for got, want, name, tol in ((float(unc), r["未平倉%"], "未平倉%", 0.011),
                                 (float(pb), base_pf, "基準線（四篇）", 0.0001),
                                 (float(addv), ADD[key][b], "附加（七/八篇）", 0.0001),
                                 (float(repv), r["獲利因子"], "替換", 0.0001),
                                 (float(d0), base_days, "原持有天", 0.5),
                                 (float(d1), r["平均持有天"], "新持有天", 0.5)):
        checked += 1
        if abs(got - want) > tol:
            fail(f"表2 {b}×{key} {name}: 文 {got} vs 來源 {want}")
    n_add += r["獲利因子"] > ADD[key][b]
    n_base += r["獲利因子"] > base_pf

print("  正文統計：")
for name, actual, claimed in (("替換勝過附加", n_add, 23),
                              ("替換勝過基準線", n_base, 14)):
    checked += 1
    ok = actual == claimed
    if not ok:
        errs += 1
    print(f"    {'OK' if ok else 'NG'} {name}：文中 {claimed}，實際 {actual}")

print(f"\n對照 {checked} 個數值，錯誤 {errs} 個")
sys.exit(1 if errs else 0)
