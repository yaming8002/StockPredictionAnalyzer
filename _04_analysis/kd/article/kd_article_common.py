# -*- coding: utf-8 -*-
"""
KD 交叉系列（一）～（八）出表／驗證共用工具（分析段，不跑回測）。

資料來源全部是 SPA 的回測輸出（回測程式都在 SPA / GitHub，這裡只排版與對帳）：
  單股：_02_strategy/kd_strategy/result/single_kd/<變體>/single_kd_trades.parquet（kd_sweep.py）
  蒙地卡羅：result/kd_mc/kd_montecarlo.csv（_04_analysis/kd/kd_montecarlo.py）
  多股：result/kd_multi/kd_multi_result.csv、units.csv（_03_multi_strategy/kd/kd_multi_driver.py）
  結論：result/kd_multi/kd_conclusion*.csv（_04_analysis/kd/kd_multi_conclusion.py）

四捨五入一律「.5 遠離 0」（Decimal、以字串轉避開二進位誤差）。單股表的數字**不用 aggregate CSV
的已四捨五入值**，而是從逐筆交易重算未四捨五入的精確值再進位——aggregate 存兩位小數，顯示一位時
常剛好落在 .x5，兩次進位會跟一次進位不同（例如 7.149 → 7.15 → 7.2，正確是 7.1）。
多股／結論 CSV 由 driver 另存「<欄>_精確」未四捨五入值（kd_multi_driver.exact_stats、
kd_multi_conclusion.exact_curve），出表一律用它；舊版 CSV 沒有精確欄時，落在 .5 上的格子才退回用
能還原的資訊（勝率＝整數勝場、總獲利＝期望值×筆數）判方向，還原不了的記成「未決」列為警告。
"""
import os
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _02_strategy.base.vbt import common  # noqa: E402

SWEEP = common.result_dir("kd_strategy", "single_kd")
MC_DIR = common.result_dir("kd_strategy", "kd_mc")
MULTI = common.result_dir("kd_strategy", "kd_multi")
RANK_CSV = os.path.join(common.result_dir("kd_strategy", "kd_ranking"), "kd_entry_ranking.csv")
MINUS = "−"

# 規格 9 欄（順序即文章表格欄序）
SPEC = ["交易次數", "勝率%", "平均持有天", "獲利平均%", "虧損平均%", "中位數%",
        "期望值/筆", "獲利因子", "總獲利(萬)"]
# 存檔 aggregate 欄名（sanity check 用）
_AGG = {"交易次數": "交易次數", "勝率%": "勝率(%)", "平均持有天": "平均持有天數",
        "獲利平均%": "平均獲利報酬率(%)", "虧損平均%": "平均虧損報酬率(%)",
        "中位數%": "中位數報酬率(%)", "期望值/筆": "期望報酬值(EV)", "獲利因子": "獲利因子(PF)"}

# 落在 .5 上、存檔資料還原不了方向的格子（驗證腳本列警告、最終報告列未決）
UNRESOLVED = []


# ── 數字格式 ───────────────────────────────────────────────────────────────────
def half_up(v, nd: int) -> Decimal:
    """四捨五入（.5 遠離 0）；以字串轉 Decimal，避開 0.1 之類的二進位誤差。"""
    d = v if isinstance(v, Decimal) else Decimal(repr(float(v)))
    return d.quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)


def fmt(v, nd: int, sign: bool = False, comma: bool = True) -> str:
    """文章數字格式：全形減號、千分位、sign=True 時正數加「+」；四捨五入成 0 不帶正負號。"""
    q = half_up(v, nd)
    if q == 0:
        q = abs(q)
    s = f"{q:,}" if comma else f"{q}"
    s = s.replace("-", MINUS)
    if sign and q > 0:
        s = "+" + s
    return s


def is_tie(v, nd: int) -> bool:
    """存檔值是否剛好落在顯示位數的 .5 上（此時存檔值不足以決定進位方向）。"""
    d = Decimal(repr(float(v))).scaleb(nd)
    return abs(d - d.to_integral_value(rounding="ROUND_DOWN")) == Decimal("0.5")


# ── 單股：從逐筆交易重算精確值 ─────────────────────────────────────────────────
@lru_cache(maxsize=None)
def single_metrics(variant: str) -> dict:
    """
    照 common.summarize_trades 的定義重算規格 9 欄，但**不四捨五入**（排除淨損益 0 的交易）。
    重算後與 aggregate CSV 對一次（容許其存檔位數的誤差），對不上代表定義抄錯，直接停。
    """
    folder = os.path.join(SWEEP, variant)
    t = pd.read_parquet(os.path.join(folder, "single_kd_trades.parquet"))
    t = t[pd.to_numeric(t["real_pnl"], errors="coerce").fillna(0.0) != 0]
    pnl = t["real_pnl"].astype(float)
    rate = (t["sell_price"] - t["buy_price"]) / t["buy_price"] * 100
    days = (pd.to_datetime(t["sell_date"]) - pd.to_datetime(t["buy_date"])).dt.days
    win, lose = pnl > 0, pnl < 0
    n = len(t)
    out = {"交易次數": n, "勝率%": win.sum() / n * 100, "平均持有天": float(days.mean()),
           "獲利平均%": float(rate[win].mean()), "虧損平均%": float(rate[lose].mean()),
           "中位數%": float(rate.median()), "期望值/筆": float(pnl.sum()) / n,
           "獲利因子": float(pnl[win].sum()) / abs(float(pnl[lose].sum())),
           "總獲利(萬)": float(pnl.sum()) / 10_000}
    agg = pd.read_csv(os.path.join(folder, "single_kd_aggregate.csv"), encoding="utf-8-sig").iloc[0]
    for k, col in _AGG.items():
        tol = 0.0051 if k != "獲利因子" else 0.00051
        if abs(float(agg[col]) - out[k]) > tol:
            raise SystemExit(f"{variant} 重算 {k}={out[k]} 與 aggregate {agg[col]} 不符，定義需檢查")
    if abs(float(agg["總獲利"]) / 10_000 - out["總獲利(萬)"]) > 0.001:
        raise SystemExit(f"{variant} 重算總獲利與 aggregate 不符")
    return out


def has_variant(variant: str) -> bool:
    return os.path.isfile(os.path.join(SWEEP, variant, "single_kd_trades.parquet"))


# ── 多股：存檔值＋ .5 方向還原 ─────────────────────────────────────────────────
def _resolve_winrate(n: int, stored: float):
    """勝率＝整數勝場 ÷ 筆數；找出唯一能四捨五入成存檔值的勝場數，回傳精確勝率。"""
    guess = int(stored / 100 * n)
    hits = [w for w in range(guess - 3, guess + 4) if round(w / n * 100, 2) == stored]
    return hits[0] / n * 100 if len(hits) == 1 else None


def _resolve_total(n: int, ev: float, stored: float):
    """總獲利＝期望值 × 筆數；用期望值（兩位小數）推回總獲利的區間，整段落在 .5 同一側就能定方向。"""
    lo = max((ev - 0.005) * n, (stored - 0.05) * 10_000) / 10_000
    hi = min((ev + 0.005) * n, (stored + 0.05) * 10_000) / 10_000
    edge = float(half_up(stored, 1))           # 存檔值本身就是 .5 的那條線
    if hi < edge:
        return edge - 0.01
    if lo > edge:
        return edge + 0.01
    return None


EXACT = "_精確"


def with_exact(df: pd.DataFrame) -> pd.DataFrame:
    """
    多股／結論 CSV 若帶「<欄>_精確」（driver 從逐筆交易或權益曲線算的未四捨五入值），
    就用它覆蓋同名的存檔欄，出表與正文統計一律從精確值一次進位。
    """
    out = df.copy()
    for c in df.columns:
        if c.endswith(EXACT) and c[:-len(EXACT)] in out.columns:
            base = c[:-len(EXACT)]
            out[base] = out[c].where(out[c].notna(), out[base])
    return out


def multi_cell(row: pd.Series, col: str, nd: int, where: str):
    """
    取多股 CSV 某格要拿去進位的值：不在 .5 上就用存檔值；在 .5 上先試著還原方向，
    還原不了就照存檔值進位、記進 UNRESOLVED（顯示值可能差最後一位，需重跑才能定）。
    """
    v = float(row[col])
    if col + EXACT in row.index and pd.notna(row[col + EXACT]):
        return float(row[col + EXACT])            # driver 有存精確值就直接用
    if not is_tie(v, nd):
        return v
    exact = None
    if col == "勝率%":
        exact = _resolve_winrate(int(row["交易次數"]), v)
    elif col == "總獲利(萬)":
        exact = _resolve_total(int(row["交易次數"]), float(row["期望值/筆"]), v)
    if exact is None:
        UNRESOLVED.append(f"{where}｜{col} 存檔 {v} 落在 .5、無逐筆資料可定方向（暫依存檔值進位）")
        return v
    return exact


# ── 文章讀取與表格解析 ─────────────────────────────────────────────────────────
def read_post(slug: str) -> str:
    path = os.path.join(common.require_blog_dir(), "site", "content", "posts", f"{slug}.md")
    with open(path, encoding="utf-8") as f:
        return f.read()


def tables_of(text: str) -> list:
    """依出現順序回傳文章裡每張 bt-table 的 HTML。"""
    return re.findall(r'<table class="bt-table">(.*?)</table>', text, flags=re.S)


def rows_of(html: str) -> list:
    """回傳每列的 [(class, 內容), ...]；只取 tbody 的 <td> 列。"""
    out = []
    for tr in re.findall(r"<tr>(.*?)</tr>", html, flags=re.S):
        if "<td" not in tr:
            continue
        cells = []
        for attr, content in re.findall(r"<td([^>]*)>(.*?)</td>", tr, flags=re.S):
            m = re.search(r'class="([^"]*)"', attr)
            cells.append((m.group(1) if m else "", content))
        out.append(cells)
    return out


def html_row(cells: list) -> str:
    """[(class, 內容)] → <tr>（builder 輸出用；class 空字串就不寫屬性）。"""
    tds = "".join(f'<td class="{c}">{v}</td>' if c else f"<td>{v}</td>" for c, v in cells)
    return f"<tr>{tds}</tr>"


def strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).strip()


# ── 檢查器 ─────────────────────────────────────────────────────────────────────
class Checker:
    """計數＋收集錯誤；每篇驗證腳本一個實例，最後 done() 印總結並回傳 exit code。"""

    def __init__(self, title: str):
        self.title, self.n, self.errs, self.warns = title, 0, [], []
        print(f"=== {title} ===")

    def check(self, ok: bool, msg: str) -> bool:
        self.n += 1
        if not ok:
            self.errs.append(msg)
            print("  X " + msg)
        return ok

    def contains(self, text: str, phrase: str, what: str) -> bool:
        """正文統計句：由資料算出應有的句子，確認文章真的這樣寫。"""
        return self.check(phrase in text, f"{what}：文章找不到「{phrase}」")

    def warn(self, msg: str):
        self.warns.append(msg)
        print("  ! " + msg)

    def tables(self, text: str, want: list, start: int = 0):
        """
        want＝[(表名, [列 cells...]), ...]，依序對文章第 start 張起的表。
        每格「內容字串」與「class」都要完全相同（數字位數、正負號、千分位、底色一起驗）。
        """
        got = tables_of(text)
        for i, (name, rows) in enumerate(want):
            if not self.check(start + i < len(got), f"{name}：文章表格數不足"):
                continue
            art = rows_of(got[start + i])
            self.check(len(art) == len(rows), f"{name}：列數 文 {len(art)} vs 應 {len(rows)}")
            for r, (a_row, w_row) in enumerate(zip(art, rows), start=1):
                if not self.check(len(a_row) == len(w_row),
                                  f"{name} 第 {r} 列欄數 文 {len(a_row)} vs 應 {len(w_row)}"):
                    continue
                for c, ((ac, av), (wc, wv)) in enumerate(zip(a_row, w_row), start=1):
                    self.check(av.strip() == wv and ac == wc,
                               f"{name} 第 {r} 列第 {c} 欄：文 [{ac}]{av.strip()} vs 應 [{wc}]{wv}")
            print(f"  {name}：{len(rows)} 列")

    def done(self) -> int:
        for u in UNRESOLVED:
            if u not in self.warns:
                self.warn("未決 " + u)
        print(f"--- {self.title}：檢查 {self.n} 項，錯誤 {len(self.errs)}，警告 {len(self.warns)}")
        return 1 if self.errs else 0
