# -*- coding: utf-8 -*-
"""
均線交叉系列（一）～（七）文章表格產生器與驗證腳本的共用工具（分析段，不跑回測）。

資料來源（全部是 SPA 回測段／分析段的產出，這裡只讀）：
  單股：_02_strategy/ma_strategy/result/ma_cross/<variant>/ma_cross_<短>_<長>_trades.parquet
        （_02_strategy/ma_strategy/ma_cross_sweep.py）
  蒙地卡羅：_02_strategy/ma_strategy/result/ma_cross/_mc_realistic/mc_realistic.csv
        （_04_analysis/ma_cross/ma_cross_montecarlo.py）
  多股：_03_multi_strategy/ma_cross/result/<task>/orderings.csv、random_dist.csv
        （_03_multi_strategy/ma_cross/ma_cross_multi_driver.py）

單股數字一律由逐筆交易「重算未捨入值」再四捨五入到文章顯示位數，不讀 aggregate CSV 的
兩位小數——aggregate 已先捨入過一次，顯示一位小數時若第二位剛好是 5（如 −5.75），
二次捨入會跟真值的四捨五入不一致（真值 −5.747 應顯示 −5.7）。

四捨五入一律「.5 遠離 0」（Decimal ROUND_HALF_UP），與讀者認知一致；Python 的 round
是銀行家進位加二進位誤差，40.65 會變 40.6。
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
from _02_strategy.ma_strategy.ma_cross_sweep import PAIRS  # noqa: E402

SINGLE_DIR = common.result_dir("ma_strategy", "ma_cross")
MC_CSV = os.path.join(SINGLE_DIR, "_mc_realistic", "mc_realistic.csv")
MULTI_DIR = os.path.join(_root, "_03_multi_strategy", "ma_cross", "result")
MC_VARIANT = "angle20_adx25_liq1000"      # （四）～（七）的代表配置：夾角>20°＋ADX<25＋1000 張
TRADING_DAYS = 5_949                      # 2002–2025 交易日數（蒙地卡羅抽樣下限＝×3）

# 文章表頭 → 指標鍵（同一個指標在不同篇的表頭寫法略有不同）
HEADER_KEY = {
    "交易次數": "n", "交易數": "n", "勝率%": "wr", "平均持有天": "hold",
    "獲利平均%": "aw", "虧損平均%": "al", "中位數%": "med",
    "期望值(EV)/筆(元)": "ev", "期望值(EV)/筆": "ev", "期望值/筆(元)": "ev",
    "獲利因子(PF)": "pf", "獲利因子": "pf", "總獲利(萬)": "tot", "擋單": "blk",
}


# ── 數字格式 ────────────────────────────────────────────────────────────────
def half_up(v, nd: int) -> Decimal:
    """四捨五入到 nd 位（.5 遠離 0）；以字串轉 Decimal 避開二進位誤差。"""
    return Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)


def fmt(v, nd: int = 0, comma: bool = True, sign: bool = False) -> str:
    """文章用格式：全形減號、千分位；sign=True 時正數加「+」。"""
    d = half_up(v, nd)
    if d == 0:
        d = abs(d)                       # 避免出現「−0」
    s = f"{d:,}" if comma else f"{d}"
    if sign and d > 0:
        s = "+" + s
    return s.replace("-", "−")


def num(text: str) -> float:
    """文章數字還原：全形減號、千分位、正號、%。"""
    t = re.sub(r"<[^>]+>", "", text)
    t = t.replace("−", "-").replace(",", "").replace("%", "").replace("+", "").strip()
    return float(t)


def decimals(text: str) -> int:
    t = re.sub(r"<[^>]+>", "", text).strip().rstrip("%萬元倍")
    return len(t.split(".")[1]) if "." in t else 0


def same(cell: str, exact) -> bool:
    """文章顯示值是否等於真值四捨五入到「文章顯示的位數」（嚴格相等，不給容差）。"""
    nd = decimals(cell)
    return Decimal(repr(num(cell))).quantize(Decimal(1).scaleb(-nd)) == half_up(exact, nd)


# ── 驗證計數 ────────────────────────────────────────────────────────────────
class Checker:
    """累計檢查數與錯誤；每支 verify 腳本一個。"""

    def __init__(self, name: str):
        self.name, self.n, self.errs = name, 0, []

    def check(self, ok: bool, msg: str) -> bool:
        self.n += 1
        if not ok:
            self.errs.append(msg)
            print("  X " + msg)
        return ok

    def phrase(self, text: str, s: str, why: str = "") -> bool:
        """由數據組出的句子片段必須原樣出現在文章裡（數據或文章任一邊變了都會報錯）。"""
        return self.check(s in text, f"文章找不到「{s}」{('（' + why + '）') if why else ''}")

    def done(self) -> int:
        print(f"{self.name}：檢查 {self.n} 項，錯誤 {len(self.errs)}")
        return 1 if self.errs else 0


# ── 文章讀取與 HTML 表格解析 ─────────────────────────────────────────────────
def post_text(slug: str) -> str:
    path = os.path.join(common.require_blog_dir(), "site", "content", "posts", f"{slug}.md")
    with open(path, encoding="utf-8") as f:
        return f.read()


def section(text: str, start: str, end: str = None) -> str:
    """取 start 標題到 end 標題之間；找不到標題直接停（文章改版要先改驗證腳本）。"""
    if start not in text:
        raise SystemExit(f"文章結構與驗證腳本不符，找不到：{start}")
    body = text.split(start, 1)[1]
    if end:
        if end not in body:
            raise SystemExit(f"文章結構與驗證腳本不符，找不到：{end}")
        body = body.split(end, 1)[0]
    return body


def tables(html: str) -> list:
    """回傳每張 <table> 的原始 HTML（依出現順序）。"""
    return re.findall(r"<table.*?</table>", html, flags=re.S)


def headers(table: str) -> list:
    """最後一列表頭的 <th> 文字（去掉 <br> 與標籤）。"""
    rows = re.findall(r"<tr>(.*?)</tr>", table.split("</thead>")[0], flags=re.S)
    ths = re.findall(r"<th[^>]*>(.*?)</th>", rows[-1], flags=re.S)
    return [re.sub(r"<[^>]+>", "", t).replace("&lt;", "<").strip() for t in ths]


def body_rows(table: str) -> list:
    """[(td 屬性 class, td 內容), ...] 的列表（只取 tbody）。"""
    body = table.split("<tbody>", 1)[1] if "<tbody>" in table else table
    out = []
    for tr in re.findall(r"<tr>(.*?)</tr>", body, flags=re.S):
        tds = re.findall(r"<td([^>]*)>(.*?)</td>", tr, flags=re.S)
        if tds:
            out.append([(cls_of(a), c.strip()) for a, c in tds])
    return out


def cls_of(attr: str) -> str:
    m = re.search(r"class=['\"]([^'\"]+)['\"]", attr)
    return m.group(1) if m else ""


# ── 單股：逐筆重算 ──────────────────────────────────────────────────────────
@lru_cache(maxsize=None)
def trades(variant: str, pair: str) -> pd.DataFrame:
    tag = pair.replace("/", "_")
    return pd.read_parquet(os.path.join(SINGLE_DIR, variant, f"ma_cross_{tag}_trades.parquet"))


@lru_cache(maxsize=None)
def metrics(variant: str, pair: str) -> dict:
    """
    與 common.summarize_trades 同定義、但不捨入：排除淨損益＝0；報酬率為買賣價毛報酬；
    金額（期望值、總獲利）為淨損益。
    """
    t = trades(variant, pair)
    t = t[t["real_pnl"] != 0]
    p = t["real_pnl"].to_numpy(np.float64)
    rate = ((t["sell_price"] - t["buy_price"]) / t["buy_price"] * 100).to_numpy(np.float64)
    hold = (pd.to_datetime(t["sell_date"]) - pd.to_datetime(t["buy_date"])).dt.days.to_numpy()
    win, lose = p > 0, p < 0
    wr = win.mean()
    ev = wr * p[win].mean() + (1 - wr) * p[lose].mean()
    return {"n": len(p), "wr": wr * 100, "hold": hold.mean(), "aw": rate[win].mean(),
            "al": rate[lose].mean(), "med": float(np.median(rate)), "ev": ev,
            "pf": p[win].sum() / -p[lose].sum(), "tot": p.sum() / 10_000,
            "stocks": t["stock_id"].nunique()}


def all_metrics(variant: str) -> dict:
    return {p: metrics(variant, p) for p in PAIRS}


# ── 蒙地卡羅 ────────────────────────────────────────────────────────────────
def mc() -> pd.DataFrame:
    return pd.read_csv(MC_CSV).set_index("短/長")


@lru_cache(maxsize=None)
def mc_exact_wr(pair: str) -> float:
    """蒙地卡羅表的勝率（net_stats 口徑：分母含淨損益＝0 的筆）未捨入值，只供 .5 邊界判斷。"""
    t = trades(MC_VARIANT, pair)
    return float((t["real_pnl"] > 0).mean() * 100)


# ── 多股 ────────────────────────────────────────────────────────────────────
EXACT = "_精確"


def with_exact(df: pd.DataFrame) -> pd.DataFrame:
    """
    多股 CSV 若帶「<欄>_精確」（driver 從逐筆交易算的未四捨五入值），就用它覆蓋同名的存檔欄，
    出表與驗證一律從精確值一次進位（同 KD 的 kd_article_common.with_exact）；舊 CSV 沒有就照存檔值。
    """
    out = df.copy()
    for c in df.columns:
        if c.endswith(EXACT) and c[:-len(EXACT)] in out.columns:
            base = c[:-len(EXACT)]
            out[base] = out[c].where(out[c].notna(), out[base])
    return out


def multi(task: str = "angle_adx") -> pd.DataFrame:
    """固定排序列＋隨機代表列（最終權益中位數那一次）合成一張表；有精確欄就以精確值為準。"""
    d = os.path.join(MULTI_DIR, task)
    det = pd.read_csv(os.path.join(d, "orderings.csv"))
    rnd_path = os.path.join(d, "random_dist.csv")
    rnd = pd.read_csv(rnd_path) if os.path.isfile(rnd_path) else pd.DataFrame()
    return with_exact(pd.concat([det, rnd], ignore_index=True))


def mrow(df: pd.DataFrame, pair: str, sizing: str, order: str) -> pd.Series:
    r = df[(df["短/長"] == pair) & (df["投法"] == sizing) & (df["排序"] == order)]
    if len(r) != 1:
        raise SystemExit(f"多股結果找不到唯一列：{pair}｜{sizing}｜{order}（{len(r)} 列）")
    return r.iloc[0]


def multi_key(r: pd.Series) -> dict:
    """多股 CSV 列 → 指標鍵（與單股同一組鍵，方便共用比對）。"""
    return {"n": r["交易次數"], "wr": r["勝率%"], "hold": r["平均持有天"], "aw": r["獲利平均%"],
            "al": r["虧損平均%"], "med": r["中位數%"], "ev": r["期望值/筆"], "pf": r["獲利因子"],
            "tot": r["總獲利(萬)"], "blk": r["擋單"]}


# ── 單股表格驗證（（一）～（三）共用）──────────────────────────────────────────
def verify_value_table(ck: Checker, table: str, variant: str, label: str,
                       color: str = None) -> None:
    """
    一般值表（（一）兩張）：每格對逐筆重算值；color="sign" 時再驗正紅負綠（pos／neg）。
    上色欄＝獲利平均、虧損平均、中位數、期望值、總獲利；其餘欄不得上色。
    """
    keys = [HEADER_KEY[h] for h in headers(table)[1:]]
    rows = body_rows(table)
    ck.check([r[0][1] for r in rows] == PAIRS, f"{label}：列順序／組數與 21 組不符")
    for row in rows:
        pair = row[0][1]
        m = metrics(variant, pair)
        for key, (cls, cell) in zip(keys, row[1:]):
            ck.check(same(cell, m[key]), f"{label} {pair} {key}：文 {cell} vs 真值 {m[key]:.6f}")
            if color == "sign":
                want = ("pos" if m[key] > 0 else "neg") if key in ("aw", "al", "med", "ev", "tot") else ""
                ck.check(cls == want, f"{label} {pair} {key} 底色：文 {cls or '無'} vs 應 {want or '無'}")


def verify_arrow_table(ck: Checker, table: str, base: str, variant: str, label: str) -> None:
    """「基準→調整」表：兩側數字都對逐筆重算值；五個方向欄依真值比較上色。"""
    keys = [HEADER_KEY[h] for h in headers(table)[1:]]
    rows = body_rows(table)
    ck.check([r[0][1] for r in rows] == PAIRS, f"{label}：列順序／組數與 21 組不符")
    for row in rows:
        pair = row[0][1]
        b, a = metrics(base, pair), metrics(variant, pair)
        for key, (cls, cell) in zip(keys, row[1:]):
            left, right = cell.split("→")
            ck.check(same(left, b[key]), f"{label} {pair} {key} 基準：文 {left} vs {b[key]:.6f}")
            ck.check(same(right, a[key]), f"{label} {pair} {key} 調整：文 {right} vs {a[key]:.6f}")
            want = ("up" if a[key] > b[key] else "down") if key in ("wr", "aw", "med", "ev", "pf") else ""
            ck.check(cls == want, f"{label} {pair} {key} 底色：文 {cls or '無'} vs 應 {want or '無'}")
