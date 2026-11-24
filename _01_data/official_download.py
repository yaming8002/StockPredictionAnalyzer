"""
取得股票資料（官方篇）：從證交所／櫃買中心下載個股日成交資料

yfinance 的台股歷史有部分股票不完整（整段缺漏、沒成交卻有價格、興櫃期間混在一起等），
這支程式改向官方免費、免金鑰的端點查詢，一次查「一檔股票的一個月」：

- 上市（.TW）：證交所 STOCK_DAY，單股查詢最早只到 2010-01（2026-11 實測）。
- 上櫃（.TWO）：櫃買中心 tradingStock，至少到 2002-01；成交量單位是「張」，這裡一律換成「股」。

官方資料的特性（使用前要知道）：
- 價格是原始價：沒有還原股票股利、減資、分割。
- 日期是民國年，偶爾帶 `*`／`＊` 註記；當天沒成交時價格是 `--`、量 0。
- 不含興櫃；上櫃轉上市的股票，轉上市前要改查櫃買。
- 查太快會被擋（回 HTML 不是 JSON）。實測間隔 2～2.5 秒約 50 次就被擋，
  3～3.5 秒＋被擋後冷卻 5 分鐘可以連跑兩天；全部跑完大約每小時 900 個月。

執行（查 2330.TW 2024-01～2024-03，存到 _01_data/result/official_raw/2330.TW.csv，已查過的月份會跳過）：
  python _01_data/official_download.py 2330.TW 2024-01 2024-03
"""
import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(_HERE, "result", "official_raw")   # 官方原始資料快取（一檔一個 CSV；result/ 不進版控）

TWSE_MIN_MONTH = pd.Period("2010-01", "M")   # 證交所單股查詢最早的月份
SLEEP_SEC, SLEEP_JITTER = 3.0, 0.5           # 每次查詢間隔 3～3.5 秒
BLOCK_COOLDOWN_SEC = 5 * 60                  # 被擋後先等 5 分鐘再查
UA = {"User-Agent": "Mozilla/5.0"}
RAW_COLS = ["date", "volume", "open", "high", "low", "close", "month", "title"]


class BlockedError(RuntimeError):
    """官方擋下（連線失敗或回傳不是 JSON）。"""


def http_json(url: str, body: bytes = None) -> dict:
    """GET（或帶 body 的 POST）並解析 JSON；被擋時官方回的是 HTML，轉成 BlockedError。"""
    req = urllib.request.Request(url, data=body, headers=UA)
    try:
        text = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
    except urllib.error.URLError as e:
        raise BlockedError(f"連線失敗：{e}") from e
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise BlockedError(f"非 JSON 回應（可能被擋）：{text[:80]!r}") from e


def to_num(text) -> float:
    """「1,234.5」→ 1234.5；沒成交的 "--" → NaN。"""
    text = str(text).replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return np.nan


def roc_to_date(text) -> pd.Timestamp:
    """民國日期「113/01/02」→ 2024-01-02；先去掉 `*`／`＊` 這類註記。"""
    y, m, d = re.sub(r"[^0-9/]", "", str(text)).split("/")
    return pd.Timestamp(int(y) + 1911, int(m), int(d))


def fetch_month(sid: str, month: str) -> tuple:
    """查一檔一個月，回傳（逐日 DataFrame，官方標題）；量一律是「股」。"""
    code = sid.split(".")[0]
    y, m = month.split("-")
    if sid.endswith(".TW"):
        js = http_json("https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY"
                       f"?date={y}{m}01&stockNo={code}&response=json")
        rows, title, vol_mult = js.get("data") or [], js.get("title", js.get("stat", "")), 1
    else:
        js = http_json("https://www.tpex.org.tw/www/zh-tw/afterTrading/tradingStock",
                       f"code={code}&date={y}/{m}/01&response=json".encode())
        table = (js.get("tables") or [{}])[0]
        rows, title, vol_mult = table.get("data") or [], js.get("name", table.get("title", "")), 1000
    # 兩邊欄位順序相同：日期、成交量、成交金額、開、高、低、收…
    recs = [{"date": roc_to_date(r[0]), "volume": to_num(r[1]) * vol_mult, "open": to_num(r[3]),
             "high": to_num(r[4]), "low": to_num(r[5]), "close": to_num(r[6])} for r in rows]
    return pd.DataFrame(recs), str(title)


def fetch_month_polite(sid: str, month: str) -> tuple:
    """fetch_month 加上查詢間隔；被擋就冷卻 5 分鐘再查同一個月（不跳過，避免漏資料）。"""
    while True:
        try:
            result = fetch_month(sid, month)
            break
        except BlockedError as e:
            print(f"被擋，冷卻 {BLOCK_COOLDOWN_SEC // 60} 分鐘：{e}", flush=True)
            time.sleep(BLOCK_COOLDOWN_SEC)
    time.sleep(SLEEP_SEC + random.random() * SLEEP_JITTER)
    return result


def fetched_months(sid: str, raw_dir: str = RAW_DIR) -> set:
    """快取裡已查過的月份（含官方「該月無資料」的標記列）。"""
    path = os.path.join(raw_dir, f"{sid}.csv")
    if not os.path.exists(path):
        return set()
    return set(pd.read_csv(path, usecols=["month"])["month"].astype(str))


def save_month(sid: str, df: pd.DataFrame, month: str, title: str, raw_dir: str = RAW_DIR) -> None:
    """附加寫入快取；官方該月沒資料（未上市、興櫃、停牌）也留一列標記，下次就不會再查。"""
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, f"{sid}.csv")
    df = df.assign(month=month, title=title)
    if len(df) == 0:
        df = pd.DataFrame([{"date": pd.NaT, "month": month, "title": title}])
    df = df.reindex(columns=RAW_COLS)   # 固定欄位順序：附加寫入不帶表頭，欄位不齊會錯位
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False, encoding="utf-8-sig")


def download_months(sid: str, months: list, raw_dir: str = RAW_DIR) -> int:
    """逐月下載，已在快取的月份跳過（中斷後重跑即可續傳）；回傳這次實際查詢的月份數。"""
    done = fetched_months(sid, raw_dir)
    todo = [m for m in months if m not in done]
    for month in todo:
        df, title = fetch_month_polite(sid, month)
        save_month(sid, df, month, title, raw_dir)
    return len(todo)


def load_official(sid: str, raw_dir: str = RAW_DIR, own_market_only: bool = False) -> pd.DataFrame:
    """讀快取：去掉「該月無資料」標記列、重複日留最後一筆、只留真的有成交的日子。

    own_market_only=True：不含上櫃轉上市補查的櫃買列（判斷本機是不是還原價時用）。
    """
    raw = pd.read_csv(os.path.join(raw_dir, f"{sid}.csv"), parse_dates=["date"])
    if own_market_only:
        raw = raw[~raw["title"].astype(str).str.startswith("TPEx:")]
    raw = raw.dropna(subset=["date"]).drop_duplicates("date", keep="last").set_index("date").sort_index()
    raw = raw[(raw["volume"] > 0) & raw["close"].notna()]
    return raw[["open", "high", "low", "close", "volume"]].astype(float)


def download_transfer_months(sid: str, months: list, raw_dir: str = RAW_DIR) -> int:
    """上櫃轉上市：上市股在證交所查不到的月份，改查櫃買（代號加 O），標題前綴 `TPEx:` 寫進同一個快取。

    例：1795.TW 2019-12 才轉上市，之前的資料要到櫃買查 1795.TWO。已補查過的月份跳過。
    """
    raw = pd.read_csv(os.path.join(raw_dir, f"{sid}.csv"), usecols=["month", "title"])
    done = set(raw.loc[raw["title"].astype(str).str.startswith("TPEx:"), "month"].astype(str))
    todo = [m for m in months if m not in done]
    for month in todo:
        df, title = fetch_month_polite(sid + "O", month)
        save_month(sid, df, month, "TPEx:" + title, raw_dir)
    return len(todo)


def month_has_data(sid: str, month: pd.Period, raw_dir: str = RAW_DIR) -> bool:
    """官方這個月有沒有成交資料（沒查過就先查）。"""
    download_months(sid, [str(month)], raw_dir)
    raw = pd.read_csv(os.path.join(raw_dir, f"{sid}.csv"), parse_dates=["date"])
    month_rows = raw[raw["month"].astype(str) == str(month)]
    return bool((month_rows["volume"] > 0).any())


def first_listed_month(sid: str, lo: pd.Period, hi: pd.Period, raw_dir: str = RAW_DIR) -> pd.Period:
    """二分搜尋官方第一個有成交資料的月份（上市櫃起始月）。

    前提：lo 已知沒資料、hi 已知有資料（或是未來月份）。回傳 hi；hi 在未來＝至今官方都查不到（興櫃）。
    """
    while (hi - lo).n > 1:
        mid = lo + (hi - lo).n // 2
        if month_has_data(sid, mid, raw_dir):
            hi = mid
        else:
            lo = mid
    return hi


def main() -> int:
    ap = argparse.ArgumentParser(description="從證交所／櫃買中心下載個股日成交資料（逐月、可續傳）")
    ap.add_argument("stock_id", help="例：2330.TW、6488.TWO")
    ap.add_argument("start", help="起始月份 YYYY-MM")
    ap.add_argument("end", help="結束月份 YYYY-MM")
    ap.add_argument("--raw-dir", default=RAW_DIR)
    a = ap.parse_args()
    first = pd.Period(a.start, "M")
    if a.stock_id.endswith(".TW") and first < TWSE_MIN_MONTH:
        print(f"證交所單股查詢最早到 {TWSE_MIN_MONTH}，起始月改成 {TWSE_MIN_MONTH}")
        first = TWSE_MIN_MONTH
    months = [str(m) for m in pd.period_range(first, pd.Period(a.end, "M"))]
    n = download_months(a.stock_id, months, a.raw_dir)
    df = load_official(a.stock_id, a.raw_dir)
    print(f"{a.stock_id}：本次查詢 {n} 個月，快取共 {len(df)} 個有成交的交易日 → {a.raw_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
