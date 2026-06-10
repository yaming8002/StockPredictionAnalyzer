"""
取得股票清單

從 TWSE/TPEx 官方 ISIN 表抓取最新台股清單，輸出 stock_list.csv。

來源（C_public.jsp 的 strMode 參數對應不同市場）：
  - 上市：https://isin.twse.com.tw/isin/C_public.jsp?strMode=2
  - 上櫃：https://isin.twse.com.tw/isin/C_public.jsp?strMode=4
  - 興櫃：https://isin.twse.com.tw/isin/C_public.jsp?strMode=5

輸出 CSV 欄位：
  stock_id,name,market,industry,isin,listed_date,is_emerging

- stock_id 加 .TW（上市）或 .TWO（上櫃 / 興櫃）後綴，與 yfinance 代號一致
- is_emerging = 1 表示興櫃（無漲跌幅限制）

執行：
  python _01_data/fetch_stock_list.py
"""

import os
import re
import time

import requests
import pandas as pd


OUT_LIST = os.path.join(os.path.dirname(__file__), "stock_list.csv")

# strMode → 市場別
SOURCES = [
    (2, "上市"),    # TWSE 上市
    (4, "上櫃"),    # TPEx 上櫃
    (5, "興櫃"),    # TPEx 興櫃（無漲跌幅限制）
]


def fetch_one(str_mode: int, market_name: str) -> pd.DataFrame:
    """從 ISIN 表抓一個市場別的清單。"""
    url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={str_mode}"
    print(f"📥 抓取 {market_name}（strMode={str_mode}）...", end=" ", flush=True)

    r = requests.get(url, timeout=60)
    r.encoding = "ms950"   # TWSE 網頁用 Big5 系編碼

    rows = re.findall(r"<tr>(.*?)</tr>", r.text, re.DOTALL)
    records = []
    for row in rows:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
        if len(cells) < 6:
            continue
        cells = [c.strip().replace("&nbsp;", "").replace("　", " ") for c in cells]

        # 第一欄格式為 "代號 名稱"，用空白分割
        id_name = cells[0]
        parts = id_name.split(maxsplit=1)
        if len(parts) < 2:
            continue
        stock_id_raw, name = parts[0].strip(), parts[1].strip()

        # 只保留純數字代號（過濾權證、TDR、特別股等）
        if not stock_id_raw.replace(" ", "").isdigit():
            continue
        # 過濾分類標題或異常 row（代號過長）
        if len(stock_id_raw) > 6:
            continue

        isin = cells[1] if len(cells) > 1 else ""
        listed_date = cells[2] if len(cells) > 2 else ""
        industry = cells[4] if len(cells) > 4 else ""
        cfi = cells[5] if len(cells) > 5 else ""

        # 用 CFICode 過濾類型：ES... = 普通股；CE... = ETF；其餘排除（權證、特別股等）
        if not (cfi.startswith("ES") or cfi.startswith("CE")):
            continue

        records.append({
            "stock_id_raw": stock_id_raw,
            "name": name,
            "isin": isin,
            "listed_date": listed_date,
            "market": market_name,
            "industry": industry,
            "cfi": cfi,
        })

    print(f"{len(records):>4} 檔")
    return pd.DataFrame(records)


def to_full_id(row) -> str:
    """根據市場別加 .TW / .TWO 後綴（與 yfinance 一致）。"""
    sid = row["stock_id_raw"]
    if row["market"] == "上市":
        return f"{sid}.TW"
    return f"{sid}.TWO"   # 上櫃 / 興櫃


def main():
    frames = []
    for str_mode, market_name in SOURCES:
        df = fetch_one(str_mode, market_name)
        frames.append(df)
        time.sleep(0.5)   # 對伺服器客氣一點

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["stock_id_raw"], keep="first")

    df["stock_id"] = df.apply(to_full_id, axis=1)
    df["is_emerging"] = (df["market"] == "興櫃").astype(int)

    df = df[["stock_id", "name", "market", "industry", "isin", "listed_date", "is_emerging"]]
    df = df.sort_values("stock_id").reset_index(drop=True)

    df.to_csv(OUT_LIST, index=False, encoding="utf-8-sig")
    print(f"\n✅ 寫入 {OUT_LIST}（{len(df)} 檔）")

    # 統計
    print("\n📊 統計：")
    for m, n in df["market"].value_counts().items():
        print(f"   {m:<6} {n:>5}")
    print(f"   {'合計':<6} {len(df):>5}")
    print(f"   興櫃比例: {df['is_emerging'].mean() * 100:.1f}%")
    print("\n🎯 完成")


if __name__ == "__main__":
    main()
