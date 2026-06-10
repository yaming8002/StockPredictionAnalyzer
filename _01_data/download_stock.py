"""
取得股票資料（基礎篇）

示範如何用 yfinance 下載台股歷史 OHLCV 資料，並存成 CSV 或 parquet。
提供兩種下載方式：
  - download_stock_data()        ：單檔下載，最穩定（適合少量、教學示範）
  - download_stock_data_multi()  ：批次下載（yf.download 一次多檔，較快但較易被限流）

執行範例：
  python _01_data/download_stock.py
"""

import os
import time
import random
import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


# 標準 logging（公開版不依賴專案內部 logger）
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("download_stock")


def save_prices(df: pd.DataFrame, save_path: str, symbol: str, fmt: str = "csv") -> str:
    """
    將股價 DataFrame 存檔，支援 CSV 與 parquet 兩種格式。

    為什麼提供兩種：CSV 人眼可直接讀（Excel 可開），適合初學與檢視；
    parquet 體積小、讀寫快、保留型別，適合大量資料長期儲存。

    fmt: "csv" 或 "parquet"
    回傳：實際存檔路徑
    """
    os.makedirs(save_path, exist_ok=True)

    if fmt == "parquet":
        file_path = os.path.join(save_path, f"{symbol}.parquet")
        # parquet 保留 DatetimeIndex（date 為 index）
        df.to_parquet(file_path, index=True)
    elif fmt == "csv":
        file_path = os.path.join(save_path, f"{symbol}.csv")
        # CSV 用 utf-8-sig，Excel 開中文不亂碼
        df.to_csv(file_path, encoding="utf-8-sig")
    else:
        raise ValueError(f"不支援的格式：{fmt}（只接受 'csv' 或 'parquet'）")

    return file_path


def _tidy_ohlcv(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    整理 yfinance 回傳的原始 DataFrame：
    欄位轉小寫、只留 OHLCV、date 設為 index。
    """
    stock_data = stock_data.reset_index()
    stock_data = stock_data.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })

    required_columns = ["date", "open", "high", "low", "close", "volume"]
    stock_data = stock_data[[c for c in required_columns if c in stock_data.columns]]
    stock_data["date"] = pd.to_datetime(stock_data["date"])
    stock_data = stock_data.set_index("date")
    return stock_data


def download_stock_data(symbol, start_date="2009-01-01", end_date=None,
                        save_path="data", fmt="csv", delay_range=(1, 3)):
    """
    下載單一股票的歷史資料並存檔。

    使用 yfinance.Ticker().history() 而非 yf.download，可避開部分限流問題、較穩定。
    end_date 預設為「明天」：yfinance 的 end 是 exclusive（不含當天），+1 天才會抓到今天。

    回傳：存檔路徑；若無資料或下載失敗回傳 None。
    """
    if end_date is None:
        end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info(f"下載 {symbol}（{start_date} ~ {end_date}）")

    try:
        ticker = yf.Ticker(symbol)
        stock_data = ticker.history(start=start_date, end=end_date, auto_adjust=False)
    except Exception as e:
        logger.error(f"{symbol} 下載失敗：{e}")
        return None

    if stock_data.empty:
        logger.warning(f"{symbol}：沒有資料")
        return None

    stock_data = _tidy_ohlcv(stock_data)
    file_path = save_prices(stock_data, save_path, symbol, fmt=fmt)
    logger.info(f"{symbol}：下載完成，共 {len(stock_data)} 筆，存入 {file_path}")

    # 加隨機延遲，避免短時間大量請求觸發 Yahoo 限流
    if delay_range and delay_range != (0, 0):
        time.sleep(random.uniform(*delay_range))

    return file_path


def _load_stock_list(stock_list_file: str) -> list[str]:
    """
    讀股票清單，支援新格式（有 stock_id 欄）與舊格式（無 header、第一欄為股號）。
    只保留台股（代號含 "TW"）。
    """
    df_list = pd.read_csv(stock_list_file)
    if "stock_id" in df_list.columns:
        stock_list = df_list["stock_id"].astype(str).str.strip().tolist()
    else:
        stock_list = df_list.iloc[:, 0].astype(str).str.strip().tolist()
    return [s for s in stock_list if "TW" in s]


def download_stock_data_multi(stock_list_file, start_date=None, end_date=None,
                              save_path="data", fmt="csv", batch_size=20, retry=3):
    """
    使用 yf.download 一次下載多支股票（高效版）。

    流程：
      - 從 stock_list_file 讀清單（只留台股）
      - 每 batch_size 檔為一批次下載，失敗自動重試
      - 批次之間加隨機延遲，避免 Yahoo 限流
      - 每支股票拆成獨立檔案（CSV 或 parquet）
    """
    os.makedirs(save_path, exist_ok=True)

    if not os.path.exists(stock_list_file):
        logger.error(f"找不到股票清單：{stock_list_file}")
        return

    # 日期範圍：預設抓最近 7 天（適合每日增量更新）
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    stock_list = _load_stock_list(stock_list_file)
    total = len(stock_list)
    logger.info(f"開始批次下載 {total} 檔股票（{start_date} ~ {end_date}）")

    for i in range(0, total, batch_size):
        batch = stock_list[i:i + batch_size]
        logger.info(f"下載批次 {i + 1} ~ {i + len(batch)}（{len(batch)} 檔）")

        # 批次重試
        df = None
        for attempt in range(retry):
            try:
                df = yf.download(
                    batch,
                    start=start_date,
                    end=end_date,
                    auto_adjust=False,
                    threads=True,
                    group_by="ticker",
                    progress=False,
                )
                if df.empty:
                    raise ValueError("無法從 Yahoo 取得資料")
                break
            except Exception as e:
                logger.warning(f"批次下載失敗（第 {attempt + 1} 次）：{e}")
                time.sleep(3 + random.random() * 2)
                df = None

        if df is None or df.empty:
            logger.error(f"批次 {batch} 下載失敗，略過")
            continue

        # 將每支股票從多層欄位中拆出，分別存檔
        for symbol in batch:
            try:
                data = df[symbol].copy()
                if data.empty:
                    logger.warning(f"{symbol} 無有效資料，跳過")
                    continue

                data = _tidy_ohlcv(data)
                # 價格 round 2 位、成交量轉整數
                price_cols = ["open", "high", "low", "close"]
                data[price_cols] = data[price_cols].apply(pd.to_numeric, errors="coerce").round(2)
                data["volume"] = pd.to_numeric(data["volume"], errors="coerce").fillna(0).round().astype(int)

                file_path = save_prices(data, save_path, symbol, fmt=fmt)
                logger.info(f"{symbol} 已下載，共 {len(data)} 筆，存入 {file_path}")
            except Exception as e:
                logger.error(f"{symbol} 拆分儲存失敗：{e}")

        # 批次之間加隨機延遲（防限流）
        time.sleep(random.uniform(2, 4))

    logger.info(f"全部股票下載完成，儲存於 {save_path}")


if __name__ == "__main__":
    # 範例 1：下載單一股票（台積電）近年資料，存成 CSV
    download_stock_data("2330.TW", start_date="2020-01-01", save_path="data", fmt="csv")

    # 範例 2：同一支改存 parquet
    # download_stock_data("2330.TW", start_date="2020-01-01", save_path="data", fmt="parquet")

    # 範例 3：依清單批次下載（需先用 fetch_stock_list.py 產生 stock_list.csv）
    # download_stock_data_multi("stock_list.csv", save_path="data", fmt="parquet")
