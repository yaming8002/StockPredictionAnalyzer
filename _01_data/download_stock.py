import os
import time
import random
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from modules.config_loader import load_config
from modules.logger import setup_logger

# 載入設定與 logger
config = load_config()
logger = setup_logger(log_file="../logs/download.log")


def download_stock_data(symbol, start_date="2009-01-01", end_date=None, save_path="data", delay_range=(1, 3)):
    """
    下載單一股票的歷史數據並存檔
    使用 yfinance.Ticker().history() 以避免 yfinance.download 的部分限流問題
    """
    if end_date is None:
        end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info(f"Downloading {symbol} from {start_date} to {end_date}")

    try:
        ticker = yf.Ticker(symbol)
        stock_data = ticker.history(start=start_date, end=end_date, auto_adjust=False)
    except Exception as e:
        logger.error(f"{symbol} 下載失敗: {e}")
        return None

    if stock_data.empty:
        logger.warning(f"{symbol}: 沒有數據")
        return None

    # 整理欄位
    stock_data.reset_index(inplace=True)
    stock_data.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    }, inplace=True)

    # 保留需要的欄位
    required_columns = ["date", "open", "high", "low", "close", "volume"]
    stock_data = stock_data[[col for col in required_columns if col in stock_data.columns]]
    stock_data.set_index("date", inplace=True)

    # 儲存檔案
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f"{symbol}.csv")
    stock_data.to_csv(file_path)
    logger.info(f"{symbol}: 下載完成，存入 {file_path}")

    # 加隨機延遲，避免觸發限流
    time.sleep(random.uniform(*delay_range))

    return file_path


def download_stock_data_multi(start_date=None, end_date=None, save_path=None, batch_size=20, retry=3):
    """
    📈 使用 yf.download 一次下載多支股票歷史資料（高效版）
    - 從 stock_List.csv 讀取清單
    - 批次下載（預設一次 20 檔）
    - 自動重試與延遲，避免 Yahoo 限流
    """
    stock_list_file = config.get("stock_list_path", "./stock_List.csv")
    save_path = save_path or config.get("original_data_folder", "./data")
    os.makedirs(save_path, exist_ok=True)

    if not os.path.exists(stock_list_file):
        logger.error(f"⚠️ 找不到股票清單：{stock_list_file}")
        return

    # 日期範圍
    if end_date is None:
        end_date = (datetime.now()).strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    # 載入股票清單
    stock_list = pd.read_csv(stock_list_file, header=None)[0].tolist()
    stock_list = [s for s in stock_list if "TW" in s]

    total = len(stock_list)
    logger.info(f"📊 開始批次下載 {total} 檔股票 ({start_date} ~ {end_date})")

    for i in range(0, total, batch_size):
        batch = stock_list[i:i + batch_size]
        logger.info(f"📦 下載批次 {i+1} ~ {i+len(batch)} ({len(batch)} 檔)")

        # 嘗試重試下載
        for attempt in range(retry):
            try:
                df = yf.download(
                    batch,
                    start=start_date,
                    end=end_date,
                    auto_adjust=False,
                    threads=True,
                    group_by='ticker',
                    progress=False
                )
                if df.empty:
                    raise ValueError("無法從 Yahoo 取得資料")
                break
            except Exception as e:
                logger.warning(f"⚠️ 批次下載失敗（第 {attempt+1} 次）：{e}")
                time.sleep(3 + random.random() * 2)
                df = None

        if df is None or df.empty:
            logger.error(f"🚫 批次 {batch} 下載失敗，略過")
            continue

        # --- 將每支股票拆分為獨立 CSV ---
        for symbol in batch:
            try:
                data = df[symbol].copy()
                if data.empty:
                    logger.warning(f"{symbol} 無有效資料，跳過")
                    continue

                data.reset_index(inplace=True)
                data.rename(columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }, inplace=True)

                data = data[["date", "open", "high", "low", "close", "volume"]]
                cols = ["open", "high", "low", "close"]
                data[cols] = data[cols].apply(pd.to_numeric, errors="coerce").round(2)

                # ✅ 成交量轉為整數（取最接近整數）
                data["volume"] = pd.to_numeric(data["volume"], errors="coerce").fillna(0).round().astype(int)
                file_path = os.path.join(save_path, f"{symbol}.csv")
                data.to_csv(file_path, index=False, encoding="utf-8-sig")
                logger.info(f"✅ {symbol} 已下載，共 {len(data)} 筆，存入 {file_path}")

            except Exception as e:
                logger.error(f"❌ {symbol} 拆分儲存失敗：{e}")

        # 批次之間加隨機延遲（防限流）
        time.sleep(random.uniform(2, 4))

    logger.info(f"🎯 全部股票下載完成，儲存於 {save_path}")
    print(f"\n📊 股票資料已全部更新 ({start_date} ~ {end_date}) → {save_path}")



def process_stock(symbol, save_path):
    """下載單一股票數據"""
    return download_stock_data(symbol, save_path=save_path, delay_range=(10, 15))


def process_all_stocks(stock_list_file, save_path):
    """從 stock_List.csv 讀取股票清單並下載所有股票數據"""
    stock_list = pd.read_csv(stock_list_file, header=None)[0].tolist()
    for stock_id in stock_list:
        if "TW" in stock_id:
            process_stock(stock_id, save_path)


def download_all_stocks():
    """依照設定檔中的 stock_List.csv，批次下載所有股票數據"""
    stock_list_file = config.get("stock_list_path", "./stock_List.csv")
    original_data_folder = config.get("original_data_folder", "./data")
    process_all_stocks(stock_list_file, original_data_folder)
