import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from modules.config_loader import load_config
from modules.logger import setup_logger

config = load_config()
logger = setup_logger()


def download_stock_data(symbol, start_date="2009-01-01", end_date=None, save_path="data"):
    """下載股票數據並轉換為正確的 pandas DataFrame 格式"""
    if end_date is None:
        end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info(f"Downloading {symbol} from {start_date} to {end_date}")
    stock_data = yf.download(symbol, start=start_date, end=end_date, rounding=True, actions=False)

    if stock_data.empty:
        logger.warning(f"{symbol}: 沒有數據")
        return None

    # 確保 DataFrame 格式正確
    stock_data.reset_index(inplace=True)  # 讓日期變成 DataFrame 的一部分

    # **確保欄位名稱為字串，並轉小寫**
    stock_data.columns = [str(col[0]).lower() for col in stock_data.columns.values]
    # 只保留所需欄位，確保欄位存在
    required_columns = ["date", "open", "high", "low", "close", "volume"]
    stock_data = stock_data[[col for col in required_columns if col in stock_data.columns]]
    # 設定 "date" 為索引
    stock_data.set_index("date", inplace=True)

    # 確保輸出資料夾存在
    os.makedirs(save_path, exist_ok=True)

    # 儲存為 CSV
    file_path = os.path.join(save_path, f"{symbol}.csv")
    stock_data.to_csv(file_path)
    logger.info(f"{symbol}: 下載完成，存入 {file_path}")

    return file_path


def process_stock(symbol, save_path):
    """下載股票數據並存入對應資料夾"""
    download_stock_data(symbol, save_path=save_path)


def process_all_stocks(stock_list_file, save_path):
    """從 stock_List.csv 讀取股票清單並下載所有股票數據"""
    stock_list = pd.read_csv(stock_list_file, header=None)[0].tolist()
    for stock_id in stock_list:
        if "TW" in stock_id:
            process_stock(stock_id, save_path)


def download_all_stocks():
    """從 stock_List.csv 讀取股票清單並下載所有股票數據"""
    stock_list_file = config.get("stock_list_path", "./stock_List.csv")
    original_data_folder = config.get("original_data_folder", "./data")
    process_all_stocks(stock_list_file, original_data_folder)
