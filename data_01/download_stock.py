import os
import pandas as pd
import yfinance as yf
import json
from datetime import datetime, timedelta


def load_config(config_file="config.json"):
    """從 config.json 讀取設定"""
    with open(config_file, "r") as file:
        config = json.load(file)
    return config


def download_stock_data(symbol, start_date="2009-01-01", end_date=None, save_path="data"):
    """下載股票數據，存入對應資料夾"""
    if end_date is None:
        end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Downloading {symbol} from {start_date} to {end_date}")
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    if stock_data.empty:
        print(f"{symbol}: 沒有數據")
        return None

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f"{symbol}.csv")
    stock_data.to_csv(file_path)
    print(f"{symbol}: 下載完成，存入 {file_path}")
    return file_path


def process_stock(symbol, save_path):
    """下載股票數據並存入對應資料夾"""
    download_stock_data(symbol, save_path=save_path)


def process_all_stocks(stock_list_file, save_path):
    """從 stock_List.csv 讀取股票清單並下載所有股票數據"""
    stock_list = pd.read_csv(stock_list_file, header=None)[0].tolist()
    for stock_id in stock_list:
        process_stock(stock_id, save_path)


if __name__ == "__main__":
    config = load_config()
    stock_list_file = config.get("stock_list_path", "./stock_List.csv")
    original_data_folder = config.get("original_data_folder", "./data")
    process_all_stocks(stock_list_file, original_data_folder)
