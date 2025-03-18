import pandas as pd
import argparse

from data_01.download_stock import process_all_stocks
from modules.config_loader import load_config
from modules.logger import setup_logger

config = load_config()
logger = setup_logger()


def download_all_stocks():
    """從 stock_List.csv 讀取股票清單並下載所有股票數據"""
    stock_list_file = config.get("stock_list_path", "./stock_List.csv")
    original_data_folder = config.get("original_data_folder", "./data")
    process_all_stocks(stock_list_file, original_data_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="股票數據下載工具")
    parser.add_argument("command", nargs="?", default="download", help="執行指令 (download)")
    args = parser.parse_args()

    if args.command == "download":
        download_all_stocks()
