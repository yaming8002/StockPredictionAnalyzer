import argparse

from _01_data.download_stock import process_all_stocks
from _01_data.to_mongoDB import process_csv_files, remove_mongoDB

from _02_strategy.analyze_log import extract_log_summary
from _02_strategy.ma_strategy import run_ma_backtest
from modules.config_loader import load_config
from modules.process_mongo import close_mongo_client

config = load_config()


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
    if args.command == "tomongo":
        process_csv_files()
    if args.command == "removemongo":
        remove_mongoDB()
    if args.command == "test":
        run_ma_backtest()
    if args.command == "col_log":
        extract_log_summary()
    close_mongo_client()
