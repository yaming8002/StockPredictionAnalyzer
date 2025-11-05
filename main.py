import argparse

from _01_data.download_stock import process_all_stocks
from _01_data.to_mongoDB import process_csv_files, remove_mongoDB
from _01_data.unit import STOCK_LIST
from _02_strategy.dow_strategy import run_dow_backtest
from _04_analysis.analyze_log import extract_log_summary, stock_targer_win

from _05_deeplearning.export_mongo_to_csv import export_mongo_to_csv
from _05_deeplearning.policy_evolve import run_deep_backtest
from _09_market import generate_dow_tomorrow_buy_cmf_list
from _09_market.generate_dow_tomorrow_buy_list import generate_tomorrow_buy_list, generate_tomorrow_sell_list
from modules.config_loader import load_config
from modules.process_mongo import close_mongo_client
import tensorflow as tf

import platform

config = load_config()


def download_all_stocks():
    """從 stock_List.csv 讀取股票清單並下載所有股票數據"""
    stock_list_file = config.get("stock_list_path", "./stock_List.csv")
    original_data_folder = config.get("original_data_folder", "./data")
    process_all_stocks(stock_list_file, original_data_folder)


def is_wsl():
    return "microsoft" in platform.uname().release.lower()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="股票數據下載工具")
    parser.add_argument("command", nargs="?", default="download", help="執行指令 (download)")
    parser.add_argument("--today", type=str, help="指定今日日期 (格式 YYYY-MM-DD)", default=None)
    parser.add_argument("--notdownload", action="store_true", help="不下載股票資料")
    args = parser.parse_args()
    if is_wsl():
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
                tf.config.set_visible_devices(gpus[0], "GPU")
                print("✅ 已啟用 GPU 加速！")
            except RuntimeError as e:
                print(f"⚠️ 設定 GPU 失敗：{e}")
        else:
            print("🚫 沒有找到 GPU，將使用 CPU 運算。")
    else:
        print("🧩 非 WSL 環境，略過 GPU 設定")

    if args.command == "download":
        download_all_stocks()
    if args.command == "tomongo":
        process_csv_files()
    if args.command == "removemongo":
        remove_mongoDB()

    if args.command == "col_log":
        extract_log_summary()
    # if args.command == "test_03_data_view":
    #     test_03_data_view()

    if args.command == "stock_targer_win":
        stock_targer_win()

    if args.command == "test_04_train_model":
        file_path = "./stock_data/leaning_label/sma_20_sma_50_trades.csv"
        model_path = "model_20_50_stream_cnn.h5"
        # test_04_train_model(file_path, model_path)

    if args.command == "test_05_predict":
        file_path = "./stock_data/leaning_label/sma_20_sma_50_trades.csv"
        model_path = "model_stream_cnn.h5"
        # test_05_predict_from_model(file_path, model_path)

    if args.command == "dow_strategy":
        run_dow_backtest()

    if args.command == "tomorrow_buy":
        notdonwload = args.notdownload is None
        generate_tomorrow_buy_list(today=args.today,notdonwload=notdonwload)
        generate_tomorrow_sell_list(today=args.today)

    close_mongo_client()
