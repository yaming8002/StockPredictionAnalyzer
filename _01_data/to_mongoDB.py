import os
import pandas as pd
import pymongo

from modules.config_loader import load_config
from modules.logger import setup_logger
from modules.process_mongo import get_mongo_client
from .stock_technical import calculate_bollinger_bands, calculate_ema, calculate_sma

config = load_config()
logger = setup_logger()


def process_csv_files():
    """處理 CSV 檔案，增加技術指標並寫入 MongoDB 或輸出為 CSV"""
    original_data_folder = config.get("original_data_folder", "./data")
    output_data_folder = config.get("output_data_folder", "./processed_data")
    use_mongodb = config.get("use_mongodb", False)

    os.makedirs(output_data_folder, exist_ok=True)

    db = None
    if use_mongodb:
        db = get_mongo_client()

    for file in os.listdir(original_data_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(original_data_folder, file)
            df = pd.read_csv(file_path, parse_dates=["date"], index_col="date")

            # 計算 EMA
            df = calculate_working(df)
            collection_name = file.replace(".csv", "")  # 根據檔案名稱建立集合名稱
            if use_mongodb:
                collection = db[collection_name]  # 依照檔案名稱建立不同集合
                records = df.reset_index().to_dict(orient="records")
                collection.insert_many(records)
                logger.info(f"{file} 已寫入 MongoDB")
            else:
                output_path = os.path.join(output_data_folder, file)
                df.to_csv(output_path)
                logger.info(f"{file} 已存入 {output_path}")


def calculate_working(df):
    df = calculate_sma(df, window=5)
    df = calculate_sma(df, window=20)
    df = calculate_sma(df, window=50)
    df = calculate_sma(df, window=60)
    df = calculate_sma(df, window=120)
    df = calculate_sma(df, window=200)
    df = calculate_ema(df, span=5)
    df = calculate_ema(df, span=20)
    df = calculate_ema(df, span=50)
    df = calculate_ema(df, span=60)
    df = calculate_ema(df, span=120)
    df = calculate_ema(df, span=200)
    df = calculate_bollinger_bands(df)

    return df.iloc[200:]


def remove_mongoDB():
    """清空 MongoDB 中的 stock_analysis 集合"""
    use_mongodb = config.get("use_mongodb", False)
    if use_mongodb:
        mongo_uri = config.get("mongo_uri", "mongodb://localhost:27017")
        client = pymongo.MongoClient(mongo_uri)
        db_name = config.get("mongo_db", "stock_db")
        client.drop_database(db_name)
        print(f"MongoDB 數據庫 {db_name} 中的所有集合內容已清空")
