import sys
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from modules.process_mongo import get_mongo_client

DEFAULT_FEATURE_COLUMNS = ["open", "high", "low", "close", "volume", "sma_5", "sma_20", "sma_50", "sma_60", "sma_120", "sma_200", "ema_5", "ema_20", "ema_50", "ema_60", "ema_120", "ema_200", "bollinger_Upper", "bollinger_Lower"]


def get_stock_features(stock_id, buy_date, columns=DEFAULT_FEATURE_COLUMNS, normalize=True):
    """
    從 MongoDB 抓出指定股票在指定日期前 60 天的資料，回傳最後 30 天的指定欄位特徵。

    參數:
        stock_id (str): 股票代碼
        buy_date (datetime): 買入日期
        columns (list): 想保留的欄位（不含 "_id", "date"）
        columns = ["open", "high", "low", "close", "volume",
                   "sma_5", "sma_20", "sma_50", "sma_60", "sma_120", "sma_200",
                   "ema_5", "ema_20", "ema_50", "ema_60", "ema_120", "ema_200",
                   "bollinger_Upper", "bollinger_Lower"]
    回傳:
        pd.DataFrame: 最終 shape 為 (30, len(columns)) 的特徵資料
    """
    db = get_mongo_client()
    collection = db[stock_id]

    start_date = buy_date - timedelta(days=60)
    end_date = buy_date - timedelta(days=1)

    cursor = collection.find({"date": {"$gte": start_date, "$lte": end_date}})
    df = pd.DataFrame(list(cursor))

    # 預設：去除非數值欄位，只留下技術指標
    df = df.tail(30)

    df = df[columns]  # 若有遺失欄位 pandas 會報錯，視情況可用 df.get()
    if normalize:
        df = df.copy()  # 避免修改原始資料

        # 分開處理 volume 與其他欄位
        if "volume" in df.columns:
            volume_values = df[["volume"]].values
            df["volume"] = MinMaxScaler().fit_transform(volume_values)

        other_cols = [col for col in df.columns if col != "volume"]
        if other_cols:
            other_values = df[other_cols].values
            df[other_cols] = MinMaxScaler().fit_transform(other_values)
    return df


def build_training_data_gen(trade_file):
    # 讀取並打亂資料順序
    trades = pd.read_csv(trade_file, parse_dates=["buy_date"])
    trades = trades.sample(frac=1).reset_index(drop=True)

    count = 0
    for _, row in trades.iterrows():
        stock_id = row["stock_id"]
        buy_date = row["buy_date"]
        profit = row["profit"]

        features = get_stock_features(stock_id, buy_date)

        # print(stock_id, buy_date, profit)
        # print(features)
        # sys.exit(0)
        if len(features) == 30:
            X = features.values.astype(np.float32)
            y = 1 if profit > 0 else 0
            yield X, y
            count += 1


def build_training_data_gen_from_df(trades_df):
    """
    接收已切割的 trades DataFrame，逐筆取得資料與 label
    """
    for _, row in trades_df.iterrows():
        stock_id = row["stock_id"]
        buy_date = row["buy_date"]
        profit = row["profit"]

        features = get_stock_features(stock_id, buy_date)

        if len(features) == 30:
            X = features.values.astype(np.float32)
            y = 1 if profit > 0 else 0
            yield X, y


def test_03_data_view():
    file_path = "./stock_data/leaning_label/sma_20_sma_50_trades.csv"
    # df = pd.read_csv(file_path, parse_dates=["buy_date", "sell_date"])
    gen = build_training_data_gen(file_path, limit=20)

    try:
        X_sample, y_sample = next(gen)
        # print(X_sample.shape, y_sample)
    except StopIteration:
        print("⚠️ 找不到足夠的有效樣本")
