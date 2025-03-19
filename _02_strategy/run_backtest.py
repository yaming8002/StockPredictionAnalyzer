import random
import backtrader as bt
import pandas as pd

from _01_data.to_mongoDB import get_mongo_client
from modules.myStockPandasData import CustomPandasData


def run_backtest(strategy: bt.Strategy):

    db = get_mongo_client()
    collections = [col for col in db.list_collection_names() if ".TW" in col]

    begin = "2013-01-01"
    end = "2023-12-31"
    set_cast = 300000
    curch_ttal = set_cast
    total_win = 0.0
    total_lose = 0.0
    total_hold_days = []
    ma_list = ["sma5", "sma10", "sma20", "sma50", "sma60", "sma100", "sma120", "sma200"]

    for i in range(len(ma_list)):
        if i + 1 == len(ma_list):
            continue
        for j in range(i + 1, len(ma_list)):

            random.shuffle(collections)
            curch_ttal = set_cast

            for file in collections:
                results = db[file].find()
                result_list = list(results)
                if not result_list:
                    continue
                df = pd.DataFrame(result_list)
                if "_id" in df.columns:
                    df = df.drop(columns=["_id"])

                df = df[(df["date"] >= begin) & (df["date"] <= end)]
                df["datetime"] = pd.to_datetime(df["date"])
                df.set_index("datetime", inplace=True)

                data = CustomPandasData(dataname=df)
                cerebro = bt.Cerebro()
                cerebro.adddata(data)
                cerebro.addstrategy(strategy, sma1=ma_list[i], sma2=ma_list[j])
                cerebro.broker.setcash(curch_ttal)
                strategies = cerebro.run()
                curch_ttal = cerebro.broker.getcash()
                strategy = strategies[0]
                total_win += strategy.win
                total_lose += strategy.lose
                total_hold_days += strategy.hold_days

                if curch_ttal < set_cast * 0.4:
                    print("loss out")
                    break

            if total_win > 0:
                print(f"{begin}~{end} win {total_win}, lose {total_lose}  勝率{total_win/(total_win+total_lose):.2f}% 剩餘{curch_ttal:.2f} 本金比{curch_ttal/set_cast:.2f} 倍")
                print(max(total_hold_days), min(total_hold_days), sum(total_hold_days) / (total_win + total_lose))
            total_hold_days = []
            total_win = 0.0
            total_lose = 0.0


def fetch_data_from_mongo(db, collection_name, begin, end):
    results = db[collection_name].find()
    result_list = list(results)
    if not result_list:
        return None
    df = pd.DataFrame(result_list).drop(columns=["_id"], errors="ignore")
    df = df.rename(columns={"Date": "datetime", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[(df["datetime"] >= begin) & (df["datetime"] <= end)]
    df.set_index("datetime", inplace=True)
    return df


def print_results(begin, end, total_win, total_lose, curch_ttal, set_cast, total_hold_days):
    win_rate = total_win / (total_win + total_lose) if total_win + total_lose > 0 else 0
    print(f"{begin}~{end} win {total_win}, lose {total_lose}  勝率{win_rate:.2f}% 剩餘{curch_ttal:.2f} 本金比{curch_ttal/set_cast:.2f} 倍")
    print(max(total_hold_days, default=0), min(total_hold_days, default=0), sum(total_hold_days) / (total_win + total_lose) if total_win + total_lose > 0 else 0)
