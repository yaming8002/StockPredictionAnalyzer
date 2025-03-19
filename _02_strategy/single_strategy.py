import pandas as pd
import backtrader as bt

from modules.logger import setup_logger
from modules.process_mongo import get_mongo_client


class StockBacktest:
    def __init__(self, stock_id, start_date, end_date, initial_cash=100000, logger_file="backtest.log"):
        self.stock_id = stock_id
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.logger = setup_logger(logger_file)
        self.db = get_mongo_client()
        self.transactions = []  # 記錄交易
        self.win_count = 0
        self.lose_count = 0
        self.final_cash = initial_cash
        self.commission = 0.001  # 假設手續費為 0.1%
        self.date_index = 0

    def fetch_data(self):
        collection = self.db[self.stock_id]
        data = collection.find({"date": {"$gte": self.start_date, "$lte": self.end_date}})
        df = pd.DataFrame(data)
        if df.empty:
            self.logger.warning(f"{self.stock_id}: 無法從 MongoDB 獲取數據")
            return None
        df.set_index("date", inplace=True)
        return df

    def buy_signal(self, data, index):
        """簡單判斷買入信號：當今天的收盤價高於昨日收盤價"""
        return data.iloc[self.date_index]["close"] > data.iloc[self.date_index - 1]["close"]

    def sell_signal(self, data, index):
        """簡單判斷賣出信號：當今天的收盤價低於昨日收盤價"""
        return data.iloc[self.date_index]["close"] < data.iloc[self.date_index - 1]["close"]

    def run_backtest(self):
        df = self.fetch_data()
        if df is None or len(df) < 2:
            return

        for i in range(1, len(df)):
            current_date = df.index[i]
            close_price = df.iloc[i]["close"]

            if self.position > 0:
                if self.sell_signal(df, i):
                    sell_price = close_price
                    profit = (sell_price - self.buy_price) * self.position - (sell_price * self.position * self.commission)
                    self.cash += sell_price * self.position - (sell_price * self.position * self.commission)
                    if profit > 0:
                        self.win_count += 1
                    else:
                        self.lose_count += 1
                    self.transactions.append((current_date, "SELL", sell_price, self.position))
                    self.position = 0
                    self.buy_price = None
            else:
                if self.buy_signal(df, i):
                    self.buy_price = close_price
                    self.position = self.cash // close_price
                    self.cash -= self.position * close_price + (self.position * close_price * self.commission)
                    self.transactions.append((current_date, "BUY", self.buy_price, self.position))

        self.final_cash = self.cash + (self.position * df.iloc[-1]["close"] if self.position > 0 else 0)
        win_rate = self.win_count / (self.win_count + self.lose_count) if (self.win_count + self.lose_count) > 0 else 0
        self.logger.info(f"{self.stock_id}: 總金額 {self.final_cash}, 勝率 {win_rate:.2%}")


if __name__ == "__main__":
    backtest = StockBacktest(stock_id="2330.TW", start_date="2022-01-01", end_date="2023-12-31", initial_cash=100000)
    backtest.run_backtest()
