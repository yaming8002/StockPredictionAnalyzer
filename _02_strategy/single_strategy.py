from datetime import datetime
import math
import pandas as pd
from modules.logger import setup_logger
from modules.process_mongo import get_mongo_client


class StockBacktest:
    def __init__(self, stock_id: str, start_date: str, end_date: str, initial_cash=100000, split_cash=0, logger_file=".\\backtest.log"):
        self.stock_id = stock_id
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.initial_cash = initial_cash
        self.logger = setup_logger(log_file=logger_file)  # 避免 logger 進入 __setattr__ 監聽
        self.win_count = 0
        self.lose_count = 0
        self.final_cash = initial_cash
        self.commission = 0.001425
        self.dues = 0.003
        self.position = 0
        self.cash = initial_cash
        self.split_cash = split_cash if split_cash != 0 else math.floor(initial_cash * 0.05)
        self.fetch_data()

    def fetch_data(self) -> None:
        db = get_mongo_client()
        collection = db[self.stock_id]
        data = collection.find({"date": {"$gte": self.start_date, "$lte": self.end_date}})
        self.data = pd.DataFrame(data)
        if self.data.empty:
            self.logger.warning(f"{self.stock_id}: 無法從 MongoDB 獲取數據")
            return None
        self.data["date"] = pd.to_datetime(self.data["date"])
        self.data.set_index("date", inplace=True)

    def log_transaction(self, transaction_type, i, price, position, tax):
        """統一記錄交易資訊"""
        log_msg = f"id:{self.stock_id}, 日期:{self.data.index[i]}, type:{transaction_type},價格: {price},股數:{position},現金餘額:{self.cash}, 手續費(含稅): {tax}"
        self.logger.info(log_msg)

    def buy_signal(self, i):
        return self.data.iloc[i]["close"] > self.data.iloc[i - 1]["close"]

    def sell_signal(self, i):
        return self.data.iloc[i]["close"] < self.data.iloc[i - 1]["close"]

    def count_tax(self, price, position, is_sell=False):
        amount = price * position
        commission = max(amount * self.commission, 20)
        tax = amount * self.dues + commission if is_sell else commission
        return math.ceil(tax)

    def process_buy(self, i):
        price = self.data.iloc[i]["close"]
        self.buy_price = price
        self.position = math.floor(self.cash) // price
        tax = self.count_tax(price, self.position)
        self.cash -= math.ceil(self.position * price) + tax
        self.log_transaction("BUY", i, self.buy_price, self.position, tax)

    def process_sell(self, i):
        sell_price = self.data.iloc[i]["close"]
        tax = self.count_tax(sell_price, self.position, is_sell=True)
        profit = (sell_price - self.buy_price) * self.position - tax
        self.cash += sell_price * self.position - tax
        self.win_count += 1 if profit > 0 else 0
        self.lose_count += 1 if profit <= 0 else 0
        self.log_transaction("SELL", i, sell_price, self.position, tax)
        self.position = 0
        self.buy_price = None

    def run_backtest(self):
        if self.data is None or len(self.data) < 2:
            return

        size = len(self.data)
        for self.index in range(1, size):
            if self.cash < self.split_cash * 5:  # 賠錢到一定的金額就跳出
                break
            if self.position > 0:
                if self.sell_signal(self.index):
                    self.process_sell(self.index)
            else:
                if self.buy_signal(self.index):
                    self.process_buy(self.index)

        if self.position > 0:
            self.process_sell(size - 1)
        buy_count = self.win_count + self.lose_count
        self.win_rate = self.win_count / buy_count if buy_count > 0 else 0
        self.logger.info(f"{self.stock_id}: 總金額 {self.cash}, 下注次數 {buy_count} , 獲利次數{self.win_count} 勝率 {self.win_rate:.2%}")


if __name__ == "__main__":
    backtest = StockBacktest(stock_id="2330.TW", start_date="2019-01-01", end_date="2023-12-31", initial_cash=100000)
    backtest.run_backtest()
