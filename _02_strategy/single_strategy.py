from datetime import datetime
import logging
import math
import pandas as pd
from modules.config_loader import load_config
from modules.logger import setup_logger
from modules.process_mongo import get_mongo_client

config = load_config()


class StockBacktest:
    def __init__(self, stock_id: str, start_date: str, end_date: str, initial_cash=100000, split_cash=0, label="backtest", loglevel=logging.INFO):
        self.stock_id = stock_id
        self.label = label
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.initial_cash = initial_cash
        self.db = get_mongo_client()
        self.win_count = 0
        self.lose_count = 0
        self.final_cash = initial_cash
        self.commission = 0.001425
        self.dues = 0.003
        self.position = 0
        self.cash = initial_cash
        self.split_cash = split_cash if split_cash != 0 else math.floor(initial_cash * 0.05)
        self.win_rate = 0.0
        self.buy_index = None
        self.hold_days: list[int] = []
        self.fetch_data()
        self.trade_records = []
        log_file_path = f"{config.get('strategy_log_folder', './strategy_log')}/{start_date}_to_{end_date}-{label}.log"
        self.logger = setup_logger(log_file=log_file_path, loglevel=loglevel)  # 避免 logger 進入 __setattr__ 監聽

    def fetch_data(self) -> None:
        collection = self.db[self.stock_id]
        cursor = collection.find({"date": {"$gte": self.start_date, "$lte": self.end_date}}, no_cursor_timeout=True)
        self.data = pd.DataFrame(list(cursor))  # 一次抓出避免 cursor 被中斷
        if self.data.empty:
            self.logger.warning(f"{self.stock_id}: 無法從 MongoDB 獲取數據")
            return None
        self.data["date"] = pd.to_datetime(self.data["date"])
        self.data.set_index("date", inplace=True)

    def insert_trade_record(self, buy_date, buy_price, buy_tax, quantity, sell_date, sell_price, sell_tax, days, profit) -> None:
        record = {"stock_id": self.stock_id, "buy_date": buy_date.strftime("%Y-%m-%d"), "buy_price": buy_price, "buy_tax": buy_tax, "quantity": quantity, "sell_date": sell_date.strftime("%Y-%m-%d"), "sell_price": sell_price, "hold_days": days, "profit": profit}
        self.trade_records.append(record)

    def log_transaction(self, transaction_type, i, price, position, tax):
        """統一記錄交易資訊"""
        log_msg = f"id:{self.stock_id}, 日期:{self.data.index[i]}, type:{transaction_type},價格: {price},股數:{position},現金餘額:{self.cash}, 手續費(含稅): {tax}"
        self.logger.debug(log_msg)

    def buy_signal(self, i):
        return self.data.iloc[i]["close"] > self.data.iloc[i - 1]["close"]

    def sell_signal(self, i):
        return self.data.iloc[i]["close"] < self.data.iloc[i - 1]["close"]

    def tw_ticket_gap(self, price):
        """依照台股股價級距調整價格，無條件進位"""
        if price < 10:
            tick_size = 0.01
        elif price < 50:
            tick_size = 0.05
        elif price < 100:
            tick_size = 0.1
        elif price < 500:
            tick_size = 0.5
        elif price < 1000:
            tick_size = 1
        else:
            tick_size = 5

        return float(f"{(math.ceil(price / tick_size) * tick_size):.2f}")  # 無條件進位

    def count_tax(self, price, position, is_sell=False):
        amount = price * position
        commission = max(amount * self.commission, 20)
        tax = amount * self.dues + commission if is_sell else commission
        return math.ceil(tax)

    def buy_price_select(self, i):
        return self.tw_ticket_gap(self.data.iloc[i]["close"])

    def sell_price_select(self, i):
        return self.tw_ticket_gap(self.data.iloc[i]["close"])

    def run_backtest(self):
        if self.data is None or len(self.data) < 2:
            return

        size = len(self.data)
        for self.index in range(1, size):
            if self.cash < self.split_cash * 5:  # 賠錢到一定的金額就跳出
                break
            if self.position > 0:
                if self.sell_signal(self.index):
                    sell_price = self.sell_price_select(self.index)
                    sell_tax = self.count_tax(sell_price, self.position, is_sell=True)
                    profit = (sell_price - self.buy_price) * self.position - sell_tax - self.buy_tax
                    self.cash += math.ceil(sell_price * self.position) - sell_tax
                    self.win_count += 1 if profit > 0 else 0
                    self.lose_count += 1 if profit <= 0 else 0
                    self.log_transaction("SELL", self.index, sell_price, self.position, sell_tax)
                    days_difference = (self.data.index[self.index] - self.data.index[self.buy_index]).days
                    self.hold_days.append(days_difference)
                    # 新增儲存交易記錄
                    self.insert_trade_record(buy_date=self.data.index[self.buy_index], buy_price=self.buy_price, buy_tax=self.buy_tax, quantity=self.position, sell_date=self.data.index[self.index], sell_price=sell_price, sell_tax=sell_tax, days=days_difference, profit=profit)

                    self.position = 0
                    self.buy_price = None
                    self.buy_tax = 0
                    self.buy_index = None

            else:
                if self.buy_signal(self.index) and self.index + 1 < size:
                    self.buy_price = self.buy_price_select(self.index)
                    self.position = self.split_cash // self.buy_price
                    if self.position <= 0:
                        return
                    self.buy_tax = self.count_tax(self.buy_price, self.position)
                    self.cash -= math.ceil(self.position * self.buy_price) + self.buy_tax
                    self.buy_index = self.index
                    self.log_transaction("BUY", self.index, self.buy_price, self.position, self.buy_tax)

        if self.position > 0:
            tax = self.count_tax(self.buy_price, self.position)
            self.cash += math.ceil(self.position * self.buy_price) + tax
            self.log_transaction("Null", self.index, self.buy_price, self.position, tax)

        buy_count = self.win_count + self.lose_count
        self.win_rate = self.win_count / buy_count if buy_count > 0 else 0
        self.logger.info(f"{self.stock_id}: 總金額 {self.cash}, 下注次數 {buy_count} , 獲利次數{self.win_count} 勝率 {self.win_rate:.2%}")


if __name__ == "__main__":
    backtest = StockBacktest(stock_id="2330.TW", start_date="2019-01-01", end_date="2023-12-31", initial_cash=100000)
    backtest.run_backtest()
