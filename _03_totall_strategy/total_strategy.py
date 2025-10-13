from datetime import datetime, timedelta
import logging
import math
from random import random
import pandas as pd
from modules.config_loader import load_config
from modules.logger import setup_logger
from modules.process_mongo import get_mongo_client

# 載入全域配置
config = load_config()


class SingleStockLog:
    """
    紀錄單一股票的運作狀況：
    - 是否持有
    - 買賣次數
    - 每次交易獲利
    - 持有天數
    - 買進金額與稅金計算
    """

    def __init__(self, stock_id, commission=0.001425, dues=0.003) -> None:
        self.stock_id = stock_id
        self.is_buy = False           # 是否持有中
        self.buy_count = 0            # 總買進次數
        self.total_profit_log = []    # 每筆交易的獲利紀錄
        self.hold_days_log = []       # 每筆交易持有天數紀錄
        self.quantity = 0             # 買進股數
        self.buy_price = 0.0          # 買進價格
        self.buy_tax = 0.0            # 買進手續費
        self.buy_date = None          # 買進日期
        self.commission = commission  # 交易手續費率
        self.dues = dues              # 證交稅率 (僅賣出時收)

    def count_tax(self, price, position, is_sell=False):
        """
        計算交易稅金
        - price: 單價
        - position: 股數
        - is_sell: 是否為賣出
        """
        amount = price * position
        commission = max(amount * self.commission, 20)  # 手續費最低 20 元
        tax = amount * self.dues + commission if is_sell else commission
        return math.ceil(tax)

    def record_buy(self, money, price, buy_date):
        """
        紀錄買進交易
        - money: 投入資金
        - price: 買進價格
        - buy_date: 買進日期
        """
        self.quantity = math.ceil(money / price)  # 計算可買股數 (四捨五入)
        tax = self.count_tax(price, self.quantity, is_sell=False)
        total_cost = math.ceil(price * self.quantity) + tax  # 總成本 (含稅)
        self.is_buy = True
        self.buy_count += 1
        self.buy_price = price
        self.buy_tax = tax
        self.buy_date = buy_date
        return total_cost

    def record_sell(self, price, sell_date):
        """
        紀錄賣出交易
        - price: 賣出價格
        - sell_date: 賣出日期
        """
        tax = self.count_tax(price, self.quantity, is_sell=True)
        gross_income = math.ceil(price * self.quantity)  # 總收入 (未扣稅)
        profit = (
            gross_income - tax - self.buy_tax - math.ceil(self.buy_price * self.quantity)
        )  # 淨利
        hold_days = (sell_date - self.buy_date).days  # 持有天數

        # 更新交易紀錄
        self.is_buy = False
        self.quantity = 0
        self.total_profit_log.append(profit)
        self.hold_days_log.append(hold_days)

        return gross_income - tax  # 回傳實際收入 (扣稅後)

    def to_csv_row(self):
        """
        匯出成 CSV 列 (交易紀錄)
        """
        buy_date_str = self.buy_date.strftime("%Y-%m-%d") if self.buy_date else ""
        profit_sum = sum(self.total_profit_log)
        avg_hold_days = (
            round(sum(self.hold_days_log) / len(self.hold_days_log), 2)
            if self.hold_days_log
            else 0
        )
        row = f"{self.stock_id},{self.is_buy},{self.buy_count},{profit_sum},{avg_hold_days},{self.quantity},{self.buy_price},{self.buy_tax},{buy_date_str}"
        return row


class MultiStockBacktest:
    """
    多股票回測框架
    - 支援同時追蹤多支股票
    - 每天依照策略選股、分批買入
    - 自動紀錄交易過程
    """

    def __init__(
        self,
        stock_list,
        start_date,
        end_date,
        lookback_days=20,
        label="multi_backtest",
        loglevel=logging.INFO,
        initial_cash=100000,
        split_cash=0,
    ):
        self.stock_list = stock_list
        self.stock_trade_log = {x: SingleStockLog(x) for x in stock_list}
        self.label = label
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.today = self.start_date
        self.initial_cash = initial_cash
        self.db = get_mongo_client()   # 連接 MongoDB
        self.commission = 0.001425
        self.dues = 0.003
        self.cash = initial_cash
        self.split_cash = (
            split_cash if split_cash != 0 else math.floor(initial_cash * 0.05)
        )  # 每次分配資金 (預設 5%)
        self.lookback_days = lookback_days  # 回顧天數
        self.one_day_total_buy = 5          # 單日最多買進標的數量

        # 設定 log
        log_file_path = f"{config.get('strategy_log_folder', './strategy_log')}/{start_date}_to_{end_date}-{label}.log"
        self.logger = setup_logger(log_file=log_file_path, loglevel=loglevel)

    def fetch_data(self, today) -> None:
        """
        撈取股票資料 (lookback_days 內)
        - 檢查賣出條件
        - 檢查買入條件
        """
        lookback_start_date = today - timedelta(days=self.lookback_days)
        self.selected_stock_list = []  # 每天重置

        for stock_id in self.stock_list:
            collection = self.db[stock_id]
            cursor = collection.find(
                {"date": {"$gte": lookback_start_date, "$lte": self.today}},
                no_cursor_timeout=True,
            )
            df = pd.DataFrame(list(cursor))

            if df.empty:
                self.logger.debug(f"{stock_id}: 沒有資料，跳過")
                continue

            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)

            # 資料量不足，略過
            if len(df) < self.lookback_days // 2:
                self.logger.debug(f"{stock_id}: 資料不足，跳過")
                continue

            # 如果該股票持有 → 檢查是否要賣出
            if self.stock_trade_log[stock_id].is_buy:
                if self.sell_signal(df):
                    price = self.sell_price_select(df)
                    self.cash += self.stock_trade_log[stock_id].record_sell(
                        price, df.index[-1]
                    )

            # 檢查是否要買進
            if self.buy_signal(df):
                self.selected_stock_list.append((stock_id, df))
                self.logger.info(f"{stock_id}: 符合買進條件，加入選股清單")

        # 決定當天要買哪些股票
        self.select_buy_stock()

    def select_buy_stock(self):
        """
        從符合條件的股票中，隨機挑選 (最多 self.one_day_total_buy 檔)
        """
        if not self.selected_stock_list:
            return

        buy_list = (
            random.sample(self.selected_stock_list, self.one_day_total_buy)
            if len(self.selected_stock_list) > self.one_day_total_buy
            else self.selected_stock_list
        )

        for id, df in buy_list:
            price = self.buy_price_select(df)
            if self.split_cash < self.cash:
                self.stock_trade_log[id].record_buy(self.split_cash, price, df.index[-1])

    def tw_ticket_gap(self, price: float):
        """
        台股價格跳動單位 (依價格區間決定)
        """
        if price < 10.0:
            tick_size = 0.01
        elif price < 50.0:
            tick_size = 0.05
        elif price < 100.0:
            tick_size = 0.1
        elif price < 500.0:
            tick_size = 0.5
        elif price < 1000.0:
            tick_size = 1
        else:
            tick_size = 5
        return float(f"{(math.ceil(price / tick_size) * tick_size):.2f}")

    # === 策略條件 (可以覆寫) ===
    def buy_signal(self, df):
        return df.iloc[-1]["close"] > df.iloc[-2]["close"]  # 簡單：連續上漲

    def sell_signal(self, df):
        return df.iloc[-1]["close"] > df.iloc[-2]["close"]  # 範例：仍是上漲即賣出 (之後可改)

    def buy_price_select(self, df):
        return self.tw_ticket_gap(df.iloc[-1]["open"])  # 用開盤價買

    def sell_price_select(self, df):
        return self.tw_ticket_gap(df.iloc[-1]["open"])  # 用開盤價賣

    def run_backtest(self):
        """
        執行回測
        - 模擬每日操作
        - 最後輸出交易紀錄 CSV
        """
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq="B")  # "B" → 僅交易日 (平日)

        for current_date in date_range:
            self.fetch_data(current_date)

        # 輸出結果
        output_file = f"{self.label}_trade_log.csv"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("stock_id,is_buy,buy_count,total_profit_sum,avg_hold_days,quantity,buy_price,buy_tax,buy_date\n")
            for _, log in self.stock_trade_log.items():
                f.write(log.to_csv_row())

        self.logger.info(f"交易紀錄已輸出至 {output_file}")
