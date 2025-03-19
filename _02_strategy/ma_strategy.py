import backtrader as bt
from modules.config_loader import load_config
from modules.logger import setup_logger


config = load_config()
logger = setup_logger()


class MAStrategy(bt.Strategy):
    params = (
        ("lookback", 80),  # 設定回溯多少根K線來識別訂單區域
        ("risk", 0.03),  # 風險百分比，用於設置停損
        ("reward_ratio", 5),  # 獎勵風險比率，用於設置停利
        ("pivot_range", 7),  # 檢測轉折點的區間大小
        ("sma1", "sma50"),  # 檢測轉折點的區間大小
        ("sma2", "sma200"),  # 檢測轉折點的區間大小
    )

    def __init__(self) -> None:
        self.sell_tax = 0.003  # Transaction tax 0.3% for selling
        self.commission = 0.001425  # Commission 0.1425% for both buy and sell
        self.wait_dat = 30
        self.win = 0
        self.lose = 0
        self.buy_price = None  # 買入價格
        self.buy_size = 0  # 買入量
        self.is_show = False  # 是否打印
        self.tax_total = 0  # 總手續費
        self.count_day = 0  # 持有天數
        self.stop_loss = None
        self.take_profit = None  # 目標價格
        self.entry_price = None
        self.buy_date = None  # 買入日期
        self.touch_list = []
        self.hold_days = []  # 記錄持有天數

        pass

    def next(self):
        # 確保至少有足夠的歷史數據進行訂單區域的計算
        if len(self) < self.wait_dat:
            return
        if len(self) >= self.data.buflen() - 5:
            if self.buy_size > 0:
                self.sell(price=self.data.close[0])  # Sell at current close price
            return

        if self.buy_size > 0:
            check_b = self.data.close[0] <= self.buy_price * 1.02
            if self.data.low[0] <= self.stop_loss:
                self.lose += 1
                self.sell(self.stop_loss, check_b)
            elif self.data.high[0] >= self.take_profit:
                self.win += 1
                self.sell(self.take_profit, check_b)

        else:
            if getattr(self.data, self.params.sma1)[-2] <= getattr(self.data, self.params.sma2)[-2] and getattr(self.data, self.params.sma1)[-1] > getattr(self.data, self.params.sma2)[-1]:
                self.stop_loss = round((1 - self.params.risk) * self.data.close[0], 2)
                self.take_profit = round(self.data.close[0] * (1 + self.params.risk * self.params.reward_ratio), 2)
                # cash_level = int(self.broker.cash /500000) if self.broker.cash > 500000 else 1
                cash_level = 1
                buy_size = int(cash_level * 5000 / self.data.close[0])
                if self.broker.cash >= self.data.close[0] * buy_size:
                    # print(f'目標{self.take_profit} 停損 {self.stop_loss }')
                    self.buy_size = buy_size
                    # print(self.data.datetime.date(0))
                    self.buy_date = self.data.datetime.date(0)

                    self.buy(price=self.data.close[0])  # 進場
                self.touch_list = []
                self.touch_list = []

    def buy(self, price=None):
        self.buy_price = round(price, 2)
        self.count_fee(self.buy_price)
        if self.is_show:
            print(f"買 {self.data.datetime.date(0)}價格 {self.buy_price} 張數{self.buy_size} 本金:{self.broker.cash}")
        super().buy(size=self.buy_size, price=self.buy_price)

    def sell(self, price=None, sell_type=None):
        sell_date = self.data.datetime.date(0)
        # print(sell_date)
        # print(self.buy_date_first)
        hold_day = (sell_date - self.buy_date).days
        self.hold_days.append(hold_day)
        net_sell_price = round(price, 2)
        self.count_fee(net_sell_price, True)
        super().sell(size=self.buy_size, price=net_sell_price)
        if self.is_show:
            if sell_type is not None:
                info = "停利" if sell_type else "停損"
            else:
                info = "賣"
            info = f"{info} {self.data.datetime.date(0)}價格 {net_sell_price} 張數{self.buy_size} 總金額{self.buy_size * net_sell_price} 本金:{self.broker.cash}"
            print(info)
        self.move_stop_loss = None
        self.buy_size = 0
        self.buy_date = None
        self.buy_date_first = None

    def count_fee(self, price, is_sell=False):
        fee = price * self.buy_size * self.commission
        if is_sell:
            fee += price * self.buy_size * self.sell_tax
        fee = round(fee) if fee > 20 else 20
        self.tax_total += fee
        self.broker.cash -= fee
