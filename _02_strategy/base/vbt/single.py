"""
vbt 策略套件 single — 單檔回測基底（一檔一帳戶）
====================================================

開發手感：繼承 VbtSingleStrategy 後，**只覆寫 add_columns / buy_signal / sell_signal**，
其餘（隔日成交、向量化回測、台股費用後處理、tick 進位、summary）由基底處理。
底層走 vbt Portfolio.from_signals（單 column）。

成交時點（基底統一規則）：buy_signal / sell_signal 回傳的是「**當日收盤判定**」訊號，
基底自動把成交延到「**隔日開盤**」（build_signals 位移一天 + exec_price 取 open）。
故子類只描述「哪天判定」即可，不必自己處理 shift 或開盤價，也天然無 look-ahead。

最小範例：
    class MyStrat(VbtSingleStrategy):
        def add_columns(self, df):
            df["sma20"] = df["close"].rolling(20).mean()
            return df
        def buy_signal(self, df):
            return df["close"] > df["sma20"]
        def sell_signal(self, df):
            return df["close"] < df["sma20"]

    res = MyStrat(split_cash=10_000).run(df, stock_id="2330.TW")
    # res = {"trades": DataFrame, "summary": dict}

資料載入不在本套件職責內：傳入已含 OHLCV（小寫）、DatetimeIndex 的 df。
"""
import numpy as np
import pandas as pd
import vectorbt as vbt

from _02_strategy.base.vbt import common


class VbtSingleStrategy:
    """單檔策略基底。子類覆寫下列 hook。"""

    def __init__(self, initial_cash: float = 1_000_000.0,
                 invest_ratio: float = None, split_cash: float = 10_000.0,
                 min_invest: float = 5_000.0, freq: str = "1D"):
        # invest_ratio 有給 → 每筆投入「當前現金的比例」；否則用固定 split_cash 金額
        self.initial_cash = initial_cash
        self.invest_ratio = invest_ratio
        self.split_cash = split_cash
        self.min_invest = min_invest
        self.freq = freq

    # ── 子類覆寫的 hook ───────────────────────────────────────
    def add_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """#1 加指標欄位（跟著該檔日期跑）。預設不加。必須回傳 df。"""
        return df

    def buy_signal(self, df: pd.DataFrame) -> pd.Series:
        """#2 回傳 bool Series（進場）。子類必須覆寫。"""
        raise NotImplementedError

    def sell_signal(self, df: pd.DataFrame) -> pd.Series:
        """#3 回傳 bool Series（出場）。子類必須覆寫。"""
        raise NotImplementedError

    def build_signals(self, df: pd.DataFrame):
        """
        回傳 (entries, exits) 兩條 bool Series，語意為「**實際成交日**」。

        預設：buy_signal / sell_signal 視為「當日收盤判定」，基底自動往後位移
        一天 → 隔日成交（搭配 exec_price 取開盤，達成「收盤判定 → 隔日開盤成交」、
        無 look-ahead）。

        路徑相依出場（如進場以來最高）請覆寫此函式做單檔逐根掃描；覆寫後即由你
        自行產出「成交日」訊號，基底不再代為位移（時點責任轉移到子類）。
        """
        entries = self.buy_signal(df).fillna(False).astype(bool).shift(1, fill_value=False)
        exits = self.sell_signal(df).fillna(False).astype(bool).shift(1, fill_value=False)
        return entries, exits

    def exec_price(self, df: pd.DataFrame) -> pd.Series:
        """成交價：預設隔日開盤（配合 build_signals 的隔日成交；tick 進位在後處理）。"""
        return df["open"]

    # ── 主流程 ───────────────────────────────────────────────
    def run(self, df: pd.DataFrame, stock_id: str = "") -> dict:
        common.ensure_columns(df)
        df = self.add_columns(df.copy())
        entries, exits = self.build_signals(df)
        price = self.exec_price(df)

        # 倉位：invest_ratio → percent of cash；否則固定 split_cash 金額
        if self.invest_ratio is not None:
            size, size_type = self.invest_ratio, "percent"
        else:
            size, size_type = self.split_cash, "value"

        pf = vbt.Portfolio.from_signals(
            close=df["close"],
            entries=entries,
            exits=exits,
            price=price,
            init_cash=self.initial_cash,
            size=size,
            size_type=size_type,
            fees=common.COMMISSION,   # 粗估；精確費用在後處理重建
            direction="longonly",
            freq=self.freq,
            accumulate=False,         # 一檔同時只抱一個部位（不加碼）
        )

        records = self._postprocess(pf, stock_id)
        return {"trades": records, "summary": common.summarize_trades(records)}

    def _postprocess(self, pf, stock_id: str) -> pd.DataFrame:
        """取已平倉 trades → tick 進位 + 精確費用重建 → 統一欄位。"""
        rec = pf.trades.records_readable
        rec = rec[rec["Status"] == "Closed"].copy()
        cols = ["stock_id", "buy_date", "sell_date", "buy_price", "sell_price",
                "qty", "buy_fee", "sell_fee", "real_pnl"]
        if rec.empty:
            return pd.DataFrame(columns=cols)

        rec["buy_price"] = common.tw_tick_arr(rec["Avg Entry Price"].to_numpy(dtype=np.float64))
        rec["sell_price"] = common.tw_tick_arr(rec["Avg Exit Price"].to_numpy(dtype=np.float64))
        rec["qty"] = rec["Size"].astype(int)
        buy_fee, sell_fee = common.reconstruct_fees(rec["buy_price"], rec["sell_price"], rec["qty"])
        rec["buy_fee"], rec["sell_fee"] = buy_fee, sell_fee
        rec["real_pnl"] = common.net_pnl(rec["buy_price"], rec["sell_price"], rec["qty"])
        rec["stock_id"] = stock_id
        rec = rec.rename(columns={"Entry Timestamp": "buy_date", "Exit Timestamp": "sell_date"})
        return rec[cols].reset_index(drop=True)
