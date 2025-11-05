from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

from _02_strategy.base.single_strategy import StockBacktest


class DLStockBacktest(StockBacktest):

    def __init__(self, *args, buy_model_path=None, sell_model_path=None, window=120, **kwargs):
        super().__init__(*args, **kwargs)

        self.window = window

        # 載入買 / 賣模型
        self.buy_model = load_model(buy_model_path) if buy_model_path else None
        self.sell_model = load_model(sell_model_path) if sell_model_path else None

        # 欄位來自你提供的 MongoDB Document
        self.feature_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "sma_5",
            "sma_20",
            "sma_50",
            "sma_60",
            "sma_120",
            "sma_200",
            "bollinger_Lower",
            "bollinger_Upper",
        ]
        self.price_cols = [
            "open",
            "high",
            "low",
            "close",
            "sma_5",
            "sma_20",
            "sma_50",
            "sma_60",
            "sma_120",
            "sma_200",
            "bollinger_Lower",
            "bollinger_Upper",
        ]

    def _get_window_tensor(self, i):
        """
        取得 dl model input → shape = (1, 120, features)
        資料區間：i-121 ~ i-1（昨天以前）
        """
        if i - self.window - 1 < 0:
            return None  # 資料不足

        df = self.data[self.feature_cols].iloc[i - self.window - 1 : i - 1].copy()

        # --- 正規化: 價位類欄位均除以昨日 open ---
        yesterday_open = self.data.iloc[i - 1]["open"]

        if yesterday_open is None or np.isnan(yesterday_open) or yesterday_open == 0:
            return None  # 避免除以 0 或 nan

        # 對股票價位直接除昨日 open，使不同股票具可比性
        for col in self.price_cols:
            df[col] = df[col] / yesterday_open

        # --- Volume log normalize ---
        df["volume"] = df["volume"].apply(lambda v: np.log(v + 1) if v not in [None, 0, np.nan] else 0)

        # 資料不足補零
        if len(df) < self.window:
            pad = pd.DataFrame(
                np.zeros((self.window - len(df), len(self.feature_cols))),
                columns=self.feature_cols,
            )
            df = pd.concat([pad, df], ignore_index=True)

        return df.values.reshape(1, self.window, len(self.feature_cols))

    def _get_buy_day_tensor(self, buy_index, yesterday_open):
        """
        產生 SELL model 的第二個輸入 input_buy_day
        shape = (1, buy_features_count + 1)
        (包含買入當天特徵 + 昨日開盤價 normalize)
        """
        if buy_index is None or buy_index < 0:
            return None

        # 取買入當天的特徵
        row = self.data[self.feature_cols].iloc[buy_index].copy()

        buy_open = row["open"]

        # 避免 divide by zero or nan
        if buy_open is None or np.isnan(buy_open) or buy_open == 0:
            return None
        if yesterday_open is None or np.isnan(yesterday_open) or yesterday_open == 0:
            yesterday_open = buy_open  # fallback

        # ✅ normalize：買入日 → 用 buy_open 當 baseline
        for col in self.price_cols:
            row[col] = row[col] / buy_open

        # ✅ volume → log normalization
        row["volume"] = np.log(row["volume"] + 1) if row["volume"] > 0 else 0

        # ✅ appended yesterday_open normalized by buy_open
        yesterday_norm = yesterday_open / buy_open

        # 🔥 加在最後 (feature + 昨日開盤比較)
        row = np.append(row.values, yesterday_norm)

        return row.reshape(1, len(self.feature_cols) + 1)

    # ✅ override buy signal
    def buy_signal(self, i):
        if not self.buy_model:
            return super().buy_signal(i)
        if i < 150:
            return False
        X = self._get_window_tensor(i)
        prob = float(self.buy_model.predict(X, verbose=0))
        return prob > 0.6

    # ✅ override sell signal

    def sell_signal(self, i):
        if not self.sell_model:
            return super().sell_signal(i)
        if i < 150:
            return False
        if self.buy_index is None or self.position <= 0:
            return False

        # (A) 最近120天
        X_current = self._get_window_tensor(i)
        if X_current is None:
            return False

        # 昨天開盤價（用來比較 gap）
        yesterday_open = self.data.iloc[i - 1]["open"]

        # (B) 買入當天特徵 + 昨日開盤 normalize
        buy_day_features = self._get_buy_day_tensor(self.buy_index, yesterday_open)
        if buy_day_features is None:
            return False

        # (C) 買入價格（模型 third input）
        buy_price_tensor = np.array([[self.buy_price]], dtype=float)

        prob = float(
            self.sell_model.predict(
                [
                    X_current,
                    buy_day_features,
                    buy_price_tensor,
                ],
                verbose=0,
            )
        )

        return prob > 0.6

    def buy_price_select(self, i):
        return self.tw_ticket_gap(self.data.iloc[i]["open"])

    def sell_price_select(self, i):
        return self.tw_ticket_gap(self.data.iloc[i]["open"])
