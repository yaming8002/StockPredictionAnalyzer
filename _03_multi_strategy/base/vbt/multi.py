"""
vbt 多股策略套件 multi — 多檔共用現金組合回測基底
====================================================

對應「同一本金、多檔股票共用資金操作」：單一共用現金池，現金不足時擋單，
並可自訂買入優先序。繼承後覆寫 add_columns / buy_signal / sell_signal；
multi 專屬再可覆寫 priority。

依賴方向 _03 → _02：台股費用 / tick / summary 沿用 _02 的 common（不重造）。

底層走 vbt Portfolio.from_signals(cash_sharing=True, group_by=True)，
每檔投入「當前共用現金的比例」(invest_ratio)，現金用完 vbt 自動跳過後續買單。

最小範例：
    class MyMulti(VbtMultiStrategy):
        def add_columns(self, df): ...
        def buy_signal(self, df): ...
        def sell_signal(self, df): ...
        # 不覆寫 priority → 預設按 stock_id 排序（現金不足時 id 小者先買）

    res = MyMulti(invest_ratio=0.10).run(data_dict)   # data_dict={stock_id: df}

⚠️ 兩個需注意的點（OSS 限制）：
  1. priority 未覆寫時用預設 stock_id 欄序 + call_seq='auto'（保證 sell 先於 buy）最穩。
  2. failed_orders 為近似重建（vbt 不原生吐被拒單），精確需 order records 比對。
"""
import numpy as np
import pandas as pd
import vectorbt as vbt

from _02_strategy.base.vbt import common


class VbtMultiStrategy:
    """多檔共用現金策略基底。子類覆寫 hook；priority 為 multi 專屬。"""

    def __init__(self, initial_cash: float = 1_000_000.0,
                 invest_ratio: float = 1.0 / 30.0, min_invest: float = 5_000.0,
                 freq: str = "1D"):
        self.initial_cash = initial_cash
        self.invest_ratio = invest_ratio   # 每檔投入共用現金的比例
        self.min_invest = min_invest
        self.freq = freq

    # ── 子類覆寫的 hook（per-stock，傳入單檔 df）──────────────
    def add_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """#1 加指標欄位。預設不加。回傳 df。"""
        return df

    def buy_signal(self, df: pd.DataFrame) -> pd.Series:
        """#2 進場 bool Series。子類覆寫。"""
        raise NotImplementedError

    def sell_signal(self, df: pd.DataFrame) -> pd.Series:
        """#3 出場 bool Series。子類覆寫。"""
        raise NotImplementedError

    def build_signals(self, df: pd.DataFrame):
        """回傳 (entries, exits)；路徑相依出場覆寫此函式做單檔掃描。"""
        entries = self.buy_signal(df).fillna(False).astype(bool)
        exits = self.sell_signal(df).fillna(False).astype(bool)
        return entries, exits

    def priority(self, df: pd.DataFrame, stock_id: str):
        """
        #4 multi 專屬：買入優先分數（分數高者現金不足時優先成交）。
        預設回 None → 改用 stock_id 欄序（id 小者優先）。
        覆寫範例：return df["xgb_prob"]（機率高者優先）。
        """
        return None

    def exec_price(self, df: pd.DataFrame) -> pd.Series:
        return df["close"]

    # ── 主流程 ───────────────────────────────────────────────
    def run(self, data_dict: dict, start_date: str = None, end_date: str = None) -> dict:
        """data_dict: {stock_id: df（含 OHLCV 小寫 + DatetimeIndex）}。"""
        stock_ids = sorted(data_dict.keys())
        close_cols, entry_cols, exit_cols, price_cols, prio_cols = {}, {}, {}, {}, {}
        has_priority = False

        for sid in stock_ids:
            df = data_dict[sid]
            common.ensure_columns(df)
            df = self.add_columns(df.copy())
            if start_date or end_date:
                df = df.loc[(df.index >= (start_date or df.index.min())) &
                            (df.index <= (end_date or df.index.max()))]
            entries, exits = self.build_signals(df)
            close_cols[sid] = df["close"]
            entry_cols[sid] = entries.fillna(False).astype(bool)
            exit_cols[sid] = exits.fillna(False).astype(bool)
            price_cols[sid] = self.exec_price(df)
            prio = self.priority(df, sid)
            if prio is not None:
                has_priority = True
                prio_cols[sid] = prio

        # 對齊成 (T, N) panel（union 日期軸；個股缺資料 → NaN/不交易）
        close = pd.DataFrame(close_cols)
        entries = pd.DataFrame(entry_cols).reindex_like(close).fillna(False).astype(bool)
        exits = pd.DataFrame(exit_cols).reindex_like(close).fillna(False).astype(bool)
        price = pd.DataFrame(price_cols).reindex_like(close)

        # call_seq：有自訂 priority → 每列按分數降序；否則 'auto'（sell 先 + 穩定）
        call_seq = "auto"
        if has_priority:
            prio = pd.DataFrame(prio_cols).reindex_like(close).fillna(-np.inf)
            call_seq = np.argsort(-prio.to_numpy(), axis=1).astype(np.int_)

        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            price=price,
            init_cash=self.initial_cash,
            size=self.invest_ratio,
            size_type="percent",       # 共用現金下 = 當前可用現金的比例
            fees=common.COMMISSION,
            direction="longonly",
            freq=self.freq,
            accumulate=False,
            cash_sharing=True,
            group_by=True,             # 全部併成一個資金組合
            call_seq=call_seq,
        )

        trades = self._postprocess(pf)
        failed = self._approx_failed_orders(entries, trades)
        summary = common.summarize_trades(trades)
        summary["擋單數(近似)"] = int(failed)
        return {"trades": trades, "summary": summary, "failed_orders_approx": int(failed)}

    def _postprocess(self, pf) -> pd.DataFrame:
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
        rec = rec.rename(columns={"Column": "stock_id",
                                  "Entry Timestamp": "buy_date", "Exit Timestamp": "sell_date"})
        return rec[cols].reset_index(drop=True)

    @staticmethod
    def _approx_failed_orders(entries: pd.DataFrame, trades: pd.DataFrame) -> int:
        """
        近似擋單數 = 進場訊號總數 − 實際成交筆數。
        （一檔一部位下，持倉中重複訊號本就被忽略；故為上界近似，精確需 order records 比對。）
        """
        return max(0, int(entries.to_numpy().sum()) - len(trades))
