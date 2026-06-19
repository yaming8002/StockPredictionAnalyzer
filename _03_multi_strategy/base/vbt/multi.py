"""
vbt 多股策略套件 multi — 多檔共用現金組合回測基底
====================================================

對應「同一本金、多檔股票共用資金操作」：單一共用現金池，**支援兩種倉位模式**，
現金不足最低門檻時真擋單（不交易），並可自訂買入優先序。
繼承後覆寫 add_columns / buy_signal / sell_signal；multi 專屬再可覆寫 priority。

依賴方向 _03 → _02：台股費用 / tick / summary 沿用 _02 的 common（不重造）。

倉位模式（sizing_mode）：
  - "percent_floor"：每筆投入 = max(已實現權益 × invest_ratio, min_invest)。
        已實現權益 = 現金 + Σ(各持股 進場成本價 × 股數)（＝本金＋已實現損益−費用，排除未實現）。
        與上線系統「總金額＝本金＋已實現」一致；同一交易日多筆買單共用「當日段首」的權益快照。
  - "fixed"：每筆投入固定 = min_invest。
  共同：**現金 < min_invest → 不交易（真擋單）**；買到「能負擔的整股數」（不夠全額時買到現金上限）。

底層走 vbt Portfolio.from_order_func（cash_sharing=True, group_by=True）逐筆模擬，
能在「下單當下」讀到共用現金與持倉成本，故下限 / 擋單 / 整股 / 已實現權益基準都精確。

最小範例：
    class MyMulti(VbtMultiStrategy):
        def add_columns(self, df): ...
        def buy_signal(self, df): ...
        def sell_signal(self, df): ...
        # 不覆寫 priority → 預設賣先買後、買單按 stock_id 欄序

    res = MyMulti(sizing_mode="percent_floor", invest_ratio=0.10,
                  min_invest=10_000).run(data_dict)   # data_dict={stock_id: df}

實作備註：
  - 整股原生（下單前 floor 成整股）；不再有舊版 qty=int() 的截斷偏差或「縮到 0 股消失」。
  - 擋單數為精確計數（下單函式擋單時累加），非舊版 entries−成交 的高估近似。
  - 模擬內現金軌跡：買用手續費率、賣用手續費率＋證交稅（讓「下一筆可用現金」貼近台股實況）；
    成交損益另由 common.net_pnl 以台股費稅（含最低 20 元）精算，故模擬內未含最低手續費為唯一殘留近似。
"""
import numpy as np
import pandas as pd
import vectorbt as vbt
from numba import njit
from vectorbt.portfolio import nb
from vectorbt.portfolio.enums import SizeType, Direction, NoOrder

from _02_strategy.base.vbt import common

_MODE_CODE = {"percent_floor": 0, "fixed": 1}


@njit
def _pre_segment_nb(c, entries, exits, prio, has_prio, base_out):
    """每個交易日段：設估值價、算當日已實現權益快照、排定 call_seq（賣先 → 買按 priority 高者先）。"""
    # 估值價 = 當前 close（供 vbt 內部估值）
    for col in range(c.from_col, c.to_col):
        c.last_val_price[col] = c.close[c.i, col]

    # base = 現金 + Σ(持股 size × 進場成本價)；同段所有買單共用此快照
    base = c.last_cash[c.group]
    for col in range(c.from_col, c.to_col):
        pos = c.last_position[col]
        if pos > 1e-9:
            base += pos * c.last_pos_record[col].entry_price
    base_out[c.group] = base

    # call_seq：賣(釋放現金)最前 → 買(priority 高→低) → 無動作最後
    n = c.group_len
    key = np.empty(n, dtype=np.float64)
    for k in range(n):
        col = c.from_col + k
        is_sell = (c.last_position[col] > 1e-9) and exits[c.i, col]
        is_buy = (c.last_position[col] < 1e-9) and entries[c.i, col]
        if is_sell:
            key[k] = -1e18
        elif is_buy:
            key[k] = -prio[c.i, col] if has_prio else 0.0
        else:
            key[k] = 1e18
    order = np.argsort(key)
    for k in range(n):
        c.call_seq_now[k] = order[k]
    return ()


@njit
def _order_nb(c, entries, exits, px, mode, ratio, min_invest,
              buy_fee, sell_fee, base_out, blocked_out):
    """單檔單日下單：出場全出；進場按模式定額、整股、真擋單。"""
    col = c.col
    i = c.i
    price = px[i, col]
    if not (price > 0.0):                       # 缺資料 / 無效價 → 不動作
        return NoOrder

    # 出場：持倉且出場訊號 → 全部賣出（賣方費含證交稅）
    if c.position_now > 1e-9 and exits[i, col]:
        return nb.order_nb(size=-c.position_now, price=price, size_type=SizeType.Amount,
                           direction=Direction.LongOnly, fees=sell_fee)

    # 進場：空手且進場訊號
    if c.position_now < 1e-9 and entries[i, col]:
        cash = c.cash_now
        if cash < min_invest:                   # 現金不足最低門檻 → 不交易（真擋單）
            blocked_out[c.group] += 1.0
            return NoOrder
        if mode == 0:                           # percent_floor
            target = base_out[c.group] * ratio
            if target < min_invest:
                target = min_invest
        else:                                   # fixed
            target = min_invest
        cps = price * (1.0 + buy_fee)
        shares = np.floor(target / cps)
        afford = np.floor(cash / cps)           # 不夠全額 → 買到能負擔的整股
        if shares > afford:
            shares = afford
        if shares < 1.0:
            blocked_out[c.group] += 1.0
            return NoOrder
        return nb.order_nb(size=shares, price=price, size_type=SizeType.Amount,
                           direction=Direction.LongOnly, fees=buy_fee)
    return NoOrder


class VbtMultiStrategy:
    """多檔共用現金策略基底。子類覆寫 hook；priority 為 multi 專屬。"""

    def __init__(self, initial_cash: float = 1_000_000.0,
                 sizing_mode: str = "percent_floor",
                 invest_ratio: float = 1.0 / 30.0, min_invest: float = 10_000.0,
                 freq: str = "1D"):
        if sizing_mode not in _MODE_CODE:
            raise ValueError(f"sizing_mode 須為 {list(_MODE_CODE)}，收到 {sizing_mode!r}")
        self.initial_cash = initial_cash
        self.sizing_mode = sizing_mode
        self.invest_ratio = invest_ratio   # percent_floor 模式：佔已實現權益的比例
        self.min_invest = min_invest       # 最低門檻 / fixed 模式的固定額
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
        預設回 None → 改用 stock_id 欄序（賣先買後，買單 id 小者優先）。
        覆寫範例：return df["xgb_prob"]（機率高者優先）。
        """
        return None

    def exec_price(self, df: pd.DataFrame) -> pd.Series:
        """成交價（預設 close；子類可改 open 做隔日開盤成交，配合 build_signals 位移）。"""
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
        entries = pd.DataFrame(entry_cols).reindex_like(close).fillna(False).infer_objects(copy=False).astype(bool)
        exits = pd.DataFrame(exit_cols).reindex_like(close).fillna(False).infer_objects(copy=False).astype(bool)
        price = pd.DataFrame(price_cols).reindex_like(close)
        if has_priority:
            prio = pd.DataFrame(prio_cols).reindex_like(close)
        else:
            prio = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        ev = entries.to_numpy(dtype=np.bool_)
        xv = exits.to_numpy(dtype=np.bool_)
        px = price.to_numpy(dtype=np.float64)
        pr = prio.to_numpy(dtype=np.float64)
        pr = np.where(np.isnan(pr), -np.inf, pr)   # 缺 priority → 排最後

        base_out = np.zeros(1, dtype=np.float64)        # 1 個共用現金組合
        blocked_out = np.zeros(1, dtype=np.float64)
        mode = _MODE_CODE[self.sizing_mode]
        buy_fee = common.COMMISSION
        sell_fee = common.COMMISSION + common.DUES

        pf = vbt.Portfolio.from_order_func(
            close,
            _order_nb, ev, xv, px, mode, float(self.invest_ratio),
            float(self.min_invest), buy_fee, sell_fee, base_out, blocked_out,
            pre_segment_func_nb=_pre_segment_nb,
            pre_segment_args=(ev, xv, pr, has_priority, base_out),
            init_cash=self.initial_cash,
            cash_sharing=True,
            group_by=True,
            freq=self.freq,
        )

        trades = self._postprocess(pf)
        summary = common.summarize_trades(trades)
        blocked = int(blocked_out[0])
        summary["擋單數"] = blocked
        return {"trades": trades, "summary": summary, "blocked_orders": blocked}

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
