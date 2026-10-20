"""
多股雙底（共用資金）
====================
把單股雙底（`_02_strategy/double_bottom_strategy/single_double_bottom_strategy.py`）搬到
「單一本金、多檔共用現金」的組合回測。進場、擺動點、流動性門檻都直接呼叫單股策略的同一支函式，
執行時序一致：**收盤判定 → 隔日開盤成交**。

**為什麼要自己的下單函式（不能用基底的 entries／exits 面板）**：
  出場 ① CHoCH（lower-high）看的是「進場以來的最高擺動高點」，起點是**實際有成交的那筆進場**。
  多股時進場可能因現金不足被擋，若沿用單股逐根掃描預先算好的出場序列，被擋的那次進場會被當成
  有持倉、出場時點就跟實際部位錯配。所以這裡只把「與部位無關」的逐日資料做成面板：
    ths    ：當天（判定日）新確認的擺動高點，沒有就 NaN
    lowbrk ：當天收盤跌破最近一個已確認擺動低點（② 跌破波段低點；與部位無關，純狀態）
  兩者都先在個股自己的日曆上 shift(1) 成「成交日」再對齊（不在聯集日曆上取 [i-1]），
  然後在 vbt `from_order_func` 的 pre_segment 裡，逐日依「目前真的有沒有持倉」維護每檔的峰值：
    空手 → 峰值重設；持倉 → 先判斷出場（ths < 峰值 或 lowbrk），再把 ths 併進峰值。
  進場當天（空手）不處理 ths，正好排除「進場判定日當天確認的高點」，與單股掃描一致。

不設停損版（單股的「# 優化 #1」）：STOP_LOW = False，只留 ①。

費用口徑 fees（run_panel 參數）：
  "rate"（預設，同 VbtMultiStrategy）：現金軌跡用純費率（無最低 20 元、成交價不過 tick），
        逐筆損益另以台股費稅精算 → 權益曲線與總獲利是兩本帳。
  "tw"  ：vbt 下單時 fees=0、fixed_fees＝該筆台股實際費稅（手續費 0.1425% 最低 20 元無條件進位；
        賣方再加證交稅 0.3%，與手續費相加後一起進位），成交價先過 tw_tick_arr——公式與
        common.reconstruct_fees／fast_multi fees="tw" 同一套，權益曲線與逐筆 real_pnl 是同一本帳。
        股數＝「金額＋實際手續費 ≤ min(目標, 現金)」的最大整股（同 fast_multi）。
與部位無關的「狀態型」出場（例：海龜跌破 N 日低）也能走這個下單函式：ths 全 NaN、lowbrk＝出場面板，
見 simulate_state_exit。

倉位、擋單、優先序、費用口徑全部沿用 `VbtMultiStrategy`（同一套 _order 規則），只多了出場判斷。
關掉資金限制（本金極大、每筆固定）時，多股逐筆應與單股完全相同——見 _03_multi_strategy/candidates/verify_multi_candidates.py。
"""
import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd
import vectorbt as vbt
from numba import njit
from vectorbt.portfolio import nb
from vectorbt.portfolio.enums import SizeType, Direction, NoOrder

from _02_strategy.base.vbt import common
from _02_strategy.double_bottom_strategy.single_double_bottom_strategy import SingleDoubleBottomStrategy
from _03_multi_strategy.base.fast_multi import _tw_buy_fee, _tw_sell_fee
from _03_multi_strategy.base.vbt.multi import VbtMultiStrategy, _MODE_CODE

FEES = ("rate", "tw")


@njit
def _pre_segment_db(c, entries, ths, lowbrk, stop_low, peak, exit_now, prio, has_prio, base_out):
    """每日段首：依實際持倉更新峰值並決定出場、算已實現權益快照、排 call_seq（賣先 → 買按 priority）。"""
    for col in range(c.from_col, c.to_col):
        c.last_val_price[col] = c.close[c.i, col]
        exit_now[col] = False
        if c.last_position[col] > 1e-9:
            th = ths[c.i, col]
            hit = stop_low and lowbrk[c.i, col]                 # ② 跌破波段低點
            if not np.isnan(th):
                if th < peak[col]:                              # ① lower-high
                    hit = True
                if th > peak[col]:
                    peak[col] = th
            exit_now[col] = hit
        else:
            peak[col] = -np.inf                                 # 空手：峰值重設（下一筆進場重新起算）

    base = c.last_cash[c.group]
    for col in range(c.from_col, c.to_col):
        pos = c.last_position[col]
        if pos > 1e-9:
            base += pos * c.last_pos_record[col].entry_price
    base_out[c.group] = base

    n = c.group_len
    key = np.empty(n, dtype=np.float64)
    for k in range(n):
        col = c.from_col + k
        is_sell = (c.last_position[col] > 1e-9) and exit_now[col]
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
def _order_db(c, entries, exit_now, px, mode, ratio, min_invest,
              buy_fee, sell_fee, base_out, blocked_out, tw, comm, min_fee, tax):
    """與 multi._order_nb 相同的下單規則，只是出場旗標改讀段首算好的 exit_now；tw=True 用台股實際費稅。"""
    col = c.col
    i = c.i
    price = px[i, col]
    if not (price > 0.0):
        return NoOrder
    if c.position_now > 1e-9 and exit_now[col]:
        if tw:
            amt = c.position_now * price
            return nb.order_nb(size=-c.position_now, price=price, size_type=SizeType.Amount,
                               direction=Direction.LongOnly, fees=0.0,
                               fixed_fees=_tw_sell_fee(amt, comm, min_fee, tax))
        return nb.order_nb(size=-c.position_now, price=price, size_type=SizeType.Amount,
                           direction=Direction.LongOnly, fees=sell_fee)
    if c.position_now < 1e-9 and entries[i, col]:
        cash = c.cash_now
        if cash < min_invest:
            blocked_out[c.group] += 1.0
            return NoOrder
        if mode == 0:
            target = base_out[c.group] * ratio
            if target < min_invest:
                target = min_invest
        else:
            target = min_invest
        if tw:
            # 預算＝min(目標, 現金)（含手續費）；先用費率估股數，再逐股往下減到放得進預算（同 fast_multi）
            budget = target if target < cash else cash
            shares = np.floor(budget / (price * (1.0 + comm)))
            while shares >= 1.0 and shares * price + _tw_buy_fee(shares * price, comm, min_fee) > budget:
                shares -= 1.0
            if shares < 1.0:
                blocked_out[c.group] += 1.0
                return NoOrder
            return nb.order_nb(size=shares, price=price, size_type=SizeType.Amount,
                               direction=Direction.LongOnly, fees=0.0,
                               fixed_fees=_tw_buy_fee(shares * price, comm, min_fee))
        cps = price * (1.0 + buy_fee)
        shares = np.floor(target / cps)
        afford = np.floor(cash / cps)
        if shares > afford:
            shares = afford
        if shares < 1.0:
            blocked_out[c.group] += 1.0
            return NoOrder
        return nb.order_nb(size=shares, price=price, size_type=SizeType.Amount,
                           direction=Direction.LongOnly, fees=buy_fee)
    return NoOrder


class MultiDoubleBottom(VbtMultiStrategy):
    """多股雙底：底底高進場（可成交門檻）；出場 ① CHoCH 或 ② 跌破波段低點（STOP_LOW=False 只留 ①）。"""

    PCT = 0.05
    STOP_LOW = True
    LIQUIDITY = True     # 多股一律可成交版（共用資金真的要下單）

    def _single(self) -> SingleDoubleBottomStrategy:
        s = SingleDoubleBottomStrategy(pct=self.PCT, liquidity=self.LIQUIDITY)
        s._trade_start = getattr(self, "_trade_start", None)    # 區間前的進場不算（與單股同一閘門）
        return s

    def exec_price(self, df: pd.DataFrame) -> pd.Series:
        return df["open"]                                       # 隔日開盤價成交

    def build_panel(self, data_dict: dict, start_date: str = None, end_date: str = None) -> dict:
        """與基底 build_panel 同一流程（全史算指標 → 裁區間 → 對齊），另外多收 ths／lowbrk 兩張面板。"""
        self._trade_start = start_date
        single = self._single()
        cols = {k: {} for k in ("close", "entries", "ths", "lowbrk", "price")}
        for sid in sorted(data_dict):
            df = data_dict[sid]
            common.ensure_columns(df)
            df = single.add_columns(df.copy())
            # 全部先在個股自己的日曆上 shift(1)：判定日 → 成交日
            entries = single.entry_signal(df).fillna(False).astype(bool).shift(1, fill_value=False)
            ths = df["zigzag_turn_high"].shift(1)
            lowbrk = (df["close"] < df["zigzag_turn_low"].ffill()).shift(1, fill_value=False)
            keep = ((df.index >= pd.Timestamp(start_date or df.index.min()))
                    & (df.index <= pd.Timestamp(end_date or df.index.max())))
            if not keep.any():
                continue
            cols["close"][sid] = df.loc[keep, "close"]
            cols["entries"][sid] = entries.loc[keep]
            cols["ths"][sid] = ths.loc[keep]
            cols["lowbrk"][sid] = lowbrk.loc[keep].astype(bool)
            cols["price"][sid] = self.exec_price(df).loc[keep]
        close = pd.DataFrame(cols["close"])
        as_bool = lambda d: (pd.DataFrame(d).reindex_like(close).fillna(False)
                             .infer_objects(copy=False).astype(bool).to_numpy(np.bool_))
        return {"close": close,
                "entries": as_bool(cols["entries"]),
                "exits": np.zeros(close.shape, dtype=np.bool_),   # 出場在下單模擬內決定（見檔頭）
                "ths": pd.DataFrame(cols["ths"]).reindex_like(close).to_numpy(np.float64),
                "lowbrk": as_bool(cols["lowbrk"]),
                "price": pd.DataFrame(cols["price"]).reindex_like(close).to_numpy(np.float64),
                "prio": np.zeros(close.shape), "has_priority": False}

    def run_panel(self, panel: dict, prio: np.ndarray = None, want_equity: bool = True,
                  fees: str = "rate") -> dict:
        """同基底 run_panel，下單函式換成帶峰值狀態的 _pre_segment_db／_order_db。"""
        return simulate_state_exit(self, panel, panel["ths"], panel["lowbrk"], bool(self.STOP_LOW),
                                   prio, want_equity, fees)


def simulate_state_exit(inst, panel: dict, ths: np.ndarray, lowbrk: np.ndarray, stop_low: bool,
                        prio: np.ndarray = None, want_equity: bool = True, fees: str = "rate") -> dict:
    """
    共用的 vbt 下單模擬：出場＝（stop_low 且 lowbrk）或（ths 低於實際持倉以來峰值）。
    inst 只取 sizing 設定與 _postprocess。海龜這類狀態型出場：ths 全 NaN、lowbrk＝出場面板。
    """
    if fees not in FEES:
        raise ValueError(f"fees 須為 {FEES}，收到 {fees!r}")
    tw = fees == "tw"
    close = panel["close"]
    ev = panel["entries"]
    if tw:
        if "price_tw" not in panel:                      # 成交價先過 tick（同 fast_multi.tick_price）
            panel["price_tw"] = common.tw_tick_arr(panel["price"])
        px = panel["price_tw"]
    else:
        px = panel["price"]
    if prio is None:
        pr, has_priority = panel["prio"], panel["has_priority"]
    else:
        pr = prio if not np.isnan(prio).any() else np.where(np.isnan(prio), -np.inf, prio)
        has_priority = True
    n = close.shape[1]
    peak = np.full(n, -np.inf)
    exit_now = np.zeros(n, dtype=np.bool_)
    base_out = np.zeros(1)
    blocked_out = np.zeros(1)
    buy_fee = common.COMMISSION
    sell_fee = common.COMMISSION + common.DUES
    max_orders = int(ev.sum()) * 2 + 1                   # 出場次數不會超過進場次數

    pf = vbt.Portfolio.from_order_func(
        close,
        _order_db, ev, exit_now, px, _MODE_CODE[inst.sizing_mode], float(inst.invest_ratio),
        float(inst.min_invest), buy_fee, sell_fee, base_out, blocked_out,
        tw, common.COMMISSION, common.MIN_COMMISSION, common.DUES,
        pre_segment_func_nb=_pre_segment_db,
        pre_segment_args=(ev, ths, lowbrk, stop_low, peak, exit_now, pr, has_priority, base_out),
        init_cash=inst.initial_cash, cash_sharing=True, group_by=True,
        freq=inst.freq, max_orders=max_orders,
    )
    trades = inst._postprocess(pf)
    summary = common.summarize_trades(trades)
    blocked = int(blocked_out[0])
    summary["擋單數"] = blocked
    val = None
    if want_equity:
        val = pf.value()
        if hasattr(val, "columns"):
            val = val.iloc[:, 0]
        peak_v = val.cummax()
        summary["最終權益"] = round(float(val.iloc[-1]), 2)
        summary["最大回撤(%)"] = round(float(((peak_v - val) / peak_v).max()) * 100.0, 2)
    return {"trades": trades, "summary": summary, "blocked_orders": blocked, "equity": val, "pf": pf}
