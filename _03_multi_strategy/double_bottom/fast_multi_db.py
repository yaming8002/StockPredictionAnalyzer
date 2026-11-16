"""
多股雙底（共用資金）的精簡引擎：〈雙底（八）〉挑出的組合 × 兩種投法 × 四種買入排序
==================================================================================
下單規則、費用口徑（台股實際費稅）、優先序平手規則全部同 `_03_multi_strategy/base/fast_multi.py`
的 `_simulate`（fees="tw"）；唯一不同是**出場在引擎裡依實際持倉逐日判定**，不吃預先算好的出場面板。

為什麼不能用預先算好的出場面板：
  ① 高點下移看「進場以來的最高擺動高點」、附加：保本停損看「這筆的進場價＋型態高度」，
  起點都是**實際有成交的那筆進場**。多股時進場可能因現金不足被擋，若照單股逐根掃描的出場序列，
  被擋掉的那次進場會被當成有持倉，出場時點就跟實際部位錯開（同 multi_double_bottom.py 檔頭）。

面板（全部先在個股自己的日曆上 shift(1)，判定日 → 成交日，再對齊聯集日曆）：
  entries ：進場（組合的進場條件＋可成交門檻），同單股 DoubleBottomEntryVariant.entry_signal
  ths     ：判定日新確認的擺動高點（① 用），沒有就 NaN
  lowbrk  ：判定日收盤 < 最近一個已確認擺動低點（②，與部位無關）
  mtop    ：判定日新確認的擺動高點 < 上一個已確認擺動高點（附加：高檔 M 頭，與部位無關）
  dclose  ：判定日收盤（附加：保本停損用）
  height  ：判定日的頸線 − 第一個底（型態高度；只在進場時讀，附加：保本停損用）
  price   ：成交日開盤（成交價；保本停損的「進場價」也用它，同單股 ExtraExit.start 的 open[i+1]）
條件式與單股 `single_double_bottom_strategy.build_signals`／`ExtraExit` 逐條對應：
  持倉中每個判定日：hit = ①（新高點 < 峰值）或 ② 或 附加規則；之後才把新高點併進峰值、更新保本狀態。
  進場判定日本身不判出場（單股 start 後 continue）——對應到這裡＝成交當天不看當天面板。
關掉資金限制時逐筆必須與單股相同，見 verify_db_multi.py。
"""
import numpy as np
import pandas as pd
from numba import njit

from _02_strategy.base.vbt import common
from _02_strategy.double_bottom_strategy.combo_sweep import make_combo
from _03_multi_strategy.base.fast_multi import EXACT, _tw_buy_fee, _tw_sell_fee

_MODE_CODE = {"percent_floor": 0, "fixed": 1}
_ADD = {"base": 0, "add_breakeven": 1, "add_mtop": 2}     # 這次用到的三種出場


def build_panel(data: dict, entry: str, exit_key: str, pct: float, start: str, end: str) -> dict:
    """全史算指標 → 個股日曆上 shift(1) → 裁區間 → 對齊聯集日曆。"""
    if exit_key not in _ADD:
        raise ValueError(f"多股只支援 {list(_ADD)}，收到 {exit_key}")
    single = make_combo(entry, exit_key, pct, True)
    single._trade_start = start
    keys = ("close", "entries", "ths", "lowbrk", "mtop", "dclose", "height", "price")
    cols = {k: {} for k in keys}
    for sid in sorted(data):
        df = data[sid]
        common.ensure_columns(df)
        df = single.add_columns(df.copy())
        th = df["zigzag_turn_high"]
        shifted = {
            "entries": single.entry_signal(df).fillna(False).astype(bool),
            "ths": th,
            "lowbrk": df["close"] < df["zigzag_turn_low"].ffill(),
            "mtop": th.notna() & (th < th.ffill().shift(1)),
            "dclose": df["close"],
            "height": df["db_neck"] - df["db_l1"],
        }
        keep = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        if not keep.any():
            continue
        cols["close"][sid] = df.loc[keep, "close"]
        cols["price"][sid] = df.loc[keep, "open"]
        for k, s in shifted.items():
            fill = False if s.dtype == bool else np.nan
            cols[k][sid] = s.shift(1, fill_value=fill).loc[keep]
    close = pd.DataFrame(cols["close"])
    out = {"close": close, "exit_code": _ADD[exit_key]}
    for k in keys[1:]:
        p = pd.DataFrame(cols[k]).reindex_like(close)
        if k in ("entries", "lowbrk", "mtop"):
            out[k] = (p == True).to_numpy(np.bool_)   # noqa: E712  對齊補出的 NaN → False
        else:
            out[k] = p.to_numpy(np.float64)
    out["price_tw"] = common.tw_tick_arr(out["price"])
    return out


@njit(cache=True)
def _exit_hit(j, i, add, ths, lowbrk, mtop, dclose, peak, armed, entry_px, height):
    """持倉中某個判定日的出場判定（順序同單股 build_signals：先判、再更新峰值與保本狀態）。"""
    th = ths[i, j]
    new_high = th == th
    hit = lowbrk[i, j] or (new_high and th < peak[j])
    if add == 2 and mtop[i, j]:
        hit = True
    if add == 1:
        c = dclose[i, j]
        if armed[j] and c < entry_px[j]:
            hit = True
        if c >= entry_px[j] + height[j]:
            armed[j] = True
    if new_high and th > peak[j]:
        peak[j] = th
    return hit


@njit(cache=True)
def _simulate(entries, ths, lowbrk, mtop, dclose, height, raw_px, px, close, prio, add,
              mode, ratio, min_invest, init_cash, want_equity, comm, min_fee, tax):
    T, N = px.shape
    pos = np.zeros(N)
    cost = np.zeros(N)
    entry_i = np.full(N, -1)
    peak = np.full(N, -np.inf)
    armed = np.zeros(N, np.bool_)
    entry_px = np.full(N, np.nan)
    hgt = np.full(N, np.nan)
    pending = np.zeros(N, np.bool_)      # 已判定出場、但成交日沒有價格 → 下一個有價日賣
    last_close = np.full(N, np.nan)
    cash = init_cash
    blocked = 0
    min_base = init_cash
    equity = np.full(T if want_equity else 0, np.nan)
    max_tr = 0
    for i in range(T):
        for j in range(N):
            if entries[i, j]:
                max_tr += 1
    tr_col = np.empty(max_tr, np.int64)
    tr_in = np.empty(max_tr, np.int64)
    tr_out = np.empty(max_tr, np.int64)
    tr_pin = np.empty(max_tr)
    tr_pout = np.empty(max_tr)
    tr_size = np.empty(max_tr)
    n_tr = 0
    cand = np.empty(N, np.int64)
    keys = np.empty(N)
    for i in range(T):
        base = cash
        for j in range(N):
            if pos[j] > 0.0:
                base += pos[j] * cost[j]
        if base < min_base:
            min_base = base
        sold = np.zeros(N, np.bool_)
        for j in range(N):
            if pos[j] > 0.0 and entry_i[j] < i:
                if not pending[j]:
                    pending[j] = _exit_hit(j, i, add, ths, lowbrk, mtop, dclose, peak, armed,
                                           entry_px, hgt)
                p = px[i, j]
                if pending[j] and p > 0.0:
                    amt = pos[j] * p
                    cash += amt - _tw_sell_fee(amt, comm, min_fee, tax)
                    tr_col[n_tr] = j
                    tr_in[n_tr] = entry_i[j]
                    tr_out[n_tr] = i
                    tr_pin[n_tr] = cost[j]
                    tr_pout[n_tr] = p
                    tr_size[n_tr] = pos[j]
                    n_tr += 1
                    pos[j] = 0.0
                    entry_i[j] = -1
                    pending[j] = False
                    sold[j] = True
        nc = 0
        for j in range(N):
            if pos[j] == 0.0 and entries[i, j] and not sold[j]:
                cand[nc] = j
                keys[nc] = -prio[i, j]
                nc += 1
        if nc > 0:
            order = np.argsort(keys[:nc], kind="mergesort")
            for k in range(nc):
                j = cand[order[k]]
                p = px[i, j]
                if not (p > 0.0):
                    continue
                if cash < min_invest:
                    blocked += 1
                    continue
                target = min_invest
                if mode == 0:
                    target = max(base * ratio, min_invest)
                budget = target if target < cash else cash
                shares = np.floor(budget / (p * (1.0 + comm)))
                while shares >= 1.0 and shares * p + _tw_buy_fee(shares * p, comm, min_fee) > budget:
                    shares -= 1.0
                if shares < 1.0:
                    blocked += 1
                    continue
                amt = shares * p
                cash -= amt + _tw_buy_fee(amt, comm, min_fee)
                pos[j] = shares
                cost[j] = p
                entry_i[j] = i
                peak[j] = -np.inf
                armed[j] = False
                entry_px[j] = raw_px[i, j]
                hgt[j] = height[i, j]
        if want_equity:
            val = cash
            for j in range(N):
                c = close[i, j]
                if c == c:
                    last_close[j] = c
                if pos[j] > 0.0:
                    val += pos[j] * last_close[j]
            equity[i] = val
    return (tr_col[:n_tr], tr_in[:n_tr], tr_out[:n_tr], tr_pin[:n_tr], tr_pout[:n_tr],
            tr_size[:n_tr], blocked, min_base, equity, pos)


def run_panel(sizing, panel: dict, prio: np.ndarray, want_equity: bool = True) -> dict:
    """
    sizing 只取 sizing_mode／invest_ratio／min_invest／initial_cash。回傳結構同 fast_multi.run_panel_fast
    （trades／summary／blocked_orders／equity），summary 另含已實現權益最低、最終權益、最大回撤、年化報酬。
    """
    close = panel["close"]
    pr = np.where(np.isnan(prio), -np.inf, prio)
    res = _simulate(panel["entries"], panel["ths"], panel["lowbrk"], panel["mtop"], panel["dclose"],
                    panel["height"], panel["price"], panel["price_tw"], close.to_numpy(np.float64), pr,
                    panel["exit_code"], _MODE_CODE[sizing.sizing_mode], float(sizing.invest_ratio),
                    float(sizing.min_invest), float(sizing.initial_cash), want_equity,
                    common.COMMISSION, common.MIN_COMMISSION, common.DUES)
    col, i_in, i_out, p_in, p_out, size, blocked, min_base, eq, pos = res
    idx, cols = close.index, close.columns
    trades = pd.DataFrame({"stock_id": cols.to_numpy()[col], "buy_date": idx.to_numpy()[i_in],
                           "sell_date": idx.to_numpy()[i_out], "buy_price": p_in, "sell_price": p_out,
                           "qty": size.astype(int)})
    if len(trades):
        bf, sf = common.reconstruct_fees(trades["buy_price"], trades["sell_price"], trades["qty"])
        trades["buy_fee"], trades["sell_fee"] = bf, sf
        trades["real_pnl"] = common.net_pnl(trades["buy_price"], trades["sell_price"], trades["qty"])
    trades = trades.sort_values(["sell_date", "stock_id"], kind="mergesort").reset_index(drop=True)
    summary = common.summarize_trades(trades)
    summary["擋單數"] = int(blocked)
    summary["已實現權益最低(%)" + EXACT] = min_base / sizing.initial_cash * 100.0
    summary["期末未平倉"] = int((pos > 0).sum())
    val = None
    if want_equity:
        val = pd.Series(eq, index=idx)
        dd = ((val.cummax() - val) / val.cummax()).fillna(0.0)
        years = (idx[-1] - idx[0]).days / 365.25
        summary["最終權益" + EXACT] = float(val.iloc[-1])
        summary["最大回撤(%)" + EXACT] = float(dd.max()) * 100.0
        summary["年化報酬(%)" + EXACT] = ((float(val.iloc[-1]) / sizing.initial_cash) ** (1 / years) - 1) * 100
        # 期末清算淨額：最後一天以收盤價把未平倉部位全部賣出、扣賣出手續費與證交稅後的帳戶淨額。
        # 跟 0050「期末全部賣出」比時用這個數，總獲利與資金倍數才是同一個口徑（最終權益只是市值、不扣賣出費稅）
        last_close = close.ffill().iloc[-1].to_numpy(np.float64)
        held = pos > 0.0
        _, sell_fee = common.reconstruct_fees(last_close[held], last_close[held], pos[held])
        summary["期末清算淨額" + EXACT] = float(val.iloc[-1]) - float(sell_fee.sum())
    return {"trades": trades, "summary": summary, "blocked_orders": int(blocked), "equity": val}
