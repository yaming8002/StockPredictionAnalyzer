"""
多股共用資金的精簡引擎（給「隨機買入順序 × 上千次」用）
========================================================
`VbtMultiStrategy.run_panel` 走 vbt `from_order_func`，全市場跑一次約 10 秒；
文章的「隨機排序取中位」要對每一組重抽 1,000 次，幾十組就是幾十萬次，vbt 版要跑好幾天。
這支用 numba 直接重寫同一套下單規則，**語意逐項對齊 multi.py 的 `_pre_segment_nb`／`_order_nb`**：

  - 每天先賣後買：持倉且有出場訊號 → 全部賣出；之後空手且有進場訊號的，依優先序高→低下單。
  - 成交價 ＝ 面板 price；價格缺值或 ≤0 的那一格不動作（賣也不賣）。
  - 每筆投入：fixed ＝ min_invest；percent_floor ＝ max(當日段首已實現權益 × ratio, min_invest)。
    已實現權益 ＝ 現金 ＋ Σ(持股數 × 進場成本價)，同一天所有買單共用段首快照。
  - 現金 < min_invest → 擋單；買不到 1 股也算擋單。
  - 期末未平倉的部位不列入交易（與 vbt 版只取 Closed 一致），也不做期末強制平倉。

**費用口徑（fees 參數，2026-10-10 起預設 "tw"）**
  - "tw"（預設）：現金軌跡改用**台股實際費稅**，與逐筆 real_pnl 的重建公式（common.reconstruct_fees）
    完全同一套：
      成交價 ＝ common.tw_tick_arr(面板 price)（升降單位無條件進位；逐筆表的買賣價也是這個值）
      買方手續費 ＝ ceil(max(價×股×0.1425%, 20))
      賣方費用   ＝ ceil(max(價×股×0.1425%, 20) ＋ 價×股×0.3%)（手續費＋證交稅，一起無條件進位）
      買進扣現金 ＝ 價×股 ＋ 買方手續費；賣出入現金 ＝ 價×股 − 賣方費用
    股數 ＝ 「價×股＋買方手續費 ≤ min(目標投入, 現金)」的最大整股（目標投入含手續費，與 vbt 口徑
    「目標 ÷ (價×(1＋費率))」同義，只是把費用換成實際公式）。
    所以「最終權益、已實現權益最低、最大回撤」與「總獲利」終於是同一本帳：
      已實現權益 ＝ 本金 ＋ Σ已平倉 real_pnl − Σ未平倉部位的買方手續費
      最終權益   ＝ 本金 ＋ Σ已平倉 real_pnl ＋ Σ未平倉(股數×最後收盤 − 股數×買價 − 買方手續費)
    **未平倉部位以最後收盤價計市值、不扣賣出費稅**（與 vbt 版 pf.value() 及先前各篇的口徑相同）。
  - "vbt"：舊行為（純費率：買付 價×股×(1＋0.1425%)、賣收 價×股×(1−0.1425%−0.3%)，無最低 20 元、
    不進位、成交價不過 tick），只留給 verify_fast_multi.py／verify_kd_fast.py 跟 vbt 版逐筆對帳用，
    數值與改版前逐位元相同。這個模式下最終權益與「本金＋總獲利」對不上（例：均線 5/50 定額
    344 萬 vs 263 萬），不可再拿來出文章數字。

唯一刻意不同：**優先序平手時，按股票欄序（代號升冪）先買**。vbt 版用 argsort，平手時
順序不保證；餵沒有平手的分數時兩者逐筆相同（見 verify_fast_multi.py）。

另外多算一個 vbt 版沒有的診斷值：**已實現權益的最低點**（佔本金比例）——文章的
「本金曾跌破 8 成」用的是已實現權益（本金＋已實現損益−費用），不是含未實現的市值權益。
"""
import numpy as np
import pandas as pd
from numba import njit

from _02_strategy.base.vbt import common

_MODE_CODE = {"percent_floor": 0, "fixed": 1}
_FEES = ("tw", "vbt")
EXACT = "_精確"    # 未四捨五入欄的後綴（文章出表要從精確值一次進位，見 exact_stats）


@njit(cache=True)
def _tw_buy_fee(amt, comm, min_fee):
    """買方手續費（同 common.reconstruct_fees：最低 20 元、無條件進位）。"""
    return np.ceil(max(amt * comm, min_fee))


@njit(cache=True)
def _tw_sell_fee(amt, comm, min_fee, tax):
    """賣方手續費＋證交稅（同 common.reconstruct_fees：兩者相加後一起無條件進位）。"""
    return np.ceil(max(amt * comm, min_fee) + amt * tax)


@njit(cache=True)
def _simulate(entries, exits, px, close, prio, mode, ratio, min_invest, init_cash,
              buy_fee, sell_fee, want_equity, tw, comm, min_fee, tax):
    """
    tw=False：舊的純費率口徑（buy_fee／sell_fee 為費率），與改版前逐位元相同。
    tw=True ：px 須是已過 tick 的成交價；費用照台股實際公式（comm／min_fee／tax）。
    """
    T, N = px.shape
    pos = np.zeros(N)                       # 持股數
    cost = np.zeros(N)                      # 進場成本價
    entry_i = np.full(N, -1)
    last_close = np.full(N, np.nan)         # 估值用：停牌日沿用最後收盤
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
        # 段首已實現權益快照（同一天所有買單共用）
        base = cash
        for j in range(N):
            if pos[j] > 0.0:
                base += pos[j] * cost[j]
        if base < min_base:
            min_base = base

        # 先賣
        sold = np.zeros(N, np.bool_)
        for j in range(N):
            if pos[j] > 0.0 and exits[i, j]:
                p = px[i, j]
                if not (p > 0.0):
                    continue
                if tw:
                    amt = pos[j] * p
                    cash += amt - _tw_sell_fee(amt, comm, min_fee, tax)
                else:
                    cash += pos[j] * p * (1.0 - sell_fee)
                tr_col[n_tr] = j
                tr_in[n_tr] = entry_i[j]
                tr_out[n_tr] = i
                tr_pin[n_tr] = cost[j]
                tr_pout[n_tr] = p
                tr_size[n_tr] = pos[j]
                n_tr += 1
                pos[j] = 0.0
                entry_i[j] = -1
                sold[j] = True

        # 再買：持倉為 0 且有買訊（含今天剛賣掉的不會再買——vbt 一格一天只回一張單）
        nc = 0
        for j in range(N):
            if pos[j] == 0.0 and entries[i, j] and not sold[j]:
                cand[nc] = j
                keys[nc] = -prio[i, j]
                nc += 1
        if nc > 0:
            order = np.argsort(keys[:nc], kind="mergesort")   # 穩定排序：平手按欄序
            for k in range(nc):
                j = cand[order[k]]
                p = px[i, j]
                if not (p > 0.0):
                    continue
                if cash < min_invest:
                    blocked += 1
                    continue
                if mode == 0:
                    target = base * ratio
                    if target < min_invest:
                        target = min_invest
                else:
                    target = min_invest
                if tw:
                    # 預算＝min(目標, 現金)，含手續費。先用費率估股數（估出來的只會多不會少：
                    # 實際費用 ≥ 費率×金額），再逐股往下減到「金額＋實際手續費 ≤ 預算」為止。
                    budget = target if target < cash else cash
                    shares = np.floor(budget / (p * (1.0 + comm)))
                    while shares >= 1.0 and shares * p + _tw_buy_fee(shares * p, comm, min_fee) > budget:
                        shares -= 1.0
                    if shares < 1.0:
                        blocked += 1
                        continue
                    amt = shares * p
                    cash -= amt + _tw_buy_fee(amt, comm, min_fee)
                else:
                    cps = p * (1.0 + buy_fee)
                    shares = np.floor(target / cps)
                    afford = np.floor(cash / cps)
                    if shares > afford:
                        shares = afford
                    if shares < 1.0:
                        blocked += 1
                        continue
                    cash -= shares * cps
                pos[j] = shares
                cost[j] = p
                entry_i[j] = i

        if want_equity:
            # 市值權益：未平倉部位以當日（停牌沿用最後）收盤計價，不扣賣出費稅
            val = cash
            for j in range(N):
                c = close[i, j]
                if c == c:
                    last_close[j] = c
                if pos[j] > 0.0:
                    val += pos[j] * last_close[j]
            equity[i] = val
    return (tr_col[:n_tr], tr_in[:n_tr], tr_out[:n_tr], tr_pin[:n_tr], tr_pout[:n_tr],
            tr_size[:n_tr], blocked, min_base, equity, pos, cost, entry_i, cash)


def tick_price(panel: dict) -> np.ndarray:
    """
    "tw" 口徑的成交價面板＝tw_tick_arr(面板 price)，第一次算完快取在 panel["price_tw"]。
    重抽上千次時不必每場重算；代價是多佔一份 (天數 × 檔數) 的 float64（全市場約 100 MB）。
    不在引擎裡逐格算 tick：tw_tick_arr 有浮點邊界（對已進位的價再進位一次不保證不變），
    逐筆表的買賣價必須與現金軌跡用的是同一個值，所以兩邊都直接用這份面板。
    """
    if "price_tw" not in panel:
        panel["price_tw"] = common.tw_tick_arr(panel["price"])
    return panel["price_tw"]


def run_panel_fast(strategy, panel: dict, prio: np.ndarray = None,
                   want_equity: bool = True, fees: str = "tw") -> dict:
    """
    與 `VbtMultiStrategy.run_panel` 同介面、同回傳結構；多一個 summary 欄
    「已實現權益最低(%)」（相對本金）。strategy 只取它的 sizing 設定。
    fees："tw"（預設，台股實際費稅）／"vbt"（舊純費率，只給對帳腳本用），見檔頭。
    另回傳 "open"：期末未平倉部位（stock_id／buy_date／buy_price／qty／buy_fee），
    供恆等式檢查「最終權益＝本金＋Σreal_pnl＋未平倉市值損益」。
    """
    if fees not in _FEES:
        raise ValueError(f"fees 須為 {_FEES}，收到 {fees!r}")
    tw = fees == "tw"
    close = panel["close"]
    pr = panel["prio"] if prio is None else np.where(np.isnan(prio), -np.inf, prio)
    if prio is None and not panel["has_priority"]:
        pr = np.zeros_like(panel["price"])
    px = tick_price(panel) if tw else panel["price"]
    res = _simulate(panel["entries"], panel["exits"], px,
                    close.to_numpy(np.float64), pr, _MODE_CODE[strategy.sizing_mode],
                    float(strategy.invest_ratio), float(strategy.min_invest),
                    float(strategy.initial_cash), common.COMMISSION,
                    common.COMMISSION + common.DUES, want_equity,
                    tw, common.COMMISSION, common.MIN_COMMISSION, common.DUES)
    col, i_in, i_out, p_in, p_out, size, blocked, min_base, eq, pos, cost, entry_i, _ = res

    idx, cols = close.index, close.columns
    # tw：引擎用的就是已進位價，直接沿用（再過一次 tw_tick_arr 可能因浮點多進一檔）；
    # vbt：引擎用原始價，逐筆表照舊在這裡進位
    tick = (lambda a: a) if tw else common.tw_tick_arr
    trades = pd.DataFrame({
        "stock_id": cols.to_numpy()[col],
        "buy_date": idx.to_numpy()[i_in],
        "sell_date": idx.to_numpy()[i_out],
        "buy_price": tick(p_in),
        "sell_price": tick(p_out),
        "qty": size.astype(int),
    })
    if len(trades):
        bf, sf = common.reconstruct_fees(trades["buy_price"], trades["sell_price"], trades["qty"])
        trades["buy_fee"], trades["sell_fee"] = bf, sf
        trades["real_pnl"] = common.net_pnl(trades["buy_price"], trades["sell_price"], trades["qty"])
    else:
        trades = trades.assign(buy_fee=[], sell_fee=[], real_pnl=[])
    trades = trades.sort_values(["sell_date", "stock_id"], kind="mergesort").reset_index(drop=True)

    held = np.flatnonzero(pos > 0.0)
    open_pos = pd.DataFrame({
        "stock_id": cols.to_numpy()[held],
        "buy_date": idx.to_numpy()[entry_i[held]],
        "buy_price": tick(cost[held]),
        "qty": pos[held].astype(int),
    })
    open_pos["buy_fee"] = (common.reconstruct_fees(open_pos["buy_price"], open_pos["buy_price"],
                                                   open_pos["qty"])[0] if len(open_pos) else [])

    summary = common.summarize_trades(trades)
    summary["擋單數"] = int(blocked)
    min_pct = min_base / strategy.initial_cash * 100.0
    summary["已實現權益最低(%)"] = round(min_pct, 2)
    summary["已實現權益最低(%)" + EXACT] = min_pct
    val = None
    if want_equity:
        val = pd.Series(eq, index=idx)
        peak = val.cummax()
        dd = ((peak - val) / peak).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        max_dd = float(dd.max()) * 100.0 if len(val) else 0.0
        summary["最終權益"] = round(float(val.iloc[-1]), 2) if len(val) else strategy.initial_cash
        summary["最大回撤(%)"] = round(max_dd, 2)
        summary["最終權益" + EXACT] = float(val.iloc[-1]) if len(val) else strategy.initial_cash
        summary["最大回撤(%)" + EXACT] = max_dd
    return {"trades": trades, "summary": summary, "blocked_orders": int(blocked),
            "equity": val, "open": open_pos}


def exact_stats(trades: pd.DataFrame) -> dict:
    """
    規格 9 欄的**未四捨五入**值（定義同 common.summarize_trades：排除淨損益 0、報酬率用毛報酬），
    鍵名＝「<規格欄>_精確」。summarize_trades 存兩位小數，文章再取一位時若剛好落在 .x5 會變成
    兩次進位、方向可能錯，所以另存精確值給出表端用；不改 summarize_trades 本身（其他系列共用）。
    與 kd_multi_driver.exact_stats 同定義（MACD／均線交叉 driver 用這一份）。
    """
    t = trades[pd.to_numeric(trades["real_pnl"], errors="coerce").fillna(0.0) != 0]
    n = len(t)
    if n == 0:
        return {}
    pnl = t["real_pnl"].astype(float)
    rate = (t["sell_price"] - t["buy_price"]) / t["buy_price"] * 100
    days = (pd.to_datetime(t["sell_date"]) - pd.to_datetime(t["buy_date"])).dt.days
    win, lose = pnl > 0, pnl < 0
    loss_sum = abs(float(pnl[lose].sum()))
    vals = {"勝率%": win.sum() / n * 100, "平均持有天": float(days.mean()),
            "獲利平均%": float(rate[win].mean()) if win.any() else 0.0,
            "虧損平均%": float(rate[lose].mean()) if lose.any() else 0.0,
            "中位數%": float(rate.median()), "期望值/筆": float(pnl.sum()) / n,
            "獲利因子": float(pnl[win].sum()) / loss_sum if loss_sum > 0 else float("inf"),
            "總獲利(萬)": float(pnl.sum()) / 10_000}
    return {k + EXACT: v for k, v in vals.items()}
