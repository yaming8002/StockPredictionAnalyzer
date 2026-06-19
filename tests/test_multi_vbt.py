"""
VbtMultiStrategy 對驗測試
==========================
用一支「獨立、透明的純 Python 參考引擎」重算多股共用現金回測，逐筆比對 vbt 版
（`_03_multi_strategy/base/vbt/multi.py`）。兩者吻合 → vbt 的下單/擋單/下限/已實現權益基準/
優先序語意正確、數字可信。涵蓋：percent_floor 已實現權益基準、fixed、真擋單、下限、整股、
sell-先於-buy、優先序，以及隨機多股情境。

執行：F:/stock-analyzer/.venv/Scripts/python.exe -m pytest tests/test_multi_vbt.py -q
"""
import math
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from _02_strategy.base.vbt import common
from _03_multi_strategy.base.vbt.multi import VbtMultiStrategy

BUY_FEE = common.COMMISSION
SELL_FEE = common.COMMISSION + common.DUES


# ── 測試用策略：直接吃預先算好的訊號欄 ─────────────────────────
class _SignalStrategy(VbtMultiStrategy):
    def buy_signal(self, df):
        return df["_buy"]

    def sell_signal(self, df):
        return df["_sell"]

    def priority(self, df, stock_id):
        return df["_prio"] if "_prio" in df.columns else None


# ── 獨立參考引擎（純 Python；刻意用迴圈、與 vbt 實作無關）───────────
def reference_run(data_dict, sizing_mode, invest_ratio, min_invest, initial_cash,
                  use_priority=False):
    """回傳 (closed_trades_set, blocked_count)。語意比照 multi.py 設計。"""
    stock_ids = sorted(data_dict)
    index = sorted(set().union(*[df.index for df in data_dict.values()]))

    def col(sid, name):
        s = data_dict[sid][name]
        return {t: s.loc[t] for t in s.index}

    buy = {s: col(s, "_buy") for s in stock_ids}
    sell = {s: col(s, "_sell") for s in stock_ids}
    px = {s: col(s, "close") for s in stock_ids}
    prio = {s: (col(s, "_prio") if "_prio" in data_dict[s].columns else {}) for s in stock_ids}

    cash = float(initial_cash)
    holding = {s: None for s in stock_ids}       # None 或 dict(qty, entry_price, buy_date)
    closed = []
    blocked = 0

    for t in index:
        # base 快照（段首：含本日將賣出的部位、以成本計）
        base = cash + sum(h["qty"] * h["entry_price"] for h in holding.values() if h)
        # 段首持倉狀態快照：一天一檔至多一個動作，買入資格看「段首是否空手」
        flat_at_start = {s: holding[s] is None for s in stock_ids}

        # 1) 賣（釋放現金，賣方費含證交稅）
        for sid in stock_ids:
            h = holding[sid]
            if h is not None and bool(sell[sid].get(t, False)):
                p = px[sid].get(t, np.nan)
                if p > 0:
                    cash += h["qty"] * p * (1.0 - SELL_FEE)
                    closed.append((sid, h["buy_date"], t, int(h["qty"])))
                    holding[sid] = None

        # 2) 買（段首空手 + 買訊號；優先序：分數高→低，平手 stock_id 小→大）
        cands = [s for s in stock_ids if flat_at_start[s]
                 and bool(buy[s].get(t, False)) and px[s].get(t, np.nan) > 0]
        if use_priority:
            cands.sort(key=lambda s: (-(prio[s].get(t, -np.inf)), s))
        else:
            cands.sort()

        for sid in cands:
            p = px[sid][t]
            if cash < min_invest:                       # 現金不足門檻 → 真擋單
                blocked += 1
                continue
            if sizing_mode == "percent_floor":
                target = max(base * invest_ratio, min_invest)
            else:
                target = min_invest
            cps = p * (1.0 + BUY_FEE)
            shares = math.floor(target / cps)
            afford = math.floor(cash / cps)
            if shares > afford:
                shares = afford
            if shares < 1:
                blocked += 1
                continue
            cash -= shares * p * (1.0 + BUY_FEE)
            holding[sid] = {"qty": shares, "entry_price": p, "buy_date": t}

    return set(closed), blocked


# ── 輔助：把陣列組成 data_dict ───────────────────────────────
def _mk(prices, buys, sells, prios=None, start="2020-01-01"):
    n = len(prices)
    idx = pd.date_range(start, periods=n, freq="D")
    d = {"open": prices, "high": prices, "low": prices, "close": prices,
         "volume": [1e6] * n,
         "_buy": np.array(buys, dtype=bool), "_sell": np.array(sells, dtype=bool)}
    if prios is not None:
        d["_prio"] = np.array(prios, dtype=float)
    return pd.DataFrame(d, index=idx)


def _engine_trades(strat, data_dict):
    res = strat.run(data_dict)
    tr = res["trades"]
    s = set((r.stock_id, r.buy_date, r.sell_date, int(r.qty)) for r in tr.itertuples())
    return s, res["blocked_orders"]


def _assert_match(data_dict, sizing_mode, invest_ratio, min_invest, initial_cash,
                  use_priority=False):
    strat = _SignalStrategy(sizing_mode=sizing_mode, invest_ratio=invest_ratio,
                            min_invest=min_invest, initial_cash=initial_cash)
    eng_tr, eng_blk = _engine_trades(strat, data_dict)
    ref_tr, ref_blk = reference_run(data_dict, sizing_mode, invest_ratio, min_invest,
                                    initial_cash, use_priority=use_priority)
    assert eng_tr == ref_tr, (
        f"trades 不一致\n只在引擎: {sorted(eng_tr - ref_tr)}\n只在參考: {sorted(ref_tr - eng_tr)}")
    assert eng_blk == ref_blk, f"擋單數不一致 引擎={eng_blk} 參考={ref_blk}"
    return eng_tr, eng_blk


# ── 測試案例 ─────────────────────────────────────────────────
def test_percent_floor_realized_equity_base():
    """percent_floor：分日買、各以~初始已實現權益為基準（非遞減現金）。"""
    P = [100.0] * 6
    dd = {
        "A": _mk(P, [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]),
        "B": _mk(P, [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]),
        "C": _mk(P, [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]),
    }
    tr, blk = _assert_match(dd, "percent_floor", 0.1, 10_000.0, 1_000_000.0)
    qtys = sorted(int(x[3]) for x in tr)
    assert qtys == [998, 998, 998], qtys   # 各以~100萬基準
    assert blk == 0


def test_fixed_mode_hard_block():
    """fixed：3 檔同買、現金只夠 2 筆 → 第 3 真擋單。"""
    P = [100.0] * 6
    dd = {s: _mk(P, [1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]) for s in "ABC"}
    tr, blk = _assert_match(dd, "fixed", 0.0, 40_000.0, 100_000.0)
    assert len(tr) == 2 and blk == 1


def test_floor_binds_and_blocks():
    """percent_floor：小現金 → 比例低於下限改用下限；再低於下限 → 擋單。"""
    P = [100.0] * 6
    dd = {"A": _mk(P, [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]),
          "B": _mk(P, [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0])}
    # init 8000, ratio0.1 → 10%*8000=800<5000下限→用5000; 買後 cash~3000<5000 → B 擋
    _assert_match(dd, "percent_floor", 0.1, 5_000.0, 8_000.0)


def test_priority_order_under_constraint():
    """優先序：只夠 1 筆、C 優先級高 → 買 C 擋 A。"""
    P = [100.0] * 4
    dd = {"A": _mk(P, [1, 0, 0, 0], [0, 0, 1, 0], prios=[1, 1, 1, 1]),
          "C": _mk(P, [1, 0, 0, 0], [0, 0, 1, 0], prios=[9, 9, 9, 9])}
    tr, blk = _assert_match(dd, "fixed", 0.0, 60_000.0, 100_000.0, use_priority=True)
    assert {x[0] for x in tr} == {"C"} and blk == 1


def test_sell_frees_cash_same_day():
    """sell 先於 buy：同日 A 賣釋放現金，B 才買得到。"""
    P = [100.0] * 5
    dd = {"A": _mk(P, [1, 0, 0, 0, 0], [0, 0, 1, 0, 0]),   # 買D1 賣D3
          "B": _mk(P, [0, 0, 1, 0, 0], [0, 0, 0, 0, 1])}   # 買D3(同A賣日)
    tr, blk = _assert_match(dd, "fixed", 0.0, 90_000.0, 100_000.0)
    assert ("B", dd["B"].index[2], dd["B"].index[4], 899) in tr or any(x[0] == "B" for x in tr)


def test_random_multi_stock():
    """隨機多股情境（固定種子）：參考引擎 vs vbt 引擎逐筆全等。"""
    rng = np.random.default_rng(42)
    T, N = 60, 8
    dd = {}
    for j in range(N):
        sid = f"S{j:02d}"
        prices = (50 + np.cumsum(rng.normal(0, 1, T))).clip(5, None).round(2).tolist()
        buys = (rng.random(T) < 0.12)
        sells = (rng.random(T) < 0.12)
        prios = rng.random(T).tolist()
        dd[sid] = _mk(prices, buys, sells, prios=prios)
    # 兩模式 + 有無優先序都驗
    _assert_match(dd, "percent_floor", 0.1, 10_000.0, 500_000.0, use_priority=True)
    _assert_match(dd, "fixed", 0.0, 20_000.0, 300_000.0, use_priority=True)
    _assert_match(dd, "percent_floor", 1 / 30, 10_000.0, 1_000_000.0, use_priority=False)


def test_whole_shares_only():
    """整股：成交股數一律為整數。"""
    P = [137.0] * 5   # 非整除價
    dd = {"A": _mk(P, [1, 0, 0, 0, 0], [0, 0, 1, 0, 0])}
    strat = _SignalStrategy(sizing_mode="fixed", min_invest=50_000.0, initial_cash=200_000.0)
    res = strat.run(dd)
    for r in res["trades"].itertuples():
        assert int(r.qty) == r.qty
