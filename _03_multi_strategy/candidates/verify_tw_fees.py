"""
對帳：vbt 多股「台股實際費稅」模式（multi_double_bottom.simulate_state_exit, fees="tw"）
==================================================================================
三道檢查（全市場、比例投入、低價／高價／流動性三種排序）：
  1. 海龜系統 2：vbt tw  vs  精簡引擎 fast_multi.run_panel_fast(fees="tw")——逐筆（股票、買賣日、買賣價、股數）
     與逐日權益都要一致（兩邊是獨立實作的同一套台股費稅規則）。
  2. 海龜系統 2：simulate_state_exit(fees="rate")  vs  基底 VbtMultiStrategy.run_panel——逐筆一致，
     確認「狀態型出場走同一個下單函式」的轉接沒有改變行為。
  3. 雙底（7%、8% 有停損；9% 只用 CHoCH）：精簡引擎沒有路徑相依出場、無法直接比，改驗恆等式
     最終權益 ＝ 本金 ＋ Σ已平倉 real_pnl ＋ Σ未平倉(股數×最後收盤 − 股數×買價 − 買方手續費)。

執行：python _03_multi_strategy/candidates/verify_tw_fees.py [--limit N]
"""
import argparse
import os
import sys
import time

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _02_strategy.base.vbt import common  # noqa: E402
from _02_strategy.base.vbt.common import DEFAULT_END, DEFAULT_START, GLITCH  # noqa: E402
from _03_multi_strategy.base.fast_multi import run_panel_fast  # noqa: E402
from _03_multi_strategy.candidates.cases import CASES  # noqa: E402
from _03_multi_strategy.double_bottom.multi_double_bottom import simulate_state_exit  # noqa: E402
from _03_multi_strategy.macd.macd_multi_driver import prio_panels  # noqa: E402

KEYS = ["stock_id", "buy_date", "sell_date", "buy_price", "sell_price", "qty"]


def norm(df):
    out = df[KEYS].copy()
    out["stock_id"] = out["stock_id"].astype(str)
    for c in ("buy_date", "sell_date"):
        out[c] = pd.to_datetime(out[c]).dt.normalize()
    for c in ("buy_price", "sell_price"):
        out[c] = out[c].astype(float).round(4)
    out["qty"] = out["qty"].astype(int)
    return out.sort_values(KEYS).reset_index(drop=True)


def same_trades(a, b) -> int:
    m = norm(a).merge(norm(b), how="outer", indicator=True)
    return int((m["_merge"] != "both").sum())


def state_turtle(inst, panel, prio, fees):
    """海龜走共用下單函式：ths 全 NaN（不用峰值）、lowbrk＝出場面板。"""
    return simulate_state_exit(inst, panel, np.full(panel["price"].shape, np.nan),
                               panel["exits"], True, prio, True, fees)


def identity_gap(inst, res, panel) -> float:
    """最終權益 − (本金＋Σreal_pnl＋未平倉市值損益)；tw 模式應為 0（浮點誤差內）。"""
    pf = res["pf"]
    last_close = panel["close"].ffill().iloc[-1]
    op = pf.positions.records_readable
    op = op[op["Status"] == "Open"]
    open_pnl = 0.0
    for _, r in op.iterrows():
        q = float(r["Size"])
        p_in = float(r["Avg Entry Price"])
        fee = float(common.reconstruct_fees(pd.Series([p_in]), pd.Series([p_in]), pd.Series([q]))[0][0])
        open_pnl += q * float(last_close[r["Column"]]) - q * p_in - fee
    expect = inst.initial_cash + float(res["trades"]["real_pnl"].sum()) + open_pnl
    return float(res["equity"].iloc[-1]) - expect


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    t0 = time.time()
    data = common.load_market(common.DATA_DIR, columns=["open", "high", "low", "close", "volume"],
                              limit=a.limit, exclude=GLITCH, min_rows=2)
    data = {s: d for s, d in data.items() if len(d.loc[DEFAULT_START:DEFAULT_END]) >= 2}
    print(f"載入 {len(data)} 檔｜{time.time() - t0:.0f} 秒", flush=True)
    by_key = {c["key"]: c for c in CASES}
    kw = {"initial_cash": 1_000_000.0, "sizing_mode": "percent_floor",
          "invest_ratio": 1.0 / 126, "min_invest": 10_000.0}
    bad = 0

    case = by_key["turtle_s2"]
    panel = case["make"]().build_panel(data, DEFAULT_START, DEFAULT_END)
    prios = prio_panels(data, panel["close"])
    inst = case["make"](**kw)
    for kind, pr in prios.items():
        v = state_turtle(inst, panel, pr, "tw")
        f = run_panel_fast(inst, panel, pr, want_equity=True, fees="tw")
        d_tr = same_trades(v["trades"], f["trades"])
        d_eq = float(np.nanmax(np.abs(v["equity"].to_numpy() - f["equity"].to_numpy())))
        ok1 = d_tr == 0 and d_eq < 0.01 and v["blocked_orders"] == f["blocked_orders"]
        r = state_turtle(inst, panel, pr, "rate")
        b = inst.run_panel(panel, pr, want_equity=False)
        d_rb = same_trades(r["trades"], b["trades"])
        ok2 = d_rb == 0 and r["blocked_orders"] == b["blocked_orders"]
        bad += (not ok1) + (not ok2)
        print(f"{'OK' if ok1 else 'NG'}｜海龜系統 2｜{kind}｜vbt tw vs 精簡引擎 tw：{len(v['trades']):,} 筆，"
              f"逐筆不符 {d_tr}｜權益最大差 {d_eq:.4f} 元｜擋單 {v['blocked_orders']:,}/{f['blocked_orders']:,}",
              flush=True)
        print(f"{'OK' if ok2 else 'NG'}｜海龜系統 2｜{kind}｜共用下單函式 rate vs 基底 run_panel：逐筆不符 {d_rb}",
              flush=True)
    del panel, prios

    for key, n_pct in (("db_p7_stop", 139), ("db_p8_stop", 139), ("db_p9_nostop", 104)):
        case = by_key[key]
        panel = case["make"]().build_panel(data, DEFAULT_START, DEFAULT_END)
        prios = prio_panels(data, panel["close"])
        inst = case["make"](**(kw | {"invest_ratio": 1.0 / n_pct}))
        for kind, pr in prios.items():
            res = inst.run_panel(panel, pr, want_equity=True, fees="tw")
            gap = identity_gap(inst, res, panel)
            ok = abs(gap) < 1.0
            bad += not ok
            print(f"{'OK' if ok else 'NG'}｜{case['label']}｜{kind}｜恆等式差 {gap:.4f} 元｜"
                  f"{len(res['trades']):,} 筆｜擋單 {res['blocked_orders']:,}", flush=True)
        del panel, prios
    print("\n" + ("全部一致" if bad == 0 else f"{bad} 項不一致") + f"｜{time.time() - t0:.0f} 秒")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
