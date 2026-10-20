"""
對帳：關掉資金限制時，多股逐筆必須等於單股逐筆（海龜 vs 雙底 20 組）
====================================================================
本金設到極大、每筆固定 1 萬（fixed），任何進場都不會被擋——這時多股引擎的每一筆
（股票、買進日、賣出日、買價、賣價）應與單股回測輸出逐筆相同。
雙底的出場在多股是「依實際持倉逐日維護峰值」（見 multi_double_bottom.py），這裡就是在
驗證這套狀態跟單股逐根掃描一致；海龜則驗證通道出場與區間閘門。

第二道（--capped，只對雙底）：本金 100 萬、每筆 3.3 萬，會大量擋單；逐筆檢查「從這筆實際成交的進場
起算」單股掃描規則推出的出場日，是否等於多股的賣出日——驗證被擋的進場沒有污染峰值狀態。

執行：python _03_multi_strategy/candidates/verify_multi_candidates.py [--limit 300] [--keys turtle_s1,db_p5_stop] [--capped]
"""
import argparse
import os
import sys
import time

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd

from _02_strategy.base.vbt import common
from _02_strategy.base.vbt.common import DEFAULT_END, DEFAULT_START, GLITCH
from _03_multi_strategy.candidates.cases import CASES

KEY_COLS = ["stock_id", "buy_date", "sell_date", "buy_price", "sell_price"]


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    out = df[KEY_COLS].copy()
    out["stock_id"] = out["stock_id"].astype(str)
    out["buy_date"] = pd.to_datetime(out["buy_date"]).dt.normalize()
    out["sell_date"] = pd.to_datetime(out["sell_date"]).dt.normalize()
    out["buy_price"] = out["buy_price"].astype(float).round(4)
    out["sell_price"] = out["sell_price"].astype(float).round(4)
    return out.sort_values(KEY_COLS).reset_index(drop=True)


def expected_sell(df: pd.DataFrame, buy_date, stop_low: bool):
    """從實際成交的買進日起，照單股掃描規則推出應有的賣出日（成交日）；期末未出場回 None。"""
    idx = df.index
    e = idx.get_loc(pd.Timestamp(buy_date))
    close = df["close"].to_numpy()
    th = df["zigzag_turn_high"].to_numpy()
    last_low = df["zigzag_turn_low"].ffill().to_numpy()
    peak = -np.inf
    for i in range(e, len(df) - 1):                      # 判定日 i ≥ 進場判定日+1（＝買進日）
        hit = stop_low and (not np.isnan(last_low[i])) and close[i] < last_low[i]
        if not np.isnan(th[i]):
            if th[i] < peak:
                hit = True
            peak = max(peak, th[i])
        if hit:
            return idx[i + 1]
    return None


def capped_check(data, case) -> int:
    """資金受限時，雙底每筆賣出日都要等於從實際進場起算的單股規則結果。回傳不一致筆數。"""
    inst = case["make"](initial_cash=1_000_000.0, sizing_mode="fixed", min_invest=33_000.0)
    res = inst.run(data, DEFAULT_START, DEFAULT_END)
    single = inst._single()
    cache, bad = {}, 0
    for r in res["trades"].itertuples():
        if r.stock_id not in cache:
            cache[r.stock_id] = single.add_columns(data[r.stock_id].copy())
        exp = expected_sell(cache[r.stock_id], r.buy_date, inst.STOP_LOW)
        if exp is None or pd.Timestamp(exp) != pd.Timestamp(r.sell_date):
            bad += 1
    print(f"{'OK' if bad == 0 else 'NG'}｜受限資金｜{case['label']}｜{len(res['trades']):,} 筆｜"
          f"擋單 {res['blocked_orders']:,}｜出場不符 {bad}", flush=True)
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--keys", default=None, help="只對這幾組（逗號分隔 key）")
    ap.add_argument("--capped", action="store_true", help="改跑受限資金的出場一致性檢查（只對雙底）")
    a = ap.parse_args()
    keys = set(a.keys.split(",")) if a.keys else None

    t0 = time.time()
    data = common.load_market(common.DATA_DIR, columns=["open", "high", "low", "close", "volume"],
                              limit=a.limit, exclude=GLITCH, min_rows=2)
    data = {s: d for s, d in data.items() if len(d.loc[DEFAULT_START:DEFAULT_END]) >= 2}
    print(f"載入 {len(data)} 檔｜{time.time() - t0:.0f} 秒", flush=True)

    bad = 0
    if a.capped:
        for case in CASES:
            if case["key"].startswith("db_") and (not keys or case["key"] in keys):
                bad += capped_check(data, case) > 0
        print("\n" + ("全部一致" if bad == 0 else f"{bad} 組不一致"))
        return 1 if bad else 0
    for case in CASES:
        if keys and case["key"] not in keys:
            continue
        inst = case["make"](initial_cash=1e12, sizing_mode="fixed", min_invest=10_000.0)
        res = inst.run(data, DEFAULT_START, DEFAULT_END)
        multi = _norm(res["trades"])
        single = pd.read_csv(case["trades"], encoding="utf-8-sig", dtype={"stock_id": str})
        single = _norm(single[single["stock_id"].isin(set(data))])
        merged = multi.merge(single, how="outer", indicator=True)
        only_m = int((merged["_merge"] == "left_only").sum())
        only_s = int((merged["_merge"] == "right_only").sum())
        ok = only_m == 0 and only_s == 0 and res["blocked_orders"] == 0
        bad += 0 if ok else 1
        print(f"{'OK' if ok else 'NG'}｜{case['label']}｜多股 {len(multi):,}／單股 {len(single):,} 筆｜"
              f"只在多股 {only_m}｜只在單股 {only_s}｜擋單 {res['blocked_orders']}｜"
              f"{time.time() - t0:.0f} 秒", flush=True)
        if not ok:
            print(merged[merged["_merge"] != "both"].head(10).to_string(index=False))
    print(f"\n{'全部一致' if bad == 0 else f'{bad} 組不一致'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
