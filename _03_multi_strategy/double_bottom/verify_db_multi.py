"""
對帳：關掉資金限制時，多股雙底逐筆必須等於單股組合掃描的逐筆
================================================================
本金極大、每筆固定 1 萬，任何進場都不會被擋——這時 fast_multi_db 的每一筆（股票、買進日、賣出日、
買價、賣價）應與 combo_sweep 存下的單股逐筆（result/single_double_bottom_combo/<tag>/..._trades.csv）相同。
驗的是「出場在引擎裡依實際持倉判定」這套狀態（① 峰值、② 跌破低點、附加：保本停損／高檔 M 頭）
與單股逐根掃描一致。

執行：python _03_multi_strategy/double_bottom/verify_db_multi.py [--limit N]
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _02_strategy.base.vbt import common  # noqa: E402
from _02_strategy.base.vbt.common import DEFAULT_END, DEFAULT_START  # noqa: E402
from _03_multi_strategy.double_bottom.db_multi_driver import PICKS, load_all  # noqa: E402
from _03_multi_strategy.double_bottom.fast_multi_db import build_panel, run_panel  # noqa: E402
from _04_analysis.double_bottom.double_bottom_montecarlo import trades_path  # noqa: E402

KEY_COLS = ["stock_id", "buy_date", "sell_date", "buy_price", "sell_price"]
UNLIMITED = SimpleNamespace(sizing_mode="fixed", invest_ratio=0.0, min_invest=10_000.0, initial_cash=1e15)


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    out = df[KEY_COLS].copy()
    out["stock_id"] = out["stock_id"].astype(str)
    for c in ("buy_date", "sell_date"):
        out[c] = pd.to_datetime(out[c]).dt.normalize()
    for c in ("buy_price", "sell_price"):
        out[c] = out[c].astype(float).round(4)
    return out.sort_values(KEY_COLS).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    t0 = time.time()
    data = load_all(a.limit)
    print(f"載入 {len(data)} 檔｜{time.time() - t0:.0f} 秒", flush=True)
    bad = 0
    for label, pct, entry, exit_key, _s in PICKS:
        panel = build_panel(data, entry, exit_key, pct, DEFAULT_START, DEFAULT_END)
        multi = _norm(run_panel(UNLIMITED, panel, np.zeros(panel["price"].shape), want_equity=False)["trades"])
        single = common.read_result_csv(trades_path(entry, exit_key, pct))
        single = _norm(single[single["stock_id"].astype(str).isin(set(data))])
        diff = multi.merge(single, how="outer", indicator=True)
        n_bad = int((diff["_merge"] != "both").sum())
        bad += n_bad
        print(f"{'OK' if n_bad == 0 else 'NG'}｜{label}｜多股 {len(multi):,}／單股 {len(single):,}｜"
              f"不一致 {n_bad}｜{time.time() - t0:.0f} 秒", flush=True)
        if n_bad:
            print(diff[diff["_merge"] != "both"].head(10).to_string(index=False))
    print("錯誤 0，逐筆一致" if bad == 0 else f"不一致合計 {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
