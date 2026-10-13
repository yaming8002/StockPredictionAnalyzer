"""
KD 多股：精簡引擎 vs vbt 多股引擎 逐筆對帳
==========================================
`kd_multi_driver.py` 全部走精簡引擎（fast_multi.run_panel_fast）。精簡引擎已用 MACD 面板
對過帳（base/verify_fast_multi.py），這裡換成 KD 的面板再對一次：同一份 MultiKD 面板、
同一組「沒有平手」的亂數優先序，分別丟給 `run_panel`（vbt）與 `run_panel_fast(fees="vbt")`（舊純費率口徑；正式出數字用的
"tw" 台股費稅口徑本來就不會跟 vbt 一致，恆等式檢查見 base/verify_fast_multi.py），比對逐筆交易
（股票、買賣日、價格、股數、淨損益）、擋單數、最終權益、最大回撤。
兩投法都用 driver 的實際份數（S 給定）跑。

優先序必須無平手：vbt 版平手時順序不保證，有平手就無從逐筆比對（精簡版平手按欄序）。

執行：python _03_multi_strategy/kd/verify_kd_fast.py [--entry all] [--s N] [--limit N]
    預設 --entry all＝文章多股表的 6 個進場全跑，各用單股 MC 表的 S（kd_montecarlo.csv）；
    指定單一進場時可用 --s 覆寫 S。預設全市場（2,258 檔，排 GLITCH）；--limit N 只取前 N 檔快檢，0 也視為全市場。
"""
import argparse
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np  # noqa: E402

from _02_strategy.base.vbt import common  # noqa: E402
from _02_strategy.base.vbt.common import DEFAULT_END, DEFAULT_START  # noqa: E402
from _03_multi_strategy.base.fast_multi import run_panel_fast  # noqa: E402
from _03_multi_strategy.kd.kd_multi_driver import (INIT_CASH, MC_CSV, load_all,  # noqa: E402
                                                    read_s_table, sizings)
from _03_multi_strategy.kd.multi_kd import ANCHORS, ENTRIES, MultiKD  # noqa: E402

KEYS = ["stock_id", "buy_date", "sell_date", "buy_price", "sell_price", "qty", "real_pnl"]


def check_entry(data: dict, entry: str, s: int) -> int:
    """單一進場 × 兩投法逐筆對帳；回傳不一致組數。"""
    builder = MultiKD()
    builder.ENTRY = entry
    panel = builder.build_panel(data, DEFAULT_START, DEFAULT_END)
    prio = np.random.default_rng(7).random(panel["price"].shape)     # 無平手
    print(f"{entry}（S={s}）｜{len(data)} 檔｜{panel['price'].shape[0]} 天｜"
          f"買訊 {int(panel['entries'].sum()):,}", flush=True)

    bad = 0
    for mname, mode, ratio, floor, n_units in sizings(s):
        m = MultiKD(initial_cash=INIT_CASH, sizing_mode=mode, invest_ratio=ratio, min_invest=floor)
        m.ENTRY = entry
        v = m.run_panel(panel, prio=prio)
        f = run_panel_fast(m, panel, prio=prio, fees="vbt")   # 對帳用舊純費率口徑
        tv = v["trades"].sort_values(["sell_date", "stock_id"]).reset_index(drop=True)[KEYS]
        tf = f["trades"][KEYS]
        same = len(tv) == len(tf) and tv.astype(str).equals(tf.astype(str))
        eq_diff = abs(v["summary"]["最終權益"] - f["summary"]["最終權益"])
        dd_diff = abs(v["summary"]["最大回撤(%)"] - f["summary"]["最大回撤(%)"])
        ok = same and v["blocked_orders"] == f["blocked_orders"] and eq_diff < 1.0 and dd_diff < 0.01
        bad += not ok
        print(f"[{'OK' if ok else 'NG'}] {mname}({n_units})｜交易 vbt {len(tv):,}／精簡 {len(tf):,}｜"
              f"擋單 {v['blocked_orders']:,}／{f['blocked_orders']:,}｜最終權益差 {eq_diff:.2f}｜"
              f"回撤差 {dd_diff:.4f}", flush=True)
        if not same:
            diff = tv.merge(tf, how="outer", indicator=True)
            print(diff[diff["_merge"] != "both"].head(10).to_string())
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", default="all", choices=["all"] + ENTRIES + ANCHORS)
    ap.add_argument("--s", type=int, default=None, help="最大連敗 S（決定兩投法份數）；省略＝讀 MC 表")
    ap.add_argument("--limit", type=int, default=None, help="只取前 N 檔快檢；省略或 0＝全市場")
    ap.add_argument("--folder", default=common.DATA_DIR)
    a = ap.parse_args()

    entries = ENTRIES if a.entry == "all" else [a.entry]
    s_table = read_s_table(MC_CSV) if a.s is None else {}
    data = load_all(a.folder, a.limit or None)

    bad = 0
    for e in entries:
        s = a.s if a.s is not None else s_table.get(e)
        if s is None:
            raise SystemExit(f"MC 表沒有 {e} 的 S，請用 --s 指定")
        bad += check_entry(data, e, s)
    print(f"不一致 {bad} 組")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
