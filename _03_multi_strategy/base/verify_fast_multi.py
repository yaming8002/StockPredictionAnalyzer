"""
精簡多股引擎的兩道檢查
========================
一、vbt 對帳（fees="vbt"）：同一份面板、同一組優先序，分別丟給 `VbtMultiStrategy.run_panel`（vbt）
    與 `fast_multi.run_panel_fast(fees="vbt")`，比對逐筆交易（股票、買賣日、價格、股數、淨損益）、
    擋單數、最終權益、最大回撤。兩種倉位模式 × 兩種資金鬆緊都跑。
    優先序用「沒有平手」的亂數：vbt 版平手時順序不保證，有平手就無從逐筆比對
    （精簡版平手按股票欄序，屬規格意圖，見 fast_multi 檔頭）。

二、台股費稅帳恆等式（fees="tw"，正式出數字用的口徑）：同樣四組設定，檢查
    （a）最終權益 ＝ 本金 ＋ Σ已平倉 real_pnl ＋ Σ未平倉(股數×最後收盤 − 股數×買價 − 買方手續費)，差 < 0.01 元；
    （b）已實現權益最低 ＝ 由逐筆表逐日重建的「本金 ＋ 已平倉 real_pnl − 持倉中部位的買方手續費」最低點，差 < 0.01 元；
    （c）每筆買進「金額＋手續費」不超過當筆目標投入（fixed 才檢查，比例的目標隨權益變動）。
    並列出同一組設定在舊口徑（vbt）與新口徑（tw）的最終權益、本金＋總獲利，看改版前後差多少。

執行：python _03_multi_strategy/base/verify_fast_multi.py [--limit 300]
"""
import argparse
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _02_strategy.base.vbt.common import DEFAULT_END, DEFAULT_START  # noqa: E402
from _03_multi_strategy.base.fast_multi import EXACT, run_panel_fast  # noqa: E402
from _03_multi_strategy.macd.macd_multi_driver import load_all  # noqa: E402
from _03_multi_strategy.macd.multi_macd import MultiMACD  # noqa: E402

KEYS = ["stock_id", "buy_date", "sell_date", "buy_price", "sell_price", "qty", "real_pnl"]
CASES = [  # (說明, sizing_mode, invest_ratio, min_invest)
    ("定額 1 萬（資金寬鬆）", "fixed", 1.0, 10_000.0),
    ("定額 5 萬（常擋單）", "fixed", 1.0, 50_000.0),
    ("比例 1/30 下限 1 萬", "percent_floor", 1 / 30, 10_000.0),
    ("比例 1/10 下限 1 萬", "percent_floor", 1 / 10, 10_000.0),
]
CENT = 0.01


def check_vbt(panel: dict, prio: np.ndarray) -> int:
    """一、fees="vbt" 對 vbt 版逐筆對帳。回傳不一致組數。"""
    bad = 0
    for label, mode, ratio, floor in CASES:
        m = MultiMACD(sizing_mode=mode, invest_ratio=ratio, min_invest=floor)
        v = m.run_panel(panel, prio=prio)
        f = run_panel_fast(m, panel, prio=prio, fees="vbt")
        tv = v["trades"].sort_values(["sell_date", "stock_id"]).reset_index(drop=True)[KEYS]
        tf = f["trades"][KEYS]
        same = (len(tv) == len(tf)
                and tv.astype(str).equals(tf.astype(str)))
        eq_diff = abs(v["summary"]["最終權益"] - f["summary"]["最終權益"])
        dd_diff = abs(v["summary"]["最大回撤(%)"] - f["summary"]["最大回撤(%)"])
        ok = same and v["blocked_orders"] == f["blocked_orders"] and eq_diff < 1.0 and dd_diff < 0.01
        bad += not ok
        print(f"[{'OK' if ok else 'NG'}] {label}｜交易 vbt {len(tv):,}／精簡 {len(tf):,}｜"
              f"擋單 {v['blocked_orders']:,}／{f['blocked_orders']:,}｜最終權益差 {eq_diff:.2f}｜"
              f"回撤差 {dd_diff:.4f}｜已實現權益最低 {f['summary']['已實現權益最低(%)']}%", flush=True)
        if not same:
            diff = tv.merge(tf, how="outer", indicator=True)
            print(diff[diff["_merge"] != "both"].head(10).to_string())
    return bad


def ledger_min_base(res: dict, cal: pd.DatetimeIndex, init_cash: float) -> float:
    """
    由逐筆表＋未平倉表逐日重建「段首已實現權益」的最低點（引擎在每天賣買之前取快照）：
    第 i 天段首 ＝ 本金 ＋ Σ(賣出日 < i 的 real_pnl) − Σ(買進日 < i 且賣出日 ≥ i 或未平倉 的買方手續費)。
    """
    t, o = res["trades"], res["open"]
    pos_of = cal.get_indexer
    T = len(cal)
    delta = np.zeros(T + 1)
    sell_i = pos_of(pd.DatetimeIndex(t["sell_date"]))
    buy_i = pos_of(pd.DatetimeIndex(t["buy_date"]))
    # 已平倉：賣出隔天起計入 real_pnl；持有期間（買進隔天～賣出當天）扣著買方手續費
    np.add.at(delta, sell_i + 1, t["real_pnl"].to_numpy(float))
    np.add.at(delta, buy_i + 1, -t["buy_fee"].to_numpy(float))
    np.add.at(delta, sell_i + 1, t["buy_fee"].to_numpy(float))
    np.add.at(delta, pos_of(pd.DatetimeIndex(o["buy_date"])) + 1, -o["buy_fee"].to_numpy(float))
    base = init_cash + np.cumsum(delta[:T])
    return min(init_cash, float(base.min()))


def check_tw(panel: dict, prio: np.ndarray) -> int:
    """二、fees="tw" 的帳恆等式＋舊／新口徑對照。回傳不通過組數。"""
    close = panel["close"]
    last_close = close.ffill().iloc[-1]
    bad = 0
    for label, mode, ratio, floor in CASES:
        m = MultiMACD(sizing_mode=mode, invest_ratio=ratio, min_invest=floor)
        old = run_panel_fast(m, panel, prio=prio, fees="vbt")
        new = run_panel_fast(m, panel, prio=prio, fees="tw")
        s, t, o = new["summary"], new["trades"], new["open"]
        pnl = float(t["real_pnl"].sum())
        mtm = float((o["qty"] * (last_close.reindex(o["stock_id"]).to_numpy() - o["buy_price"])
                     - o["buy_fee"]).sum())
        want_eq = m.initial_cash + pnl + mtm
        d_eq = abs(s["最終權益" + EXACT] - want_eq)
        want_min = ledger_min_base(new, close.index, m.initial_cash)
        d_min = abs(s["已實現權益最低(%)" + EXACT] / 100 * m.initial_cash - want_min)
        over = 0
        if mode == "fixed":
            over = int(((t["buy_price"] * t["qty"] + t["buy_fee"]) > floor + 1e-9).sum()
                       + ((o["buy_price"] * o["qty"] + o["buy_fee"]) > floor + 1e-9).sum())
        ok = d_eq < CENT and d_min < CENT and over == 0
        bad += not ok
        print(f"[{'OK' if ok else 'NG'}] {label}｜交易 {len(t):,}｜未平倉 {len(o)}｜"
              f"最終權益 {s['最終權益' + EXACT]:,.2f} vs 帳 {want_eq:,.2f}（差 {d_eq:.4f}）｜"
              f"已實現最低 差 {d_min:.4f}｜超出目標 {over}", flush=True)
        so = old["summary"]
        print(f"      舊口徑 vbt：最終權益 {so['最終權益']:,.0f}｜本金＋總獲利 "
              f"{m.initial_cash + so['總獲利']:,.0f}｜已實現最低 {so['已實現權益最低(%)']}%｜"
              f"回撤 {so['最大回撤(%)']}%｜交易 {so['交易次數']:,}｜擋單 {old['blocked_orders']:,}")
        print(f"      新口徑 tw ：最終權益 {s['最終權益']:,.0f}｜本金＋總獲利 "
              f"{m.initial_cash + s['總獲利']:,.0f}（＋未平倉 {mtm:,.0f}）｜已實現最低 "
              f"{s['已實現權益最低(%)']}%｜回撤 {s['最大回撤(%)']}%｜交易 {s['交易次數']:,}｜"
              f"擋單 {new['blocked_orders']:,}", flush=True)
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    a = ap.parse_args()

    data = load_all(a.limit)
    builder = MultiMACD()
    builder.BASE, builder.ENTRY, builder.PRIO = "cross", "none", "low_price"
    panel = builder.build_panel(data, DEFAULT_START, DEFAULT_END)
    rng = np.random.default_rng(7)
    prio = rng.random(panel["price"].shape)          # 無平手
    print(f"{len(data)} 檔｜{panel['price'].shape[0]} 天", flush=True)

    print("\n== 一、vbt 對帳（fees='vbt'）==")
    bad_v = check_vbt(panel, prio)
    print("\n== 二、台股費稅帳恆等式（fees='tw'）==")
    bad_t = check_tw(panel, prio)
    print(f"\nvbt 對帳不一致 {bad_v} 組｜tw 恆等式不通過 {bad_t} 組")
    return 1 if bad_v or bad_t else 0


if __name__ == "__main__":
    sys.exit(main())
