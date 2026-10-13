"""
均線交叉多股：精簡引擎 vs vbt 多股引擎 逐筆對帳 ＋ 正式結果重現
================================================================
`ma_cross_multi_driver.py` 全部走精簡引擎（fast_multi.run_panel_fast）。精簡引擎已用 MACD 面板
對過帳（base/verify_fast_multi.py），這裡換成均線交叉自己的面板（`MultiMACross`）逐組再對一次。
三個任務 × 21 組 × driver 實際用的每一種投法全部都跑，兩道檢查：

一、vbt 對帳（fees="vbt"）：同一份面板、同一組「沒有平手」的優先序，分別丟給 `run_panel`（vbt）與
    `run_panel_fast(fees="vbt")`，比對逐筆交易（股票、買賣日、價格、股數、淨損益）、擋單數、最終權益、
    最大回撤。正式出數字用的 "tw" 台股費稅口徑 vbt 表達不了（最低手續費 20 元），帳恆等式見
    base/verify_fast_multi.py。
    - angle_adx／no_filter：亂數優先序（vbt 平手時順序不保證，有平手就無從逐筆比對）。
    - stock_id：代號序＝「優先序 N−欄序」（無平手、等價於代號升冪）；另確認精簡引擎用 0（driver 的寫法）
      與用 N−欄序跑出來逐筆相同。
二、正式結果重現（fees="tw"）：用 driver 同一套函式（_prio_panels／_row）重算三種固定排序（stock_id 為代號序）
    的每一列，必須與 result/<task>/fixed_<pair>.csv 逐欄相同（數值容許 1e-9 相對誤差）。
    隨機 1,000 次的列不重算（每組要再跑 1,000 次），隨機列用的是同一支引擎與同一份面板。

執行：python _03_multi_strategy/ma_cross/verify_ma_cross_fast.py [--task all] [--pairs ...] [--workers 3] [--limit N]
    預設全市場（2,258 檔，排 GLITCH）、三任務 × 21 組；--limit N 只取前 N 檔快檢（不做檢查二），0 也視為全市場。
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _02_strategy.base.vbt import common  # noqa: E402
from _02_strategy.base.vbt.common import DEFAULT_END, DEFAULT_START  # noqa: E402
from _02_strategy.ma_strategy.ma_cross_sweep import PAIRS  # noqa: E402
from _03_multi_strategy.base.fast_multi import run_panel_fast  # noqa: E402
from _03_multi_strategy.ma_cross import ma_cross_multi_driver as drv  # noqa: E402
from _03_multi_strategy.ma_cross.multi_ma_cross import MultiMACross  # noqa: E402

KEYS = ["stock_id", "buy_date", "sell_date", "buy_price", "sell_price", "qty", "real_pnl"]
REL_TOL = 1e-9


def _compare_vbt(m, panel, prio, label: str) -> tuple:
    """vbt run_panel vs 精簡引擎 fees="vbt"；回傳 (是否一致, 一行說明, 差異明細)。"""
    v = m.run_panel(panel, prio=prio)
    f = run_panel_fast(m, panel, prio=prio, fees="vbt")
    tv = v["trades"].sort_values(["sell_date", "stock_id"]).reset_index(drop=True)[KEYS]
    tf = f["trades"][KEYS]
    same = len(tv) == len(tf) and tv.astype(str).equals(tf.astype(str))
    eq_diff = abs(v["summary"]["最終權益"] - f["summary"]["最終權益"])
    dd_diff = abs(v["summary"]["最大回撤(%)"] - f["summary"]["最大回撤(%)"])
    ok = same and v["blocked_orders"] == f["blocked_orders"] and eq_diff < 1.0 and dd_diff < 0.01
    line = (f"[{'OK' if ok else 'NG'}] {label}｜交易 vbt {len(tv):,}／精簡 {len(tf):,}｜"
            f"擋單 {v['blocked_orders']:,}／{f['blocked_orders']:,}｜最終權益差 {eq_diff:.2f}｜回撤差 {dd_diff:.4f}")
    detail = ""
    if not same:
        diff = tv.astype(str).merge(tf.astype(str), how="outer", indicator=True)
        detail = diff[diff["_merge"] != "both"].head(10).to_string()
    return ok, line, detail


def _same_row(want: dict, got: dict) -> list:
    """逐欄比對存檔列與重算列；回傳不一致的欄位說明。"""
    bad = []
    for k, w in want.items():
        g = got.get(k, np.nan)
        if isinstance(w, (bool, np.bool_)) or isinstance(g, (bool, np.bool_)):
            if str(w).lower() != str(g).lower():
                bad.append(f"{k}: 存檔 {w} vs 重算 {g}")
            continue
        try:
            wf, gf = float(w), float(g)
        except (TypeError, ValueError):
            if str(w) != str(g):
                bad.append(f"{k}: 存檔 {w} vs 重算 {g}")
            continue
        if np.isnan(wf) and np.isnan(gf):
            continue
        if not np.isclose(wf, gf, rtol=REL_TOL, atol=1e-9):
            bad.append(f"{k}: 存檔 {wf} vs 重算 {gf}")
    return bad


def check_pair(task: str, pair: str, s_table: dict, reproduce: bool) -> tuple:
    """一個任務 × 一組均線的兩道檢查；回傳 (不一致數, 輸出文字)。"""
    t0 = time.time()
    short, long = (int(x) for x in pair.split("/"))
    m = MultiMACross(initial_cash=drv.INIT_CASH)
    m.SHORT_MA, m.LONG_MA = short, long
    for k, v in drv.TASKS[task].items():
        setattr(m, k, v)
    panel = m.build_panel(drv._DATA, DEFAULT_START, DEFAULT_END)
    n_cols = panel["price"].shape[1]
    if task == "stock_id":
        prio = np.broadcast_to(np.arange(n_cols, 0, -1, dtype=np.float64),
                               panel["price"].shape).copy()       # 代號升冪、無平手
    else:
        prio = np.random.default_rng(7).random(panel["price"].shape)   # 無平手
    out = [f"── {task} {pair}｜{n_cols} 檔｜{panel['price'].shape[0]} 天｜買訊 {int(panel['entries'].sum()):,}"]
    bad = 0

    det_rows = []
    prios = ({"代號": np.zeros_like(panel["price"])} if task == "stock_id"
             else drv._prio_panels(panel))
    for sname, mode, ratio, floor, n_units in drv.sizings(task, pair, s_table):
        m.sizing_mode, m.invest_ratio, m.min_invest = mode, ratio, floor
        ok, line, detail = _compare_vbt(m, panel, prio, f"{sname}({n_units})")
        bad += not ok
        out.append("  " + line)
        if detail:
            out.append(detail)
        if task == "stock_id":
            a = run_panel_fast(m, panel, prio=np.zeros_like(panel["price"]))["trades"][KEYS]
            b = run_panel_fast(m, panel, prio=prio)["trades"][KEYS]
            same = a.astype(str).equals(b.astype(str))
            bad += not same
            out.append(f"  [{'OK' if same else 'NG'}] {sname} 精簡引擎 優先序 0 ≡ N−欄序（{len(a):,} 筆）")
        if reproduce:
            for oname, pr in prios.items():
                det_rows.append(drv._row(pair, sname, n_units, floor, oname,
                                         run_panel_fast(m, panel, prio=pr)))

    if reproduce:
        path = os.path.join(drv.OUT_ROOT, task, f"fixed_{pair.replace('/', '_')}.csv")
        stored = pd.read_csv(path).to_dict("records")
        if len(stored) != len(det_rows):
            bad += 1
            out.append(f"  [NG] 正式結果列數：存檔 {len(stored)} vs 重算 {len(det_rows)}")
        else:
            n_bad_rows = 0
            for want, got in zip(stored, det_rows):
                diff = _same_row(want, got)
                if diff:
                    n_bad_rows += 1
                    out.append(f"  [NG] {want['投法']}｜{want['排序']}：" + "；".join(diff[:5]))
            bad += n_bad_rows
            out.append(f"  [{'OK' if not n_bad_rows else 'NG'}] 正式結果重現 {len(det_rows)} 列"
                       f"（{os.path.basename(path)}）")
    out.append(f"  耗時 {time.time() - t0:.0f} 秒")
    return bad, "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", choices=["all"] + list(drv.TASKS))
    ap.add_argument("--pairs", nargs="+", default=PAIRS)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--folder", default=common.DATA_DIR)
    ap.add_argument("--limit", type=int, default=None, help="只取前 N 檔快檢；省略或 0＝全市場")
    a = ap.parse_args()
    limit = a.limit or None
    reproduce = limit is None                # 正式結果是全市場跑的，快檢無從比對

    s_table = {}
    mc = pd.read_csv(drv.MC_CSV)
    s_table = dict(zip(mc["短/長"], mc["最大連敗_P95"].astype(int)))
    tasks = list(drv.TASKS) if a.task == "all" else [a.task]
    jobs = [(t, p) for t in tasks for p in a.pairs]
    print(f"{len(jobs)} 組（{'、'.join(tasks)} × {len(a.pairs)}）｜{DEFAULT_START}~{DEFAULT_END}｜"
          f"{'全市場' if limit is None else f'前 {limit} 檔'}｜{a.workers} 進程", flush=True)

    bad = 0
    with ProcessPoolExecutor(max_workers=a.workers, initializer=drv._init,
                             initargs=(a.folder, limit)) as pool:
        futs = [pool.submit(check_pair, t, p, s_table, reproduce) for t, p in jobs]
        for f in futs:
            n, text = f.result()
            bad += n
            print(text, flush=True)
    print(f"不一致 {bad} 項")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
