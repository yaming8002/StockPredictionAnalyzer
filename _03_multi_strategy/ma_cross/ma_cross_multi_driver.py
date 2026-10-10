"""
均線交叉：多股共用資金 × 21 組 × 投法 × 買入排序（多股回測段）
================================================================
文章（五）（六）（七）與多股 reference 的數字都從這支產生。引擎用
`_03_multi_strategy/base/fast_multi.py`（與 vbt 多股引擎逐筆對帳一致，見 verify_fast_multi.py），
因為「隨機買入順序 × 1,000 次」用 vbt 版要跑好幾天。

三個任務（--task）：
  angle_adx   文章（五）（六）（七）：夾角>20°＋ADX<25＋5 日均量>1000 張。
              份數由單股蒙地卡羅的最大連敗 P95（S）決定（讀 _04 的 mc_realistic.csv）：
                定額（fixed）：份數＝round(S/0.2)，每筆＝100 萬／份數；
                比例（dyn）  ：份數＝round(1/(1−0.8^(1/S)))，每筆＝已實現權益／份數，下限 100 萬／份數。
              另跑「等分 20 份」對照（定額每筆 5 萬；比例 1/20、下限 5 萬）。
  no_filter   reference 2026-07-08：純黃金交叉、無濾網；定額每筆 1 萬；比例 1/30、下限 1 萬。
  stock_id    reference 2026-06-19：純黃金交叉、定額每筆 1 萬、買入順序＝股票代號（不挑單）。

排序：流動性（5 日均量×收盤）／低價／高價 三種固定排序 ＋ 隨機排序 × N 次
（每次每天每檔抽一個亂數當優先序，種子 SEED0＋第 k 次；取「最終權益」中位數那一次的完整列，
另附最終權益與總獲利的 5%／95% 分位（總獲利另含中位數）與「已實現權益曾跌破本金 8 成」的次數比例）。stock_id 任務只跑代號序。

引擎費用口徑＝台股實際費稅（fast_multi 預設 fees="tw"，2026-10-10 起）：最終權益、已實現權益最低、
最大回撤與逐筆 real_pnl（總獲利）同一本帳。之前的輸出是純費率口徑，重跑前要先清掉 result/<task>/
（有 fixed_<pair>.csv 的組會被當成已完成而跳過）。
各列另附「<欄名>_精確」＝未四捨五入值（fast_multi.exact_stats ＋ 最終權益／已實現權益最低／最大回撤），
隨機的總獲利 P5／中位／P95 也有精確版；文章出表從精確值一次進位。

交易區間＝標準區間 2002～2025，指標吃全史暖身。每組跑完就落地，中斷後重跑會跳過已完成的組。

輸出（result/ 不進版控）：_03_multi_strategy/ma_cross/result/<task>/
  fixed_<pair>.csv、random_<pair>.csv（逐組）→ 全部完成後合併成 orderings.csv、random_dist.csv
執行：python _03_multi_strategy/ma_cross/ma_cross_multi_driver.py --task angle_adx [--runs 1000] [--workers 6]
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
from _02_strategy.base.vbt.common import DEFAULT_END, DEFAULT_START, GLITCH  # noqa: E402
from _02_strategy.ma_strategy.ma_cross_sweep import PAIRS  # noqa: E402
from _03_multi_strategy.base.fast_multi import EXACT, exact_stats, run_panel_fast  # noqa: E402
from _03_multi_strategy.ma_cross.multi_ma_cross import MultiMACross  # noqa: E402

INIT_CASH = 1_000_000.0
FLOOR = 0.80                   # 份數公式的資金防線：連敗 S 筆後權益仍 ≥ 8 成
SEED0 = 20261008
OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
MC_CSV = os.path.join(common.result_dir("ma_strategy", "ma_cross"), "_mc_realistic", "mc_realistic.csv")
WANT = ["open", "high", "low", "close", "volume"] + [f"sma_{n}" for n in (5, 10, 20, 50, 60, 120, 200)]

TASKS = {
    "angle_adx": {"ANGLE_DEG": 20.0, "VOL_ADX": 25.0, "MIN_VOL_ZHANG": 1000},
    "no_filter": {},
    "stock_id": {},
}

_DATA = None


def units(s: int):
    """S → (定額份數, 比例份數)。兩投法公式不同、份數不可共用。"""
    return int(round(s / (1.0 - FLOOR))), int(round(1.0 / (1.0 - FLOOR ** (1.0 / s))))


def sizings(task: str, pair: str, s_table: dict) -> list:
    """回傳 [(投法名, sizing_mode, invest_ratio, min_invest, 份數)]。"""
    if task == "angle_adx":
        n_fixed, n_pct = units(s_table[pair])
        return [("定額｜公式", "fixed", 1.0, INIT_CASH / n_fixed, n_fixed),
                ("定額｜等分20", "fixed", 1.0, INIT_CASH / 20, 20),
                ("比例｜公式", "percent_floor", 1.0 / n_pct, INIT_CASH / n_pct, n_pct),
                ("比例｜等分20", "percent_floor", 1.0 / 20, INIT_CASH / 20, 20)]
    if task == "no_filter":
        return [("定額｜1萬", "fixed", 1.0, 10_000.0, 100),
                ("比例｜1/30", "percent_floor", 1.0 / 30, 10_000.0, 30)]
    return [("定額｜1萬", "fixed", 1.0, 10_000.0, 100)]


def _init(folder: str, limit):
    global _DATA
    data = common.load_market(folder, columns=WANT, limit=limit, exclude=GLITCH, min_rows=2)
    _DATA = {sid: df for sid, df in data.items() if len(df.loc[DEFAULT_START:DEFAULT_END]) >= 2}


def _prio_panels(panel: dict) -> dict:
    close = panel["close"]
    turn = pd.DataFrame({sid: _DATA[sid]["volume"].rolling(5).mean() * _DATA[sid]["close"]
                         for sid in close.columns}).reindex(index=close.index, columns=close.columns)
    c = close.to_numpy(np.float64)
    return {"流動性": turn.to_numpy(np.float64), "低價": -c, "高價": c}


def _row(pair, sname, n_units, per, order, res) -> dict:
    s = res["summary"]
    return {"短/長": pair, "投法": sname, "份數": n_units, "每筆": round(per),
            "排序": order, **common.spec_row(s), "擋單": res["blocked_orders"],
            # 總獲利(萬) 只留 1 位小數；文章顯示到萬位，落在 .5 時要靠元為單位的值判斷進位
            "總獲利(元)": s["總獲利"],
            "已實現權益最低%": s["已實現權益最低(%)"],
            # 破底線用精確值判（存檔值兩位小數，79.995 會被捨成 80.00 而漏判）
            "本金曾<80%": s["已實現權益最低(%)" + EXACT] < FLOOR * 100,
            "最終權益": s["最終權益"], "最大回撤%": s["最大回撤(%)"],
            "已實現權益最低%" + EXACT: s["已實現權益最低(%)" + EXACT],
            "最終權益" + EXACT: s["最終權益" + EXACT],
            "最大回撤%" + EXACT: s["最大回撤(%)" + EXACT]} | exact_stats(res["trades"])


def run_pair(task: str, pair: str, s_table: dict, runs: int, out_dir: str) -> str:
    t0 = time.time()
    short, long = (int(x) for x in pair.split("/"))
    m = MultiMACross()
    m.SHORT_MA, m.LONG_MA = short, long
    for k, v in TASKS[task].items():
        setattr(m, k, v)
    panel = m.build_panel(_DATA, DEFAULT_START, DEFAULT_END)
    prios = {"代號": np.zeros_like(panel["price"])} if task == "stock_id" else _prio_panels(panel)

    det, rnd = [], []
    for sname, mode, ratio, floor, n_units in sizings(task, pair, s_table):
        m.sizing_mode, m.invest_ratio, m.min_invest = mode, ratio, floor
        for oname, pr in prios.items():
            det.append(_row(pair, sname, n_units, floor, oname, run_panel_fast(m, panel, prio=pr)))
        if task == "stock_id" or runs <= 0:
            continue
        rows, pnl = [], []
        for k in range(runs):
            pr = np.random.default_rng(SEED0 + k).random(panel["price"].shape)
            res = run_panel_fast(m, panel, prio=pr)
            rows.append(_row(pair, sname, n_units, floor, "隨機", res))
            pnl.append(rows[-1].get("總獲利(萬)" + EXACT, 0.0))     # 精確總獲利（萬）；無交易＝0
        df = pd.DataFrame(rows)
        eq = df["最終權益"].to_numpy()
        med = df.iloc[int(np.argsort(eq)[len(eq) // 2])].to_dict()
        # 總獲利的分位另外記：最終權益含未平倉部位的市值，總獲利只算已平倉的逐筆淨損益，
        # 文章的「最差／中位／最好」用後者（2026-10-10 前引擎是純費率、兩者還差在費用口徑，現已同一本帳）。
        # 用未捨入的總獲利算分位，避免 1 位小數捨入後再取分位落在 .5 邊界。
        pnl = np.asarray(pnl)
        med.update({"隨機次數": runs, "最終權益_P5": round(float(np.percentile(eq, 5))),
                    "最終權益_P95": round(float(np.percentile(eq, 95))),
                    "總獲利(萬)_P5": round(float(np.percentile(pnl, 5)), 4),
                    "總獲利(萬)_中位": round(float(np.median(pnl)), 4),
                    "總獲利(萬)_P95": round(float(np.percentile(pnl, 95)), 4),
                    "總獲利(萬)_P5" + EXACT: float(np.percentile(pnl, 5)),
                    "總獲利(萬)_中位" + EXACT: float(np.median(pnl)),
                    "總獲利(萬)_P95" + EXACT: float(np.percentile(pnl, 95)),
                    "本金曾<80%_比例%": round(float(df["本金曾<80%"].mean() * 100), 1),
                    "擋單_中位": int(df["擋單"].median())})
        rnd.append(med)
    tag = pair.replace("/", "_")
    pd.DataFrame(det).to_csv(os.path.join(out_dir, f"fixed_{tag}.csv"), index=False, encoding="utf-8-sig")
    if rnd:
        pd.DataFrame(rnd).to_csv(os.path.join(out_dir, f"random_{tag}.csv"), index=False,
                                 encoding="utf-8-sig")
    return f"{pair}｜{time.time() - t0:.0f} 秒"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--pairs", nargs="+", default=PAIRS)
    ap.add_argument("--folder", default=common.DATA_DIR)
    ap.add_argument("--limit", type=int, default=None, help="冒煙用；會寫進正式 result")
    a = ap.parse_args()

    s_table = {}
    if a.task == "angle_adx":
        if not os.path.isfile(MC_CSV):
            raise SystemExit(f"找不到 {MC_CSV}\n先跑：python _04_analysis/ma_cross/ma_cross_montecarlo.py")
        mc = pd.read_csv(MC_CSV)
        s_table = dict(zip(mc["短/長"], mc["最大連敗_P95"].astype(int)))
    out_dir = os.path.join(OUT_ROOT, a.task)
    os.makedirs(out_dir, exist_ok=True)
    todo = [p for p in a.pairs
            if not os.path.isfile(os.path.join(out_dir, f"fixed_{p.replace('/', '_')}.csv"))]
    print(f"{a.task}｜{len(todo)}/{len(a.pairs)} 組待跑｜隨機 {a.runs} 次｜{a.workers} 進程", flush=True)

    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                             initargs=(a.folder, a.limit)) as pool:
        futs = [pool.submit(run_pair, a.task, p, s_table, a.runs, out_dir) for p in todo]
        for f in futs:
            print(f.result(), flush=True)

    tags = [p.replace("/", "_") for p in a.pairs]
    pd.concat([pd.read_csv(os.path.join(out_dir, f"fixed_{t}.csv")) for t in tags]).to_csv(
        os.path.join(out_dir, "orderings.csv"), index=False, encoding="utf-8-sig")
    rnd = [os.path.join(out_dir, f"random_{t}.csv") for t in tags]
    if all(os.path.isfile(p) for p in rnd):
        pd.concat([pd.read_csv(p) for p in rnd]).to_csv(
            os.path.join(out_dir, "random_dist.csv"), index=False, encoding="utf-8-sig")
    print(f"→ {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
