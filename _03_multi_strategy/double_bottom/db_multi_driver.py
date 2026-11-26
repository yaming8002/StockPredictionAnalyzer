"""
多股雙底 × 兩種投法 × 四種買入排序（〈雙底（九）〉〈雙底（十）〉）
====================================================================
〈雙底（八）〉蒙地卡羅那五組（全是 ZigZag 9%、可成交）搬進「單一本金、多檔共用資金」的組合回測，
每組跑兩種投法 × 四種買入排序（低價／高價／成交金額／隨機 1,000 次取中位＋P5/P95）。
引擎＝fast_multi_db（出場依實際持倉判定，原因見該檔檔頭）；對帳＝verify_db_multi.py。

**份數（同 MACD 多股 driver，兩投法公式不同、絕不共用；floor=0.80）**：
  S＝〈雙底（八）〉蒙地卡羅表的最大連敗 P95（已發佈數字，寫死在 PICKS，不在這裡重跑）。
  固定金額投入：份數＝S/0.2，每筆＝100 萬÷份數，不複利。
  固定比例投入：份數＝round(1/(1−0.8^(1/S)))，每筆＝max(已實現權益÷份數, 1 萬)。

**買入排序看「判定日」（成交日前一根）的收盤與成交金額**（prio_prev；用戶 2026-11-13 定）：
  MACD／KD 多股共用的 macd_multi_driver.prio_panels 用的是成交當天的收盤，而成交在當天開盤，
  排序等於看到開盤時還不知道的收盤價。用戶：「的確是不應該看未來。後續是要看前一天收盤來排序。
  但是買入還是一樣是隔天」。成交仍是判定日隔天開盤，只有排序依據往前一根；已發佈的 MACD／KD 不動。
隨機排序：每場一整面板的亂數優先序，種子 SEED0＋k，固定住才能重現；面板落 .npy 給子進程讀。
每格（組合 × 投法 × 排序）跑完就寫檔，中斷後重下同一行會跳過已完成的格子（--restart 強制重來）。

執行：python _03_multi_strategy/double_bottom/db_multi_driver.py [--runs 1000] [--workers 4] [--limit N]
輸出：_02_strategy/double_bottom_strategy/result/double_bottom_multi/db_multi_result.csv
"""
import argparse
import gc
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _02_strategy.base.vbt import common  # noqa: E402
from _02_strategy.base.vbt.common import DEFAULT_END, DEFAULT_START, GLITCH  # noqa: E402
from _03_multi_strategy.base.fast_multi import EXACT, exact_stats  # noqa: E402
from _03_multi_strategy.double_bottom.fast_multi_db import build_panel, run_panel  # noqa: E402

WANT = ["open", "high", "low", "close", "volume"]
INIT_CASH = 1_000_000.0
FLOOR = 0.80
PCT_MIN_INVEST = 10_000.0
SEED0 = 20261113
OUT = common.result_dir("double_bottom_strategy", "double_bottom_multi")
PATH = os.path.join(OUT, "db_multi_result.csv")
ORDERS = ("低價", "高價", "成交金額", "隨機")

# (顯示名, 門檻, 進場, 出場, S)；S＝〈雙底（八）〉蒙地卡羅表的最大連敗 P95
PICKS = [
    ("第二底墊高 ≥ 2% × 基本出場", 0.09, "rise", "base", 32),
    ("第二底墊高 ≥ 2% × 附加：保本停損", 0.09, "rise", "add_breakeven", 33),
    ("第二底墊高 ≥ 2% × 附加：高檔 M 頭", 0.09, "rise", "add_mtop", 34),
    ("型態高度 ≥ 10% × 基本出場", 0.09, "height", "base", 32),
    ("基本版", 0.09, "base", "base", 32),
]


def units(s: int):
    """回傳 (定額份數, 比例份數)。兩式不同，並列避免日後混用。"""
    n_fixed = int(round(s / (1.0 - FLOOR)))
    n_pct = int(round(1.0 / (1.0 - FLOOR ** (1.0 / s))))
    if n_fixed == n_pct:
        raise AssertionError(f"S={s} 兩投法份數相同，公式寫錯了")
    return n_fixed, n_pct


def sizings(s: int):
    """[(投法, 份數, sizing)]；sizing 只帶 run_panel 需要的四個欄位。"""
    n_fixed, n_pct = units(s)
    return [("定額", n_fixed, SimpleNamespace(sizing_mode="fixed", invest_ratio=0.0,
                                              min_invest=INIT_CASH / n_fixed, initial_cash=INIT_CASH)),
            ("比例", n_pct, SimpleNamespace(sizing_mode="percent_floor", invest_ratio=1.0 / n_pct,
                                            min_invest=PCT_MIN_INVEST, initial_cash=INIT_CASH))]


def load_all(limit=None) -> dict:
    """全市場、保留全史（指標暖身），交易區間在 build_panel 才裁。"""
    data = common.load_market(common.DATA_DIR, columns=WANT, limit=limit, exclude=GLITCH, min_rows=2)
    return {sid: df for sid, df in data.items() if len(df.loc[DEFAULT_START:DEFAULT_END]) >= 2}


def prio_prev(data: dict, close_panel: pd.DataFrame) -> dict:
    """
    低價／高價／成交金額三種優先序，依據＝判定日（成交日前一根，個股自己的日曆）的收盤與 5 日均量×收盤。
    先在個股日曆上 shift(1) 再對齊聯集日曆，與進場面板的判定日 → 成交日位移同一套。
    """
    idx, cols = close_panel.index, close_panel.columns
    align = lambda f: pd.DataFrame({sid: f(data[sid]).shift(1) for sid in cols}).reindex(
        index=idx, columns=cols).to_numpy(np.float64)
    c = align(lambda df: df["close"])
    turn = align(lambda df: df["volume"].rolling(5).mean() * df["close"])
    return {"低價": -c, "高價": c, "成交金額": turn}


def row_of(res: dict, **labels) -> dict:
    """通用 10 欄（spec_row）＋擋單＋期末未平倉＋各欄精確值＋多股的期末／年化／回撤。"""
    s = res["summary"]
    row = common.spec_row(s, **labels, 擋單=res["blocked_orders"]) | exact_stats(res["trades"])
    for k in ("期末未平倉", "已實現權益最低(%)" + EXACT, "最終權益" + EXACT,
              "最大回撤(%)" + EXACT, "年化報酬(%)" + EXACT):
        row[k] = s[k]
    return row


# ── 隨機排序：子進程讀 .npy 面板 ─────────────────────────────────────────
_ARRAYS = ("entries", "ths", "lowbrk", "mtop", "dclose", "height", "price", "price_tw")
_CACHE = {}


def dump_panel(panel: dict, folder: str) -> None:
    os.makedirs(folder, exist_ok=True)
    close = panel["close"]
    np.save(os.path.join(folder, "close.npy"), close.to_numpy(np.float64))
    np.save(os.path.join(folder, "index.npy"), close.index.to_numpy())
    np.save(os.path.join(folder, "columns.npy"), np.array(close.columns, dtype=object), allow_pickle=True)
    np.save(os.path.join(folder, "exit_code.npy"), np.array([panel["exit_code"]]))
    for k in _ARRAYS:
        np.save(os.path.join(folder, f"{k}.npy"), panel[k])


def load_panel(folder: str) -> dict:
    if folder not in _CACHE:
        _CACHE.clear()
        close = pd.DataFrame(np.load(os.path.join(folder, "close.npy")),
                             index=pd.DatetimeIndex(np.load(os.path.join(folder, "index.npy"))),
                             columns=np.load(os.path.join(folder, "columns.npy"), allow_pickle=True))
        panel = {"close": close, "exit_code": int(np.load(os.path.join(folder, "exit_code.npy"))[0])}
        panel |= {k: np.load(os.path.join(folder, f"{k}.npy")) for k in _ARRAYS}
        _CACHE[folder] = panel
    return _CACHE[folder]


def run_seeds(task: tuple) -> list:
    """子進程：同一格跑一串種子，各回一列。"""
    folder, sizing, seeds = task
    panel = load_panel(folder)
    prio = np.empty(panel["price"].shape, dtype=np.float64)
    rows = []
    for seed in seeds:
        np.random.default_rng(SEED0 + seed).random(out=prio)
        rows.append(row_of(run_panel(sizing, panel, prio, want_equity=True)))
    return rows


def summarize_random(rows: list, labels: dict) -> dict:
    """1,000 次取中位當代表值；精確欄取逐次精確值的中位；總獲利與期末另出 P5／P95。"""
    rep = pd.DataFrame(rows)
    out = dict(labels)
    for c in rep.columns:
        if rep[c].dtype.kind in "fi":
            out[c] = float(rep[c].median())
    for c in ("總獲利(萬)" + EXACT, "最終權益" + EXACT):
        out[c.replace(EXACT, "P5" + EXACT)] = float(rep[c].quantile(0.05))
        out[c.replace(EXACT, "P95" + EXACT)] = float(rep[c].quantile(0.95))
    out["隨機次數"] = len(rep)
    return out


def run_random(folder: str, sizing, runs: int, workers: int) -> list:
    seeds = list(range(runs))
    tasks = [(folder, sizing, seeds[i::workers]) for i in range(workers) if seeds[i::workers]]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [r for part in pool.map(run_seeds, tasks) for r in part]


def run_pick(data, pick, done, rows, a, tmp, t0) -> None:
    """一個組合：建面板 → 三種固定排序 → 隨機；每格跑完就寫檔。"""
    label, pct, entry, exit_key, s = pick
    panel = build_panel(data, entry, exit_key, pct, DEFAULT_START, DEFAULT_END)
    prios = prio_prev(data, panel["close"])
    folder = os.path.join(tmp, f"{entry}_{exit_key}")
    dump_panel(panel, folder)
    print(f"\n{label}｜S={s}｜份數 定額/比例 {units(s)}｜面板 {time.time() - t0:.0f} 秒", flush=True)
    for mname, n_units, sizing in sizings(s):
        for kind in ORDERS:
            if (label, mname, kind) in done:
                continue
            tc = time.time()
            labels = {"交易策略": label, "投法": mname, "排序": kind, "份數": n_units}
            if kind == "隨機":
                r = summarize_random(run_random(folder, sizing, a.runs, a.workers), labels)
            else:
                r = row_of(run_panel(sizing, panel, prios[kind], want_equity=True), **labels)
            rows.append(r)
            pd.DataFrame(rows).to_csv(PATH, index=False, encoding="utf-8-sig")
            print(f"  {mname}({n_units}) {kind}：{r['交易次數']:,.0f} 筆｜擋單 {r['擋單']:,.0f}｜"
                  f"PF {r['獲利因子(PF)' + EXACT]:.3f}｜總獲利 {r['總獲利(萬)' + EXACT]:,.0f} 萬｜"
                  f"年化 {r['年化報酬(%)' + EXACT]:.2f}%｜MDD {r['最大回撤(%)' + EXACT]:.1f}%｜"
                  f"{time.time() - tc:.0f} 秒", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--restart", action="store_true", help="忽略既有結果、整輪從頭跑")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows, done = [], set()
    if os.path.exists(PATH) and not a.restart:
        rows = common.read_result_csv(PATH).to_dict("records")
        done = {(r["交易策略"], r["投法"], r["排序"]) for r in rows}
        print(f"接續上次：已完成 {len(done)} 格", flush=True)
    t0 = time.time()
    data = load_all(a.limit)
    print(f"載入 {len(data)} 檔｜{DEFAULT_START}~{DEFAULT_END}｜隨機 {a.runs} 次 × {a.workers} 進程", flush=True)
    tmp = tempfile.mkdtemp(prefix="db_panel_")
    try:
        for pick in PICKS:
            if all((pick[0], m, k) in done for m in ("定額", "比例") for k in ORDERS):
                continue
            run_pick(data, pick, done, rows, a, tmp, t0)
            shutil.rmtree(os.path.join(tmp, f"{pick[2]}_{pick[3]}"), ignore_errors=True)
            gc.collect()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out = pd.DataFrame(rows)
    common.assert_spec_columns(out)
    out.to_csv(PATH, index=False, encoding="utf-8-sig")
    print(f"\n耗時 {time.time() - t0:.0f} 秒｜{len(out)} 列 → {PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
