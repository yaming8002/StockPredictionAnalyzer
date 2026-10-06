"""
多股 MACD × 隨機買入順序基準線（1,000 次）
=============================================
四種買入排序裡的「隨機」這一格：把同一天多個買訊的成交順序整個打亂，重抽 1,000 次
取中位＋P5/P95，當「亂買」的基準線——低價優先到底是真有優勢，還是只是剛好抽中一種
順序，得跟這條線比才知道。

**為什麼要多進程**：vbt 的 `from_order_func` 逐格走 6,000 天 × 2,258 檔，單場全市場
實測約 10.5 秒；5 個交易策略 × 2 種投法 × 1,000 次 ＝ 10,000 場，單進程要約 29 小時。
訊號面板跟優先序無關（面板只看價格與規則，不看現金），所以面板建一次就好，之後丟給
子進程各自重跑下單模擬。面板落成 .npy 再由子進程讀回，比用 pickle 傳 200 MB 陣列省。

**⚠️ 記憶體的真正瓶頸是系統的 commit 額度，不是「可用實體記憶體」。** 建面板需要全市場
資料在手，父進程的 commit 會衝到 20 GB 以上；子進程要跑好幾個小時。兩件事重疊的話光
父進程就把 commit 吃光，子進程一配置就 `MemoryError: Allocation failed`——而那時
「可用實體記憶體」還顯示二十幾 GB，完全看不出來。所以流程是**先把所有面板建完落地、
清掉 data、才開子進程池**。查 commit 用：
`Get-Counter '\\Memory\\Committed Bytes','\\Memory\\Commit Limit'`。

**⚠️ 進程數受系統 commit 額度限制**：每個子進程的記憶體是鋸齒狀的，全市場的峰值約
15~16 GB（每場 +1.5 GB、每 5~6 場自己歸零），這個峰值壓不下去。實測這台機器（64 GB
RAM，同時開著 Android Studio／WSL2／Chrome）**4 個子進程可以、6 個會 MemoryError**。
開跑前先看 commit 餘裕：`Get-Counter '\\Memory\\Committed Bytes','\\Memory\\Commit Limit'`。

這一輪要跑好幾個小時，所以**每跑完一格就立刻寫檔**，再開一次會自動跳過已完成的格子
（`--restart` 強制從頭跑）。中途掛掉不必整輪重來。

執行：
    python _03_multi_strategy/macd/macd_multi_random.py [--runs 1000] [--workers 6] [--limit N]
輸出：result/macd_multi/macd_multi_random.csv（每格跑完即更新）
"""
import argparse
import gc
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd
import vectorbt as vbt

# vbt 預設會把每個物件的屬性結果記憶化。這裡每場都是一個用完即丟的 Portfolio，快取
# 只會把它們留住：實測（900 檔）記憶體鋸齒的峰值從 10.74 GB 降到 7.72 GB（−28%）。
# 關掉只是不做屬性記憶化，數值完全不受影響。
vbt.settings["caching"]["enabled"] = False

from _02_strategy.base.vbt import common
from _03_multi_strategy.macd.multi_macd import STRATEGIES, MultiMACD
from _03_multi_strategy.macd.macd_multi_driver import (INIT_CASH, OUT, PCT_MIN_INVEST,
                                            SEED0, S_BY_STRATEGY, load_all,
                                            units)

# 子進程內的面板快取：同一格的種子共用一份，但**只留最新一個策略的面板**——
# 子進程在整輪都活著，五個策略的面板全留會各佔數百 MB、乘上進程數就爆掉。
_CACHE = {}


def dump_panel(panel: dict, folder: str) -> None:
    """把面板落成 .npy（close 的日期軸與欄名另存，子進程要拼回 DataFrame）。"""
    os.makedirs(folder, exist_ok=True)
    close = panel["close"]
    np.save(os.path.join(folder, "close.npy"), close.to_numpy(np.float64))
    np.save(os.path.join(folder, "index.npy"), close.index.to_numpy())
    np.save(os.path.join(folder, "columns.npy"), np.array(close.columns, dtype=object),
            allow_pickle=True)
    for key in ("entries", "exits", "price"):
        np.save(os.path.join(folder, f"{key}.npy"), panel[key])


def load_panel(folder: str) -> dict:
    """子進程讀回面板；prio 由呼叫端每次重抽，這裡給占位陣列。"""
    if folder in _CACHE:
        return _CACHE[folder]
    _CACHE.clear()                      # 換策略了，舊面板立刻釋放
    close = pd.DataFrame(
        np.load(os.path.join(folder, "close.npy")),
        index=pd.DatetimeIndex(np.load(os.path.join(folder, "index.npy"))),
        columns=np.load(os.path.join(folder, "columns.npy"), allow_pickle=True))
    panel = {"close": close,
             "entries": np.load(os.path.join(folder, "entries.npy")),
             "exits": np.load(os.path.join(folder, "exits.npy")),
             "price": np.load(os.path.join(folder, "price.npy")),
             "prio": np.zeros(close.shape), "has_priority": True}
    _CACHE[folder] = panel
    return panel


def run_chunk(task: tuple) -> list:
    """子進程：同一格（策略 × 投法）跑一串種子，各回一列規格 10 欄。"""
    folder, mode, kwargs, seeds = task
    panel = load_panel(folder)
    inst = MultiMACD(initial_cash=INIT_CASH, sizing_mode=mode, **kwargs)
    # 優先序緩衝區重用：每場都配一塊 (天數 × 檔數) 的 float64 要 103 MB，
    # 每場重配再乘上進程數很快就把記憶體吃光。配一次、每場填新亂數。
    prio = np.empty(panel["entries"].shape, dtype=np.float64)
    rows = []
    for seed in seeds:
        np.random.default_rng(SEED0 + seed).random(out=prio)
        # want_equity=False：這裡只要 summary，不必展開整條逐日權益曲線
        res = inst.run_panel(panel, prio, want_equity=False)
        row = common.spec_row(res["summary"])
        row["擋單"] = res["blocked_orders"]
        rows.append(row)
        del res
        if len(rows) % 50 == 0:
            gc.collect()
    return rows


def chunks(seeds: list, n: int) -> list:
    """把種子切成 n 份（子進程各拿一份，面板只讀一次）。"""
    return [seeds[i::n] for i in range(n) if seeds[i::n]]


def run_cell(folder: str, mode: str, kwargs: dict, runs: int, workers: int,
             per_worker: int) -> list:
    """
    跑完一格（同一個策略 × 投法的全部 runs 場），**每一小批就把子進程池重建一次**。

    記憶體實況（逐場量 PrivateUsage 量出來的，900 檔）：**每場約 +1.5 GB，但每 5~6 場
    會自己掉回起點**，是鋸齒狀、峰值有上限，不是無上限洩漏。全市場（2,258 檔）的鋸齒
    峰值約 15~16 GB／子進程——**這個峰值由 vbt 內部決定，調 per_worker 壓不下去**。
    所以進程數要照「峰值 × 進程數 < 系統 commit 餘裕」抓，實測這台 4 個可以、6 個會掛。

    重建子進程池仍然保留：長跑時讓位址空間定期回到作業系統，比較不會被其他程式的
    commit 波動夾死。per_worker=25 時一批約 4 分鐘、開銷約 4%。

    ⚠️ 判斷有沒有問題要看**鋸齒峰值**，不是看某兩個時間點連線外推——我曾因此把一輪
    其實正常的 4 進程回測誤判成「注定跑不完」而砍掉。
    """
    got = []
    step = workers * per_worker
    for start in range(0, runs, step):
        seeds = list(range(start, min(start + step, runs)))
        tasks = [(folder, mode, kwargs, c) for c in chunks(seeds, workers)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for part in pool.map(run_chunk, tasks):
                got.extend(part)
    return got


def summarize(rows: list, labels: dict) -> dict:
    """1,000 次的中位數當代表值；總獲利另出 P5/P95 看「亂買」的分布寬度。"""
    rep = pd.DataFrame(rows)
    out = dict(labels)
    for c in rep.columns:
        out[c] = (round(float(rep[c].median()), 4)
                  if rep[c].dtype.kind in "fi" else rep[c].iloc[0])
    out["總獲利(萬)P5"] = round(float(rep["總獲利(萬)"].quantile(0.05)), 1)
    out["總獲利(萬)P95"] = round(float(rep["總獲利(萬)"].quantile(0.95)), 1)
    return out


def build_all_panels(data: dict, tmp: str, done: set, t0: float) -> list:
    """
    把每個還沒跑完的策略的面板都建好、落成 .npy，回傳 [(顯示名, 資料夾, 定額份數,
    比例份數, 還要跑哪些投法)]。

    面板落地之後 data 就可以清掉——這是整支的記憶體關鍵（見檔頭說明）。
    """
    plan = []
    for label, base, entry in STRATEGIES:
        todo = [m for m in ("定額", "比例") if (label, m) not in done]
        if not todo:
            print(f"{label}｜兩格都已完成，跳過", flush=True)
            continue
        n_fixed, n_pct = units(S_BY_STRATEGY[label])
        builder = MultiMACD()
        builder.BASE, builder.ENTRY, builder.PRIO = base, entry, "low_price"
        folder = os.path.join(tmp, f"{base}_{entry}")
        dump_panel(builder.build_panel(data), folder)
        plan.append((label, folder, n_fixed, n_pct, todo))
        print(f"{label}｜面板落地｜{time.time() - t0:.0f} 秒", flush=True)
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1000)
    # 進程數受**系統 commit 額度**限制，不是受實體記憶體限制（見檔頭）。開跑前先看
    # commit 還剩多少：每個子進程約要 5 GB。
    ap.add_argument("--workers", type=int, default=6)
    # 每個子進程跑幾場就換掉（vbt/numba 每場漏約 0.4 GB commit，見 run_cell）
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--restart", action="store_true",
                    help="忽略既有結果、整輪從頭跑")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "macd_multi_random.csv")
    rows, done = [], set()
    if os.path.exists(path) and not a.restart:
        prev = pd.read_csv(path)
        rows = prev.to_dict("records")
        done = {(r["交易策略"], r["投法"]) for r in rows}
        print(f"接續上次：已完成 {len(done)} 格", flush=True)

    t0 = time.time()
    data = load_all(a.limit)
    print(f"載入 {len(data)} 檔｜隨機 {a.runs} 次 × {a.workers} 進程｜"
          f"{time.time() - t0:.0f} 秒", flush=True)

    tmp = tempfile.mkdtemp(prefix="macd_panel_")
    try:
        plan = build_all_panels(data, tmp, done, t0)
        data.clear()
        gc.collect()
        n_cells = sum(len(x[4]) for x in plan)
        print(f"資料已釋放，要跑 {n_cells} 格 × {a.runs} 次"
              f"｜{time.time() - t0:.0f} 秒", flush=True)
        if not n_cells:
            print("沒有要跑的格子。")
            return 0

        for label, folder, n_fixed, n_pct, todo in plan:
            print(f"\n{label}", flush=True)
            for mname, mode, kwargs, n_units in (
                    ("定額", "fixed", {"min_invest": INIT_CASH / n_fixed}, n_fixed),
                    ("比例", "percent_floor",
                     {"invest_ratio": 1.0 / n_pct,
                      "min_invest": PCT_MIN_INVEST}, n_pct)):
                if mname not in todo:
                    continue
                t = time.time()
                got = run_cell(folder, mode, kwargs, a.runs, a.workers, a.batch)
                rows.append(summarize(got, {"交易策略": label, "投法": mname,
                                            "排序": "隨機", "份數": n_units}))
                pd.DataFrame(rows).to_csv(path, index=False,
                                          encoding="utf-8-sig")   # 跑完一格就落地
                r = rows[-1]
                print(f"  {mname}({n_units}) 隨機×{len(got)}："
                      f"{r['交易次數']:,.0f} 筆｜PF {r['獲利因子']}｜"
                      f"總獲利中位 {r['總獲利(萬)']} 萬"
                      f"（P5 {r['總獲利(萬)P5']}／P95 {r['總獲利(萬)P95']}）｜"
                      f"{time.time() - t:.0f} 秒", flush=True)
            shutil.rmtree(folder, ignore_errors=True)   # 這個策略跑完就清面板
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    out = pd.DataFrame(rows)
    common.assert_spec_columns(out)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n耗時 {time.time() - t0:.0f} 秒｜{len(out)} 列 → {path}")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
