"""
多股 MACD × 兩種投法 × 四種買入排序
=====================================
把 MACD 系列挑出的五組交易策略（見 `_03_multi_strategy/macd/multi_macd.py`）搬進
「單一本金、多檔共用資金」的組合回測，每組都跑兩種投法 × 四種買入排序。

**兩種投法的份數公式不同、份數也不同，絕不可共用**（資金防線 floor=0.80，
連敗時權益至少保住八成本金；S＝該策略單股逐筆損益的蒙地卡羅「最大連敗 P95」）：
  固定金額投入（定額，線性相加）：r=(1−0.8)/S → 份數＝S/0.2；每筆＝100 萬÷份數，不複利。
  固定比例投入（比例，幾何複利）：r=1−0.8^(1/S) → 份數＝round(1/r)；每筆＝已實現權益×(1/份數)。
S 取自 MACD 系列第十篇蒙地卡羅那張表的已發佈數字（見下方 S_BY_STRATEGY），
不在這裡重跑一輪——重跑的隨機種子不同會與文章對不上，份數跟著飄。

**四種買入排序**（同一天多個買訊、現金不夠時先買誰）：低價／高價／流動性／隨機。
這支只出前三種；隨機那一格全市場要重跑 1,000 次，走多進程的 `macd_multi_random.py`。
隨機＝隨機化買入順序、重複多次取中位＋5/95，當「亂買」基準線，用來檢驗低價優先
是不是真的有優勢。訊號與價格面板只建一次（`build_panel`），之後只重跑下單模擬
（`run_panel`）換優先序，否則每抽一次都要重掃全市場。

**下單模擬走精簡引擎 `run_panel_fast`（台股實際費稅，2026-10-10 起）**：隨機那一格（macd_multi_random.py）
早已走精簡引擎，固定排序若還用 vbt 版（純費率、無最低 20 元），同一張表的「贏不贏隨機」就是拿兩套
費用口徑在比，所以三種固定排序也改走同一支。vbt 版與精簡引擎的唯一行為差異是優先序平手時的順序
（精簡引擎按股票代號序；低價／高價排序同價時會碰到），見 fast_multi 檔頭。
各列另附「<欄名>_精確」＝未四捨五入值（fast_multi.exact_stats），文章出表從精確值一次進位。

執行：
    python _03_multi_strategy/macd/macd_multi_driver.py [--random-runs 1000] [--limit N]
輸出：result/macd_multi/macd_multi_result.csv ＋ 終端表。
"""
import argparse
import os
import sys
import time

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd

from _02_strategy.base.vbt import common
from _02_strategy.base.vbt.common import DEFAULT_END, DEFAULT_START, GLITCH
from _03_multi_strategy.base.fast_multi import EXACT, exact_stats, run_panel_fast
from _03_multi_strategy.macd.multi_macd import STRATEGIES, MultiMACD

DATA = common.DATA_DIR
WANT = ["open", "high", "low", "close", "volume", "macd", "signal_line"]
INIT_CASH = 1_000_000.0
FLOOR = 0.80
PCT_MIN_INVEST = 10_000.0        # 固定比例投入的每筆下限
SEED0 = 20260929                 # 隨機排序的種子起點；固定住才能重現同一批抽樣
OUT = common.result_dir("macd_strategy", "macd_multi")

# S＝最大連敗 P95，取自第十篇蒙地卡羅表（已發佈）
S_BY_STRATEGY = {
    "交叉 × 均線多頭排列 × 跌破年線": 32,
    "交叉 × ADX>25 × 跌破年線": 21,
    "交叉 × 無濾網 × 跌破年線": 25,
    "零軸 × 收盤>MA200 × 跌破年線": 43,
    "背離 × RSI<50且上升 × 跌破年線": 19,
}


def units(s: int):
    """回傳 (定額份數, 比例份數)。兩式不同，這裡並列避免日後又混用。"""
    n_fixed = int(round(s / (1.0 - FLOOR)))                    # 線性
    n_pct = int(round(1.0 / (1.0 - FLOOR ** (1.0 / s))))       # 幾何
    if n_fixed == n_pct:
        raise AssertionError(f"S={s} 兩投法份數相同，公式寫錯了")
    return n_fixed, n_pct


def load_all(limit=None):
    """
    讀全市場，每檔只留 WANT 那幾欄、保留全史（指標暖身吃起日前的資料）。
    交易區間在 build_panel(DEFAULT_START, DEFAULT_END) 才裁；這裡只篩掉區間內不到 2 根的檔。

    走 `common.load_market`（欄位清單讀 parquet metadata，不整檔讀；原因見該函式說明）。
    """
    data = common.load_market(DATA, columns=WANT, limit=limit, exclude=GLITCH, min_rows=2)
    return {sid: df for sid, df in data.items()
            if len(df.loc[DEFAULT_START:DEFAULT_END]) >= 2}


def prio_panels(data, close_panel):
    """低價／高價／流動性三種優先序，直接由原始資料對齊出來，不必重掃訊號。"""
    idx, cols = close_panel.index, close_panel.columns
    turn = pd.DataFrame(
        {sid: data[sid]["volume"].rolling(5).mean() * data[sid]["close"]
         for sid in cols}).reindex(index=idx, columns=cols)
    c = close_panel.to_numpy(np.float64)
    return {"低價": -c, "高價": c, "流動性": turn.to_numpy(np.float64)}


def row_of(label, mname, kind, n_units, res):
    """出「回測結果表規格」的固定 10 欄；額外掛上擋單數當診斷欄，另附各欄精確值。"""
    return common.spec_row(res["summary"], 交易策略=label, 投法=mname, 排序=kind,
                           份數=n_units, 擋單=res["blocked_orders"]) | exact_stats(res["trades"])


def main():
    ap = argparse.ArgumentParser()
    # 隨機排序預設關掉：這裡是單進程實作，全市場 1,000 次要跑近 30 小時。
    # 正式的隨機那一格走 macd_multi_random.py（多進程、面板重用），這支只留小樣本冒煙用。
    ap.add_argument("--random-runs", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    t0 = time.time()
    data = load_all(a.limit)
    print(f"載入 {len(data)} 檔｜{DEFAULT_START}~{DEFAULT_END}｜"
          f"{time.time() - t0:.0f} 秒", flush=True)

    rows = []
    for label, base, entry in STRATEGIES:
        s = S_BY_STRATEGY[label]
        n_fixed, n_pct = units(s)
        builder = MultiMACD()
        builder.BASE, builder.ENTRY, builder.PRIO = base, entry, "low_price"
        tb = time.time()
        panel = builder.build_panel(data, DEFAULT_START, DEFAULT_END)
        prios = prio_panels(data, panel["close"])
        print(f"\n{label}｜S={s}｜定額 {n_fixed} 份（每筆 "
              f"{INIT_CASH / n_fixed:,.0f}）｜比例 {n_pct} 份｜"
              f"面板 {time.time() - tb:.0f} 秒", flush=True)

        for mname, mode, kwargs, n_units in (
                ("定額", "fixed", {"min_invest": INIT_CASH / n_fixed}, n_fixed),
                ("比例", "percent_floor",
                 {"invest_ratio": 1.0 / n_pct, "min_invest": PCT_MIN_INVEST}, n_pct)):
            inst = MultiMACD(initial_cash=INIT_CASH, sizing_mode=mode, **kwargs)
            for kind in ("低價", "高價", "流動性"):
                tr = time.time()
                res = run_panel_fast(inst, panel, prios[kind], want_equity=False)
                r = row_of(label, mname, kind, n_units, res)
                rows.append(r)
                print(f"  {mname}({n_units}) {kind}：{r['交易次數']:,} 筆"
                      f"｜擋單 {r['擋單']:,}｜勝率 {r['勝率%']}%｜"
                      f"PF {r['獲利因子']}｜總獲利 {r['總獲利(萬)']} 萬｜"
                      f"{time.time() - tr:.1f} 秒", flush=True)
            # 隨機排序：重抽 N 次、每次換一整面板的亂數優先序，取中位當代表值
            # （單次抽樣受運氣影響太大，中位＋5/95 才看得出「亂買」的分布落在哪）。
            if a.random_runs > 0:
                tr = time.time()
                reps = [row_of(label, mname, "隨機", n_units,
                               run_panel_fast(inst, panel, np.random.default_rng(
                                   SEED0 + seed).random(panel["entries"].shape),
                                   want_equity=False))
                        for seed in range(a.random_runs)]
                rep = pd.DataFrame(reps)
                # 精確欄取「逐次精確值」的中位、不捨入（同 macd_multi_random.summarize）
                med = {c: ((float(rep[c].median()) if c.endswith(EXACT)
                            else round(float(rep[c].median()), 4))
                           if rep[c].dtype.kind in "fi" else rep[c].iloc[0])
                       for c in rep.columns}
                med["總獲利(萬)P5"] = round(float(rep["總獲利(萬)"].quantile(0.05)), 1)
                med["總獲利(萬)P95"] = round(float(rep["總獲利(萬)"].quantile(0.95)), 1)
                med["總獲利(萬)P5" + EXACT] = float(rep["總獲利(萬)" + EXACT].quantile(0.05))
                med["總獲利(萬)P95" + EXACT] = float(rep["總獲利(萬)" + EXACT].quantile(0.95))
                rows.append(med)
                print(f"  {mname}({n_units}) 隨機×{a.random_runs}："
                      f"PF {med['獲利因子']}｜總獲利中位 {med['總獲利(萬)']} 萬"
                      f"（P5 {med['總獲利(萬)P5']}／P95 {med['總獲利(萬)P95']}）｜"
                      f"{time.time() - tr:.0f} 秒", flush=True)

    out = pd.DataFrame(rows)
    common.assert_spec_columns(out)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "macd_multi_result.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n耗時 {time.time() - t0:.0f} 秒｜{len(out)} 列 → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
