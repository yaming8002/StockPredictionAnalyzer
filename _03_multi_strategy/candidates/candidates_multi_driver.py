"""
新系列候選（海龜 vs 雙底）多股回測：20 組 × 兩種投法 × 三種固定排序（vbt）
==========================================================================
組合清單見 cases.py（全部可成交 1,000 萬版）。每組跑兩段：
  全期  2002-01-01 ~ 2025-12-31：交易統計（規格 9 欄）＋擋單＋資金倍數＋最大回撤；
  對照  2015-01-01 ~ 2025-12-31：逐日市值權益存檔，分析段（_04_analysis/candidates/）挑每組全期
        總獲利最高的投法×排序，跟 0050 買進持有（含息）比。指標暖身一律吃全史，下單才從起日開始。

份數（S＝單股蒙地卡羅最大連敗 P95，讀 _04_analysis/candidates/candidates_montecarlo.py 的輸出）：
  固定金額投入：份數＝round(S/0.2)，每筆＝100 萬÷份數，不複利；
  固定比例投入：份數＝round(1/(1−0.8^(1/S)))，每筆＝已實現權益÷份數（下限 1 萬）。
三種買入排序（同日現金不夠時先買誰）：低價／高價／流動性（排序面板沿用 MACD 多股 driver 的 prio_panels）。
隨機排序 ×1000 這一輪不跑。

**引擎：vbt `from_order_func`**（`VbtMultiStrategy.run_panel`；雙底用 multi_double_bottom 的帶狀態下單函式，
出場以實際成交的進場起算）。vbt 版的現金軌跡是純費率（無最低 20 元），逐筆損益另以台股費稅精算，
所以「資金倍數／回撤」（權益曲線）與「總獲利」（逐筆加總）是兩本帳，量級差異很小但不會逐位相等。

執行：python _03_multi_strategy/candidates/candidates_multi_driver.py [--workers 3] [--limit N] [--keys ...]
輸出：_02_strategy/double_bottom_strategy/result/candidates_multi/
  runs/<key>.csv           每組 6 列（兩投法 × 三排序）全期結果
  equity_2015/<key>.parquet 每組 6 條 2015–2025 逐日權益
  candidates_multi_result.csv  全部合併
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd  # noqa: E402

from _02_strategy.base.vbt import common  # noqa: E402
from _02_strategy.base.vbt.common import DEFAULT_END, DEFAULT_START, GLITCH  # noqa: E402
from _03_multi_strategy.candidates.cases import CASES  # noqa: E402
from _03_multi_strategy.macd.macd_multi_driver import prio_panels  # noqa: E402

INIT_CASH = 1_000_000.0
PCT_MIN_INVEST = 10_000.0
WIN_START = "2015-01-01"          # 必須等於 0050 對照起點（分析段會檢查）
ORDERINGS = ("低價", "高價", "流動性")
OUT = common.result_dir("double_bottom_strategy", "candidates_multi")
MC_FILE = os.path.join(OUT, "candidates_montecarlo.csv")
RESULT_FILE = os.path.join(OUT, "candidates_multi_result.csv")

_DATA = None


def _init(limit):
    """每個子進程只讀一次全市場（全史，供指標暖身）。"""
    global _DATA
    data = common.load_market(common.DATA_DIR, columns=["open", "high", "low", "close", "volume"],
                              limit=limit, exclude=GLITCH, min_rows=2)
    _DATA = {s: d for s, d in data.items() if len(d.loc[DEFAULT_START:DEFAULT_END]) >= 2}


def _methods(n_fixed, n_pct):
    return (("定額", n_fixed, {"sizing_mode": "fixed", "min_invest": INIT_CASH / n_fixed}),
            ("比例", n_pct, {"sizing_mode": "percent_floor", "invest_ratio": 1.0 / n_pct,
                             "min_invest": PCT_MIN_INVEST}))


def run_case(key: str, n_fixed: int, n_pct: int) -> pd.DataFrame:
    """一組：全期 6 格的統計 ＋ 2015–2025 的 6 條權益（存檔）。傳 key（組合含 lambda，不能跨進程 pickle）。"""
    case = next(c for c in CASES if c["key"] == key)
    t0 = time.time()
    rows, curves = [], {}
    for start, tag in ((DEFAULT_START, "全期"), (WIN_START, "2015")):
        panel = case["make"]().build_panel(_DATA, start, DEFAULT_END)
        prios = prio_panels(_DATA, panel["close"])
        for mname, n_units, kw in _methods(n_fixed, n_pct):
            inst = case["make"](initial_cash=INIT_CASH, **kw)
            for kind in ORDERINGS:
                res = inst.run_panel(panel, prios[kind], want_equity=True)
                if tag == "2015":
                    curves[f"{mname}｜{kind}"] = res["equity"]
                    continue
                s = res["summary"]
                rows.append(common.spec_row(s, key=case["key"], 交易策略=case["label"], 投法=mname,
                                            排序=kind, 份數=n_units, 擋單=res["blocked_orders"])
                            | {"資金倍數": round(s["最終權益"] / INIT_CASH, 2),
                               "最大回撤%": round(s["最大回撤(%)"], 1)})
        del panel, prios
    out = pd.DataFrame(rows)
    os.makedirs(os.path.join(OUT, "runs"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "equity_2015"), exist_ok=True)
    out.to_csv(os.path.join(OUT, "runs", f"{case['key']}.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(curves).to_parquet(os.path.join(OUT, "equity_2015", f"{case['key']}.parquet"))
    print(f"完成 {case['label']}｜{time.time() - t0:.0f} 秒", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--keys", default=None)
    a = ap.parse_args()
    if not os.path.isfile(MC_FILE):
        raise SystemExit(f"找不到 S 表：{MC_FILE}\n先跑 python _04_analysis/candidates/candidates_montecarlo.py")
    mc = pd.read_csv(MC_FILE, encoding="utf-8-sig").set_index("key")
    keys = set(a.keys.split(",")) if a.keys else None
    todo = [c for c in CASES if not keys or c["key"] in keys]

    t0 = time.time()
    frames = []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(a.limit,)) as pool:
        futs = {pool.submit(run_case, c["key"], int(mc.loc[c["key"], "定額份數"]),
                            int(mc.loc[c["key"], "比例份數"])): c for c in todo}
        for f in as_completed(futs):
            frames.append(f.result())
            print(f"  已完成 {len(frames)}/{len(todo)}｜{time.time() - t0:.0f} 秒", flush=True)
    if not keys:
        order = {c["key"]: i for i, c in enumerate(CASES)}
        out = pd.concat(frames).sort_values("key", key=lambda s: s.map(order), kind="stable")
        common.assert_spec_columns(out)
        out.to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")
        print(f"→ {RESULT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
