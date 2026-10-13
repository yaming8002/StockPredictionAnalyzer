# -*- coding: utf-8 -*-
"""
KD 單股全變體：正式結果全市場重現（vbt 單股引擎，逐筆比對）
==============================================================
`kd_sweep.py` 為了省時間，每檔只 add_columns 一次、所有變體共用（KdVariantShared）。這支對每個變體
**各自從原始 K 棒重算一次**，在全市場 2,258 檔、標準區間跑完，逐筆比對 result/single_kd/<變體>/
single_kd_trades.parquet（股票、買賣日、買賣價、股數、費用、淨損益全部要相同）：

- 策略檔有對應註解狀態的 12 個變體（baseline、opt1、opt3~9、#2a、#2b…，對照表見
  verify_kd_variants.EQUIV）：用**策略檔 single_kd_strategy.py 本身**（取消註解後 exec）重算——
  也就是公開教學檔照文章切換後跑出來的結果，必須等於存檔。
- 其餘變體（策略檔沒有對應註解行）：用 KdVariant 各自 add_columns 重算。抓的是「共用備欄」與
  「單獨跑」不一致（例如某條件的欄位依門檻參數而變、共用備欄卻用預設參數算），以及存檔過期。

另確認每個變體的 per_stock.csv 檔數＝實際參與的股票數。

執行：python _02_strategy/kd_strategy/verify_kd_sweep_full.py [--set all] [--workers 4] [--limit N]
    預設全市場、全部 121 變體；--limit N 只跑前 N 檔（只驗流程，存檔是全市場、無從比對，會略過比對）。
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd  # noqa: E402

from _02_strategy.base.vbt import common  # noqa: E402
from _02_strategy.base.vbt.common import DEFAULT_END, DEFAULT_START, GLITCH  # noqa: E402
from _02_strategy.kd_strategy import kd_sweep  # noqa: E402
from _02_strategy.kd_strategy.kd_variants import KdVariant  # noqa: E402
from _02_strategy.kd_strategy.single_kd_strategy import RESULT_DIR  # noqa: E402
from _02_strategy.kd_strategy.verify_kd_variants import EQUIV, TOGGLES, toggled_class  # noqa: E402

STORED_ROOT = os.path.join(RESULT_DIR, "single_kd")
TEACHING = {v: k for k, v in EQUIV.items()}     # 變體名 → 策略檔註解狀態


def fresh_strategy(name: str):
    """回傳 (說明, 會自己 add_columns 的策略物件)。"""
    if name in TEACHING:
        state = TEACHING[name]
        return f"策略檔 {state}", toggled_class(TOGGLES[state])()
    spec = kd_sweep.VARIANTS[name]
    v = KdVariant()
    v.ENTRY, v.EXIT = spec["entry"], spec["exit"]
    v.LIQUIDITY, turnover = kd_sweep._LIQ[spec["liq"]]
    if turnover is not None:
        v.TURNOVER = turnover
    if spec["zone"] is not None:
        v.OVERSOLD, v.OVERBOUGHT = spec["zone"]
    return "KdVariant 單獨備欄", v


def check_group(names: list, folder: str, limit) -> list:
    """一組變體逐檔重算；回傳 [(變體, 說明, 筆數, 檔數, 問題)]。"""
    strategies = {n: fresh_strategy(n) for n in names}
    trades = {n: [] for n in names}
    n_stock = 0
    for sid, df in common.iter_market(folder, limit=limit, exclude=GLITCH):
        if len(df.loc[DEFAULT_START:DEFAULT_END]) < 2:
            continue
        common.ensure_columns(df)
        n_stock += 1
        for n, (_, s) in strategies.items():
            tr = s.run(df, sid, start=DEFAULT_START, end=DEFAULT_END)["trades"]
            if not tr.empty:
                trades[n].append(tr)
    out = []
    for n in names:
        got = pd.concat(trades[n], ignore_index=True) if trades[n] else pd.DataFrame()
        problem = ""
        if limit is None:
            stored = pd.read_parquet(os.path.join(STORED_ROOT, n, "single_kd_trades.parquet"))
            per_stock = pd.read_csv(os.path.join(STORED_ROOT, n, "single_kd_per_stock.csv"))
            try:
                pd.testing.assert_frame_equal(stored.reset_index(drop=True), got.reset_index(drop=True),
                                              check_dtype=False, check_exact=True)
            except AssertionError as exc:
                problem = f"逐筆不同（存檔 {len(stored):,}／重算 {len(got):,}）：{str(exc).splitlines()[0]}"
            if len(per_stock) != n_stock:
                problem += f"｜per_stock 檔數 {len(per_stock)} ≠ 實際 {n_stock}"
        out.append((n, strategies[n][0], len(got), n_stock, problem))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="+", default=["all"], choices=sorted(kd_sweep.SETS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--folder", default=common.DATA_DIR)
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 檔（略過比對）；省略或 0＝全市場")
    a = ap.parse_args()
    limit = a.limit or None

    names = list(dict.fromkeys(n for s in a.set for n in kd_sweep.SETS[s]))
    print(f"{len(names)} 變體（策略檔註解狀態 {sum(n in TEACHING for n in names)} 個）｜"
          f"{DEFAULT_START}~{DEFAULT_END}｜{'全市場' if limit is None else f'前 {limit} 檔（不比對）'}"
          f"｜{a.workers} 進程", flush=True)
    t0 = time.time()
    groups = kd_sweep._split(names, max(1, a.workers))
    bad = 0
    with ProcessPoolExecutor(max_workers=len(groups)) as pool:
        futs = [pool.submit(check_group, g, a.folder, limit) for g in groups]
        for f in futs:
            for n, how, n_tr, n_stock, problem in f.result():
                bad += bool(problem)
                print(f"[{'NG' if problem else 'OK'}] {n}｜{how}｜{n_tr:,} 筆｜{n_stock} 檔"
                      + (f"\n    {problem}" if problem else ""), flush=True)
    print(f"不一致 {bad} 變體｜{time.time() - t0:.0f} 秒")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
