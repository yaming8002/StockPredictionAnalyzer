"""
雙均線交叉單股全變體：正式結果全市場重現（vbt 單股引擎，逐筆比對）
====================================================================
`ma_cross_sweep.py` 為了省時間，每組均線每檔只 add_columns 一次、所有變體共用（MACrossVariant 的
add_columns 是空的）。這支對每個「變體 × 均線組」**各自從原始 K 棒重算**，在全市場 2,258 檔、標準區間
跑完，逐筆比對 result/ma_cross/<變體>/ma_cross_<短>_<長>_trades.parquet（股票、買賣日、買賣價、股數、
費用、淨損益全部要相同），並確認 per_stock.csv 檔數＝實際參與的股票數：

- 旗標全部是策略檔 `MACrossStrategy` 本身旗標的變體（文章與 reference 用的絕大多數）：直接用**策略檔**
  設旗標、照 run() 原流程重算＝讀者照 CLI 跑出來的結果，必須等於存檔。
- 重產區只存在於 driver 的兩個旗標（CHOCH_STATELESS、GAP_BASE）：用 driver 的 MACrossVariant，但每個
  變體各自用策略檔 add_columns（CHOCH=True 才有 ZigZag）備欄，抓「共用備欄」與「單獨跑」不一致與存檔過期。

執行：python _02_strategy/ma_strategy/verify_ma_cross_sweep_full.py [--set all] [--pairs ...] [--workers 6] [--limit N]
    預設全市場、全部變體 × 21 組（部分變體只跑舊 CSV 有的組，同 driver 的 VARIANT_PAIRS）；
    --limit N 只跑前 N 檔（只驗流程，存檔是全市場、無從比對，會略過比對）。
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
from _02_strategy.ma_strategy import ma_cross_sweep as sweep  # noqa: E402
from _02_strategy.ma_strategy.ma_cross_strategy import RESULT_DIR, MACrossStrategy  # noqa: E402

STORED_ROOT = os.path.join(RESULT_DIR, "ma_cross")


def is_teaching(name: str) -> bool:
    return all(hasattr(MACrossStrategy, k) for k in sweep.VARIANTS[name])


def run_one(name: str, short: int, long: int, df: pd.DataFrame, sid: str) -> pd.DataFrame:
    """單一變體、單一檔，從原始 K 棒重算的逐筆交易。"""
    if is_teaching(name):
        s = MACrossStrategy()
        s.SHORT_MA, s.LONG_MA = short, long
        for k, v in sweep.VARIANTS[name].items():
            setattr(s, k, v)
        return s.run(df, sid, start=DEFAULT_START, end=DEFAULT_END)["trades"]
    prep = MACrossStrategy()
    prep.SHORT_MA, prep.LONG_MA, prep.CHOCH = short, long, True     # CHOCH=True 才補算 ZigZag
    prepared = prep.add_columns(df.copy())
    return sweep.make_variant(name, short, long).run(prepared, sid, start=DEFAULT_START,
                                                      end=DEFAULT_END)["trades"]


def check_pair(pair: str, names: list, folder: str, limit) -> list:
    """一組均線的所有變體逐檔重算；回傳 [(設定, 說明, 筆數, 檔數, 問題)]。"""
    short, long = (int(x) for x in pair.split("/"))
    names = [n for n in names if pair in sweep.VARIANT_PAIRS.get(n, sweep.PAIRS)]
    trades = {n: [] for n in names}
    n_stock = 0
    for sid, df in common.iter_market(folder, limit=limit, exclude=GLITCH):
        if len(df.loc[DEFAULT_START:DEFAULT_END]) < 2:
            continue
        common.ensure_columns(df)
        n_stock += 1
        for n in names:
            tr = run_one(n, short, long, df, sid)
            if not tr.empty:
                trades[n].append(tr)
    label = f"ma_cross_{short}_{long}"
    out = []
    for n in names:
        got = pd.concat(trades[n], ignore_index=True) if trades[n] else pd.DataFrame()
        problem = ""
        if limit is None:
            stored = pd.read_parquet(os.path.join(STORED_ROOT, n, f"{label}_trades.parquet"))
            per_stock = pd.read_csv(os.path.join(STORED_ROOT, n, f"{label}_per_stock.csv"))
            if not (stored.empty and got.empty):
                try:
                    pd.testing.assert_frame_equal(stored.reset_index(drop=True), got.reset_index(drop=True),
                                                  check_dtype=False, check_exact=True)
                except AssertionError as exc:
                    problem = f"逐筆不同（存檔 {len(stored):,}／重算 {len(got):,}）：{str(exc).splitlines()[0]}"
            if len(per_stock) != n_stock:
                problem += f"｜per_stock 檔數 {len(per_stock)} ≠ 實際 {n_stock}"
        how = "策略檔" if is_teaching(n) else "driver 變體＋單獨備欄"
        out.append((f"{n} {pair}", how, len(got), n_stock, problem))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="all", choices=list(sweep.SETS))
    ap.add_argument("--variants", nargs="+", default=None, choices=list(sweep.VARIANTS))
    ap.add_argument("--pairs", nargs="+", default=sweep.PAIRS)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--folder", default=common.DATA_DIR)
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 檔（略過比對）；省略或 0＝全市場")
    a = ap.parse_args()
    limit = a.limit or None

    names = a.variants or sweep.SETS[a.set]
    n_cfg = sum(p in sweep.VARIANT_PAIRS.get(n, sweep.PAIRS) for n in names for p in a.pairs)
    print(f"{len(names)} 變體 × {len(a.pairs)} 組＝{n_cfg} 設定（策略檔 "
          f"{sum(is_teaching(n) for n in names)} 變體）｜{DEFAULT_START}~{DEFAULT_END}｜"
          f"{'全市場' if limit is None else f'前 {limit} 檔（不比對）'}｜{a.workers} 進程", flush=True)
    t0 = time.time()
    bad = 0
    with ProcessPoolExecutor(max_workers=min(a.workers, len(a.pairs))) as pool:
        futs = [pool.submit(check_pair, p, names, a.folder, limit) for p in a.pairs]
        for f in futs:
            for n, how, n_tr, n_stock, problem in f.result():
                bad += bool(problem)
                print(f"[{'NG' if problem else 'OK'}] {n}｜{how}｜{n_tr:,} 筆｜{n_stock} 檔"
                      + (f"\n    {problem}" if problem else ""), flush=True)
    print(f"不一致 {bad} 設定｜{time.time() - t0:.0f} 秒")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
