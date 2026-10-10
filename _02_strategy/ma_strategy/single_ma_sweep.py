"""
單一均線突破：全變體 × 全 MA 掃描（單股回測段）
=================================================
`single_ma_strategy.py` 的優化用「註解切換」管理，一次只能跑一個變體；reference
（blog/reference/single_ma/）的對照表卻是 19 個變體 × 7 條均線。照 CLI 一個一個切註解
要手動改檔 19 次、每次重讀全市場，所以這裡把「讀檔 ＋ 算指標」對每條均線只做一次，
再讓所有變體共用同一份備妥的 DataFrame。

**變體條件一律照抄策略檔 buy_signal／sell_signal 的原文、照原檔的行序疊加**
（策略檔是唯一定義；改了那邊要回來同步這裡）。策略檔沒有、只留在 reference 的兩個寫法：
  - opt2 的「量增」：`vol_ma5 > vol_ma20`（reference 2026-06-15 opt2 第 19 行原文）
  - opt4 v1 出場：`dist <= dist.shift(1)` 取代下穿（reference 2026-06-15 opt4 第 16 行原文）
opt5／opt6 疊在 baseline 上（策略檔現行寫法；當年是否疊 #1 沒有紀錄）。

交易區間＝標準區間 2002～2025，指標在全史上算（起日前當暖身，見 single.py）。

輸出（result/ 不進版控）：
  result/single_ma/<變體>/single_ma_<N>_per_stock.csv、_aggregate.csv   與單一變體 CLI 同格式
  result/single_ma/<變體>/single_ma_<N>_trades.parquet                    逐筆交易（體檢表等分析段用）

執行：
  python _02_strategy/ma_strategy/single_ma_sweep.py [--ma 20 200] [--variants opt11 opt12] [--limit N]
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

from _02_strategy.base.vbt import batch, common  # noqa: E402
from _02_strategy.base.vbt.common import DEFAULT_END, DEFAULT_START, GLITCH  # noqa: E402
from _02_strategy.ma_strategy.single_ma_strategy import (  # noqa: E402
    DEFAULT_PERIODS, RESULT_DIR, SingleMAStrategy)

# 變體 → (進場優化編號, 出場, 參數)。進場編號依策略檔 buy_signal 的行序套用，與列出順序無關。
VARIANTS = {
    "baseline": ((), "cross", {}),
    "opt1": ((1,), "cross", {}),
    "opt2": ((1, "2vol"), "cross", {}),
    "opt2b": ((1, 2), "cross", {}),
    "opt3": ((1, 3), "cross", {}),
    "opt4": ((1,), "shrink3", {}),          # v2：下穿 或 距離連 3 日縮水
    "opt4v1": ((1,), "dist_v1", {}),        # v1：距離不再擴大就賣（取代下穿）
    "opt5": ((5,), "cross", {}),
    "opt6": ((6,), "cross", {}),
    "opt7": ((7,), "cross", {}),
    "opt8": ((7, 1), "cross", {}),
    "opt9": ((7, 9), "cross", {}),
    "opt10": ((10,), "cross", {"adx_max": 25}),
    "opt10b": ((10,), "cross", {"adx_max": 20}),
    "opt11": ((7, 10), "cross", {"adx_max": 20}),
    "opt12b": ((7, 10, 12), "cross", {"adx_max": 20, "liq_min": 300_000}),
    "opt12": ((7, 10, 12), "cross", {"adx_max": 20, "liq_min": 1_000_000}),
    "opt13": ((13,), "cross", {}),
    "opt13_liq": ((13, 12), "cross", {"liq_min": 1_000_000}),
}


class SingleMAVariant(SingleMAStrategy):
    """
    以設定值決定疊哪些優化（取代註解切換），條件文字照抄策略檔。
    df 由 prepare_stock 備妥（已 add_columns），這裡的 add_columns 不再重算。
    """

    OPTS = ()
    EXIT = "cross"
    ADX_MAX = 20
    LIQ_MIN = 300_000

    def add_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def buy_signal(self, df: pd.DataFrame) -> pd.Series:
        opts = self.OPTS
        above = df["close"] > df["ma"]
        signal = above & ~above.shift(1, fill_value=False)
        if 7 in opts:
            signal = signal & (df["long_red"] | df["gap_over_ma"])
        if 10 in opts:
            signal = signal & (df["adx"].shift(1) < self.ADX_MAX)
        if 13 in opts:
            breakout_ctx = df["had_breakout"].shift(1)
            touch = df["low_near_ma"].shift(1)
            signal = breakout_ctx & touch & (df["close"] > df["open"]) & (df["body"] > df["body"].shift(1)) & (df["close"] > df["ma"])
        if 12 in opts:
            signal = signal & (df["vol_ma5"] > self.LIQ_MIN)
        if 1 in opts:
            signal = above & signal.shift(1, fill_value=False)
        if 9 in opts:
            signal = signal.shift(1, fill_value=False) & (df["open"] > df["close"]) & above
        if "2vol" in opts:
            signal = signal & (df["vol_ma5"] > df["vol_ma20"]) & (df["vol_ma5"] > 1_000_000)
        if 2 in opts:
            signal = signal & (df["vol_ma5"] > 1_000_000)
        if 3 in opts:
            signal = signal & (df["cmf"] > 0.1)
        if 5 in opts:
            signal = signal & df["consolidated"] & (df["volume"] > df["vol_man"])
        if 6 in opts:
            signal = signal & df["below_enough"] & (df["volume"] > df["vol_man"])
        return signal

    def sell_signal(self, df: pd.DataFrame) -> pd.Series:
        dist = df["close"] - df["ma"]
        if self.EXIT == "dist_v1":
            return dist <= dist.shift(1)
        below = df["close"] < df["ma"]
        signal = below & ~below.shift(1, fill_value=False)
        if self.EXIT == "shrink3":
            shrink = dist < dist.shift(1)
            signal = signal | (shrink & shrink.shift(1, fill_value=False) & shrink.shift(2, fill_value=False))
        return signal


def make_variant(name: str, ma: int) -> SingleMAVariant:
    opts, exit_rule, params = VARIANTS[name]
    v = SingleMAVariant()
    v.MA_PERIOD, v.OPTS, v.EXIT = ma, opts, exit_rule
    v.ADX_MAX = params.get("adx_max", SingleMAVariant.ADX_MAX)
    v.LIQ_MIN = params.get("liq_min", SingleMAVariant.LIQ_MIN)
    return v


def run_ma(ma: int, names: list, folder: str, limit: int) -> str:
    """一條均線：逐檔讀全史、算一次指標，所有變體共用；結果依變體落地。"""
    t0 = time.time()
    prep = SingleMAStrategy()
    prep.MA_PERIOD = ma
    variants = {n: make_variant(n, ma) for n in names}
    rows = {n: [] for n in names}
    trades = {n: [] for n in names}
    n_stock = 0
    for sid, df in common.iter_market(folder, limit=limit, exclude=GLITCH):
        if len(df.loc[DEFAULT_START:DEFAULT_END]) < 2:
            continue
        common.ensure_columns(df)
        df = prep.add_columns(df.copy())
        n_stock += 1
        for n, v in variants.items():
            res = v.run(df, sid, start=DEFAULT_START, end=DEFAULT_END)
            rows[n].append({"stock_id": sid, **res["summary"]})
            if not res["trades"].empty:
                trades[n].append(res["trades"])
    for n in names:
        tr = pd.concat(trades[n], ignore_index=True) if trades[n] else pd.DataFrame()
        result = {"per_stock": pd.DataFrame(rows[n]), "trades": tr,
                  "aggregate": batch._aggregate(tr, n_stocks=len(rows[n]), n_failed=0)}
        out_dir = os.path.join(RESULT_DIR, "single_ma", n)
        batch.write_results(result, out_dir, f"single_ma_{ma}")
        tr.to_parquet(os.path.join(out_dir, f"single_ma_{ma}_trades.parquet"), index=False)
    return f"MA{ma}｜{n_stock} 檔 × {len(names)} 變體｜{time.time() - t0:.0f} 秒"


def main() -> int:
    ap = argparse.ArgumentParser(description="單一均線突破：全變體 × 全 MA 掃描")
    ap.add_argument("--ma", type=int, nargs="+", default=list(DEFAULT_PERIODS))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--folder", default=common.DATA_DIR)
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 檔（冒煙用；會覆寫正式結果）")
    ap.add_argument("--workers", type=int, default=7, help="平行跑幾條均線")
    a = ap.parse_args()

    print(f"{len(a.variants)} 變體 × MA {a.ma}｜{DEFAULT_START}~{DEFAULT_END}", flush=True)
    with ProcessPoolExecutor(max_workers=min(a.workers, len(a.ma))) as pool:
        futs = [pool.submit(run_ma, ma, a.variants, a.folder, a.limit) for ma in a.ma]
        for f in futs:
            print(f.result(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
