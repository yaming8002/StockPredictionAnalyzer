"""
單一均線突破：賣出日成交量檢查（分析段，不跑回測）
====================================================
reference（blog/reference/single_ma/ opt11、opt12、summary、master）的「賣出流動性體檢」表：
對某個變體的每一筆交易，取「賣出當天的成交量（張＝1000 股）」看分布——
回測假設隔日開盤一定賣得掉，賣出日只有幾十張的交易在實盤其實出不了場。

讀 `_02_strategy/ma_strategy/single_ma_sweep.py` 存的逐筆交易 parquet，
到股價 parquet 查賣出日成交量。平均值會被成交量異常值灌爆，所以只列中位數與分位數。

輸出：result/single_ma/_sell_volume_<變體>.csv ＋ 終端表。
執行：python _04_analysis/single_ma/single_ma_sell_volume.py [--variant opt11] [--ma 20 200]
"""
import argparse
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _02_strategy.base.vbt import common  # noqa: E402

RESULT = os.path.join(common.result_dir("ma_strategy", "single_ma"))
THRESHOLDS = (50, 100, 300, 1000)        # 張


def sell_day_volume(trades: pd.DataFrame) -> np.ndarray:
    """每筆交易賣出當天的成交量（張）；按股票分組，一檔只讀一次 parquet。"""
    out = np.full(len(trades), np.nan)
    for sid, idx in trades.groupby("stock_id").groups.items():
        vol = pd.read_parquet(os.path.join(common.DATA_DIR, f"{sid}.parquet"),
                              columns=["volume"])["volume"]
        days = pd.to_datetime(trades.loc[idx, "sell_date"])
        out[trades.index.get_indexer(idx)] = vol.reindex(days).to_numpy() / 1000.0
    if np.isnan(out).any():
        raise ValueError(f"{int(np.isnan(out).sum())} 筆交易的賣出日在股價檔查不到成交量")
    return out


def stats(zhang: np.ndarray) -> dict:
    row = {"筆數": len(zhang), "中位(張)": round(float(np.median(zhang)))}
    for q in (10, 25, 75):
        row[f"p{q}(張)"] = round(float(np.percentile(zhang, q)))
    for t in THRESHOLDS:
        row[f"<{t}張(%)"] = round(float((zhang < t).mean() * 100), 1)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="opt11")
    ap.add_argument("--ma", type=int, nargs="+", default=[20, 200])
    a = ap.parse_args()

    rows = []
    for ma in a.ma:
        path = os.path.join(RESULT, a.variant, f"single_ma_{ma}_trades.parquet")
        if not os.path.isfile(path):
            raise SystemExit(f"找不到 {path}，請先跑 _02_strategy/ma_strategy/single_ma_sweep.py")
        trades = pd.read_parquet(path)
        rows.append({"MA": ma, **stats(sell_day_volume(trades))})
    table = pd.DataFrame(rows).set_index("MA")
    out = os.path.join(RESULT, f"_sell_volume_{a.variant}.csv")
    table.to_csv(out, encoding="utf-8-sig")
    print(f"{a.variant} 賣出日成交量（張）\n{table.T.to_string()}\n→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
