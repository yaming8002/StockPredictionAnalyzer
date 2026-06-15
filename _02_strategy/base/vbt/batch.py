"""
vbt 策略套件 batch — 指定資料夾的批次回測層
================================================

single.py 只跑單檔；本檔補上「掃整個資料夾、每檔各自獨立回測、彙總」這層，
可重用給任何 VbtSingleStrategy 子類（全市場掃描用）。

職責：
  - run_folder：逐檔讀 parquet → strategy.run(df, stock_id) → 收每檔 summary + 彙總所有交易
  - write_results：把彙總 / 每檔結果落地成 CSV（可選逐筆交易）

資料夾位置不限：由呼叫端傳入路徑（讀哪、寫哪由上層決定）。
"""
import glob
import os

import pandas as pd

from _02_strategy.base.vbt import common


def list_parquet(folder: str) -> list:
    """列資料夾下所有 .parquet（排序）。"""
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"找不到資料夾: {folder}")
    return sorted(glob.glob(os.path.join(folder, "*.parquet")))


def _load_prices(parquet_path: str, start=None, end=None) -> pd.DataFrame:
    """讀單檔 OHLCV parquet；index 須為 DatetimeIndex；可選日期區間裁切。"""
    df = pd.read_parquet(parquet_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("index 必須是 DatetimeIndex（日期）")
    df = df.sort_index()
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df


def run_folder(strategy, folder: str, start=None, end=None, limit: int = None) -> dict:
    """
    對資料夾所有 parquet 各自獨立跑同一個 strategy，彙總結果。

    回傳 {"per_stock": DataFrame, "aggregate": dict, "trades": DataFrame, "failed": list}。
    單檔失敗（缺欄 / 壞檔 / 區間內無資料）記進 failed、不中斷整批（不靜默吞）。
    """
    files = list_parquet(folder)
    if limit is not None:
        files = files[:limit]

    per_stock_rows = []
    all_trades = []
    failed = []

    for path in files:
        stock_id = os.path.splitext(os.path.basename(path))[0]
        try:
            df = _load_prices(path, start, end)
            if df.empty:
                failed.append((stock_id, "區間內無資料"))
                continue
            res = strategy.run(df, stock_id=stock_id)
        except Exception as exc:  # 單檔失敗不影響整批，但要記錄下來
            failed.append((stock_id, str(exc)))
            continue

        per_stock_rows.append({"stock_id": stock_id, **res["summary"]})
        if not res["trades"].empty:
            all_trades.append(res["trades"])

    per_stock = pd.DataFrame(per_stock_rows)
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    aggregate = _aggregate(trades, n_stocks=len(per_stock_rows), n_failed=len(failed))
    return {"per_stock": per_stock, "aggregate": aggregate,
            "trades": trades, "failed": failed}


def _aggregate(trades: pd.DataFrame, n_stocks: int, n_failed: int) -> dict:
    """把所有股票的交易匯成一份全市場 summary，並補上股票數 / 失敗數。"""
    agg = common.summarize_trades(trades)
    return {"參與股票數": n_stocks, "失敗檔數": n_failed, **agg}


def write_results(result: dict, out_dir: str, label: str, write_trades: bool = False) -> list:
    """
    落地結果到 out_dir：
      <label>_per_stock.csv  每檔一列
      <label>_aggregate.csv  全市場彙總一列
      <label>_trades.csv     僅 write_trades=True 才輸出
    回傳已寫出的檔案路徑 list。
    """
    os.makedirs(out_dir, exist_ok=True)
    written = []

    per_stock_path = os.path.join(out_dir, f"{label}_per_stock.csv")
    result["per_stock"].to_csv(per_stock_path, index=False, encoding="utf-8-sig")
    written.append(per_stock_path)

    agg_path = os.path.join(out_dir, f"{label}_aggregate.csv")
    pd.DataFrame([result["aggregate"]]).to_csv(agg_path, index=False, encoding="utf-8-sig")
    written.append(agg_path)

    if write_trades and not result["trades"].empty:
        trades_path = os.path.join(out_dir, f"{label}_trades.csv")
        result["trades"].to_csv(trades_path, index=False, encoding="utf-8-sig")
        written.append(trades_path)

    return written
