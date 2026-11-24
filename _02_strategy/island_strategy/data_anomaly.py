"""
島型系列的資料異常股票清單：逐檔統計 2002–2025 區間內的三種異常，任一條超過門檻就從島型回測排除。

起因（用戶 2026-11-17）：持有 20 根卻出現持有數年的交易（3135.TW 2005→2021、3321.TW 2009→2014），
追查後是股價資料的問題——島型策略只數 volume>0 的 K 棒，資料缺列或長期量為 0 時，20 根會橫跨數年；
裸版持有 20 根「超過 60 天」的 940 筆（2.7%）貢獻了總淨損益的 28%。用戶選「把有異常的挑出來，重新分析」。

三種異常（每檔各算一個值，ANOMALY_RULES 是唯一門檻定義）：
  max_gap_days   相鄰兩列的最大日曆天間隔 > 20：整段缺列（全市場 99% 分位數是 13 天＝春節）
  zero_vol_moved volume=0 但收盤 ≠ 前一列收盤的天數 ≥ 20：沒有成交卻有價格變動，資料矛盾（95% 分位數 16）
  max_zero_run   連續 volume=0 的最長列數 > 60：連續約 3 個月以上沒有任何成交（停止買賣、長期停牌一類）
只在島型系列使用，不動 common.GLITCH（跨策略回測宇宙另議，見 session/_open_mode4.md）。
不含「單日價格破表」（mode 1：90 檔單日破表故意保留），例如 3114.TWO 2025-04 一個月 −99% 不在清單內。

2026-11-23 改版（股價資料已依證交所／櫃買中心官方資料修過，見 docs/reference/data_quality_known_issues.md）：
  修完後三種異常只剩 22 檔「缺列 > 20 天」，全是官方確認的真實停牌，用戶裁定照常回測，
  所以島型母體不再用 ANOMALY_RULES 排除（scan() 仍輸出三項統計供檢查）。
  島型母體＝全部資料檔扣掉 island_exclude()（唯一定義）：
    1. DATA_INSUFFICIENT：官方修復後 2002–2025 剩不到 250 列的 16 檔（照抄 lab backtest_untestable.csv「資料不足」；
       只限修過的股票——2024–2025 新上市股列數少但資料正常，照常納入）
    2. DATA_ERROR：修完仍有單日價格翻倍、未經查證的 2 檔（用戶 2026-11-23：「3666 和 6283 不納入算是異常資訊」）
  common.GLITCH 另外 3 檔（3591／8039／8027）已換成官方資料、價格正常，島型納入（只在島型系列，不動 common.GLITCH）。
  → 島型母體 2,161 − 16 − 2＝2,143 檔。

執行（輸出逐檔統計＋是否異常）：
    STOCK_DATA_DIR=... python _02_strategy/island_strategy/data_anomaly.py
  → result/data_anomaly_scan.csv
"""
import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd

from _02_strategy.base.vbt import batch, common
from _02_strategy.base.vbt.common import DEFAULT_START, DEFAULT_END

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result", "data_anomaly_scan.csv")
ANOMALY_RULES = {"max_gap_days": 20, "zero_vol_moved": 19, "max_zero_run": 60}   # 統計值 > 門檻＝異常
# 官方修復後資料不足（模式 3 2026-11-23；SPA 是公開 repo 不讀 lab 檔，清單照抄，lab 改了要同步）
DATA_INSUFFICIENT = frozenset({
    "1780.TWO", "2809.TW", "2938.TWO", "3135.TW", "3158.TWO", "3485.TWO", "4166.TWO", "4804.TWO", "5481.TWO", "6028.TWO", "6423.TWO", "6474.TWO", "6498.TWO", "6597.TWO", "6620.TWO", "8291.TWO",
})
DATA_ERROR = {
    "3666.TWO": "2022-05-05 單日 +100%，不在官方修復清單",
    "6283.TW": "2016-10-21 單日 +102%，修復後仍在，未查證",
}


def longest_run(flags: np.ndarray) -> int:
    """布林陣列中連續 True 的最長長度。"""
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def scan_one(df: pd.DataFrame) -> dict:
    """單檔在標準區間內的三項統計。"""
    seg = df.loc[DEFAULT_START:DEFAULT_END]
    if len(seg) < 2:
        return {"rows": len(seg), "max_gap_days": 0, "gap_at": None, "max_zero_run": 0,
                "zero_vol_moved": 0, "zero_vol": 0}
    zero = (seg["volume"] == 0).to_numpy()
    moved = zero & (seg["close"] != seg["close"].shift(1)).to_numpy()
    gaps = seg.index.to_series().diff().dt.days
    return {"rows": len(seg), "max_gap_days": int(gaps.max()), "gap_at": gaps.idxmax().date(),
            "max_zero_run": longest_run(zero), "zero_vol_moved": int(moved.sum()), "zero_vol": int(zero.sum())}


def scan(folder: str = None) -> pd.DataFrame:
    """全部資料檔逐檔統計＋abnormal（三種異常）／insufficient（資料不足）／data_error 欄。"""
    rows = []
    for path in batch.list_parquet(folder or common.DATA_DIR):
        sid = os.path.splitext(os.path.basename(path))[0]
        rows.append({"stock_id": sid, **scan_one(batch._load_prices(path))})
    df = pd.DataFrame(rows)
    df["abnormal"] = np.logical_or.reduce([df[c] > th for c, th in ANOMALY_RULES.items()])
    df["insufficient"] = df["stock_id"].isin(DATA_INSUFFICIENT)
    df["data_error"] = df["stock_id"].isin(DATA_ERROR)
    return df


def island_exclude(folder: str = None) -> frozenset:
    """島型母體要排除的股票（2026-11-23 起的唯一定義）：DATA_INSUFFICIENT ∪ DATA_ERROR。"""
    df = scan(folder)
    present = set(df["stock_id"])
    return frozenset((DATA_INSUFFICIENT | frozenset(DATA_ERROR)) & present)


def anomaly_set(folder: str = None) -> set:
    """三種異常的股票代號（2026-11-17～11-22 舊母體用；資料修過後不再拿來排除，留作檢查）。"""
    df = scan(folder)
    return set(df.loc[df["abnormal"], "stock_id"])


def main() -> int:
    df = scan()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"{len(df)} 檔，異常 {int(df['abnormal'].sum())} 檔 → {OUT}")
    for c, th in ANOMALY_RULES.items():
        print(f"  {c} > {th}：{int((df[c] > th).sum())} 檔")
    excl = sorted((DATA_INSUFFICIENT | set(DATA_ERROR)) & set(df["stock_id"]))
    print(f"島型排除 {len(excl)} 檔（資料不足 {int(df['insufficient'].sum())}＋DATA_ERROR {len(DATA_ERROR)}）"
          f"→ 母體 {len(df) - len(excl)} 檔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
