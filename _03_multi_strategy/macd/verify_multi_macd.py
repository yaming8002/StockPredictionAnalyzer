"""
多股引擎的正確性錨點：關掉資金限制後必須等於單股回測
=========================================================
多股引擎多了「共用現金、先賣後買、錢不夠就擋單、買入優先序」這些東西，這些正是
最容易寫錯的地方，而正式結果裡看不出來——數字本來就該跟單股不一樣。

對帳方法：**把資金限制關掉**（本金給到天文數字）、**每筆投入與單股一致**，
各檔之間就不再互相搶資金，兩邊算的是同一件事，逐筆交易（股票、買賣日、買賣價）必須完全相同。
任何一筆對不上就是引擎有問題（先賣後買的順序、出場配對、同根衝突、檔位），
這時不要拿正式結果去讀。股數與淨損益的設計差異見下方 PNL_TOL 註解。

⚠️ 出場規則要挑單股與多股都支援的：`MultiMACD` 的出場固定「跌破年線」取代原生出場，
所以單股那邊也要設成同一條（EXIT="ma200"、EXIT_MODE="replace"）。

執行（預設全市場 2,258 檔，排 GLITCH；--limit N 只取前 N 檔快檢，0 也視為全市場）：
    python _03_multi_strategy/macd/verify_multi_macd.py [--limit N]
"""
import argparse
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd

from _02_strategy.base.vbt import common
from _02_strategy.base.vbt.common import DEFAULT_END, DEFAULT_START
from _03_multi_strategy.macd.multi_macd import STRATEGIES, MultiMACD
from _03_multi_strategy.macd.macd_multi_driver import load_all
from _02_strategy.macd_strategy.macd_sweep import prepare, variant_trades

UNLIMITED = 1e15          # 本金給到永遠花不完＝關掉資金限制
PER_TRADE = 10_000.0      # 每筆投入；與單股基底的 split_cash 預設一致

# 對帳以**逐筆交易清單**為準（2026-10-11 改）：時點與配對只要有一處不同就是引擎問題，
# 所以「股票、買進日、賣出日、買價、賣價」必須逐筆完全相同。
# 舊版比的是 summarize_trades 的交易次數等欄位，但它會排除淨損益剛好 0 的交易；
# 全市場有幾筆單股 0 元／多股 −0.5 元這種貼在 0 的交易，筆數就差 1，誤判成不一致。
MATCH = ["stock_id", "buy_date", "sell_date", "buy_price", "sell_price"]

# 股數與損益是**設計差異**：多股的預算要連手續費一起裝進 1 萬
# （shares = floor(target /(price×(1+費率)))），單股是 1 萬全部買股票、手續費另計，
# 所以多股股數只會「≤ 單股」（低價股可能少好幾股）。股數相同的交易，淨損益只容許捨入差。
PNL_TOL = 1.0             # 元


def compare_trades(single: pd.DataFrame, multi: pd.DataFrame) -> tuple:
    """逐筆對帳；回傳 (問題清單, 股數不同的筆數)。"""
    a = single.sort_values(MATCH).reset_index(drop=True)
    b = multi.sort_values(MATCH).reset_index(drop=True)
    problems = []
    if len(a) != len(b) or not a[MATCH].astype(str).equals(b[MATCH].astype(str)):
        diff = a[MATCH].astype(str).merge(b[MATCH].astype(str), how="outer", indicator=True)
        diff = diff[diff["_merge"] != "both"]
        problems.append(f"交易清單不同：單股 {len(a):,} 筆／多股 {len(b):,} 筆，"
                        f"配不上 {len(diff):,} 筆\n{diff.head(10).to_string()}")
        return problems, 0
    qa, qb = a["qty"].astype(int), b["qty"].astype(int)
    if (qb > qa).any():
        problems.append(f"多股股數大於單股 {int((qb > qa).sum())} 筆（設計上只會 ≤）")
    same_qty = qa == qb
    pnl_gap = (a["real_pnl"].astype(float) - b["real_pnl"].astype(float)).abs()
    n_pnl = int((same_qty & (pnl_gap >= PNL_TOL)).sum())
    if n_pnl:
        problems.append(f"股數相同但淨損益差 ≥ {PNL_TOL:.0f} 元 {n_pnl} 筆"
                        f"（最大差 {pnl_gap[same_qty].max():.2f}）")
    return problems, int((~same_qty).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只取前 N 檔快檢；省略或 0＝全市場")
    a = ap.parse_args()
    limit = a.limit or None

    single_data = prepare(limit=limit)            # 單股用：欄位先備妥
    multi_data = load_all(limit)                  # 多股用：原始 OHLCV
    print(f"對帳 {len(single_data)} 檔（單股）／{len(multi_data)} 檔（多股）"
          f"｜每筆 {PER_TRADE:,.0f}｜資金上限關閉", flush=True)

    bad = 0
    for label, base, entry in STRATEGIES:
        single, _ = variant_trades(single_data, base, entry, "ma200", "replace")
        m = MultiMACD(initial_cash=UNLIMITED, sizing_mode="fixed",
                      min_invest=PER_TRADE)
        m.BASE, m.ENTRY, m.PRIO = base, entry, "low_price"
        res = m.run(multi_data, DEFAULT_START, DEFAULT_END)
        problems, n_qty = compare_trades(single, res["trades"])
        print(f"[{'OK' if not problems else 'NG'}] {label}｜逐筆 {len(single):,} 筆"
              f"｜股數不同（設計差異）{n_qty:,} 筆｜擋單 {res['blocked_orders']}", flush=True)
        for p in problems:
            print("    " + p)
        bad += bool(problems)
    print(f"\n不一致 {bad} 組")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
