"""
多股引擎的正確性錨點：關掉資金限制後必須等於單股回測
=========================================================
多股引擎多了「共用現金、先賣後買、錢不夠就擋單、買入優先序」這些東西，這些正是
最容易寫錯的地方，而正式結果裡看不出來——數字本來就該跟單股不一樣。

對帳方法：**把資金限制關掉**（本金給到天文數字）、**每筆投入與單股一致**，
各檔之間就不再互相搶資金，兩邊算的是同一件事，逐欄必須完全相同。
任何一欄對不上就是引擎有問題（先賣後買的順序、出場配對、同根衝突、檔位或費用），
這時不要拿正式結果去讀。

⚠️ 出場規則要挑單股與多股都支援的：`MultiMACD` 的出場固定「跌破年線」取代原生出場，
所以單股那邊也要設成同一條（EXIT="ma200"、EXIT_MODE="replace"）。

執行（先用 300 檔快檢，再視需要跑全市場）：
    python _03_multi_strategy/macd/verify_multi_macd.py [--limit 300]
"""
import argparse
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd

from _02_strategy.base.vbt import common
from _03_multi_strategy.macd.multi_macd import STRATEGIES, MultiMACD
from _03_multi_strategy.macd.macd_multi_driver import load_all
from _02_strategy.macd_strategy.macd_sweep import prepare, variant_trades

UNLIMITED = 1e15          # 本金給到永遠花不完＝關掉資金限制
PER_TRADE = 10_000.0      # 每筆投入；與單股基底的 split_cash 預設一致

# 必須**完全相同**：進出場的時點與配對只要有一處不同，這兩欄立刻走鐘，
# 是這份對帳真正在守的東西。
EXACT = ["交易次數", "平均持有天數", "中位數報酬率(%)"]

# 容許極小的差。原因是兩邊「每筆投入 1 萬」的定義不同——多股的預算要**連手續費一起
# 裝進 1 萬**（shares = floor(target /(price×(1+費率)))），單股是 1 萬全部買股票、手續費
# 另計，所以多股每筆股數偶爾少一股。股數一變，損益貼著 0 的那幾筆可能翻邊或剛好變成
# 0（summarize_trades 會排除淨損益 0 的交易），勝率與平均報酬率就跟著差個 0.01。
# 這是設計差異不是引擎錯誤。時點若真的寫錯，這些欄位會差好幾個百分點、不會卡在容許值內。
CLOSE = {"勝率(%)": 0.001, "平均獲利報酬率(%)": 0.001, "平均虧損報酬率(%)": 0.001,
         "期望報酬值(EV)": 0.01, "獲利因子(PF)": 0.01, "總獲利": 0.01}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    a = ap.parse_args()

    single_data = prepare(limit=a.limit)          # 單股用：欄位先備妥
    multi_data = load_all(a.limit)                # 多股用：原始 OHLCV
    print(f"對帳 {len(single_data)} 檔（單股）／{len(multi_data)} 檔（多股）"
          f"｜每筆 {PER_TRADE:,.0f}｜資金上限關閉", flush=True)

    bad = 0
    for label, base, entry in STRATEGIES:
        _, single = variant_trades(single_data, base, entry, "ma200", "replace")
        m = MultiMACD(initial_cash=UNLIMITED, sizing_mode="fixed",
                      min_invest=PER_TRADE)
        m.BASE, m.ENTRY, m.PRIO = base, entry, "low_price"
        multi = m.run(multi_data)["summary"]
        diff = [f"{k}: 多股 {multi[k]} vs 單股 {single[k]}"
                for k in EXACT if multi[k] != single[k]]
        for k, tol in CLOSE.items():
            a_, b_ = float(multi[k]), float(single[k])
            if b_ and abs(a_ - b_) / abs(b_) > tol:
                diff.append(f"{k}: 多股 {a_} vs 單股 {b_}（差 "
                            f"{abs(a_ - b_) / abs(b_):.2%} > 容許 {tol:.0%}）")
        print(f"[{'OK' if not diff else 'NG'}] {label}"
              f"（{single['交易次數']:,} 筆）", flush=True)
        for d in diff:
            print("    " + d)
            bad += 1
    print(f"\n不一致的欄位共 {bad} 個")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
