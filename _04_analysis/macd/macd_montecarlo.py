"""
MACD（十）：五組交易策略的蒙地卡羅壓測（分析段）
=================================================
讀回測段 `_02_strategy/macd_strategy/macd_mc_trades.py` 存下的逐筆交易，做蒙地卡羅；
這支不跑回測。五組怎麼挑的見回測段檔頭。口徑與均線、KD 兩個系列一致：

- **交易頻率基準**：每個交易日約 3–4 筆，乘上回測期間的交易日數得「部署總筆數區間 T」。
  2002–2025 約 5,949 個交易日 → T ∈ [17,847, 23,796]。
- **抽樣**：歷史筆數 N ≥ 區間下限 → 每條路徑先隨機抽一個 T，再有放回抽 T 筆；
  N < 下限 → T = N（全部交易納入），一樣有放回重抽，不硬灌到區間、避免虛增筆數。
- 每組跑 10,000 條路徑。
- 帳戶：起始 100 萬、定額累加（每筆固定金額、不複利）。
- **破產**：權益跌破 50 萬（本金的五成）。本金腰斬多半已經撐不住，這條線設早一點當警戒。

這張表的「最大連敗 P95」就是後續多股回測算份數用的 S（見 _03 的 macd_multi_driver）。

執行（先跑回測段，再跑這支）：
    python _02_strategy/macd_strategy/macd_mc_trades.py
    python _04_analysis/macd/macd_montecarlo.py
輸出：result/macd_mc/macd_montecarlo.csv（欄位同第十篇蒙地卡羅表）＋ 終端表。
"""
import os
import sys
import time

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd  # noqa: E402

from _02_strategy.macd_strategy.macd_mc_trades import CASES, OUT, trades_path  # noqa: E402
from _02_strategy.base.vbt.common import summarize_trades  # noqa: E402
from _04_analysis.analyze_vbt import monte_carlo  # noqa: E402

PATHS = 10_000
INIT_CASH = 1_000_000.0
RUIN_RATIO = 0.5              # 破產＝權益跌破本金五成
TRADING_DAYS = 5_949          # 2002-01-01 ~ 2025-12-31 的交易日數
T_LOW, T_HIGH = TRADING_DAYS * 3, TRADING_DAYS * 4


def article_row(label: str, trades: pd.DataFrame, mc: dict) -> dict:
    """
    第十篇蒙地卡羅表的欄位與欄序（與 blog reference 的 macd_matrix_montecarlo.csv 同一格式）。
    報酬% ＝ 最終資金相對本金；勝率／平均獲利／平均虧損是每筆回測統計（summarize_trades），
    不是蒙地卡羅產物；賺賠比 ＝ 平均獲利% ÷ |平均虧損%|。
    """
    s = summarize_trades(trades)
    ret = {k: round((mc[f"最終資金_{k}"] - INIT_CASH) / INIT_CASH * 100, 1)
           for k in ("P5", "中位", "P95")}
    win, loss = s["平均獲利報酬率(%)"], s["平均虧損報酬率(%)"]
    return {"組合": label, "交易數": mc["歷史筆數"],
            "抽樣模式": "抽區間" if mc["歷史筆數"] >= T_LOW else "全納",
            "報酬% P5": ret["P5"], "報酬% 中位": ret["中位"], "報酬% P95": ret["P95"],
            "信賴區間寬度": round(ret["P95"] - ret["P5"], 1),
            "破產%": mc[f"破產機率(<{RUIN_RATIO:.0%})"],
            "最大連敗 P95": mc["最大連敗_P95"], "最大回撤% P95": mc["最大回撤%_P95"],
            "勝率%": s["勝率(%)"], "平均獲利%": win, "平均虧損%": loss,
            "賺賠比": round(win / abs(loss), 2) if loss else None}


def main():
    t0 = time.time()
    print(f"{len(CASES)} 組 × {PATHS:,} 條路徑｜T ∈ [{T_LOW:,}, {T_HIGH:,}]", flush=True)

    rows = []
    for label, base, filt, ex in CASES:
        path = trades_path(base, filt, ex)
        if not os.path.isfile(path):
            raise SystemExit(f"找不到逐筆交易：{path}\n"
                             "先跑回測段：python _02_strategy/macd_strategy/macd_mc_trades.py")
        trades = pd.read_parquet(path)
        # 淨損益剛好 0 的交易不算勝負（summarize_trades 也排除），筆數才會跟矩陣表一致
        trades = trades[trades["real_pnl"] != 0]
        mc = monte_carlo(trades, initial_cash=INIT_CASH, n_sims=PATHS,
                         ruin_ratio=RUIN_RATIO, t_low=T_LOW, t_high=T_HIGH)
        rows.append(article_row(label, trades, mc))
        print(f"  {label}：歷史 {mc['歷史筆數']:,} 筆｜"
              f"最大連敗 P95 {mc['最大連敗_P95']}｜"
              f"最大回撤%中位 {mc['最大回撤%_中位']}｜"
              f"破產 {mc[f'破產機率(<{RUIN_RATIO:.0%})']}%｜"
              f"{time.time() - t0:.0f} 秒", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "macd_montecarlo.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n耗時 {time.time() - t0:.0f} 秒 → {path}")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
