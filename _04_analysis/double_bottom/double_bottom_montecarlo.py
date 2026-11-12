"""
雙底（七）：組合排名前六的蒙地卡羅壓測（分析段，不跑回測）
=========================================================
讀 combo_sweep 已存下的逐筆交易（<tag>/..._trades.csv），只做 ZigZag 6%｜可成交 這一格
（用戶 2026-11-07：取排名最好的六組進蒙地卡羅、只跑 6% 可成交）。排名見 combo_ranking.py。

口徑與 MACD（十）`_04_analysis/macd/macd_montecarlo.py` 一致（用戶 2026-11-07 定：照 MACD（十）抽後放回）：
- 部署總筆數區間 T：每個交易日約 3–4 筆 × 5,949 個交易日 → T ∈ [17,847, 23,796]。
- 歷史筆數 ≥ 下限 → 每條路徑隨機抽一個 T、有放回抽 T 筆；不足下限 → T = 歷史筆數（全納），一樣有放回。
- 每組 10,000 條路徑；起始 100 萬、定額累加不複利；破產＝權益跌破 50 萬。

執行：python _04_analysis/double_bottom/double_bottom_montecarlo.py [--drawdown]
輸出：result/single_double_bottom_combo/_montecarlo_top6.csv
      加 --drawdown：改取 combo_drawdown_filter 的逐格排名（門檻 × 進場 × 出場），跳過交易數 < T_LOW 的格子，
      取前五 → _montecarlo_drawdown_top5.csv；每組用自己那一格的逐筆交易
      （用戶 2026-11-11：〈雙底（八）〉不固定 6%，「如果成交量不符合預期就順序往下」）
"""
import os
import sys
import time

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd  # noqa: E402

from _02_strategy.base.vbt.common import summarize_trades  # noqa: E402
from _02_strategy.double_bottom_strategy.combo_sweep import SUB, _tag  # noqa: E402
from _02_strategy.double_bottom_strategy.single_double_bottom_strategy import RESULT_DIR  # noqa: E402
from _04_analysis.analyze_vbt import monte_carlo  # noqa: E402
from _04_analysis.double_bottom.combo_ranking import rank_combos  # noqa: E402

TOP_N = 6
TOP_N_DRAWDOWN = 5
PCT, LIQ = 0.06, True
PATHS = 10_000
INIT_CASH = 1_000_000.0
RUIN_RATIO = 0.5
TRADING_DAYS = 5_949
T_LOW, T_HIGH = TRADING_DAYS * 3, TRADING_DAYS * 4
OUT = os.path.join(RESULT_DIR, SUB, "_montecarlo_top6.csv")
DRAWDOWN_RANK = os.path.join(RESULT_DIR, SUB, "_drawdown_filter_rank.csv")
DRAWDOWN_CELLS = os.path.join(RESULT_DIR, SUB, "_drawdown_filter_cells.csv")
OUT_DRAWDOWN = os.path.join(RESULT_DIR, SUB, "_montecarlo_drawdown_top5.csv")


def trades_path(entry: str, exit_key: str, pct: float = PCT) -> str:
    tag = _tag(pct, LIQ, entry, exit_key)
    return os.path.join(RESULT_DIR, SUB, tag, f"single_double_bottom_{tag}_trades.csv")


def article_row(rank: int, entry: str, exit_key: str, trades: pd.DataFrame, mc: dict, pct: float = PCT) -> dict:
    """欄位同 MACD（十）蒙地卡羅表；勝率／平均獲利／平均虧損是逐筆統計，不是蒙地卡羅產物。"""
    s = summarize_trades(trades)
    exact = {k: (mc[f"最終資金_{k}"] - INIT_CASH) / INIT_CASH * 100 for k in ("P5", "中位", "P95")}
    win, loss = s["平均獲利報酬率(%)"], s["平均虧損報酬率(%)"]
    return {"名次": rank, "門檻%": round(pct * 100), "進場": entry, "出場": exit_key, "交易數": mc["歷史筆數"],
            "抽樣模式": "抽區間" if mc["歷史筆數"] >= T_LOW else "全納",
            "報酬% P5": round(exact["P5"], 1), "報酬% 中位": round(exact["中位"], 1),
            "報酬% P95": round(exact["P95"], 1), "信賴區間寬度": round(exact["P95"] - exact["P5"], 1),
            "破產%": mc[f"破產機率(<{RUIN_RATIO:.0%})"],
            "最大連敗 P95": mc["最大連敗_P95"], "最大回撤% P95": mc["最大回撤%_P95"],
            "勝率%": s["勝率(%)"], "平均獲利%": win, "平均虧損%": loss,
            "賺賠比": round(win / abs(loss), 2) if loss else None,
            "報酬% P5_精確": exact["P5"], "報酬% 中位_精確": exact["中位"], "報酬% P95_精確": exact["P95"]}


def main() -> int:
    # --drawdown：改用回檔篩選後的逐格排名（combo_drawdown_filter.py），只取交易數到抽樣下限的格子
    filtered = "--drawdown" in sys.argv[1:]
    out = OUT_DRAWDOWN if filtered else OUT
    t0 = time.time()
    n = TOP_N_DRAWDOWN if filtered else TOP_N
    if filtered:
        cells = pd.read_csv(DRAWDOWN_CELLS)
        top = cells[cells["夠筆數"]].head(n)
    else:
        top = rank_combos().head(n).assign(**{"門檻%": round(PCT * 100)})
    print(f"前 {n} 組 × {PATHS:,} 條路徑｜T ∈ [{T_LOW:,}, {T_HIGH:,}]", flush=True)
    rows = []
    for _, r in top.iterrows():
        pct = r["門檻%"] / 100
        path = trades_path(r["進場"], r["出場"], pct)
        if not os.path.isfile(path):
            raise SystemExit(f"找不到逐筆交易：{path}（先跑 combo_sweep）")
        trades = pd.read_csv(path)
        # 淨損益剛好 0 的交易不算勝負（summarize_trades 也排除），筆數才會跟主表一致
        trades = trades[trades["real_pnl"] != 0]
        mc = monte_carlo(trades, initial_cash=INIT_CASH, n_sims=PATHS,
                         ruin_ratio=RUIN_RATIO, t_low=T_LOW, t_high=T_HIGH)
        rows.append(article_row(int(r["名次"]), r["進場"], r["出場"], trades, mc, pct))
        print(f"  #{r['名次']} {r['門檻%']}% {r['進場']} × {r['出場']}：{mc['歷史筆數']:,} 筆｜{time.time() - t0:.0f} 秒", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"→ {out}")
    print(df.drop(columns=[c for c in df.columns if c.endswith("_精確")]).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
