"""
均線交叉（四）：21 組的單股實際部署蒙地卡羅＋淨口徑每筆統計（分析段）
=====================================================================
讀回測段 `_02_strategy/ma_strategy/ma_cross_sweep.py` 存的逐筆交易（每檔獨立、每筆 1 萬、
訊號全成交），對 21 組各做一次蒙地卡羅；這支不跑回測。

口徑（與 MACD、KD 系列同一套，常數取自 macd_montecarlo，不另立第二份）：
  10,000 條路徑、起始 100 萬、每筆定額累加；部署總筆數 T ∈ [交易日數×3, ×4]，
  歷史筆數不足下限 → T＝歷史筆數（全納入、有放回重抽）；本金大虧＝權益曾跌破本金 5 成。

「最大連敗 P95」就是多股回測算份數用的 S（見 _03_multi_strategy/ma_cross/ma_cross_multi_driver.py）。

淨口徑每筆統計（net_ev.csv）：勝負依 real_pnl（已扣台股費稅）分；每筆報酬率% ＝
real_pnl ÷ 買進付出的現金（價×股＋買進手續費）。`summarize_trades` 的賺賠比是
「淨分勝負、毛取率」的混合口徑，拿來談每筆期望會高估，所以這裡另算一份全淨口徑。

輸出：result/ma_cross/_mc_realistic/mc_realistic.csv、net_ev.csv
執行：python _04_analysis/ma_cross/ma_cross_montecarlo.py [--variant angle20_adx25_liq1000]
"""
import argparse
import os
import sys
import time

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _02_strategy.base.vbt import common  # noqa: E402
from _02_strategy.ma_strategy.ma_cross_sweep import PAIRS  # noqa: E402
from _04_analysis.analyze_vbt import monte_carlo  # noqa: E402
from _04_analysis.macd.macd_montecarlo import (INIT_CASH, PATHS, RUIN_RATIO,  # noqa: E402
                                               T_HIGH, T_LOW)

RESULT = common.result_dir("ma_strategy", "ma_cross")
OUT = os.path.join(RESULT, "_mc_realistic")
PER_TRADE = 10_000.0


def net_stats(trades: pd.DataFrame) -> dict:
    """全淨口徑：勝率、平均獲利／虧損%、賺賠比、每筆期望%。"""
    outlay = trades["buy_price"] * trades["qty"] + trades["buy_fee"]
    ret = trades["real_pnl"] / outlay * 100.0
    win, loss = ret[trades["real_pnl"] > 0], ret[trades["real_pnl"] < 0]
    p = len(win) / len(ret)
    avg_w = float(win.mean()) if len(win) else 0.0
    avg_l = float(loss.mean()) if len(loss) else 0.0
    return {"勝率%": round(p * 100, 2), "淨平均獲利%": round(avg_w, 2),
            "淨平均虧損%": round(avg_l, 2),
            "淨賺賠比": round(avg_w / abs(avg_l), 2) if avg_l else np.nan,
            "每筆淨期望%": round(float(ret.mean()), 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="angle20_adx25_liq1000")
    a = ap.parse_args()

    t0 = time.time()
    print(f"{a.variant}｜21 組 × {PATHS:,} 條｜T ∈ [{T_LOW:,}, {T_HIGH:,}]", flush=True)
    mc_rows, ev_rows = [], []
    for pair in PAIRS:
        s, l = pair.split("/")
        path = os.path.join(RESULT, a.variant, f"ma_cross_{s}_{l}_trades.parquet")
        if not os.path.isfile(path):
            raise SystemExit(f"找不到 {path}\n先跑：python _02_strategy/ma_strategy/ma_cross_sweep.py")
        trades = pd.read_parquet(path)
        trades = trades[trades["real_pnl"] != 0]          # 與 summarize_trades 同：損益 0 的不算勝負
        mc = monte_carlo(trades, initial_cash=INIT_CASH, n_sims=PATHS,
                         ruin_ratio=RUIN_RATIO, t_low=T_LOW, t_high=T_HIGH)
        ret = {k: round((mc[f"最終資金_{k}"] - INIT_CASH) / INIT_CASH * 100, 1)
               for k in ("P5", "中位", "P95")}
        mc_rows.append({"短/長": pair, "交易數": mc["歷史筆數"],
                        "抽樣模式": "全納入" if mc["歷史筆數"] < T_LOW else "抽區間",
                        "報酬%_P5": ret["P5"], "報酬%_中位": ret["中位"], "報酬%_P95": ret["P95"],
                        "區間寬度%": round(ret["P95"] - ret["P5"], 1),
                        "本金大虧%": mc[f"破產機率(<{RUIN_RATIO:.0%})"],
                        "最大連敗_P95": mc["最大連敗_P95"], "最大回撤%_P95": mc["最大回撤%_P95"]})
        ev_rows.append({"短/長": pair, "交易數": len(trades), **net_stats(trades)})
        print(f"  {pair}：{len(trades):,} 筆｜連敗 P95 {mc['最大連敗_P95']}｜"
              f"報酬中位 {ret['中位']}%｜{time.time() - t0:.0f} 秒", flush=True)

    os.makedirs(OUT, exist_ok=True)
    mc_df = pd.DataFrame(mc_rows).merge(pd.DataFrame(ev_rows)[["短/長", "勝率%", "每筆淨期望%"]],
                                        on="短/長")
    mc_df.to_csv(os.path.join(OUT, "mc_realistic.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(ev_rows).to_csv(os.path.join(OUT, "net_ev.csv"), index=False, encoding="utf-8-sig")
    print(mc_df.to_string(index=False))
    print(f"→ {OUT}｜{time.time() - t0:.0f} 秒")
    return 0


if __name__ == "__main__":
    sys.exit(main())
