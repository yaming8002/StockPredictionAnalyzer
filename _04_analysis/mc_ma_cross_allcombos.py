"""
多股雙均線交叉 × 蒙地卡羅（全 21 組合）
==========================================
調用既有邏輯逐一執行：_03 多股回測（MultiMACross, 固定1萬/共用100萬）取每筆損益，
再餵 _04 analyze_vbt.monte_carlo（bootstrap 10,000 次）量 最大連敗 S / 最大回撤 / 破產率。
資料只載一次。輸出 result/mc/ma_cross_mc.csv + 終端表。

執行：python _04_analysis/mc_ma_cross_allcombos.py
"""
import glob
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd

from _03_multi_strategy.ma_cross.multi_ma_cross import MultiMACross, GLITCH
from _04_analysis.analyze_vbt import monte_carlo

DATA = r"F:\stock-analyzer\data\stock_data"
MAS = [5, 10, 20, 50, 60, 120, 200]
WANT = ["open", "high", "low", "close", "volume"] + [f"sma_{n}" for n in MAS]
INIT_CASH = 1_000_000.0
N_SIMS = 10_000
RUIN_RATIO = 0.8            # 破產定義：權益跌破本金 80%（資金防線）
OUT = os.path.join(_root, "_02_strategy", "ma_strategy", "result", "mc")


def load_all():
    data = {}
    for f in sorted(glob.glob(os.path.join(DATA, "*.parquet"))):
        sid = os.path.splitext(os.path.basename(f))[0]
        if sid in GLITCH:
            continue
        try:
            cols = pd.read_parquet(f, columns=None).columns
            df = pd.read_parquet(f, columns=[c for c in WANT if c in cols])
        except Exception:
            continue
        if not df.empty:
            data[sid] = df
    return data


def main():
    os.makedirs(OUT, exist_ok=True)
    print("載入全市場 …", flush=True)
    data = load_all()
    print(f"載入 {len(data)} 檔。逐一跑 21 組合（多股固定1萬 → MC {N_SIMS} 次）…", flush=True)
    print("短/長|成交|勝率%|S中位|S_P95|S_P99|S極值|回撤%中位|回撤%P95|回撤%極值|破產%<80", flush=True)

    rows = []
    for s in MAS:
        for l in MAS:
            if s >= l:
                continue
            strat = MultiMACross(sizing_mode="fixed", min_invest=10_000.0,
                                 initial_cash=INIT_CASH)
            strat.SHORT_MA, strat.LONG_MA = s, l
            res = strat.run(data, start_date="2001-01-01", end_date="2025-12-31")
            trades = res["trades"]
            mc = monte_carlo(trades, initial_cash=INIT_CASH, n_sims=N_SIMS,
                             ruin_ratio=RUIN_RATIO)
            wr = res["summary"]["勝率(%)"]
            row = {"短": s, "長": l, "成交": mc.get("每次抽樣筆數", 0), "勝率%": wr,
                   "S中位": mc.get("最大連敗_中位"), "S_P95": mc.get("最大連敗_P95"),
                   "S_P99": mc.get("最大連敗_P99"), "S極值": mc.get("最大連敗_極值"),
                   "回撤%中位": mc.get("最大回撤%_中位"), "回撤%P95": mc.get("最大回撤%_P95"),
                   "回撤%極值": mc.get("最大回撤%_極值"),
                   "破產%<80": mc.get(f"破產機率(<{RUIN_RATIO:.0%})")}
            rows.append(row)
            print(f"{s}/{l}|{row['成交']}|{wr:.1f}|{row['S中位']}|{row['S_P95']}|"
                  f"{row['S_P99']}|{row['S極值']}|{row['回撤%中位']}|{row['回撤%P95']}|"
                  f"{row['回撤%極值']}|{row['破產%<80']}", flush=True)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT, "ma_cross_mc.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nALL_DONE -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
