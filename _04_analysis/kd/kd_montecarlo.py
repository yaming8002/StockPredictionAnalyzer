"""
KD 交叉（五）：六組交易策略的蒙地卡羅壓測（分析段）
=====================================================
讀單股掃描段 `_02_strategy/kd_strategy/kd_sweep.py` 存的逐筆交易（每檔獨立、每筆 1 萬、
訊號全成交），對「PF 前 6 進場 × 高檔死叉」各做一次蒙地卡羅；這支不跑回測。

口徑（與均線、MACD 系列同一套，常數取自 macd_montecarlo，不另立第二份）：
  10,000 條路徑、起始 100 萬、每筆定額累加；部署總筆數 T ∈ [交易日數×3, ×4]，
  歷史筆數不足下限 → T＝歷史筆數（全納入、有放回重抽）；本金大虧＝權益曾跌破本金 5 成。
  淨損益剛好 0 的交易不算勝負（summarize_trades 也排除），先濾掉再抽樣。
  ⚠️ 舊版（2026-07-28 reference、文章（五））的 MC 交易數比矩陣表多 1～2 筆，推測當時沒濾 0；
  這裡照 MACD／均線同口徑濾掉，所以「交易數」會與矩陣表一致。

三組（--set，可多選）：
  article  文章（五）蒙地卡羅表：6 進場 × 高檔死叉
  anchors  2026-07-28 reference 另外納入的兩個錨點：低檔 K,D<20（opt1）、黃金交叉（opt6）× 高檔死叉
  death    2026-07-28 reference「×一般死叉」基準對照：上面 8 個進場 × 一般死叉
  top5     2026-07-27 reference Top5：高位 K,D>50／創250日新高／多頭 5>20>60／黃金交叉／CMF>0 × 高檔死叉
           （不在預設；單跑請加 --file kd_montecarlo_top5.csv，預設檔是多股 driver 讀的）

「連敗 P95」就是多股回測算份數用的 S（見 _03_multi_strategy/kd/kd_multi_driver.py，
讀這裡的 CSV，用「進場」代號對應）。每筆期望% 用全淨口徑（real_pnl ÷ 買進付出現金，
見 ma_cross_montecarlo.net_stats），與文章表註「以扣費稅的淨報酬計算」一致。

執行（先跑單股掃描段）：
    python _02_strategy/kd_strategy/kd_sweep.py --set matrix
    python _04_analysis/kd/kd_montecarlo.py [--set article anchors death] [--paths 10000]
    python _04_analysis/kd/kd_montecarlo.py --set top5 --file kd_montecarlo_top5.csv
輸出：_02_strategy/kd_strategy/result/kd_mc/kd_montecarlo.csv ＋ 終端表。
"""
import argparse
import os
import sys
import time

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd  # noqa: E402

from _02_strategy.base.vbt import common  # noqa: E402
from _02_strategy.kd_strategy import kd_sweep  # noqa: E402
from _02_strategy.kd_strategy.kd_variants import NAME_ENTRY, NAME_EXIT  # noqa: E402
from _02_strategy.kd_strategy.single_kd_strategy import RESULT_DIR  # noqa: E402
from _04_analysis.analyze_vbt import monte_carlo  # noqa: E402
from _04_analysis.ma_cross.ma_cross_montecarlo import net_stats  # noqa: E402
from _04_analysis.macd.macd_montecarlo import (INIT_CASH, PATHS, RUIN_RATIO,  # noqa: E402
                                               T_HIGH, T_LOW)

SWEEP_DIR = os.path.join(RESULT_DIR, "single_kd")
OUT = common.result_dir("kd_strategy", "kd_mc")
MC_FILE = "kd_montecarlo.csv"

# 文章（五）的六個進場（PF 前 6，順序同 multi_kd.ENTRIES）；() ＝ 純黃金交叉
ARTICLE = ("breakout250", "breakout120", "gap", "low_redk", "breakout60", "divergence")
ANCHORS = ("low_zone", ())
# 2026-07-27 reference（kd_top5_montecarlo）：舊 6 進 × 5 出矩陣 PF 前 5（皆 × 高檔死叉）
TOP5 = ("high_zone50", "breakout250", "bull_align_5_20_60", (), "cmf_pos")
SETS = {
    "article": [(e, "high_death") for e in ARTICLE],
    "anchors": [(e, "high_death") for e in ANCHORS],
    "death": [(e, "death") for e in ARTICLE + ANCHORS],
    "top5": [(e, "high_death") for e in TOP5],
}
GROUP = {"article": "文章", "anchors": "錨點", "death": "死叉對照", "top5": "Top5"}
# 預設三組寫 kd_montecarlo.csv（多股 driver 讀這份）；top5 另存，免得單跑 top5 蓋掉它
DEFAULT_SETS = ["article", "anchors", "death"]


def entry_code(entry) -> str:
    """進場代號（CSV 的「進場」欄，多股 driver 用它對應 S）；純黃金交叉寫 golden。"""
    return entry if entry else "golden"


def label(entry, exit_) -> str:
    name = NAME_ENTRY[entry] if entry else "黃金交叉"
    return f"{name} × {NAME_EXIT[exit_]}"


def mc_csv_path(out: str = OUT) -> str:
    """MC 表路徑（分析段寫、多股 driver 讀，兩邊共用這一支）。"""
    return os.path.join(out, MC_FILE)


def run_case(entry, exit_, group: str, sweep_dir: str, paths: int) -> dict:
    name = kd_sweep._v("amt", entry, exit_)
    path = os.path.join(sweep_dir, name, "single_kd_trades.parquet")
    if not os.path.isfile(path):
        raise SystemExit(f"找不到逐筆交易：{path}\n先跑：python _02_strategy/kd_strategy/kd_sweep.py --set matrix")
    trades = pd.read_parquet(path)
    trades = trades[trades["real_pnl"] != 0]          # 損益 0 的不算勝負（同 summarize_trades）
    mc = monte_carlo(trades, initial_cash=INIT_CASH, n_sims=paths,
                     ruin_ratio=RUIN_RATIO, t_low=T_LOW, t_high=T_HIGH)
    ret = {k: round((mc[f"最終資金_{k}"] - INIT_CASH) / INIT_CASH * 100, 1)
           for k in ("P5", "中位", "P95")}
    ev = net_stats(trades)
    # 欄序同文章（五）蒙地卡羅表；分組／進場／出場／變體三欄放最前供程式對應
    return {"分組": group, "進場": entry_code(entry), "出場": exit_, "變體": name,
            "組合": label(entry, exit_), "交易數": mc["歷史筆數"],
            "抽樣模式": "全納入" if mc["歷史筆數"] < T_LOW else "抽區間",
            "勝率%": ev["勝率%"], "每筆期望%": ev["每筆淨期望%"],
            "報酬% P5": ret["P5"], "報酬% 中位": ret["中位"], "報酬% P95": ret["P95"],
            "區間寬度": round(ret["P95"] - ret["P5"], 1),
            "本金大虧%": mc[f"破產機率(<{RUIN_RATIO:.0%})"],
            "連敗 P95": mc["最大連敗_P95"], "回撤% P95": mc["最大回撤%_P95"]}


def main() -> int:
    ap = argparse.ArgumentParser(description="KD 交叉（五）蒙地卡羅")
    ap.add_argument("--set", nargs="+", default=DEFAULT_SETS, choices=list(SETS))
    ap.add_argument("--file", default=MC_FILE,
                    help="輸出檔名（單跑 top5 請改，例 kd_montecarlo_top5.csv，避免蓋掉多股 driver 讀的那份）")
    ap.add_argument("--paths", type=int, default=PATHS, help="路徑數（正式 10,000；冒煙可調小）")
    ap.add_argument("--sweep-dir", default=SWEEP_DIR, help="單股掃描結果根目錄")
    ap.add_argument("--out", default=OUT, help="輸出目錄（冒煙請改暫存目錄）")
    a = ap.parse_args()

    t0 = time.time()
    print(f"{a.set}｜{a.paths:,} 條路徑｜T ∈ [{T_LOW:,}, {T_HIGH:,}]", flush=True)
    rows = []
    for key in a.set:
        group_rows = []
        for entry, exit_ in SETS[key]:
            r = run_case(entry, exit_, GROUP[key], a.sweep_dir, a.paths)
            group_rows.append(r)
            print(f"  {r['組合']}：{r['交易數']:,} 筆｜連敗 P95 {r['連敗 P95']}｜"
                  f"報酬中位 {r['報酬% 中位']}%｜{time.time() - t0:.0f} 秒", flush=True)
        # 組內依每筆期望由高到低（文章（五）表的排序）
        rows += sorted(group_rows, key=lambda r: -r["每筆期望%"])

    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, a.file)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(df.drop(columns=["變體"]).to_string(index=False))
    print(f"→ {path}｜{time.time() - t0:.0f} 秒")
    return 0


if __name__ == "__main__":
    sys.exit(main())
