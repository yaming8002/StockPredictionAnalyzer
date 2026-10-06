"""
多股雙均線交叉全 21 組合的回測段：每組的多股逐筆交易
====================================================
跑 _03 多股回測（MultiMACross，固定 1 萬／共用 100 萬），每個短／長均線組合存一份逐筆交易。
蒙地卡羅在分析段 `_04_analysis/ma_cross/mc_ma_cross_allcombos.py`，讀這裡的輸出。
拆兩段是為了分層：_03 只做多股回測、_04 只做分析。逐筆交易存 parquet（浮點數不失真），
分析段固定種子重跑蒙地卡羅才會與合併版逐位一致。資料只載一次。

執行：python _03_multi_strategy/ma_cross/ma_cross_allcombos_trades.py
輸出：_02_strategy/ma_strategy/result/mc/trades/<短>_<長>.parquet（每組一檔）
"""
import os
import sys
import time

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

# 跨策略共用的單一定義（資料品質排除集＋標準回測區間），勿在此另立第二份
from _02_strategy.base.vbt import common  # noqa: E402
from _02_strategy.base.vbt.common import DEFAULT_END, DEFAULT_START, GLITCH  # noqa: E402
from _03_multi_strategy.ma_cross.multi_ma_cross import MultiMACross  # noqa: E402

MAS = [5, 10, 20, 50, 60, 120, 200]
WANT = ["open", "high", "low", "close", "volume"] + [f"sma_{n}" for n in MAS]
INIT_CASH = 1_000_000.0
TRADES_DIR = os.path.join(common.result_dir("ma_strategy", "mc"), "trades")


def combos():
    """全部短 < 長的組合（共 21 組），回測段與分析段共用同一份順序。"""
    return [(s, l) for s in MAS for l in MAS if s < l]


def trades_path(short: int, long: int) -> str:
    """某一組的逐筆交易檔路徑（回測段寫、分析段讀，兩邊共用這一支）。"""
    return os.path.join(TRADES_DIR, f"{short}_{long}.parquet")


def main():
    os.makedirs(TRADES_DIR, exist_ok=True)
    t0 = time.time()
    print("載入全市場 …", flush=True)
    data = common.load_market(common.DATA_DIR, columns=WANT, exclude=GLITCH)
    print(f"載入 {len(data)} 檔。逐一跑 {len(combos())} 組合（多股固定 1 萬）…", flush=True)

    for s, l in combos():
        strat = MultiMACross(sizing_mode="fixed", min_invest=10_000.0, initial_cash=INIT_CASH)
        strat.SHORT_MA, strat.LONG_MA = s, l
        res = strat.run(data, start_date=DEFAULT_START, end_date=DEFAULT_END)
        path = trades_path(s, l)
        res["trades"].to_parquet(path, index=False)
        print(f"{s}/{l}：{len(res['trades']):,} 筆｜勝率 {res['summary']['勝率(%)']}%｜"
              f"{time.time() - t0:.0f} 秒", flush=True)
    print(f"\nALL_DONE -> {TRADES_DIR}", flush=True)


if __name__ == "__main__":
    main()
