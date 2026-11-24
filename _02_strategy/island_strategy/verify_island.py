"""
底部島型策略的兩項驗證（不跑全市場回測，只抽樣檢查訊號正確性）：

1. 截斷測試（不偷看未來）：把資料截到第 T 根，重算判定欄位；截斷前每一天的 island_k／訊號
   必須和用全史算出來的完全一樣。若某天的訊號在多看幾根之後才改變，就代表用到了未來資料。
2. 持有根數：用全史跑一次策略，逐筆檢查買進日到賣出日之間（只算 volume>0 的 K 棒）剛好 HOLD 根。
3. （--exits）出場優化 #1～#13 的截斷測試：附加與替換版的整條進出場判定，截斷前與全史一致。
4. （--entries）進場優化 #14～#33 的截斷測試：每條條件的判定欄位與進場訊號，截斷前與全史一致。

執行：python _02_strategy/island_strategy/verify_island.py [--stocks 20] [--cuts 15] [--hold 20] [--exits] [--entries]
有任何不一致就以非 0 結束並印出第一筆差異。
"""
import argparse
import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd

from _02_strategy.base.vbt import batch, common
from _02_strategy.base.vbt.common import GLITCH, DEFAULT_START, DEFAULT_END
from _02_strategy.island_strategy.single_island_strategy import (
    ENTRY_FILTERS, EXIT_RULES, STOP_RULES, SingleIslandStrategy, make_exit_variant)


def check_truncation(df: pd.DataFrame, cuts: int, rng: np.random.Generator) -> str:
    """回傳第一筆差異的描述；全部一致回傳空字串。"""
    strat = SingleIslandStrategy(liquidity=True)
    full = strat.add_columns(df.copy())
    full_sig = strat.buy_signal(full)
    for t in rng.integers(300, len(df), size=cuts):
        part = strat.add_columns(df.iloc[:t].copy())
        part_sig = strat.buy_signal(part)
        same_k = full["island_k"].loc[part.index].equals(part["island_k"])
        same_sig = full_sig.loc[part.index].equals(part_sig)
        if not (same_k and same_sig):
            return f"截到 {df.index[t - 1].date()} 時判定不同"
    return ""


def check_hold(df: pd.DataFrame, hold: int) -> str:
    """逐筆確認持有根數（volume>0 的 K 棒）＝hold。"""
    trades = SingleIslandStrategy(hold=hold).run(df, start=DEFAULT_START, end=DEFAULT_END)["trades"]
    bars = df.index[df["volume"] > 0]
    for _, tr in trades.iterrows():
        n = bars.get_loc(pd.Timestamp(tr["sell_date"])) - bars.get_loc(pd.Timestamp(tr["buy_date"]))
        if n != hold:
            return f"{tr['buy_date']}→{tr['sell_date']} 持有 {n} 根"
    return ""


def exit_variants() -> list:
    """出場優化 #1～#13：附加（持有 20 根）各一條，加上替換版（停損 × 主出場）全部。"""
    out = [make_exit_variant(20, True, s, None) for s in STOP_RULES]
    out += [make_exit_variant(20, True, None, x) for x in EXIT_RULES]
    rep_main = [x for x in EXIT_RULES if x != "ma10switch"]
    out += [make_exit_variant(20, True, s, x, replace=True) for s in (None, *STOP_RULES) for x in rep_main]
    return out


def check_exit_truncation(df: pd.DataFrame, cuts: int, rng: np.random.Generator) -> str:
    """
    出場優化的截斷測試：截到第 T 根重跑逐根掃描，截斷前每一天的進場／出場判定（shift 前）都要和全史一樣。
    出場是路徑相依的，所以比的是整條訊號序列，不是單一欄位。
    """
    for strat in exit_variants():
        strat._trade_start = None
        full = strat.add_columns(df.copy())
        e_full, x_full = strat.build_signals(full)
        for t in rng.integers(300, len(df), size=cuts):
            part = strat.add_columns(df.iloc[:t].copy())
            e_part, x_part = strat.build_signals(part)
            if not (e_full.loc[part.index].equals(e_part) and x_full.loc[part.index].equals(x_part)):
                return f"{strat.STOP_RULE}／{strat.EXIT_RULE}／替換={strat.REPLACE} 截到 {df.index[t - 1].date()} 時不同"
    return ""


def check_entry_truncation(df: pd.DataFrame, cuts: int, rng: np.random.Generator) -> str:
    """進場優化 #14～#33：截到第 T 根重算，截斷前每一天的 f_<條件> 欄位與進場訊號都要和全史一樣。"""
    strat = make_exit_variant(20, True, entry_filter=ENTRY_FILTERS[0])
    full = strat.add_columns(df.copy())
    cols = [f"f_{name}" for name in ENTRY_FILTERS]
    for t in rng.integers(300, len(df), size=cuts):
        part = strat.add_columns(df.iloc[:t].copy())
        for c in cols:
            if not full[c].loc[part.index].equals(part[c]):
                return f"{c} 截到 {df.index[t - 1].date()} 時不同"
    for name in ENTRY_FILTERS:
        strat.ENTRY_FILTER = name
        full_sig = strat.buy_signal(full)
        part = strat.add_columns(df.iloc[:len(df) // 2].copy())
        if not full_sig.loc[part.index].equals(strat.buy_signal(part)):
            return f"{name} 進場訊號截半時不同"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", type=int, default=20)
    ap.add_argument("--cuts", type=int, default=15)
    ap.add_argument("--hold", type=int, default=20)
    ap.add_argument("--exits", action="store_true", help="改跑出場優化 #1～#13 的截斷測試")
    ap.add_argument("--entries", action="store_true", help="改跑進場優化 #14～#33 的截斷測試")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    files = [p for p in batch.list_parquet(common.DATA_DIR)
             if os.path.splitext(os.path.basename(p))[0] not in GLITCH]
    picked = rng.choice(len(files), size=a.stocks, replace=False)
    bad = 0
    for i in picked:
        sid = os.path.splitext(os.path.basename(files[i]))[0]
        df = batch._load_prices(files[i])
        if len(df) < 400:
            continue
        if a.entries:
            checks = (("進場截斷", check_entry_truncation(df, a.cuts, rng)),)
        elif a.exits:
            checks = (("出場截斷", check_exit_truncation(df, a.cuts, rng)),)
        else:
            checks = (("截斷", check_truncation(df, a.cuts, rng)), ("持有", check_hold(df, a.hold)))
        for name, msg in checks:
            if msg:
                bad += 1
                print(f"✗ {sid} {name}：{msg}")
    print("全部一致" if bad == 0 else f"共 {bad} 項不一致")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
