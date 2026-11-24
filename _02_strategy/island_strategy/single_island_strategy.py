"""
島型反轉（底部島型）策略（single island）— island_strategy 套件下的方案
======================================================================

繼承 VbtSingleStrategy，交易一律由 vbt 執行（Portfolio.from_signals）；本檔只產生進出場訊號。

型態（只做多、只看底部島型；判定日 t 收盤即可確認，只用 t 以前的 K 棒）：
  存在 k∈[1, MAX_K]，島＝t−k..t−1（k 根），島前一根＝b＝t−k−1：
    ① 向下缺口未回補：島內最高價 < b 的最低價
    ② 向上跳空離開：t 的最低價 > 島內最高價
  缺口大小不限；同一天有多個 k 成立時取最小的 k（記在 island_k 診斷用）。

交易（判定日收盤 → 隔日開盤成交，基底統一規則）：
  進場：t 判定底部島型 → t+1 開盤買。
  出場：持有 HOLD 根後開盤賣出（判定日＝進場成交日 + HOLD − 1，隔日開盤成交，持有剛好 HOLD 根）。
  持有中不接新訊號；賣出那天最快隔一天才再進場（與 vbt accumulate=False 一致）。

為什麼先把 volume=0 的 K 棒拿掉（add_columns 裡）：
  資料裡有休市日的幽靈 K 棒（open=high=low=close=前收、volume=0，見 docs/reference/data_quality_known_issues.md）。
  它的高低點等於前收，會憑空製造或吃掉缺口，直接改變島型判定；固定持有的「根數」也會被它灌水。
  所以型態、持有根數、成交都只在 volume>0 的 K 棒上進行（停牌日一併跳過）。

為什麼 HOLD 用 CLI 參數（不用註解切換）：
  HOLD 是 baseline 本身要並列比較的設定（5／10／20 根），比照雙底的 ZigZag 門檻 --pct。
  之後的優化（停損、出場、濾網）一律走註解切換。
可成交門檻同理用 --liq：成交金額＝5 日均量×收盤 > 1,000 萬（與其他系列同一基準），在判定日檢查。

出場優化（註解切換；類別屬性 STOP_RULE／EXIT_RULE／REPLACE 各開一行，driver 另以 make_exit_variant 產生變體，
不動本檔的註解狀態）。全部看判定日收盤、隔日開盤賣；預設「附加」＝持有 HOLD 根仍在、新規則疊上去、先觸發者算：
  停損類 STOP_RULE（島的位置取進場判定日那座島）：
    #1  lowclose  收盤 < 島內最低收盤價
    #2  islandlow 收盤 < 島內最低價（島底）
    #3  gapfill   收盤 < 島內最高價（向上缺口被回補）
    #4  time5     買進後 5 根收盤都沒有高於買進價 → 第 5 根收盤出場（時間停損）
  主出場類 EXIT_RULE：
    #5～#8  trail5／10／15／20  收盤 < 買進後最高價 ×（1 − 5／10／15／20%）（最高價含當根，同雙底 #19）
    #9  atr        收盤 < 買進後最高價 − 3 × ATR(14)（同雙底 #18）
    #10 ma10       收盤由上跌破 MA10（昨收 ≥ 昨 MA10、今收 < 今 MA10，同雙底 #20）
    #11 ma10switch 持有期間（HOLD 根內）任一天收盤 > MA10 → 改成「收盤 < MA10 出」、不再有時間上限；沒站上照 HOLD 根
    #12 gapdown    買進後出現向下跳空（當根最高價 < 前一根最低價）
    #13 topisland  出現頂部島型（島內最低價 > 島前一根最高價，且當根最高價 < 島內最低價；k≤10）
  附錄：替換 REPLACE＝True：拿掉持有 HOLD 根，出場只剩 STOP_RULE／EXIT_RULE（#11 依定義需要 HOLD，不適用）。
  出場原因記在 diag：exit_hold／exit_stop／exit_main（同根多條成立各自都算）＝生效率的分子。

進場優化（註解切換；類別屬性 ENTRY_FILTER，driver 另以 make_exit_variant(entry_filter=...) 產生變體）。
疊在「底部島型成立」上，只用判定日 t 收盤已知的資料；「島內」＝t−k..t−1：
  週線（週 K 用「本週到今天為止」：本週收盤＝今天收盤，前面的週取已完成週的最後收盤；週＝週一～週日，
        補行交易的週六算在同一週）：
    #14 wk_up        週 MA5 > 週 MA20
    #15 wk_down      週 MA5 < 週 MA20
    #16 wk_above20   週收盤（＝今天收盤）> 週 MA20
    #17 wk_ma20_up   週 MA20 上揚（本週到今天的週 MA20 > 上週完成時的週 MA20）
    #18 wk_cross4    週 MA5 > 週 MA20，且前 4 個完成週裡至少一週週 MA5 ≤ 週 MA20（4 週內剛上穿）
  日線趨勢：
    #19 above200     收盤 > MA200
    #20 below200     收盤 < MA200
    #21 align20      多頭排列 MA20 > MA60 > MA120（同雙底 #6）
    #22 ma60_120     MA60 > MA120（用戶 2026-11-17 加）
  前段跌幅（近 60 根最高價＝t−60..t−1 的最高價，跌到島底）：
    #23 drop15       1 − 島底 ÷ 近 60 根最高價 ≥ 15%
    #24 drop25       同上 ≥ 25%
  量能：
    #25 vol_break    判定日量 > 20 日均量（含當日）× 1.5（同雙底 #7）
    #26 vol_shrink   島內均量 < 島前一根往前 20 根的均量
  缺口大小（向上缺口＝判定日最低價 ÷ 島內最高價 − 1）：
    #27 gap1         ≥ 1%
    #28 gap3         ≥ 3%
  島寬：
    #29 k1           單日島（k＝1）
    #30 k3plus       k ≥ 3
  超賣：
    #31 rsi30        島內 RSI(14) 最低值 < 30
    #32 kd20         島內 KD 的 K(9,3,3) 最低值 < 20
  確認日 K 棒：
    #33 redk_upper   收紅 K（收盤 > 開盤）且收在當日振幅上半部（收盤 ≥ (高＋低)/2）

診斷（self.diag，由 driver 寫出）：訊號數、進場數、出場數、期末未平倉數、出場原因，以及進場訊號的 island_k 分布。

執行：
    python _02_strategy/island_strategy/single_island_strategy.py <資料夾> --hold 20 [--liq]
  輸出：result/single_island/h<hold>_<raw|liq1000>/
"""
import argparse
import json
import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd

from _01_data.indicators_momentum_volume import calculate_kd, calculate_rsi
from _01_data.indicators_volatility import calculate_atr
from _02_strategy.base.vbt import batch
from _02_strategy.base.vbt.single import VbtSingleStrategy
from _02_strategy.base.vbt.common import DEFAULT_START, DEFAULT_END

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
TURNOVER_MIN = 10_000_000
MAX_K = 10                # 島最多幾根 K 棒
ATR_PERIOD, CHANDELIER_K = 14, 3.0   # 優化 #9
TIME_STOP_BARS = 5                   # 優化 #4
TRAIL_DROPS = {"trail5": 0.05, "trail10": 0.10, "trail15": 0.15, "trail20": 0.20}   # 優化 #5～#8
STOP_RULES = ("lowclose", "islandlow", "gapfill", "time5")
EXIT_RULES = (*TRAIL_DROPS, "atr", "ma10", "ma10switch", "gapdown", "topisland")
# 進場優化 #14～#33 的參數
WK_CROSS_WEEKS = 4                    # 優化 #18：幾週內剛上穿
PRIOR_LOOKBACK = 60                   # 優化 #23／#24：近幾根的最高價
VOL_WINDOW, VOL_MULTIPLE = 20, 1.5    # 優化 #25／#26
RSI_PERIOD, RSI_MAX = 14, 30          # 優化 #31
KD_K_MAX = 20                         # 優化 #32
ENTRY_FILTERS = ("wk_up", "wk_down", "wk_above20", "wk_ma20_up", "wk_cross4",
                 "above200", "below200", "align20", "ma60_120", "drop15", "drop25",
                 "vol_break", "vol_shrink", "gap1", "gap3", "k1", "k3plus", "rsi30", "kd20", "redk_upper")


def bottom_island(high: pd.Series, low: pd.Series, max_k: int = MAX_K):
    """
    回傳 (island_k, island_top, island_low)：底部島型成立的判定日填值，其餘 NaN。
      island_k  ：成立的最小 k（島的根數）
      island_top：島內最高價（向上跳空要越過的位置）
      island_low：島內最低價（島底）
    第 t 根只用 t 以前（含 t）的 high／low；k 由大到小覆寫，最後留下最小的 k。
    """
    n = len(high)
    island_k = np.full(n, np.nan)
    island_top = np.full(n, np.nan)
    island_low = np.full(n, np.nan)
    for k in range(max_k, 0, -1):
        top = high.rolling(k).max().shift(1)          # 島＝t−k..t−1 的最高價
        bottom = low.rolling(k).min().shift(1)        # 島＝t−k..t−1 的最低價
        before = low.shift(k + 1)                     # 島前一根 b 的最低價
        hit = ((top < before) & (low > top)).to_numpy()
        island_k[hit] = k
        island_top[hit] = top.to_numpy()[hit]
        island_low[hit] = bottom.to_numpy()[hit]
    return island_k, island_top, island_low


def island_low_close(close: pd.Series, island_k: np.ndarray) -> np.ndarray:
    """判定日那座島（t−k..t−1）的最低收盤價；k 取 bottom_island 回傳的最小 k。"""
    out = np.full(len(close), np.nan)
    for k in range(1, MAX_K + 1):
        mask = island_k == k
        if mask.any():
            out[mask] = close.rolling(k).min().shift(1).to_numpy()[mask]
    return out


def top_island(high: pd.Series, low: pd.Series, max_k: int = MAX_K) -> np.ndarray:
    """頂部島型（底部的鏡像）成立的判定日＝True；只用 t 以前（含 t）的 high／low。"""
    hit_any = np.zeros(len(high), dtype=bool)
    for k in range(1, max_k + 1):
        floor = low.rolling(k).min().shift(1)         # 島內最低價
        before = high.shift(k + 1)                    # 島前一根的最高價
        hit_any |= ((floor > before) & (high < floor)).to_numpy()
    return hit_any


def island_stat(series: pd.Series, island_k: np.ndarray, how: str) -> np.ndarray:
    """判定日那座島（t−k..t−1）上 series 的 min／mean；k 取 bottom_island 回傳的最小 k。"""
    out = np.full(len(series), np.nan)
    for k in range(1, MAX_K + 1):
        mask = island_k == k
        if mask.any():
            out[mask] = getattr(series.rolling(k), how)().shift(1).to_numpy()[mask]
    return out


def weekly_ma(close: pd.Series, n: int):
    """
    週 MA（本週到今天為止）：回傳 (今天的週 MAn, 上週完成時的週 MAn)，對齊日線 index。
    今天的週 MAn＝（今天收盤 ＋ 前 n−1 個已完成週的週收盤）÷ n；只用今天以前的資料。
    週＝週一～週日（pandas 'W'），補行交易的週六歸在同一週。
    """
    week = close.index.to_period("W")
    wk_close = close.groupby(week).last()                     # 每週最後一筆收盤（只有已完成的週會被用到）
    prev_sum = wk_close.rolling(n - 1).sum().shift(1)         # 前 n−1 個完成週的週收盤和
    last_ma = wk_close.rolling(n).mean().shift(1)             # 上週完成時的週 MAn
    today_ma = (close.to_numpy() + prev_sum.reindex(week).to_numpy()) / n
    return (pd.Series(today_ma, index=close.index),
            pd.Series(last_ma.reindex(week).to_numpy(), index=close.index))


def weekly_cross_recent(close: pd.Series, weeks: int) -> np.ndarray:
    """前 weeks 個已完成週裡，至少一週週 MA5 ≤ 週 MA20（完成週的值）。"""
    week = close.index.to_period("W")
    wk_close = close.groupby(week).last()
    below = (wk_close.rolling(5).mean() <= wk_close.rolling(20).mean()).astype(float)
    below[wk_close.rolling(20).mean().isna()] = np.nan
    recent = below.rolling(weeks).max().shift(1)              # 前 weeks 個完成週
    return (recent.reindex(week).to_numpy() == 1.0)


def entry_filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """進場優化 #14～#33 的判定欄位 f_<名稱>（布林）；需要 island_k／island_top／island_low 已算好。"""
    close, k = df["close"], df["island_k"].to_numpy()
    wk5, _ = weekly_ma(close, 5)
    wk20, wk20_last = weekly_ma(close, 20)
    ma = {n: close.rolling(n).mean() for n in (20, 60, 120, 200)}
    peak = df["high"].rolling(PRIOR_LOOKBACK).max().shift(1)
    drop = 1.0 - df["island_low"] / peak
    vol_ma = df["volume"].rolling(VOL_WINDOW).mean()
    isl_vol = island_stat(df["volume"].astype(float), k, "mean")
    pre_vol = np.full(len(df), np.nan)                        # 島前一根往前 20 根的均量
    for kk in range(1, MAX_K + 1):
        mask = k == kk
        if mask.any():
            pre_vol[mask] = vol_ma.shift(kk + 1).to_numpy()[mask]
    gap = df["low"] / df["island_top"] - 1.0
    tmp = calculate_kd(calculate_rsi(df[["open", "high", "low", "close"]].copy(), RSI_PERIOD))
    f = {
        "wk_up": wk5 > wk20,                                                  # 優化 #14
        "wk_down": wk5 < wk20,                                                # 優化 #15
        "wk_above20": close > wk20,                                           # 優化 #16
        "wk_ma20_up": wk20 > wk20_last,                                       # 優化 #17
        "wk_cross4": (wk5 > wk20) & weekly_cross_recent(close, WK_CROSS_WEEKS),  # 優化 #18
        "above200": close > ma[200],                                          # 優化 #19
        "below200": close < ma[200],                                          # 優化 #20
        "align20": (ma[20] > ma[60]) & (ma[60] > ma[120]),                    # 優化 #21
        "ma60_120": ma[60] > ma[120],                                         # 優化 #22
        "drop15": drop >= 0.15,                                               # 優化 #23
        "drop25": drop >= 0.25,                                               # 優化 #24
        "vol_break": df["volume"] > vol_ma * VOL_MULTIPLE,                    # 優化 #25
        "vol_shrink": pd.Series(isl_vol < pre_vol, index=df.index),           # 優化 #26
        "gap1": gap >= 0.01,                                                  # 優化 #27
        "gap3": gap >= 0.03,                                                  # 優化 #28
        "k1": pd.Series(k == 1, index=df.index),                              # 優化 #29
        "k3plus": pd.Series(k >= 3, index=df.index),                          # 優化 #30
        "rsi30": pd.Series(island_stat(tmp["rsi"], k, "min") < RSI_MAX, index=df.index),  # 優化 #31
        "kd20": pd.Series(island_stat(tmp["k"], k, "min") < KD_K_MAX, index=df.index),    # 優化 #32
        "redk_upper": (close > df["open"]) & (close >= (df["high"] + df["low"]) / 2),     # 優化 #33
    }
    for name, cond in f.items():
        df[f"f_{name}"] = pd.Series(cond, index=df.index).fillna(False).astype(bool)
    return df


class ExitState:
    """
    單筆持倉的出場狀態：進場判定日呼叫 start(i)，持倉中每根呼叫 check(j) 回傳 (停損成立, 主出場成立)。
    進場價＝判定日隔天開盤；買進後最高價含當根（同雙底 ExtraExit）。
    """

    def __init__(self, stop_rule, exit_rule, df: pd.DataFrame):
        self.stop_rule, self.exit_rule = stop_rule, exit_rule
        self.open = df["open"].to_numpy(dtype=float)
        self.high = df["high"].to_numpy(dtype=float)
        self.low = df["low"].to_numpy(dtype=float)
        self.close = df["close"].to_numpy(dtype=float)
        self.isl_low = df["island_low"].to_numpy(dtype=float)
        self.isl_top = df["island_top"].to_numpy(dtype=float)
        self.isl_lowclose = df["island_lowclose"].to_numpy(dtype=float)
        self.atr = df[f"atr_{ATR_PERIOD}"].to_numpy(dtype=float)
        self.ma_break = df["ma10_break"].to_numpy(dtype=bool)
        self.top_isl = df["top_island"].to_numpy(dtype=bool)

    def start(self, i: int):
        self.sig = i
        self.entry_px = self.open[i + 1] if i + 1 < len(self.open) else np.nan
        self.peak_px = -np.inf
        self.peak_close = -np.inf
        self.levels = {"lowclose": self.isl_lowclose[i], "islandlow": self.isl_low[i], "gapfill": self.isl_top[i]}
        self.switched = False

    def stop_hit(self, j: int) -> bool:
        rule = self.stop_rule
        if rule is None:
            return False
        if rule == "time5":                                    # 優化 #4
            return j == self.sig + TIME_STOP_BARS and self.peak_close <= self.entry_px
        return self.close[j] < self.levels[rule]               # 優化 #1～#3

    def main_hit(self, j: int) -> bool:
        c, rule = self.close[j], self.exit_rule
        if rule is None or rule == "ma10switch":               # #11 在 build_signals 處理
            return False
        if rule in TRAIL_DROPS:                                # 優化 #5～#8
            return c < self.peak_px * (1.0 - TRAIL_DROPS[rule])
        if rule == "atr":                                      # 優化 #9
            return (not np.isnan(self.atr[j])) and c < self.peak_px - CHANDELIER_K * self.atr[j]
        if rule == "ma10":                                     # 優化 #10
            return bool(self.ma_break[j])
        if rule == "gapdown":                                  # 優化 #12
            return self.high[j] < self.low[j - 1]
        if rule == "topisland":                                # 優化 #13
            return bool(self.top_isl[j])
        raise ValueError(f"未知的出場優化 {rule}")

    def check(self, j: int):
        """先更新含當根的最高價／最高收盤，再判定；回傳 (停損, 主出場)。"""
        self.peak_px = max(self.peak_px, self.high[j])
        self.peak_close = max(self.peak_close, self.close[j])
        return self.stop_hit(j), self.main_hit(j)


class SingleIslandStrategy(VbtSingleStrategy):
    """底部島型進場、固定持有 HOLD 根出場。"""

    STOP_RULE = None              # baseline：不設停損
    # STOP_RULE = "lowclose"      # 優化 #1 收盤跌破島內最低收盤價
    # STOP_RULE = "islandlow"     # 優化 #2 收盤跌破島底（島內最低價）
    # STOP_RULE = "gapfill"       # 優化 #3 收盤回補向上缺口（< 島內最高價）
    # STOP_RULE = "time5"         # 優化 #4 買進後 5 根收盤沒高過買進價

    EXIT_RULE = None              # baseline：只有持有 HOLD 根
    # EXIT_RULE = "trail5"        # 優化 #5 自最高價回落 5%
    # EXIT_RULE = "trail10"       # 優化 #6 自最高價回落 10%
    # EXIT_RULE = "trail15"       # 優化 #7 自最高價回落 15%
    # EXIT_RULE = "trail20"       # 優化 #8 自最高價回落 20%
    # EXIT_RULE = "atr"           # 優化 #9 ATR 移動停損 3×ATR(14)
    # EXIT_RULE = "ma10"          # 優化 #10 收盤跌破 MA10
    # EXIT_RULE = "ma10switch"    # 優化 #11 站上 MA10 後改成跌破 MA10 出
    # EXIT_RULE = "gapdown"       # 優化 #12 出現向下跳空
    # EXIT_RULE = "topisland"     # 優化 #13 出現頂部島型

    ENTRY_FILTER = None           # baseline：底部島型成立就買
    # ENTRY_FILTER = "wk_up"      # 優化 #14 週 MA5 > 週 MA20
    # ENTRY_FILTER = "wk_down"    # 優化 #15 週 MA5 < 週 MA20
    # ENTRY_FILTER = "wk_above20" # 優化 #16 週收盤 > 週 MA20
    # ENTRY_FILTER = "wk_ma20_up" # 優化 #17 週 MA20 上揚
    # ENTRY_FILTER = "wk_cross4"  # 優化 #18 週 MA5 4 週內剛上穿週 MA20
    # ENTRY_FILTER = "above200"   # 優化 #19 收盤 > MA200
    # ENTRY_FILTER = "below200"   # 優化 #20 收盤 < MA200
    # ENTRY_FILTER = "align20"    # 優化 #21 多頭排列 MA20 > MA60 > MA120
    # ENTRY_FILTER = "ma60_120"   # 優化 #22 MA60 > MA120
    # ENTRY_FILTER = "drop15"     # 優化 #23 前段跌幅 ≥ 15%
    # ENTRY_FILTER = "drop25"     # 優化 #24 前段跌幅 ≥ 25%
    # ENTRY_FILTER = "vol_break"  # 優化 #25 判定日量 > 20 日均量 × 1.5
    # ENTRY_FILTER = "vol_shrink" # 優化 #26 島內量縮
    # ENTRY_FILTER = "gap1"       # 優化 #27 向上缺口 ≥ 1%
    # ENTRY_FILTER = "gap3"       # 優化 #28 向上缺口 ≥ 3%
    # ENTRY_FILTER = "k1"         # 優化 #29 單日島
    # ENTRY_FILTER = "k3plus"     # 優化 #30 島寬 ≥ 3 根
    # ENTRY_FILTER = "rsi30"      # 優化 #31 島內 RSI(14) < 30
    # ENTRY_FILTER = "kd20"       # 優化 #32 島內 K 值 < 20
    # ENTRY_FILTER = "redk_upper" # 優化 #33 確認日紅 K 收在上半部

    REPLACE = False      # baseline：STOP_RULE／EXIT_RULE 附加在持有 HOLD 根上，先觸發者算
    # REPLACE = True     # 附錄：替換，拿掉持有 HOLD 根，出場只剩 STOP_RULE／EXIT_RULE

    def __init__(self, hold: int = 20, liquidity: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.hold = hold
        self.liquidity = liquidity
        self._trade_end = None
        self.diag = {"signals": 0, "entries": 0, "exits": 0, "open_at_end": 0,
                     "exit_hold": 0, "exit_stop": 0, "exit_main": 0,
                     **{f"k{k}": 0 for k in range(1, MAX_K + 1)}}

    def add_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["volume"] > 0].copy()              # 拿掉幽靈 K 棒與停牌日（理由見檔頭）
        df["turnover"] = df["volume"].rolling(5).mean() * df["close"]
        k, top, bottom = bottom_island(df["high"], df["low"])
        df["island_k"], df["island_top"], df["island_low"] = k, top, bottom
        # 出場優化用欄位（#1、#9～#13）
        df["island_lowclose"] = island_low_close(df["close"], k)
        df["top_island"] = top_island(df["high"], df["low"])
        calculate_atr(df, window=ATR_PERIOD)
        df["ma10"] = df["close"].rolling(10).mean()
        df["ma10_break"] = ((df["close"] < df["ma10"])
                            & (df["close"].shift(1) >= df["ma10"].shift(1))).fillna(False)
        if self.ENTRY_FILTER:                         # 進場優化 #14～#33 用欄位
            df = entry_filter_columns(df)
        return df

    def buy_signal(self, df: pd.DataFrame) -> pd.Series:
        """判定日＝底部島型成立（＋可成交門檻）。"""
        signal = df["island_k"].notna()
        if self.liquidity:
            signal = signal & (df["turnover"] > TURNOVER_MIN)
        if self.ENTRY_FILTER:                         # 優化 #14～#33
            signal = signal & df[f"f_{self.ENTRY_FILTER}"]
        return signal.fillna(False).astype(bool)

    def sell_signal(self, df: pd.DataFrame) -> pd.Series:
        """出場依進場位置數根數，在 build_signals 逐根掃描；這裡不使用。"""
        return pd.Series(False, index=df.index)

    def window_signals(self, df: pd.DataFrame, start=None, end=None):
        self._trade_end = end                         # 診斷要知道區間終點（數期末未平倉）
        return super().window_signals(df, start, end)

    def build_signals(self, df: pd.DataFrame):
        """
        逐根掃描產生「判定日」進出場，再 shift(1) 成隔日開盤成交日交給 vbt。
        判定日 i 進場 → i+1 開盤買；出場判定日＝i+HOLD → i+HOLD+1 開盤賣，持有 HOLD 根。
        出場優化（附加）：持倉中每根收盤判定 STOP_RULE／EXIT_RULE，與持有 HOLD 根先觸發者算；
        REPLACE 時不用持有 HOLD 根。#11 站上 MA10 後改成「收盤 < MA10 出」，切換後不再看 HOLD。
        """
        if self.REPLACE and self.EXIT_RULE == "ma10switch":
            raise ValueError("ma10switch 需要持有 HOLD 根，不能用替換")
        entry_raw = self.entry_signal(df).to_numpy()
        n = len(df)
        entries = np.zeros(n, dtype=bool)
        exits = np.zeros(n, dtype=bool)
        reason = np.zeros(n, dtype=np.int8)           # 位元：1=持有 HOLD 根 2=停損 4=主出場
        state = ExitState(self.STOP_RULE, self.EXIT_RULE, df) if (self.STOP_RULE or self.EXIT_RULE) else None
        close = df["close"].to_numpy(dtype=float)
        ma10 = df["ma10"].to_numpy(dtype=float)
        switch = self.EXIT_RULE == "ma10switch"
        in_pos, exit_day = False, -1                  # exit_day＝持有 HOLD 根的出場判定日
        for i in range(n):
            if in_pos:
                hit_stop, hit_main = state.check(i) if state else (False, False)
                if switch and state.switched:                                   # 優化 #11 切換後
                    hit_main, hit_hold = bool(close[i] < ma10[i]), False
                else:
                    hit_hold = (not self.REPLACE) and i == exit_day
                    if switch and not hit_hold and close[i] > ma10[i]:
                        state.switched = True                                   # 優化 #11 站上 MA10（隔根起生效）
                if hit_hold or hit_stop or hit_main:
                    exits[i] = True
                    reason[i] = (1 if hit_hold else 0) + (2 if hit_stop else 0) + (4 if hit_main else 0)
                    in_pos = False
                continue
            if entry_raw[i]:
                entries[i] = True
                in_pos, exit_day = True, i + self.hold
                if state:
                    state.start(i)
        self._count_diag(df, entries, exits, entry_raw, reason)
        e = pd.Series(entries, index=df.index).shift(1, fill_value=False)
        x = pd.Series(exits, index=df.index).shift(1, fill_value=False)
        return e, x

    def _count_diag(self, df: pd.DataFrame, entries, exits, entry_raw, reason):
        """只數成交日落在交易區間內的事件（判定日＋1 才成交）。"""
        idx = df.index
        lo = pd.Timestamp(self._trade_start) if self._trade_start else idx[0]
        hi = pd.Timestamp(self._trade_end) if self._trade_end else idx[-1]
        exec_day = pd.Series(idx, index=idx).shift(-1)
        inside = ((exec_day >= lo) & (exec_day <= hi)).to_numpy()
        self.diag["signals"] += int((entry_raw & inside).sum())
        ent_in = entries & inside
        self.diag["entries"] += int(ent_in.sum())
        ex_in = exits & inside
        self.diag["exits"] += int(ex_in.sum())
        self.diag["exit_hold"] += int((ex_in & ((reason & 1) > 0)).sum())
        self.diag["exit_stop"] += int((ex_in & ((reason & 2) > 0)).sum())
        self.diag["exit_main"] += int((ex_in & ((reason & 4) > 0)).sum())
        ks = df["island_k"].to_numpy()[ent_in]
        for k in ks[~np.isnan(ks)].astype(int):
            self.diag[f"k{k}"] += 1
        ent_pos = np.flatnonzero(ent_in)
        if len(ent_pos) and not (exits & inside)[ent_pos[-1] + 1:].any():
            self.diag["open_at_end"] += 1


def make_exit_variant(hold: int, liquidity: bool, stop_rule=None, exit_rule=None,
                      replace: bool = False, entry_filter=None) -> SingleIslandStrategy:
    """出場／進場優化變體（與策略檔註解切換同一組設定，不去動本檔的註解狀態）。"""
    if stop_rule not in (None, *STOP_RULES) or exit_rule not in (None, *EXIT_RULES):
        raise ValueError(f"未知的出場變體 {stop_rule}／{exit_rule}")
    if entry_filter not in (None, *ENTRY_FILTERS):
        raise ValueError(f"未知的進場變體 {entry_filter}")
    v = SingleIslandStrategy(hold=hold, liquidity=liquidity)
    v.STOP_RULE, v.EXIT_RULE, v.REPLACE, v.ENTRY_FILTER = stop_rule, exit_rule, replace, entry_filter
    return v


def variant_tag(hold: int, liquidity: bool, stop_rule=None, exit_rule=None, replace: bool = False,
                entry_filter=None) -> str:
    """變體輸出資料夾名：[in-<進場>__]<h持有根數|rep>__<停損>__<主出場>_<raw|liq1000>。"""
    base = "rep" if replace else f"h{hold}"
    head = f"in-{entry_filter}__" if entry_filter else ""
    return f"{head}{base}__{stop_rule or 'none'}__{exit_rule or 'none'}_{'liq1000' if liquidity else 'raw'}"


def run_config(folder: str, hold: int, liquidity: bool, limit=None,
               start=DEFAULT_START, end=DEFAULT_END, write_trades=True, out_root=None,
               stop_rule=None, exit_rule=None, replace=False, subdir="single_island",
               extra_exclude=frozenset(), entry_filter=None) -> dict:
    """
    跑一組設定、寫出結果與診斷，回傳彙總 dict（driver 共用）。out_root 預設 RESULT_DIR。
    extra_exclude：要排除的股票；島型母體由呼叫端傳 data_anomaly.island_exclude()（2026-11-23 起不再固定排 GLITCH）。
    """
    strat = make_exit_variant(hold, liquidity, stop_rule, exit_rule, replace, entry_filter)
    result = batch.run_folder(strat, folder, start=start, end=end, limit=limit,
                              exclude=set(extra_exclude))
    plain = stop_rule is None and exit_rule is None and not replace and entry_filter is None
    tag = (f"h{hold}_{'liq1000' if liquidity else 'raw'}" if plain
           else variant_tag(hold, liquidity, stop_rule, exit_rule, replace, entry_filter))
    out_dir = os.path.join(out_root or RESULT_DIR, subdir, tag)
    batch.write_results(result, out_dir, f"single_island_{tag}", write_trades=write_trades)
    with open(os.path.join(out_dir, "diag.json"), "w", encoding="utf-8") as f:
        json.dump(strat.diag, f, ensure_ascii=False, indent=1)
    return {"tag": tag, **result["aggregate"], **strat.diag, "failed": len(result["failed"])}


def main(argv) -> int:
    parser = argparse.ArgumentParser(description="底部島型反轉：資料夾批次回測（vbt）")
    parser.add_argument("folder")
    parser.add_argument("--hold", type=int, default=20, help="持有根數（5／10／20）")
    parser.add_argument("--liq", action="store_true", help="可成交版：成交金額 > 1,000 萬")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    a = parser.parse_args(argv[1:])
    from _02_strategy.island_strategy.data_anomaly import island_exclude
    res = run_config(a.folder, a.hold, a.liq, a.limit, a.start, a.end, extra_exclude=island_exclude(a.folder))
    print(json.dumps({k: res[k] for k in ("tag", "交易次數", "勝率(%)", "獲利因子(PF)", "平均持有天數",
                                          "signals", "entries", "exits", "open_at_end")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
