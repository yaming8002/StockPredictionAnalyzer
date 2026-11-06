"""
雙底（底底高）策略（single double bottom）— double_bottom_strategy 套件下的方案
==============================================================================

繼承 VbtSingleStrategy，交易一律由 vbt 執行（Portfolio.from_signals）；本檔只產生進出場訊號。

擺動點：`_01_data.indicators_pattern.calculate_zigzag`（收盤價、門檻 pct）。它只在「反轉幅度
達 pct 的確認當日」才把擺動低點／高點填進 zigzag_turn_low／zigzag_turn_high，確認之前是 NaN，
所以用這兩欄判斷不會用到未來資料。

規則（只做多；判定日收盤 → 隔日開盤成交）：
  進場：某日確認出新擺動低點 L2，且前一個已確認擺動低點 L1 存在、L2 > L1（底底高）。
        **第二個底一定要已經確認**——訊號落在 L2 的確認日，不會在低點當天或確認前進場。
  出場（雙判斷，任一成立即出，同日都成立記為「同日」）：
    ① CHoCH（lower-high）：進場後新確認的擺動高點，低於進場以來的最高擺動高點
       （與均線交叉系列的 CHoCH 同一定義）。
    ② 跌破波段低點：收盤跌破最近一個已確認的擺動低點（持倉期間有新的已確認低點就更新）。

優化以「註解切換」管理（比照其他系列）：全部註解＝baseline（① 或 ②）。
  # 優化 #1（不設停損）：出場只留 ① CHoCH，拿掉 ② 跌破波段低點。切換＝把類別屬性 STOP_LOW 那行換成
    下一行註解的 False。沒有停損時部位可能長期不出場，一定要看診斷的期末未平倉率與未平倉部位的持有天數。
    （2026-10-16 加入；全組合掃描由 double_bottom_sweep.py --nostop 以子類設定，結果寫 single_double_bottom_nostop/）

  進場優化（2026-10-20 加入；每條單獨疊在基本版進場上，buy_signal 內一次只開一行）：
  # 優化 #2 頸線突破：第二個底確認後，收盤突破「頸線」那天才判定進場。頸線＝兩個底之間那個已確認的擺動高點
    （ZigZag 的轉折高點就是那段上升腿的最高收盤，確認日一定早於第二個底的確認日，判定時已知）。
    等待從第二個底的確認日（含當天）起算；超過 NECK_WAIT 個交易日、或期間又確認出新的擺動低點，就作廢。
  # 優化 #3 兩底間隔：兩個底「極值本身那天」相隔 ≥ GAP_MIN 個交易日。calculate_zigzag 只記確認日，
    極值日另外推回：該段下跌腿（上一次轉折確認日 ～ 本次確認日）裡收盤最低的最後一天（同 calculate_zigzag
    遇到相同低價會更新的寫法），只用確認日當天以前的資料。
  # 優化 #4 第二底墊高幅度：L2 ≥ L1 × RISE_MIN。
  # 優化 #5 站上年線：判定日收盤 > MA200（同 MACD 系列條件⑥）。
  # 優化 #6 長線多頭排列：MA20 > MA60 > MA120。2026-10-22 用戶改定，原本是 MA5 > MA20 > MA60（同 KD／MACD
    系列已發佈的定義）；改的理由：雙底是打底，第二個底確認時短均線 MA5 必然下彎，短線多頭會擋掉太多。
    舊定義保留在 add_columns 的註解行，舊結果在 result/single_double_bottom_entry/p*__align_ma5/。
  # 優化 #7 放量確認：判定日成交量 > 20 日均量 × 1.5（同 MACD 系列條件③）。
  （2026-10-21 追加 #8～#11，判定日一樣是第二個底的確認日）
  # 優化 #8 前段跌幅：第一個底之前，從前一個已確認擺動高點跌到第一個底的幅度 ≥ PRIOR_DROP_MIN。
  # 優化 #9 型態高度：頸線 ≥ 兩個底中較高者 × NECK_HEIGHT_MIN。
  # 優化 #10 第二底量縮：第二個底附近的平均量 < 第一個底附近的平均量。「附近」＝以極值日為中心前後
    VOL_HALF 根（共 5 根），但**不超過該底的確認日**——極值日之後的 K 棒在確認日當下可能還沒發生，
    所以第二個底的窗口在「確認日 − 極值日 < 2」時會被截短。極值日推法同 #3。
  # 優化 #11 確認日 K 棒：確認日收紅 K（收盤 > 開盤），或長下影線（下影線 ≥ 實體 × 2 且 ≥ 全日振幅 ÷ 3），任一。
  #5～#11 的判定日＝第二個底的確認日。全部條件只用判定日當下已知資訊（截斷測試見 verify_entry_no_lookahead.py）。
  全組合掃描用 double_bottom_variants.py（條件式逐字抄這裡的註解行）。

  出場優化（2026-10-29 加入；進場固定基本版，每條單獨「附加」在 ①② 上、先觸發者算，切換＝類別屬性
  EXIT_RULE 那幾行一次只開一行）。頸線＝進場判定日的 db_neck；型態高度＝頸線 − 第一個底（第二個底較高，
  第一個底就是最低點）；進場價＝判定日隔天開盤。來源整理見 blog/reference/double_bottom/ 出場紀錄。
  # 優化 #12 測幅滿足點：收盤 ≥ 頸線 ＋ 型態高度。
  # 優化 #13 打折測幅（Bulkowski）：收盤 ≥ 頸線 ＋ 0.65 × 型態高度。
  # 優化 #14 頸線停利（Bulkowski DB 交易設定）：收盤 ≥ 頸線。進場判定日收盤就已在頸線之上的，隔天就出場。
  # 優化 #15 突破失敗：持倉中收盤曾 > 頸線，之後收盤 < 頸線。
  # 優化 #16 跌破突破 K 棒低點：持倉中第一次收盤 > 頸線那天記下當日最低價，之後收盤 < 該價。
  # 優化 #17 保本停損：持倉中收盤曾 ≥ 進場價 ＋ 型態高度，之後收盤 < 進場價。
  # 優化 #18 ATR 移動停損：收盤 < 進場後最高價 − 3 × ATR(14)（ATR 取當日；同 MACD 系列吊燈出場）。
  # 優化 #19 回落 20%：收盤 < 進場後最高價 × 0.80（最高價寫法同 MACD 系列回落 10%）。
  # 優化 #20 跌破 MA10：收盤由上跌破 MA10（昨收 ≥ 昨日 MA10、今收 < 今日 MA10；寫法同 MACD 系列跌破年線）。
  # 優化 #21 高檔 M 頭：新確認的擺動高點 < 上一個已確認擺動高點（可在進場前，通常就是頸線）。
    與 ① 的差別：① 只跟進場以後的高點比。寫法同下方附錄的雙頂，但這裡是附加、不取代 ①。
  #12～#16 需要頸線：沒有頸線的進場（兩個底之間沒有確認過的高點）這幾條不觸發。
  診斷多記 exit_new（出場那根新規則成立，含與 ①② 同根）、exit_new_only（只有新規則成立＝搶在 ①② 之前）。
  附錄：替換（2026-11-03 加入）：類別屬性 REPLACE 換成下一行註解的 True，出場只剩 EXIT_RULE 那一條、①② 都不用。
    只當附錄對照，不是優化編號；沒有停損類的規則部位會長期不出場，看診斷的期末未平倉率
    （掃描：exit_filter_sweep.py --replace，結果寫 single_double_bottom_exit_replace/）。

附錄（舊版，2026-10-16 第一輪，結果在 result/single_double_bottom_v1_dtop/）：出場是「CHoCH 二選一
  （① 或 ②）＋雙頂（頂頂低）」——持倉中新確認高點 H2 < 上一個已確認高點 H1 就出場（H1 可能在進場前）。
  持倉中只要出現頂頂低就已出場，所以持倉期間確認的高點一路墊高、「進場以來最高」恆等於「上一個高點」，
  ① 一定被雙頂先吃掉（實測觸發 0 次，那 4 組等於只用雙頂出場）→ 本版拿掉雙頂、改成 ①② 雙判斷。
  build_signals 內「# 附錄：雙頂」幾行保留當時的雙頂寫法供對照（不是現行規則）。

為什麼 pct 用 CLI 參數（不用註解切換）：
  pct 是 baseline 本身要並列比較的設定（2%~10%），不是「在 baseline 上疊一條優化」，比照海龜用
  --system 選兩套原典系統。之後的優化（濾網、停損等）一律走註解切換。
流動性門檻（可成交版）同理用 --liq：成交金額＝5 日均量×股價 > 1,000 萬（與其他系列同一基準）。

診斷：每次回測另外累計進場數、①②各自觸發次數、同日同時成立次數、期末仍未平倉的部位數（self.diag），
由 driver 寫出——取代型的出場若常常不觸發，已平倉交易的統計會失真，要一起看。

執行：
    python _02_strategy/double_bottom_strategy/single_double_bottom_strategy.py <資料夾> --pct 0.02 [--liq]
  輸出：result/single_double_bottom/p<pct>_<raw|liq1000>/
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

from _01_data.indicators_pattern import calculate_zigzag
from _01_data.indicators_volatility import calculate_atr
from _02_strategy.base.vbt import batch
from _02_strategy.base.vbt.single import VbtSingleStrategy
from _02_strategy.base.vbt.common import GLITCH, DEFAULT_START, DEFAULT_END

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
TURNOVER_MIN = 10_000_000
NECK_WAIT = 60            # 優化 #2：第二個底確認後最多等幾個交易日突破頸線
GAP_MIN = 10              # 優化 #3：兩個底的極值日至少相隔幾個交易日
RISE_MIN = 1.02           # 優化 #4：第二個底至少比第一個底高 2%
MA_LONG = 200             # 優化 #5：年線
VOL_WINDOW, VOL_MULTIPLE = 20, 1.5   # 優化 #7：量 > 20 日均量 × 1.5
PRIOR_DROP_MIN = 0.15     # 優化 #8：第一個底之前的跌幅 ≥ 15%
NECK_HEIGHT_MIN = 1.10    # 優化 #9：頸線 ≥ 較高的底 × 1.10
VOL_HALF = 2              # 優化 #10：底的極值日前後各 2 根（共 5 根）平均量
TARGET_DISCOUNT = 0.65    # 優化 #13：Bulkowski 測幅打折（他統計到達目標的比率）
ATR_PERIOD, CHANDELIER_K = 14, 3.0   # 優化 #18：進場後最高價 − 3 × ATR(14)
TRAIL_DROP = 0.20         # 優化 #19：自進場後最高價回落 20%
MA_SHORT = 10             # 優化 #20：MA10


def _vol_near(volume: np.ndarray, ext_day: int, confirm_day: int) -> float:
    """極值日前後 VOL_HALF 根的平均量，右端不超過確認日（之後的量在確認當下還不知道）。"""
    lo, hi = max(0, ext_day - VOL_HALF), min(ext_day + VOL_HALF, confirm_day)
    return float(np.nanmean(volume[lo:hi + 1]))


def double_bottom_context(close: np.ndarray, turn_low: np.ndarray, turn_high: np.ndarray,
                          volume: np.ndarray = None):
    """
    在每個「擺動低點確認日」記下雙底需要的資訊（其餘日 NaN），全部只用確認日當天以前的資料：
      l1      ：前一個已確認擺動低點
      gap     ：前一個低點與這個低點「極值日」相隔的交易日數
      neck    ：兩個低點之間最近一個已確認的擺動高點（頸線）
    給了 volume 時另外回傳（#8、#10 用）：
      drop1   ：第一個底之前那個已確認擺動高點跌到第一個底的幅度（(H0 − L1) ÷ H0）
      vol1／vol2：第一／第二個底附近的平均量（_vol_near）
    極值日＝該段下跌腿（上一次任何轉折的確認日 ～ 本次確認日）裡收盤最低的最後一天。
    """
    n = len(close)
    l1 = np.full(n, np.nan)
    gap = np.full(n, np.nan)
    neck = np.full(n, np.nan)
    drop1, vol1, vol2 = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    last_turn = 0                 # 上一次任何轉折（高或低）的確認日
    prev_low_val, prev_low_day = np.nan, -1
    prev_low_drop, prev_low_vol = np.nan, np.nan
    last_high_val, last_high_day = np.nan, -1
    for i in range(n):
        if not np.isnan(turn_high[i]):
            last_high_val, last_high_day = turn_high[i], i
            last_turn = i
        if not np.isnan(turn_low[i]):
            leg = close[last_turn:i + 1]
            # 收盤有缺值（NaN）的檔：calculate_zigzag 比較時跳過 NaN，這裡也用 nanmin 才找得到同一個極值
            ext_day = last_turn + int(np.flatnonzero(leg == np.nanmin(leg))[-1])
            # 這個低點自己的「前段跌幅」與「附近量」，等它變成下一組的第一個底時使用
            my_drop = (last_high_val - turn_low[i]) / last_high_val if last_high_day >= 0 else np.nan
            my_vol = _vol_near(volume, ext_day, i) if volume is not None else np.nan
            if prev_low_day >= 0:
                l1[i] = prev_low_val
                gap[i] = ext_day - prev_low_ext
                if last_high_day > prev_low_day:          # 兩個底之間確認過的高點
                    neck[i] = last_high_val
                drop1[i], vol1[i], vol2[i] = prev_low_drop, prev_low_vol, my_vol
            prev_low_val, prev_low_day, prev_low_ext = turn_low[i], i, ext_day
            prev_low_drop, prev_low_vol = my_drop, my_vol
            last_turn = i
    if volume is None:
        return l1, gap, neck
    return l1, gap, neck, drop1, vol1, vol2


def neckline_breakout(close: np.ndarray, setup: np.ndarray, turn_low: np.ndarray,
                      neck: np.ndarray, wait: int = NECK_WAIT) -> np.ndarray:
    """
    優化 #2：setup（底底高成立的確認日）之後，第一天收盤 > 頸線就判定進場。
    等待從確認日當天算起 wait 個交易日；期間又確認新的擺動低點則作廢（新的低點若也成立會重新起算）。
    """
    n = len(close)
    out = np.zeros(n, dtype=bool)
    level, deadline = np.nan, -1
    for i in range(n):
        if not np.isnan(turn_low[i]):                      # 新的低點：舊的等待作廢
            level, deadline = (neck[i], i + wait) if setup[i] and not np.isnan(neck[i]) else (np.nan, -1)
        if i > deadline:
            level = np.nan
        if not np.isnan(level) and close[i] > level:
            out[i] = True
            level, deadline = np.nan, -1
    return out


class ExtraExit:
    """
    出場優化 #12～#21 的單筆持倉狀態：進場判定日呼叫 start(i)，持倉中每根呼叫 hit(...)。
    有狀態的規則（#15～#17 的「曾經突破／曾經賺到」）一律先用前一根為止的狀態判定、再更新，
    所以當根不會同時觸發與啟動；#18、#19 的最高價含當根（同 MACD 系列 _scan_path_exits）。
    """

    def __init__(self, rule: str, df: pd.DataFrame):
        self.rule = rule
        self.open = df["open"].to_numpy(dtype=float)
        self.high = df["high"].to_numpy(dtype=float)
        self.low = df["low"].to_numpy(dtype=float)
        self.close = df["close"].to_numpy(dtype=float)
        self.neck_col = df["db_neck"].to_numpy(dtype=float)
        self.l1_col = df["db_l1"].to_numpy(dtype=float)
        self.atr = df[f"atr_{ATR_PERIOD}"].to_numpy(dtype=float)
        self.ma_break = df["ma10_break"].to_numpy(dtype=bool)

    def start(self, i: int):
        self.neck = self.neck_col[i]
        self.height = self.neck - self.l1_col[i]
        self.entry_px = self.open[i + 1] if i + 1 < len(self.open) else np.nan
        self.peak_px = -np.inf
        self.broke, self.bo_low, self.armed = False, np.nan, False

    def hit(self, j: int, new_high: bool, high_val: float, prev_high: float) -> bool:
        c, rule = self.close[j], self.rule
        self.peak_px = max(self.peak_px, self.high[j])
        if rule == "target":                                   # 優化 #12
            return c >= self.neck + self.height
        if rule == "target65":                                 # 優化 #13
            return c >= self.neck + TARGET_DISCOUNT * self.height
        if rule == "neck":                                     # 優化 #14
            return c >= self.neck
        if rule == "neckfail":                                 # 優化 #15
            fired = self.broke and c < self.neck
            self.broke = self.broke or c > self.neck
            return fired
        if rule == "bolow":                                    # 優化 #16
            fired = (not np.isnan(self.bo_low)) and c < self.bo_low
            if np.isnan(self.bo_low) and c > self.neck:
                self.bo_low = self.low[j]
            return fired
        if rule == "breakeven":                                # 優化 #17
            fired = self.armed and c < self.entry_px
            self.armed = self.armed or c >= self.entry_px + self.height
            return fired
        if rule == "atr":                                      # 優化 #18
            return (not np.isnan(self.atr[j])) and c < self.peak_px - CHANDELIER_K * self.atr[j]
        if rule == "trail20":                                  # 優化 #19
            return c < self.peak_px * (1.0 - TRAIL_DROP)
        if rule == "ma10":                                     # 優化 #20
            return bool(self.ma_break[j])
        if rule == "mtop":                                     # 優化 #21
            return new_high and (not np.isnan(prev_high)) and high_val < prev_high
        raise ValueError(f"未知的出場優化 {rule}")


class SingleDoubleBottomStrategy(VbtSingleStrategy):
    """雙底（底底高）進場、CHoCH 或跌破波段低點出場。訊號由逐根掃描產生，成交交給 vbt。"""

    STOP_LOW = True      # baseline：② 跌破波段低點出場（兼停損）
    # STOP_LOW = False   # 優化 #1：不設停損，出場只留 ① CHoCH

    EXIT_RULE = None             # baseline：只有 ①②
    # EXIT_RULE = "target"       # 優化 #12 測幅滿足點
    # EXIT_RULE = "target65"     # 優化 #13 打折測幅 0.65
    # EXIT_RULE = "neck"         # 優化 #14 頸線停利
    # EXIT_RULE = "neckfail"     # 優化 #15 突破後跌回頸線
    # EXIT_RULE = "bolow"        # 優化 #16 跌破突破 K 棒低點
    # EXIT_RULE = "breakeven"    # 優化 #17 保本停損（賺到一個型態高度後）
    # EXIT_RULE = "atr"          # 優化 #18 ATR 移動停損 3×ATR(14)
    # EXIT_RULE = "trail20"      # 優化 #19 自最高價回落 20%
    # EXIT_RULE = "ma10"         # 優化 #20 跌破 MA10
    # EXIT_RULE = "mtop"         # 優化 #21 高檔 M 頭

    REPLACE = False      # baseline：EXIT_RULE 附加在 ①② 上，先觸發者算
    # REPLACE = True     # 附錄：替換，出場只剩 EXIT_RULE 那一條（EXIT_RULE 為 None 時不作用）

    def __init__(self, pct: float = 0.02, liquidity: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pct = pct
        self.liquidity = liquidity
        self._trade_end = None
        self.diag = {"entries": 0, "exit_choch": 0, "exit_low": 0, "exit_both": 0, "open_at_end": 0,
                     "open_days_sum": 0, "open_days_max": 0,
                     "exits": 0, "exit_new": 0, "exit_new_only": 0,   # 出場優化 #12～#21 的生效率用
                     "signals": 0}   # signals＝區間內的進場判定日數（含持倉中被忽略的），算生效率用   # 期末未平倉部位：進場到區間終點的日曆天

    def add_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        calculate_zigzag(df, self.pct)
        close = df["close"]
        df["turnover"] = df["volume"].rolling(5).mean() * close
        l1, gap, neck, drop1, vol1, vol2 = double_bottom_context(
            close.to_numpy(dtype=float), df["zigzag_turn_low"].to_numpy(),
            df["zigzag_turn_high"].to_numpy(), df["volume"].to_numpy(dtype=float))
        df["db_l1"], df["db_gap"], df["db_neck"] = l1, gap, neck
        df["db_drop1"], df["db_vol1"], df["db_vol2"] = drop1, vol1, vol2
        # 優化 #11：確認日 K 棒（紅 K 或長下影線）
        body = (df["close"] - df["open"]).abs()
        lower = df[["open", "close"]].min(axis=1) - df["low"]
        df["db_bull_bar"] = (df["close"] > df["open"]) | (
            (lower >= body * 2) & (lower >= (df["high"] - df["low"]) / 3))
        # 進場優化用欄位（全部只看判定日以前）
        df["ma200"] = close.rolling(MA_LONG).mean()
        # df["bull_align"] = ((close.rolling(5).mean() > close.rolling(20).mean())     # 舊 #6 MA5>MA20>MA60
        #                     & (close.rolling(20).mean() > close.rolling(60).mean()))  # 2026-10-22 用戶改定為長線版
        df["bull_align"] = ((close.rolling(20).mean() > close.rolling(60).mean())
                            & (close.rolling(60).mean() > close.rolling(120).mean()))
        df["vol_ma20"] = df["volume"].rolling(VOL_WINDOW).mean()
        # 出場優化用欄位：#18 ATR、#20 收盤由上跌破 MA10
        calculate_atr(df, window=ATR_PERIOD)
        ma10 = close.rolling(MA_SHORT).mean()
        df["ma10_break"] = ((close < ma10) & (close.shift(1) >= ma10.shift(1))).fillna(False)
        return df

    def base_signal(self, df: pd.DataFrame) -> pd.Series:
        """基本版判定日：確認出新擺動低點，且高於前一個已確認低點（底底高）。"""
        return (df["zigzag_turn_low"] > df["db_l1"]).fillna(False)   # 非確認日 NaN → False

    def buy_signal(self, df: pd.DataFrame) -> pd.Series:
        """進場判定日＝基本版（＋最多一條進場優化）＋可成交門檻。"""
        signal = self.base_signal(df)
        # signal = pd.Series(neckline_breakout(df["close"].to_numpy(dtype=float), signal.to_numpy(), df["zigzag_turn_low"].to_numpy(), df["db_neck"].to_numpy()), index=df.index)  # 優化 #2 頸線突破
        # signal = signal & (df["db_gap"] >= GAP_MIN)                       # 優化 #3 兩底間隔 ≥ 10 個交易日
        # signal = signal & (df["zigzag_turn_low"] >= df["db_l1"] * RISE_MIN)  # 優化 #4 第二底墊高 ≥ 2%
        # signal = signal & (df["close"] > df["ma200"])                     # 優化 #5 站上年線
        # signal = signal & df["bull_align"]                                # 優化 #6 長線多頭排列 20>60>120
        # signal = signal & (df["volume"] > df["vol_ma20"] * VOL_MULTIPLE)  # 優化 #7 放量 1.5 倍
        # signal = signal & (df["db_drop1"] >= PRIOR_DROP_MIN)              # 優化 #8 前段跌幅 ≥ 15%
        # signal = signal & (df["db_neck"] >= df["zigzag_turn_low"] * NECK_HEIGHT_MIN)  # 優化 #9 型態高度（L2 為較高者）
        # signal = signal & (df["db_vol2"] < df["db_vol1"])                 # 優化 #10 第二底量縮
        # signal = signal & df["db_bull_bar"]                               # 優化 #11 確認日紅 K 或長下影線
        if self.liquidity:
            signal = signal & (df["turnover"] > TURNOVER_MIN)
        return signal.fillna(False).astype(bool)

    def sell_signal(self, df: pd.DataFrame) -> pd.Series:
        """出場是路徑相依（看進場後的擺動點），在 build_signals 逐根掃描；這裡不使用。"""
        return pd.Series(False, index=df.index)

    def window_signals(self, df: pd.DataFrame, start=None, end=None):
        self._trade_end = end                               # 診斷要知道區間終點（數期末未平倉）
        return super().window_signals(df, start, end)

    def build_signals(self, df: pd.DataFrame):
        """
        逐根掃描產生「判定日」進出場，再 shift(1) 成隔日開盤成交日交給 vbt。
        持倉中不接受新進場（與 vbt accumulate=False 一致）；同一天出場後最快隔一天才再進場。
        """
        entry_raw = self.entry_signal(df).to_numpy()
        close = df["close"].to_numpy(dtype=float)
        tl = df["zigzag_turn_low"].to_numpy()
        th = df["zigzag_turn_high"].to_numpy()
        n = len(df)
        entries = np.zeros(n, dtype=bool)
        exits = np.zeros(n, dtype=bool)
        reason = np.zeros(n, dtype=np.int8)                 # 位元：1=① CHoCH 2=② 跌破低點 4=出場優化 #12～#21
        extra = ExtraExit(self.EXIT_RULE, df) if self.EXIT_RULE else None

        last_low = np.nan       # 最近一個已確認擺動低點（含當日確認）
        last_high = np.nan      # 最近一個已確認擺動高點（#21 用；不分持倉與否都要追）
        # last_high = np.nan    # 附錄：雙頂 — 上一個已確認擺動高點
        in_pos = False
        peak = -np.inf          # ① 用：進場以來最高擺動高點
        for i in range(n):
            new_high = not np.isnan(th[i])
            prev_high, last_high = last_high, (th[i] if new_high else last_high)
            # prev_high = last_high; last_high = th[i] if new_high else last_high   # 附錄：雙頂
            if not np.isnan(tl[i]):
                last_low = tl[i]
            if not in_pos:
                if entry_raw[i]:
                    in_pos = True
                    entries[i] = True
                    peak = -np.inf
                    if extra:
                        extra.start(i)
                continue
            # ── 持倉中：判定出場 ──
            hit_choch = new_high and th[i] < peak                                  # ① lower-high
            hit_low = self.STOP_LOW and (not np.isnan(last_low)) and close[i] < last_low   # ② 跌破波段低點
            # hit_choch = hit_choch or (new_high and not np.isnan(prev_high) and th[i] < prev_high)  # 附錄：雙頂
            hit_new = bool(extra and extra.hit(i, new_high, th[i], prev_high))     # 出場優化（附加）
            if self.REPLACE and extra:                                             # 附錄：替換，①② 不用
                hit_choch = hit_low = False
            if new_high:
                peak = max(peak, th[i])
            if hit_choch or hit_low or hit_new:
                exits[i] = True
                reason[i] = (1 if hit_choch else 0) + (2 if hit_low else 0) + (4 if hit_new else 0)
                in_pos = False

        self._count_diag(df.index, entries, exits, reason, entry_raw)
        e = pd.Series(entries, index=df.index).shift(1, fill_value=False)
        x = pd.Series(exits, index=df.index).shift(1, fill_value=False)
        return e, x

    def _count_diag(self, idx, entries, exits, reason, entry_raw):
        """只數成交日落在交易區間內的事件（判定日＋1 才成交）。"""
        lo = np.datetime64(pd.Timestamp(self._trade_start) if self._trade_start else idx[0], "ns")
        hi = np.datetime64(pd.Timestamp(self._trade_end) if self._trade_end else idx[-1], "ns")
        days = idx.values.astype("datetime64[ns]")
        exec_day = np.append(days[1:], np.datetime64("NaT", "ns"))   # 判定日 i 的成交日＝i+1
        inside = (exec_day >= lo) & (exec_day <= hi)
        n_in = int((entries & inside).sum())
        self.diag["signals"] += int((entry_raw & inside).sum())
        self.diag["entries"] += n_in
        ex_in = exits & inside
        self.diag["exit_choch"] += int((ex_in & (reason == 1)).sum())
        self.diag["exit_low"] += int((ex_in & (reason == 2)).sum())
        self.diag["exit_both"] += int((ex_in & (reason == 3)).sum())
        self.diag["exits"] += int(ex_in.sum())
        self.diag["exit_new"] += int((ex_in & ((reason & 4) > 0)).sum())
        self.diag["exit_new_only"] += int((ex_in & (reason == 4)).sum())
        # 期末未平倉：區間內最後一筆進場之後，區間內沒有出場
        ent_pos = np.flatnonzero(entries & inside)
        if len(ent_pos):
            last = ent_pos[-1]
            if not (ex_in[last + 1:]).any():
                self.diag["open_at_end"] += 1
                last_day = min(hi, days[-1])                          # 區間終點（該檔資料若較早結束取資料末日）
                held = int((last_day - exec_day[last]) / np.timedelta64(1, "D"))
                self.diag["open_days_sum"] += held
                self.diag["open_days_max"] = max(self.diag["open_days_max"], held)


class _NoStop(SingleDoubleBottomStrategy):
    """優化 #1（不設停損）供掃描用：與策略檔「# 優化 #1」那行同一設定，不去動策略檔的註解狀態。"""
    STOP_LOW = False


def run_config(folder: str, pct: float, liquidity: bool, limit=None,
               start=DEFAULT_START, end=DEFAULT_END, write_trades=True, nostop=False,
               out_root=None) -> dict:
    """
    跑一組設定、寫出結果與診斷，回傳彙總 dict（driver 共用）。nostop＝優化 #1。
    out_root：輸出根目錄（預設 RESULT_DIR；重跑對帳時指到另一個資料夾，不覆蓋正式結果）。
    """
    cls = _NoStop if nostop else SingleDoubleBottomStrategy
    strat = cls(pct=pct, liquidity=liquidity)
    result = batch.run_folder(strat, folder, start=start, end=end, limit=limit, exclude=GLITCH)
    tag = f"p{int(round(pct * 100))}_{'liq1000' if liquidity else 'raw'}"
    out_dir = os.path.join(out_root or RESULT_DIR,
                           "single_double_bottom_nostop" if nostop else "single_double_bottom", tag)
    batch.write_results(result, out_dir, f"single_double_bottom_{tag}", write_trades=write_trades)
    with open(os.path.join(out_dir, "diag.json"), "w", encoding="utf-8") as f:
        json.dump(strat.diag, f, ensure_ascii=False, indent=1)
    return {"tag": tag, **result["aggregate"], **strat.diag, "failed": len(result["failed"])}


def main(argv) -> int:
    parser = argparse.ArgumentParser(description="雙底（底底高）：資料夾批次回測（vbt）")
    parser.add_argument("folder")
    parser.add_argument("--pct", type=float, default=0.02, help="ZigZag 門檻（0.02～0.10）")
    parser.add_argument("--liq", action="store_true", help="可成交版：成交金額 > 1,000 萬")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    a = parser.parse_args(argv[1:])
    res = run_config(a.folder, a.pct, a.liq, a.limit, a.start, a.end)
    print(json.dumps({k: res[k] for k in ("tag", "交易次數", "勝率(%)", "獲利因子(PF)", "平均持有天數",
                                          "entries", "exit_choch", "exit_low", "exit_both",
                                          "open_at_end")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
