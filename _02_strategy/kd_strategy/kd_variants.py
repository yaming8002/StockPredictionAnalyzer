"""
KD 變體參數化（給掃描 driver 用）
==================================
`single_kd_strategy.py` 是教學主體，優化用**註解切換**管理——一次只開一行，讀的人看得出
「這一輪到底跑了什麼」。但 KD 系列文章與 reference（blog/reference/single_kd/）的對照表
加起來上百組，總不能改一行跑一次，所以這裡開一個參數化子類：同樣的條件式，改由類別屬性選。

**條件式凡是策略檔（single_kd_strategy.py）或多股檔（_03_multi_strategy/kd/multi_kd.py）
已有的，一律照抄原文**（不照中文描述重寫）；兩邊改口徑時要一起改，否則掃描表與教學主體
會各自走鐘。程式碼沒有、只留在文章／reference 的條件，照文章裡的寫法，並在條件旁註明出處；
連文章都沒寫出公式的（舊 scratchpad driver 已遺失），在註解標「⚠️ 定義待確認」與選擇理由。

四個屬性：
  ENTRY      進場條件名稱（字串或 tuple，tuple 內全部 AND）。預設基礎＝黃金交叉，
             其餘名稱是疊在上面的濾網；名稱若屬於 ENTRY_BASES（純檔位、連續確認），
             就取代黃金交叉當基礎（一個 ENTRY 至多一個基礎）。() ＝ 純黃金交叉。
  EXIT       出場條件名稱（字串或 tuple，tuple 內任一觸發即出＝OR）。
  LIQUIDITY  進場流動性門檻：'none' 不加／'lot' 5 日均量 > 1000 張（#2a）／
             'amt' 成交金額 5日均量×股價 > TURNOVER（#2b，預設 1,000 萬；舊 3,000 萬版另設 TURNOVER）。
             門檻只 gate 進場，出場不加（與策略檔、多股檔同口徑）。
  OVERSOLD / OVERBOUGHT  超賣／超買門檻（預設 20/80）；30/70、10/90 門檻變形用。
             影響：低檔區、高檔死叉、過熱 K 下彎、純檔位的 K/D 上穿門檻。
             K 跌破 80／跌破 50 是文章裡的固定線，不跟著門檻變。

出場大多是「判定日」向量化條件（含頂頂低，見 _lower_high_stateless）。唯二例外是
2026-07-26 CHoCH 校準的兩種「依進場」定義（PATH_EXITS：文章版、公開稀疏版），基準要從
進場那天算，只能逐根掃描——所以 KdVariant 覆寫 build_signals：EXIT 含 PATH_EXITS 才走掃描，
其餘仍走基底向量化；掃描時進場一律取 self.entry_signal(df)（見 single.py）。

用法：
    v = KdVariant(); v.ENTRY, v.EXIT, v.LIQUIDITY = ("breakout60",), "high_death", "amt"
    res = v.run(df_全史, sid, start=DEFAULT_START, end=DEFAULT_END)
"""
import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd

from _01_data.indicators_momentum_volume import calculate_obv
from _01_data.indicators_pattern import calculate_zigzag
from _02_strategy.kd_strategy.single_kd_strategy import (
    MID_LINE, OVERBOUGHT, OVERSOLD, TURNOVER_MIN, VOL_LOT_MIN, SingleKDStrategy)

# 頂頂低用的 ZigZag 回檔門檻（與 MACD／均線交叉的 CHoCH、文章 kd-cross-exit 同為 2%）
ZIGZAG_PCT = 0.02
# 均線（收盤 rolling 均值內算；calculate_sma 為空殼，與策略檔 bull_trend 同做法）
MA_PERIODS = (5, 20, 60, 120, 200)
# 創新高回看天數
BREAKOUT_PERIODS = (20, 60, 120, 250)
# K 線品質「上影短」：上影線 ≤ 當日振幅的 1/3（⚠️ 定義待確認，見 _short_upper）
UPPER_SHADOW_MAX = 1 / 3
# K 跌破 80：文章 kd-cross-exit 的固定線（不隨 OVERBOUGHT 門檻變形）
K_EXIT_LINE = 80


def add_extra_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    策略檔 add_columns 之外、變體條件會用到的欄位（每檔只算一次，所有變體共用）。
    都是跟門檻無關的欄；跟 OVERSOLD／OVERBOUGHT 有關的條件在條件函式裡從 k/d 直接比。
    """
    close = df["close"]
    for n in MA_PERIODS:
        df[f"ma{n}"] = close.rolling(n).mean()
    for n in BREAKOUT_PERIODS:
        df[f"hh{n}"] = close.rolling(n).max()
    # OBV：引用 _01_data.calculate_obv（parquet 沒存）；20 日均線版給 2026-07-26 的「OBV>自身20MA」用
    calculate_obv(df)
    df["obv_ma20"] = df["obv"].rolling(20).mean()
    # 頂頂低（無狀態）：ZigZag 確認日才有值的稀疏擺動高點
    calculate_zigzag(df, ZIGZAG_PCT)
    df["lower_high"] = _lower_high_stateless(df["zigzag_turn_high"])
    return df


def _lower_high_stateless(turn_high: pd.Series) -> pd.Series:
    """
    頂頂低（無狀態盤勢版）：照抄文章 kd-cross-exit 的 KDExitLowerHigh。
    市場依序的擺動高點裡，後一個比前一個低 → 在「確認日」（回檔達 2% 那天）標 True。

    為什麼不沿用 macd_variants 的 EXIT_LOWER_HIGH：那支是 _scan_path_exits 逐根掃描，
    基準是「進場以來」的擺動高點（peak 從 −∞ 起、進場重置）＝依賴進場點，
    即 2026-07-26 reference 的「公開稀疏版」（KD 黃金交叉 PF 1.093），不是這裡要的版本。
    用戶 2026-07-27 定案 KD 一律用「不依進場、直接依盤勢」的無狀態版（PF 1.033），
    它跟持倉無關，所以能寫成向量化的判定日訊號，不需要覆寫 build_signals。
    確認日只用到當日以前的收盤，無 look-ahead；成交照基底延到隔日開盤。
    """
    highs = turn_high.dropna()
    lower = highs < highs.shift(1)
    out = pd.Series(False, index=turn_high.index)
    out.loc[highs.index[lower.to_numpy()]] = True
    return out


def _short_upper(df: pd.DataFrame) -> pd.Series:
    """
    K 線品質「上影短」：上影線（high − max(open, close)）≤ 當日振幅 × 1/3；一字線（振幅 0）算短。
    ⚠️ 定義待確認：只出現在 2026-07-20 reference（PF 0.768~0.773 區間），文章與程式碼都沒公式，
    原 driver 已遺失。1/3 振幅是常見的「上影不長」判法，數字只用來重現該區間的量級。
    """
    upper = df["high"] - df[["open", "close"]].max(axis=1)
    rng = df["high"] - df["low"]
    return (upper <= rng * UPPER_SHADOW_MAX) | (rng <= 0)


def _cross_down(s: pd.Series, line: float) -> pd.Series:
    """由上往下穿越 line（昨 ≥ line、今 < line）；寫法同策略檔 #7、文章 KDExitDown80。"""
    return (s < line) & (s.shift(1) >= line)


def _both_turn(cond: pd.Series) -> pd.Series:
    """K、D 兩條「都」進入某狀態的那一天（今天兩條都成立、昨天沒有兩條都成立）。"""
    return cond & ~cond.shift(1, fill_value=False)


# ── 進場基礎（取代黃金交叉；一個 ENTRY 至多一個）────────────────────────────
ENTRY_BASES = {
    # 連續確認（2026-07-20 reference／文章 kd-cross-entry-filters「隔日 K 仍>D」）：
    # 黃金交叉的「隔天」K 仍在 D 之上才算進場判定日。⚠️ 定義待確認（原 driver 遺失）：
    # 照字面寫成「昨日黃金交叉 & 今日 K>D」；流動性門檻看判定日（今天）。
    "confirm2": lambda s, df: df["golden"].shift(1, fill_value=False) & (df["k"] > df["d"]),
    # 純檔位（2026-07-20 reference §2，不判斷交叉）：K 由下往上穿過超賣線（K↑20）就買。
    "zone_k_up20": lambda s, df: (df["k"] > s.OVERSOLD) & (df["k"].shift(1) <= s.OVERSOLD),
    # 純檔位 D 線版：D↑20 買
    "zone_d_up20": lambda s, df: (df["d"] > s.OVERSOLD) & (df["d"].shift(1) <= s.OVERSOLD),
    # 純檔位 K、D 兩條都版：⚠️ 定義待確認——取「K、D 兩條都站上 20 的那一天」
    "zone_kd_up20": lambda s, df: _both_turn((df["k"] > s.OVERSOLD) & (df["d"] > s.OVERSOLD)),
}

# ── 進場濾網（疊在基礎上，AND）──────────────────────────────────────────────
ENTRY_FILTERS = {
    # 策略檔 #1：低檔（K、D 都 < 超賣門檻）
    "low_zone": lambda s, df: (df["k"] < s.OVERSOLD) & (df["d"] < s.OVERSOLD),
    # multi_kd low_redk 原文：低檔且當日收紅（c >= o；文件寫「收>開」，以程式碼為準）
    "low_redk": lambda s, df: (df["k"] < s.OVERSOLD) & (df["d"] < s.OVERSOLD) & (df["close"] >= df["open"]),
    # 2026-07-26 MA20/OBV 篇：50 中線上下（交叉當日）
    "high_zone50": lambda s, df: (df["k"] > MID_LINE) & (df["d"] > MID_LINE),
    "d_above50": lambda s, df: df["d"] > MID_LINE,
    "low_zone50": lambda s, df: (df["k"] < MID_LINE) & (df["d"] < MID_LINE),
    "d_below50": lambda s, df: df["d"] < MID_LINE,
    # multi_kd breakoutN 原文：收盤創近 N 日新高（含當日）
    "breakout20": lambda s, df: df["close"] >= df["hh20"],
    "breakout60": lambda s, df: df["close"] >= df["hh60"],
    "breakout120": lambda s, df: df["close"] >= df["hh120"],
    "breakout250": lambda s, df: df["close"] >= df["hh250"],
    # multi_kd gap 原文：今開 > 昨高
    "gap": lambda s, df: df["open"] > df["high"].shift(1),
    # 文章 kd-cross-entry-filters KDMaBull：5 > 20 > 60；20 > 60 > 120 同寫法
    "bull_align_5_20_60": lambda s, df: (df["ma5"] > df["ma20"]) & (df["ma20"] > df["ma60"]),
    "bull_align_20_60_120": lambda s, df: (df["ma20"] > df["ma60"]) & (df["ma60"] > df["ma120"]),
    # 策略檔 #5：長線均線多頭 MA120 > MA200
    "ma120_gt_ma200": lambda s, df: df["bull_trend"],
    # 站上單一均線：當日收盤 > MA（2026-07-26 reference「MA20＝當日 close > 20 日均線」，其餘同寫法）
    "above_ma20": lambda s, df: df["close"] > df["ma20"],
    "above_ma60": lambda s, df: df["close"] > df["ma60"],
    "above_ma120": lambda s, df: df["close"] > df["ma120"],
    "above_ma200": lambda s, df: df["close"] > df["ma200"],
    # 策略檔 #3：CMF>0
    "cmf_pos": lambda s, df: df["cmf"] > 0,
    # 策略檔 #4：底背離（價創 20 日新低、K 未創 20 日新低）
    "divergence": lambda s, df: df["bull_divergence"],
    # OBV 上升兩種定義（兩份紀錄數字不同，交易數也差一截，判定是不同定義）：
    #   obv_up_ma20：2026-07-26 reference 明文「OBV > 自身 20 日均線」（PF 0.766、196,623 筆）
    #   obv_up_1d  ：2026-07-20 reference／文章總表的「OBV 上升」（PF 0.767~0.777、297,198 筆＝基準的 9 成）
    #               ⚠️ 定義待確認：原 driver 遺失；9 成通過率只有「今日 OBV > 昨日」說得通，照此實作
    "obv_up_ma20": lambda s, df: df["obv"] > df["obv_ma20"],
    "obv_up_1d": lambda s, df: df["obv"] > df["obv"].shift(1),
    # 量能（2026-07-20 reference／文章總表）：
    #   vol_above_ma5：「帶量（量>5日均量）」字面
    #   vol_x1_5／vol_x2：「量增 1.5／2 倍」⚠️ 定義待確認——基準量沒寫，取同表「帶量」的 5 日均量
    "vol_above_ma5": lambda s, df: df["volume"] > df["vol_ma5"],
    "vol_x1_5": lambda s, df: df["volume"] > df["vol_ma5"] * 1.5,
    "vol_x2": lambda s, df: df["volume"] > df["vol_ma5"] * 2,
    # 交叉強度（2026-07-26 reference：交叉當日 K−D > 門檻）
    "kd_spread2": lambda s, df: (df["k"] - df["d"]) > 2,
    "kd_spread5": lambda s, df: (df["k"] - df["d"]) > 5,
    "kd_spread10": lambda s, df: (df["k"] - df["d"]) > 10,
    # K 線品質（2026-07-20 reference「非黑K／上影短／綜合」；文章只列非黑K）
    #   非黑K：收 ≥ 開（與 low_redk 的紅K 同寫法）；綜合＝非黑K 且 上影短
    "candle_not_black": lambda s, df: df["close"] >= df["open"],
    "candle_short_upper": lambda s, df: _short_upper(df),
    "candle_combo": lambda s, df: (df["close"] >= df["open"]) & _short_upper(df),
}

# ── 出場（判定日條件；tuple 內任一觸發即出）──────────────────────────────────
EXITS = {
    # 策略檔 baseline：死亡交叉
    "death": lambda s, df: df["death"],
    # 策略檔 #1／#6：死叉且高檔（K、D 都 > 超買門檻）
    "high_death": lambda s, df: df["death"] & (df["k"] > s.OVERBOUGHT) & (df["d"] > s.OVERBOUGHT),
    # 文章 KDExitDown80：K 由上跌破 80
    "k_down80": lambda s, df: _cross_down(df["k"], K_EXIT_LINE),
    # 策略檔 #7：K 由上跌破 50
    "k_down50": lambda s, df: _cross_down(df["k"], MID_LINE),
    # 純檔位 D 線版出場：D 跌破 50／80（K 版的同寫法換成 D）
    "d_down80": lambda s, df: _cross_down(df["d"], K_EXIT_LINE),
    "d_down50": lambda s, df: _cross_down(df["d"], MID_LINE),
    # 純檔位 K、D 兩條都版出場：⚠️ 定義待確認——取「K、D 兩條都跌到線下的那一天」
    "kd_down80": lambda s, df: _both_turn((df["k"] < K_EXIT_LINE) & (df["d"] < K_EXIT_LINE)),
    "kd_down50": lambda s, df: _both_turn((df["k"] < MID_LINE) & (df["d"] < MID_LINE)),
    # 策略檔 #8：昨 K > 超買門檻、今 K 下彎
    "climax": lambda s, df: (df["k"].shift(1) > s.OVERBOUGHT) & (df["k"] < df["k"].shift(1)),
    # 策略檔 #9：頂背離（價創 20 日新高、K 未創 20 日新高）
    "top_div": lambda s, df: df["top_divergence"],
    # 文章 KDExitLowerHigh：頂頂低（無狀態盤勢版）
    "lower_high": lambda s, df: df["lower_high"],
    # 跌破 MA20（2026-07-20 reference §4/§5「跌破 MA20」）：⚠️ 定義待確認——寫成「下穿」
    # （昨收 ≥ MA20、今收 < MA20），與 MACD 跌破年線、本檔 K 跌破 80 的穿越寫法一致；
    # 狀態版（收盤 < MA20 就賣）會讓收在均線下的進場隔天立刻被洗出，不像停損本意。
    "below_ma20": lambda s, df: (df["close"] < df["ma20"]) & (df["close"].shift(1) >= df["ma20"].shift(1)),
}

# ── 路徑相依出場（2026-07-26 CHoCH 校準，reference kd_choch_public_calibration）──────
# 三種都是「純 lower-high」：持倉中出現比基準低的擺動高點就出，差別只在基準從哪算、ZigZag 怎麼算。
#   choch_sparse     ：公開稀疏版＝照 ma_cross_strategy.build_signals 的 CHoCH 掃描（peak 進場時重設為 −∞，
#                      要進場後先立一個頂、下一個頂更低才出）；⚠️ 只取 lower-high、不疊 ma_cross 的死叉
#                      （reference 說三版差異「全在基準從哪算」，判定三版同為純 lower-high）。
#   choch_article    ：文章版（無 look-ahead 寫法）＝基準＝進場判定日當下最近的擺動高點（進場前高點），
#                      之後創高就上移、更低即出；照 lab dow_strategy_20260517_w120_p50_CHoCH.py 的
#                      peak_turn_high_since_buy 寫法，ZigZag 用公開稀疏版往後填（確認日才更新）。
#   choch_article_lab：文章版（重現舊數字用）＝同上，但 ZigZag 換成 lab 的 forward-fill 寫法
#                      （_zigzag_ffill_lab，抄 lab dow_climax_strategy.compute_zigzag_close）。
#                      ⚠️ 那支在確認日會把新高點「回寫到轉折當天」，回測逐根讀到的是未來才確認的值＝look-ahead；
#                      2026-10-09 冒煙 40 檔：本版勝率 61%／PF 2.9，無 look-ahead 版只有 37%／1.19（≈無狀態版），
#                      舊 reference 的 2.13（勝率 53%、抱 14 天）輪廓只有本版對得上 → 判定舊數字來自這個 look-ahead。
#                      原 driver 已遺失、無法逐行確認，僅供重產對照，不可當策略結論。
# 值＝(擺動高點欄來源, 進場時 peak 初值來源)；來源 None＝−∞
PATH_EXITS = {
    "choch_sparse": ("sparse", None),
    "choch_article": ("sparse", "sparse_ffill"),
    "choch_article_lab": ("lab_ffill", "lab_ffill"),
}


def _zigzag_ffill_lab(close: np.ndarray, pct: float = ZIGZAG_PCT) -> np.ndarray:
    """
    抄 lab/_02_strategy/dow/dow_climax_strategy.py 的 compute_zigzag_close（只留 turn_high）。
    確認日起往後填最近擺動高點，並把新高點回寫到轉折當天（turn_highs[ext_hi_i]）——後者即 look-ahead。
    """
    n = len(close)
    turn_highs = np.full(n, np.nan)
    if n == 0:
        return turn_highs
    direction = 0
    ext_hi = ext_lo = close[0]
    ext_hi_i = 0
    last_turn_high = np.nan
    for i in range(1, n):
        c = close[i]
        if direction >= 0 and c >= ext_hi:
            ext_hi, ext_hi_i = c, i
        if direction <= 0 and c <= ext_lo:
            ext_lo = c
        made_pivot = False
        if direction >= 0 and (ext_hi - c) / ext_hi >= pct:
            last_turn_high = ext_hi
            turn_highs[ext_hi_i] = ext_hi           # 回寫轉折當天（look-ahead 來源）
            direction, ext_lo = -1, c
            made_pivot = True
        if not made_pivot and direction <= 0 and (c - ext_lo) / ext_lo >= pct:
            direction, ext_hi, ext_hi_i = 1, c, i
        if not np.isnan(last_turn_high):
            turn_highs[i] = last_turn_high
    return turn_highs


def _path_series(df: pd.DataFrame, source) -> np.ndarray:
    """PATH_EXITS 的來源 → 陣列（只在用到的變體才算，不進共用備欄）。"""
    if source is None:
        return None
    if source == "sparse":
        return df["zigzag_turn_high"].to_numpy(dtype=float)
    if source == "sparse_ffill":
        return df["zigzag_turn_high"].ffill().to_numpy(dtype=float)
    if source == "lab_ffill":
        return _zigzag_ffill_lab(df["close"].to_numpy(dtype=float))
    raise ValueError(f"未知擺動高點來源 {source}")


def _scan_lower_high(entry: np.ndarray, other_exit: np.ndarray, turn_high: np.ndarray,
                     peak_init) -> tuple:
    """
    逐根掃描：判定日進場 → 持倉中遇到「擺動高點 < peak」或其他出場條件就出（判定日）。
    turn_high 是稀疏擺動高點（確認日才有值）；peak_init 為 None 時進場重設成 −∞，
    否則取進場判定日的 peak_init[i]（NaN 視同 −∞：還沒有任何擺動高點可當基準）。
    與 ma_cross 的掃描同慣例：進場那根不檢查出場、出場那根不再進場。
    """
    n = len(entry)
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    in_pos = False
    peak = -np.inf
    for i in range(n):
        if not in_pos:
            if entry[i]:
                in_pos = True
                entries[i] = True
                base = np.nan if peak_init is None else peak_init[i]
                peak = -np.inf if np.isnan(base) else base
            continue
        th = turn_high[i]
        if not np.isnan(th):
            if th < peak:                    # 頂頂低（相對基準）→ 出場
                exits[i] = True
                in_pos = False
                continue
            peak = max(peak, th)
        if other_exit[i]:
            exits[i] = True
            in_pos = False
    return entries, exits


# 顯示名稱（表格用）
NAME_ENTRY = {
    "confirm2": "連續確認（隔日K仍>D）", "zone_k_up20": "純檔位 K↑20",
    "zone_d_up20": "純檔位 D↑20", "zone_kd_up20": "純檔位 K、D 都↑20",
    "low_zone": "低檔 K,D<20", "low_redk": "低檔＋紅K", "high_zone50": "高位 K,D>50",
    "d_above50": "只 D>50", "low_zone50": "低位 K,D<50", "d_below50": "只 D<50",
    "breakout20": "創20日新高", "breakout60": "創60日新高", "breakout120": "創120日新高",
    "breakout250": "創250日新高", "gap": "跳空", "bull_align_5_20_60": "多頭 5>20>60",
    "bull_align_20_60_120": "多頭 20>60>120", "ma120_gt_ma200": "MA120>MA200",
    "above_ma20": "站上MA20", "above_ma60": "站上MA60", "above_ma120": "站上MA120",
    "above_ma200": "站上MA200", "cmf_pos": "CMF>0", "divergence": "底背離",
    "obv_up_ma20": "OBV>20MA", "obv_up_1d": "OBV上升", "vol_above_ma5": "帶量",
    "vol_x1_5": "量增1.5倍", "vol_x2": "量增2倍", "kd_spread2": "K−D>2",
    "kd_spread5": "K−D>5", "kd_spread10": "K−D>10", "candle_not_black": "非黑K",
    "candle_short_upper": "上影短", "candle_combo": "K線品質綜合",
}
NAME_EXIT = {
    "death": "死叉", "high_death": "高檔死叉", "k_down80": "K跌破80", "k_down50": "K跌破50",
    "d_down80": "D跌破80", "d_down50": "D跌破50", "kd_down80": "K,D跌破80",
    "kd_down50": "K,D跌破50", "climax": "過熱K下彎", "top_div": "頂背離",
    "lower_high": "頂頂低", "below_ma20": "跌破MA20",
    "choch_article": "CHoCH文章版", "choch_article_lab": "CHoCH文章版(lab ZigZag)",
    "choch_sparse": "CHoCH公開稀疏版",
}


def _as_tuple(x) -> tuple:
    return (x,) if isinstance(x, str) else tuple(x)


class KdVariant(SingleKDStrategy):
    """進場 × 出場 × 流動性 × 門檻 參數化的 KD 策略；條件式與策略檔註解切換行同一份。"""

    ENTRY = ()                  # () ＝ 純黃金交叉
    EXIT = "death"
    LIQUIDITY = "none"          # 'none' / 'lot' / 'amt'
    TURNOVER = TURNOVER_MIN     # 'amt' 的金額門檻（舊 3,000 萬版改設 30_000_000）
    OVERSOLD = OVERSOLD
    OVERBOUGHT = OVERBOUGHT

    def add_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """策略檔的 add_columns（KD、交叉、CMF、背離、長多…）＋ 變體額外欄位。"""
        df = super().add_columns(df)
        return add_extra_columns(df)

    def buy_signal(self, df: pd.DataFrame) -> pd.Series:
        names = _as_tuple(self.ENTRY)
        bases = [n for n in names if n in ENTRY_BASES]
        if len(bases) > 1:
            raise ValueError(f"進場基礎只能有一個：{bases}")
        signal = ENTRY_BASES[bases[0]](self, df) if bases else df["golden"]
        for n in names:
            if n in ENTRY_BASES:
                continue
            if n not in ENTRY_FILTERS:
                raise ValueError(f"未知進場條件 {n}")
            signal = signal & ENTRY_FILTERS[n](self, df)
        liq = self.LIQUIDITY
        if liq == "lot":
            signal = signal & (df["vol_ma5"] > VOL_LOT_MIN)          # 策略檔 #2a
        elif liq == "amt":
            signal = signal & (df["turnover"] > self.TURNOVER)       # 策略檔 #2b
        elif liq != "none":
            raise ValueError(f"未知流動性門檻 {liq}")
        return signal.fillna(False).astype(bool)

    def sell_signal(self, df: pd.DataFrame) -> pd.Series:
        """向量化出場（OR）；PATH_EXITS 不在這裡算，由 build_signals 掃描疊上。"""
        names = _as_tuple(self.EXIT)
        if not names:
            raise ValueError("EXIT 至少要一條")
        signal = pd.Series(False, index=df.index)
        for n in names:
            if n in PATH_EXITS:
                continue
            if n not in EXITS:
                raise ValueError(f"未知出場條件 {n}")
            signal = signal | EXITS[n](self, df)
        return signal.fillna(False).astype(bool)

    def build_signals(self, df: pd.DataFrame):
        """EXIT 不含 PATH_EXITS → 基底向量化；含 → 逐根掃描，自己 shift(1) 成隔日開盤成交。"""
        path = [n for n in _as_tuple(self.EXIT) if n in PATH_EXITS]
        if not path:
            return super().build_signals(df)
        if len(path) > 1:
            raise ValueError(f"路徑相依出場只能有一個：{path}")
        high_src, peak_src = PATH_EXITS[path[0]]
        turn_high = _path_series(df, high_src)
        peak_init = turn_high if peak_src == high_src else _path_series(df, peak_src)
        entries, exits = _scan_lower_high(
            self.entry_signal(df).to_numpy(),           # 判定日進場（區間前不算）
            self.sell_signal(df).to_numpy(),            # 其餘向量化出場（OR）
            turn_high, peak_init)
        e = pd.Series(entries, index=df.index).shift(1, fill_value=False)
        x = pd.Series(exits, index=df.index).shift(1, fill_value=False)
        return e, x

    def label(self) -> str:
        """表格用的組合名稱，例：創60日新高 × 高檔死叉。"""
        entry = "＋".join(NAME_ENTRY[n] for n in _as_tuple(self.ENTRY)) or "黃金交叉"
        exit_ = " 或 ".join(NAME_EXIT[n] for n in _as_tuple(self.EXIT))
        return f"{entry} × {exit_}"
