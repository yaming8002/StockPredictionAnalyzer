"""
vbt 策略套件 common — 各策略共用的低階工具
================================================

只放「與引擎無關」的純函式與跨策略共用常數：台股費率 / tick 進位、精確費用重建、
欄位驗證、summary 組裝，以及標準回測區間、資料品質排除集（GLITCH）等全策略共用口徑。
策略基底（single）以「呼叫這些函式」共用，不靠類別繼承。
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

# 台股費率
COMMISSION = 0.001425      # 手續費率
DUES = 0.003               # 證交稅（賣方）
MIN_COMMISSION = 20.0      # 單筆最低手續費

# 統一小寫欄位（與資料層約定一致）
OHLCV = ("open", "high", "low", "close", "volume")

# 標準回測區間：後續測試一律以此為主；策略檔可用 --start/--end 覆蓋。
# 起點 2002 而非資料起點（2000-10）或 2001：指標本身需要暖身期才算得準（MA200 就要 200 根、
# MACD 12/26/9 也要數十根），起點壓在資料開頭會拿半成品的指標值產生訊號。掐掉 2000 殘月與 2001
# 這段暖身期，等指標穩定後再開始計算交易；尾端同理掐掉 2026 未滿年。
# 全策略共用單一定義，勿在各策略檔另立第二份。
DEFAULT_START = "2002-01-01"
DEFAULT_END = "2025-12-31"

# 價格 glitch 壞資料股（近零價/天價，見 docs data-quality 掃描）排除集：跨策略共用的資料品質口徑，
# 全市場回測一律事前剔除、與 reference 同口徑。只排那 5 檔確定非物理價的；增資/縮表等合法公司行為
# 造成的大跳不在此列、不誤殺。各策略檔一律 import 此單一定義，勿另立第二份。
GLITCH = {"3591.TW", "8039.TW", "8027.TWO", "6283.TW", "3666.TWO"}

# ── 路徑：單一定義，各 driver 一律 import 這裡，不要再寫死本機絕對路徑 ──────────
# 這是公開 repo，別人 clone 下來要能跑。三個位置都可用環境變數覆寫，沒設就用 repo 內預設：
#   STOCK_DATA_DIR  股價 parquet 全史所在目錄（預設 <repo>/stock_data）
#   CHART_OUT_DIR   產圖腳本的輸出目錄（預設 <repo>/result/charts；result/ 不進版控）
#   BLOG_DIR        文章對照／驗證腳本要讀的 blog 專案根目錄（沒有預設，見 require_blog_dir）
# 作者本機把前兩個指到 repo 外的共用位置，行為與寫死時相同。
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.environ.get("STOCK_DATA_DIR") or os.path.join(_REPO, "stock_data")
CHART_DIR = os.environ.get("CHART_OUT_DIR") or os.path.join(_REPO, "result", "charts")
DIVIDEND_FILE = (os.environ.get("DIVIDEND_FILE")
                 or os.path.join(_REPO, "dividends", "dividend_actions.parquet"))


def require_blog_dir() -> str:
    """
    文章對照／驗證類腳本用：要讀已發佈文章與其數據表，但**文章內容不在這個公開 repo**，
    只有「產生那些數字的程式」在。所以不給預設值，沒設就講清楚原因而不是丟 FileNotFound。
    """
    d = os.environ.get("BLOG_DIR")
    if not d:
        raise SystemExit(
            "這支腳本要對照已發佈的文章，請先設環境變數 BLOG_DIR 指向 blog 專案根目錄：\n"
            "    set BLOG_DIR=D:/path/to/blog               （Windows）\n"
            "    export BLOG_DIR=/path/to/blog                （bash）\n"
            "文章內容不在這個公開 repo 裡，這裡只有產生那些數字的回測與驗證程式。")
    return d


def tw_tick_arr(prices) -> np.ndarray:
    """
    台股升降單位「無條件進位」（向量化）。買賣價一律過此函式。
    NaN 會原樣保留（np.ceil(nan)=nan）。
    """
    p = np.asarray(prices, dtype=np.float64)
    tick = np.select(
        [p < 10, p < 50, p < 100, p < 500, p < 1000],
        [0.01, 0.05, 0.1, 0.5, 1.0],
        default=5.0,
    )
    return np.round(np.ceil(p / tick) * tick, 2)


def reconstruct_fees(buy_price, sell_price, qty):
    """
    精確台股費用重建（vbt 純 ratio 表達不了 min 20 + 賣方證交稅，故出 trades 後校正）。
    回傳 (buy_fee, sell_fee)，皆 np.ceil 無條件進位。
    """
    buy_price = np.asarray(buy_price, dtype=np.float64)
    sell_price = np.asarray(sell_price, dtype=np.float64)
    qty = np.asarray(qty, dtype=np.float64)

    buy_amt = buy_price * qty
    sell_amt = sell_price * qty
    buy_fee = np.ceil(np.maximum(buy_amt * COMMISSION, MIN_COMMISSION))
    sell_fee = np.ceil(np.maximum(sell_amt * COMMISSION, MIN_COMMISSION) + sell_amt * DUES)
    return buy_fee, sell_fee


def net_pnl(buy_price, sell_price, qty):
    """每筆已扣台股費用的淨損益。"""
    buy_fee, sell_fee = reconstruct_fees(buy_price, sell_price, qty)
    bp = np.asarray(buy_price, dtype=np.float64)
    sp = np.asarray(sell_price, dtype=np.float64)
    q = np.asarray(qty, dtype=np.float64)
    return (sp - bp) * q - buy_fee - sell_fee


def ensure_columns(df: pd.DataFrame, required=OHLCV) -> None:
    """檢查必要欄位齊全；缺欄直接報錯（不靜默吞）。"""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"資料缺少必要欄位: {missing}（現有: {list(df.columns)}）")


def trim_outliers(series: pd.Series):
    """
    MAD 去極值（k=3）：以中位數 ± 3×MAD 為界，回傳 (filtered, lower, upper)。
    對齊 lab strategy_runner 的 trim_outliers（auto 模式，不做 win/lose 夾擠）。
    """
    if len(series) == 0:
        return series, None, None
    median = series.median()
    mad = (series - median).abs().median()
    lower = median - 3 * mad
    upper = median + 3 * mad
    filtered = series[(series >= lower) & (series <= upper)]
    return filtered, lower, upper


def confidence_interval(series: pd.Series):
    """95% t 分布信賴區間（對齊 lab strategy_runner）。樣本 < 2 回 (nan, nan)。"""
    if len(series) < 2:
        return (np.nan, np.nan)
    mean = np.mean(series)
    std_err = stats.sem(series)
    ci = stats.t.interval(0.95, len(series) - 1, loc=mean, scale=std_err)
    return (round(float(ci[0]), 2), round(float(ci[1]), 2))


def _trim_block(win_df: pd.DataFrame, lose_df: pd.DataFrame, win_rate: float) -> dict:
    """
    去極值（MAD）後的指標。「去極值」定義：在獲利、虧損子集各自移除
    『淨損益落在 中位數 ± 3×MAD 外』的交易，再對存活交易計各項平均
    （金額 / 報酬率 / 持有天數 / EV 同屬一個族群，敘事一致）。
    """
    def keep(sub: pd.DataFrame):
        """回傳 (存活交易 df, 下界, 上界)；空集合回 (sub, None, None)。"""
        if sub.empty:
            return sub, None, None
        _, low, high = trim_outliers(sub["profit"])
        kept = sub[(sub["profit"] >= low) & (sub["profit"] <= high)]
        return kept, low, high

    win_keep, win_low, win_high = keep(win_df)
    lose_keep, lose_low, lose_high = keep(lose_df)
    kept_all = pd.concat([win_keep, lose_keep])

    avg_win_trim = float(win_keep["profit"].mean()) if not win_keep.empty else 0.0
    avg_lose_trim = float(lose_keep["profit"].mean()) if not lose_keep.empty else 0.0
    avg_win_rate_trim = float(win_keep["profit_rate"].mean()) if not win_keep.empty else 0.0
    avg_lose_rate_trim = float(lose_keep["profit_rate"].mean()) if not lose_keep.empty else 0.0
    avg_hold_trim = float(kept_all["hold_days"].mean()) if not kept_all.empty else 0.0
    expect_trim = win_rate * avg_win_trim + (1 - win_rate) * avg_lose_trim

    # Trim PF（去極值後獲利因子）：與 EV(Trim) 同一套 MAD 去極值集合上，存活獲利總額 / 存活虧損總額。
    # 量化「PF 有多少來自肥尾」——raw PF 高但 Trim PF 接近 1，代表 edge 大半靠少數極端大贏單。
    sum_win_trim = float(win_keep["profit"].sum()) if not win_keep.empty else 0.0
    sum_lose_trim = abs(float(lose_keep["profit"].sum())) if not lose_keep.empty else 0.0
    pf_trim = round(sum_win_trim / sum_lose_trim, 4) if sum_lose_trim > 0 else float("inf")

    return {
        "IQR獲利下限": round(win_low, 2) if win_low is not None else np.nan,
        "IQR獲利上限": round(win_high, 2) if win_high is not None else np.nan,
        "IQR虧損下限": round(lose_low, 2) if lose_low is not None else np.nan,
        "IQR虧損上限": round(lose_high, 2) if lose_high is not None else np.nan,
        "排除極值後平均獲利金額": round(avg_win_trim, 2),
        "排除極值後平均虧損金額": round(avg_lose_trim, 2),
        "排除極值後平均獲利報酬率(%)": round(avg_win_rate_trim, 2),
        "排除極值後平均虧損報酬率(%)": round(avg_lose_rate_trim, 2),
        "排除極值後平均持有天數": round(avg_hold_trim, 2),
        "排除極值後期望報酬值(EV,Trim)": round(expect_trim, 2),
        "排除極值後獲利因子(PF,Trim)": pf_trim,
    }


def _empty_summary() -> dict:
    """無交易時的零值 summary（欄位與正常情況一致，供下游對齊）。"""
    keys = ["交易次數", "勝率(%)", "平均獲利金額", "平均虧損金額",
            "平均獲利報酬率(%)", "平均虧損報酬率(%)", "中位數報酬率(%)",
            "最大獲利", "最大虧損",
            "最大獲利報酬率(%)", "最大虧損報酬率(%)", "平均持有天數",
            "期望報酬值(EV)", "獲利因子(PF)", "總獲利", "IQR獲利下限", "IQR獲利上限",
            "IQR虧損下限", "IQR虧損上限", "排除極值後平均獲利金額",
            "排除極值後平均虧損金額", "排除極值後平均獲利報酬率(%)",
            "排除極值後平均虧損報酬率(%)", "排除極值後平均持有天數",
            "排除極值後期望報酬值(EV,Trim)", "排除極值後獲利因子(PF,Trim)",
            "獲利信賴區間下限(95%)", "獲利信賴區間上限(95%)",
            "虧損信賴區間下限(95%)", "虧損信賴區間上限(95%)"]
    summary = {k: 0.0 for k in keys}
    summary["交易次數"] = 0
    return summary


def summarize_trades(records: pd.DataFrame) -> dict:
    """
    從 trades 表組完整 summary（指標對齊 lab strategy_runner）。
    金額類用淨損益 real_pnl（已扣台股費稅）；報酬率類用買賣價 gross %。
    排除淨損益=0 的交易（與 lab 一致）。
    """
    if len(records) == 0:
        return _empty_summary()

    df = records.copy()
    df["profit"] = pd.to_numeric(df["real_pnl"], errors="coerce").fillna(0.0)
    df = df[df["profit"] != 0]
    n = len(df)
    if n == 0:
        return _empty_summary()

    # 報酬率：買賣價毛報酬（不含費用，對齊 lab）；持有天數：賣出 − 買入
    df["profit_rate"] = (df["sell_price"] - df["buy_price"]) / df["buy_price"] * 100
    df["hold_days"] = (pd.to_datetime(df["sell_date"]) - pd.to_datetime(df["buy_date"])).dt.days

    win_df = df[df["profit"] > 0]
    lose_df = df[df["profit"] < 0]
    win_rate = len(win_df) / n
    avg_win = float(win_df["profit"].mean()) if not win_df.empty else 0.0
    avg_lose = float(lose_df["profit"].mean()) if not lose_df.empty else 0.0
    avg_win_rate = float(win_df["profit_rate"].mean()) if not win_df.empty else 0.0
    avg_lose_rate = float(lose_df["profit_rate"].mean()) if not lose_df.empty else 0.0
    expect_value = win_rate * avg_win + (1 - win_rate) * avg_lose

    # 獲利因子（PF）= 獲利總額 / 虧損總額（>1 才賺）。先前未輸出、靠下游手算，現一併納入。
    total_win = float(win_df["profit"].sum()) if not win_df.empty else 0.0
    total_lose = abs(float(lose_df["profit"].sum())) if not lose_df.empty else 0.0
    profit_factor = round(total_win / total_lose, 4) if total_lose > 0 else float("inf")

    # 去極值（MAD）後各項指標 + 95% 信賴區間
    trim = _trim_block(win_df, lose_df, win_rate)
    win_ci = confidence_interval(win_df["profit"]) if not win_df.empty else (np.nan, np.nan)
    lose_ci = confidence_interval(lose_df["profit"]) if not lose_df.empty else (np.nan, np.nan)

    base = {
        "交易次數": n,
        "勝率(%)": round(win_rate * 100, 2),
        "平均獲利金額": round(avg_win, 2),
        "平均虧損金額": round(avg_lose, 2),
        "平均獲利報酬率(%)": round(avg_win_rate, 2),
        "平均虧損報酬率(%)": round(avg_lose_rate, 2),
        # 中位數報酬率：所有交易毛報酬率的中位數（典型的一筆長怎樣）。平均被少數右尾大單抬高，
        # 中位數才看得出「一半以上的交易其實在小賠」——量化獲利對肥尾的依賴。
        "中位數報酬率(%)": round(float(df["profit_rate"].median()), 2),
        "最大獲利": round(float(df["profit"].max()), 2),
        "最大虧損": round(float(df["profit"].min()), 2),
        "最大獲利報酬率(%)": round(float(df["profit_rate"].max()), 2),
        "最大虧損報酬率(%)": round(float(df["profit_rate"].min()), 2),
        "平均持有天數": round(float(df["hold_days"].mean()), 2),
        "期望報酬值(EV)": round(expect_value, 2),
        "獲利因子(PF)": profit_factor,
        "總獲利": round(float(df["profit"].sum()), 2),
    }
    ci = {
        "獲利信賴區間下限(95%)": win_ci[0],
        "獲利信賴區間上限(95%)": win_ci[1],
        "虧損信賴區間下限(95%)": lose_ci[0],
        "虧損信賴區間上限(95%)": lose_ci[1],
    }
    return {**base, **trim, **ci}


# ── 回測結果主表「規格列」helper（對齊 memory backtest-result-table-spec）─────────────
# 規格＝單一主表固定 10 欄：策略鍵（標籤，由呼叫端傳）＋下列 9 個指標欄（順序固定）。
# 所有 driver 一律用 spec_row 出列，杜絕「手挑欄位」造成漏欄（driver 曾因手打 row 漏掉持有天/獲利均/虧損均）。
SPEC_COLUMNS = ("交易次數", "勝率%", "平均持有天", "獲利平均%", "虧損平均%",
                "中位數%", "期望值/筆", "獲利因子", "總獲利(萬)")

# 規格欄名 → summarize_trades 回傳的 summary key（"總獲利(萬)" 為特例：總獲利 ÷ 10000）
_SPEC_MAP = {
    "交易次數": "交易次數",
    "勝率%": "勝率(%)",
    "平均持有天": "平均持有天數",
    "獲利平均%": "平均獲利報酬率(%)",
    "虧損平均%": "平均虧損報酬率(%)",
    "中位數%": "中位數報酬率(%)",
    "期望值/筆": "期望報酬值(EV)",
    "獲利因子": "獲利因子(PF)",
}


def spec_row(summary: dict, **labels) -> dict:
    """把 summarize_trades 的 summary 轉成「規格固定 9 指標欄」的有序 row（標籤欄在前）。

    labels：策略鍵與任意額外標籤欄（如 進場/投法/份數/擋單），依傳入順序置於指標欄之前。
    缺任何必要 summary key → 直接 raise（源頭杜絕漏欄；別再手打 row）。
    """
    row = dict(labels)                       # 標籤欄在前（dict 保序）
    for col in SPEC_COLUMNS:
        if col == "總獲利(萬)":
            if "總獲利" not in summary:
                raise KeyError("summary 缺『總獲利』，無法出規格列")
            row[col] = round(summary["總獲利"] / 10000, 1)
            continue
        src = _SPEC_MAP[col]
        if src not in summary:
            raise KeyError(f"summary 缺『{src}』（對應規格欄『{col}』），無法出規格列")
        row[col] = summary[src]
    return row


def assert_spec_columns(rows) -> None:
    """出表前檢查：每列都含規格 9 欄，缺就 raise。rows 可為 list[dict] 或 DataFrame。"""
    if hasattr(rows, "columns"):             # DataFrame
        cols = set(rows.columns)
    elif rows:
        cols = set(rows[0])
    else:
        cols = set()
    missing = [c for c in SPEC_COLUMNS if c not in cols]
    if missing:
        raise ValueError(f"結果表缺規格欄：{missing}（規格 9 欄＝{list(SPEC_COLUMNS)}）")
