import pandas as pd
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import statistics
from scipy import stats
import logging
# 讀取 CSV（請換成你的實際檔案路徑）
# file_path = "./stock_data/leaning_label/opt_sma_120_sma_200_year_3_sell_3_AVG_trades.csv"
# df = pd.read_csv(file_path, parse_dates=["buy_date", "sell_date"])

# # 用 dict 統計每一天被幾筆交易覆蓋
# overlap_counter = defaultdict(int)

# # 每一筆交易，把持有期間的每一天都記下來
# for _, row in df.iterrows():
#     for date in pd.date_range(row["buy_date"], row["sell_date"]):
#         overlap_counter[date.date()] += 1

# # 整理成 DataFrame
# overlap_df = pd.DataFrame(overlap_counter.items(), columns=["date", "overlap_count"])

# # 找出重疊最多的日期（例如前 10 名）
# top_overlap = overlap_df.sort_values(by="overlap_count", ascending=False).head(10)

# print("🔺 重疊最多的日期（交易重疊次數最多）：")
# print(top_overlap)

# # 計算平均重疊次數
# average_overlap = overlap_df["overlap_count"].mean()
# print(f"\n📊 平均每日重疊交易次數：{average_overlap:.2f}")

# # 計算眾數（最常見的重疊次數）
# mode_overlap = overlap_df["overlap_count"].mode()
# if len(mode_overlap) == 1:
#     print(f"🎯 最常見的重疊次數（眾數）：{mode_overlap.iloc[0]}")
# else:
#     print(f"🎯 最常見的重疊次數（眾數們）：{', '.join(map(str, mode_overlap.tolist()))}")



def analyze_hold_days(hold_days, log=None):
    """
    分析持有天數分布，排除極端值後輸出更穩健的統計資料。
    
    參數:
        hold_days (list[int/float]): 每筆交易的持有天數列表
        log (logging.Logger, optional): 若提供 logger，會自動輸出分析結果

    回傳:
        dict: 包含原始統計、去極值後統計的字典
    """
    if not hold_days:
        msg = "無持有天數數據"
        if log:
            log.info(msg)
        return {"message": msg}

    df_days = pd.Series(hold_days)

    # --- 1️⃣ 原始統計 ---
    raw_max = df_days.max()
    raw_min = df_days.min()
    raw_mean = df_days.mean()
    raw_std = df_days.std(ddof=1)

    # --- 2️⃣ 去除極值 (IQR 法) ---
    Q1 = df_days.quantile(0.25)
    Q3 = df_days.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_days = df_days[(df_days >= lower_bound) & (df_days <= upper_bound)]

    # --- 3️⃣ 穩健統計 (Robust Statistics) ---
    median_days = filtered_days.median()
    mad_days = stats.median_abs_deviation(filtered_days, scale="normal")
    trimmed_mean = stats.trim_mean(filtered_days, 0.1)
    try:
        mode_days = statistics.mode(filtered_days)
    except statistics.StatisticsError:
        mode_days = "無唯一眾數"

    # --- 4️⃣ 輸出統計結果 ---
    summary = {
        "raw": {
            "count": len(df_days),
            "max": float(raw_max),
            "min": float(raw_min),
            "mean": float(raw_mean),
            "std": float(raw_std),
        },
        "filtered": {
            "count": int(len(filtered_days)),
            "median": float(median_days),
            "mad": float(mad_days),
            "trimmed_mean": float(trimmed_mean),
            "mode": mode_days,
            "iqr_lower": float(lower_bound),
            "iqr_upper": float(upper_bound),
        },
    }

    # --- 5️⃣ 輸出到 logger ---
    if log:
        log.info("📊 持有天數統計（含極值排除分析）")
        log.info(
            f"原始: 最大 {raw_max}, 最小 {raw_min}, 平均 {raw_mean:.2f}, 標準差 {raw_std:.2f}"
        )
        log.info(
            f"去極值後: 樣本 {len(filtered_days)}/{len(df_days)}, "
            f"中位數 {median_days:.2f}, 修正標準差(MAD) {mad_days:.2f}, "
            f"修剪平均 {trimmed_mean:.2f}, 眾數 {mode_days}"
        )

    return summary