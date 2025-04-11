import pandas as pd
from collections import defaultdict

# 讀取資料
file_path = "./stock_data/leaning_label/sma_120_sma_200_volume_trades.csv"
df = pd.read_csv(file_path, parse_dates=["buy_date", "sell_date"])

# 統計每天的交易重疊次數
overlap_counter = defaultdict(int)

for _, row in df.iterrows():
    for date in pd.date_range(row["buy_date"], row["sell_date"]):
        overlap_counter[date.date()] += 1

# 整理成 DataFrame
overlap_df = pd.DataFrame(overlap_counter.items(), columns=["date", "overlap_count"])

# 加入「年月」欄位（yyyy-mm 格式）
overlap_df["year_month"] = overlap_df["date"].apply(lambda x: x.strftime("%Y-%m"))

# 依照年月統計每月的平均重疊次數（也可以改成總和）
monthly_overlap = overlap_df.groupby("year_month")["overlap_count"].mean().reset_index()
monthly_overlap = monthly_overlap.sort_values(by="overlap_count", ascending=False)

# 顯示結果
print("🔺 每月平均重疊次數（越高代表當月交易活動越頻繁）：")
print(monthly_overlap.head(10))

# 若要輸出整份：monthly_overlap.to_csv("monthly_overlap.csv", index=False)
