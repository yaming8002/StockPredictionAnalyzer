import pandas as pd
from collections import defaultdict

# 讀取 CSV（請換成你的實際檔案路徑）
file_path = "./stock_data/leaning_label/sma_20_sma_50_trades.csv"
df = pd.read_csv(file_path, parse_dates=["buy_date", "sell_date"])

# 用 dict 統計每一天被幾筆交易覆蓋
overlap_counter = defaultdict(int)

# 每一筆交易，把持有期間的每一天都記下來
for _, row in df.iterrows():
    for date in pd.date_range(row["buy_date"], row["sell_date"]):
        overlap_counter[date.date()] += 1

# 整理成 DataFrame
overlap_df = pd.DataFrame(overlap_counter.items(), columns=["date", "overlap_count"])

# 找出重疊最多的日期（例如前 10 名）
top_overlap = overlap_df.sort_values(by="overlap_count", ascending=False).head(10)

# 顯示結果
print("🔺 重疊最多的日期（交易重疊次數最多）：")
print(top_overlap)

# 若要輸出整份重疊表：
# overlap_df.to_csv("date_overlap_count.csv", index=False)
