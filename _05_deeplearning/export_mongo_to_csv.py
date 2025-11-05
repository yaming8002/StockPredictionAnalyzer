# _05_deeplearning/export_mongo_to_csv.py
import os
import pandas as pd
from modules.process_mongo import get_mongo_client, close_mongo_client


# ✅ 欄位限制（避免 Mongo 多餘欄位）
FEATURE_COLS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "sma_5",
    "sma_20",
    "sma_50",
    "sma_60",
    "sma_120",
    "sma_200",
    "bollinger_Lower",
    "bollinger_Upper",
]


def export_mongo_to_csv(output_folder="./stock_data/deep_data", min_rows=200):
    """
    將 MongoDB 每個股票 collection 匯出為 CSV

    ✅ 過濾 TW 開頭的股票
    ✅ 若資料筆數小於 min_rows (預設 200) → 不匯出
    ✅ 只輸出指定 FEATURE_COLS
    """

    os.makedirs(output_folder, exist_ok=True)

    db = get_mongo_client()

    collections = db.list_collection_names()
    collections = [c for c in collections if c.endswith("TW")]  # 過濾非台股 collection

    print(f"⭐ 發現股票資料數量: {len(collections)} 檔")

    for col in collections:
        print(f"\n⬇ 正在匯出 {col} ...")

        cursor = db[col].find({}, {field: 1 for field in FEATURE_COLS}, no_cursor_timeout=True)  # 指定要的欄位

        df = pd.DataFrame(list(cursor))
        cursor.close()

        if df.empty:
            print(f"⚠ 資料為空，略過 {col}")
            continue

        # ✅ 筆數不足不輸出
        if len(df) < min_rows:
            print(f"⚠ 資料筆數不足（{len(df)} < {min_rows}），略過 {col}")
            continue

        df = df[FEATURE_COLS]
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        save_path = os.path.join(output_folder, f"{col}.csv")
        df.to_csv(save_path, index=False, encoding="utf-8-sig")

        print(f"✅ 匯出完成: {save_path}（共 {len(df)} 筆資料）")

    close_mongo_client()
    print("\n🎉 所有股票匯出完成！")
