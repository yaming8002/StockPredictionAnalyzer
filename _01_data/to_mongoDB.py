from datetime import datetime
import os
import pandas as pd
import numpy as np
import pymongo
from typing import List, Optional

from modules.config_loader import load_config
from modules.logger import setup_logger
from modules.process_mongo import get_mongo_client
from .stock_technical import calculate_bollinger_bands, calculate_ema, calculate_sma

config = load_config()
logger = setup_logger("../toMongo.log")
def _find_col(df, aliases):
    lower_map = {c.lower(): c for c in df.columns}
    for name in aliases:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None

def process_csv_files():
    """處理 CSV 檔案，增加技術指標並寫入 MongoDB 或輸出為 CSV"""
    original_data_folder = config.get("original_data_folder", "./data")
    output_data_folder   = config.get("output_data_folder", "./processed_data")
    stock_list_file      = config.get("stock_list_path", "./stock_List.csv")  # 股票名單路徑
    use_mongodb          = config.get("use_mongodb", False)

    os.makedirs(output_data_folder, exist_ok=True)

    db = None
    if use_mongodb:
        db = get_mongo_client()

    # 載入股票名單
    if os.path.exists(stock_list_file):
        stock_list_df = pd.read_csv(stock_list_file)
    else:
        stock_list_df = pd.DataFrame(columns=["stock_id"])  # 保底用

    for file in os.listdir(original_data_folder):
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(original_data_folder, file)
        df = pd.read_csv(file_path)

        # ---------- 1) 日期處理：修正 tz 警告（✅ 新增 utc=True + 轉台北時區） ----------
        date_col = _find_col(df, ["date", "Date", "DATE"])
        if date_col:
            # 先轉 UTC（aware），再轉 Asia/Taipei，最後拿掉 tz（naive）
            dt = pd.to_datetime(df[date_col], errors="coerce", utc=True)               # ✅
            dt = dt.dt.tz_convert("Asia/Taipei").dt.tz_localize(None)                  # ✅
            df[date_col] = dt
            # 移除無效日期
            df = df[~df[date_col].isna()]
            # 設成索引
            df = df.set_index(date_col)
        else:
            # 沒找到日期欄位，就嘗試把 index 轉成日期
            idx = pd.to_datetime(df.index, errors="coerce", utc=True)                  # ✅
            # idx 可能全是 NaT；僅保留有效
            mask = ~idx.isna()
            df = df.loc[mask].copy()
            if df.empty:
                logger.error(f"{file} 缺少可解析的日期欄位或資料為空，已跳過")
                continue
            idx = idx[mask].tz_convert("Asia/Taipei").tz_localize(None)                # ✅
            df.index = idx

        # 到這裡保證是 DatetimeIndex（如果不是，直接跳過這個檔）
        if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
            logger.error(f"{file} 缺少可解析的日期欄位或資料為空，已跳過")
            continue

        # 正規化為「純日期的午夜時間」，避免夾帶時分秒
        df.index = df.index.normalize()
        df.index.name = "date"

        # ---------- 2) 去重：同日只留最後一筆 ----------
        before_len = len(df)
        df = df[~df.index.duplicated(keep="last")]
        after_len = len(df)
        dup_removed = before_len - after_len

        # ---------- 3) 數值清理（✅ 新增） ----------
        # 偵測常見數值欄位並轉為 numeric
        price_cols = {
            "open":      _find_col(df, ["open", "opening", "o"]),
            "high":      _find_col(df, ["high", "h"]),
            "low":       _find_col(df, ["low", "l"]),
            "close":     _find_col(df, ["close", "c", "price"]),
            "adj_close": _find_col(df, ["adj close", "adj_close", "adjusted close", "adjusted_close", "ac"]),
        }
        vol_col = _find_col(df, ["volume", "vol", "volumn", "shares"])

        numeric_targets = [c for c in price_cols.values() if c] + ([vol_col] if vol_col else [])
        for col in numeric_targets:
            df[col] = pd.to_numeric(df[col], errors="coerce")  # 非數字 → NaN

        # 規則：
        # - 任一價格欄位（存在者）為 NaN 或 ≤ 0 → 略過該列
        # - 成交量（存在者）為 NaN 或 < 0 → 略過該列（允許 0）
        bad_mask = pd.Series(False, index=df.index)
        for key, col in price_cols.items():
            if col:
                bad_mask |= (~np.isfinite(df[col])) | (df[col] <= 0)
        if vol_col:
            bad_mask |= (~np.isfinite(df[vol_col])) | (df[vol_col] < 0)

        invalid_rows = int(bad_mask.sum())
        if invalid_rows > 0:
            df = df.loc[~bad_mask]
            logger.warning(f"{file} 移除 {invalid_rows} 筆數值有問題的列（價格<=0/非數字 或 成交量<0/非數字）")  # ✅

        if df.empty:
            logger.error(f"{file} 清理後沒有有效資料，已跳過")
            # 從股票名單移除
            stock_id = file.replace(".csv", "")
            if "stock_id" in stock_list_df.columns:
                stock_list_df = stock_list_df[stock_list_df["stock_id"] != stock_id]
            continue

        # 若有刪除（重複或壞數值），回存原始 CSV（含索引 'date'）
        if dup_removed > 0 or invalid_rows > 0:
            df.to_csv(file_path, index=True)
            logger.info(f"{file} 已更新 CSV 檔案（移除重複 {dup_removed} 筆、壞數值 {invalid_rows} 筆）")

        # ---------- 4) 2025-09 至少 18 筆 ----------
        start_date = pd.Timestamp("2025-09-01")
        end_date   = pd.Timestamp("2025-09-30")
        df_sep2025 = df[(df.index >= start_date) & (df.index <= end_date)]
        if len(df_sep2025) < 18:
            stock_id = file.replace(".csv", "")
            if "stock_id" in stock_list_df.columns:
                stock_list_df = stock_list_df[stock_list_df["stock_id"] != stock_id]
                logger.warning(f"{stock_id} 在 2025-09 只有 {len(df_sep2025)} 筆，已從股票名單移除")
            continue

        # ---------- 5) 計算技術指標 ----------
        df = calculate_working(df)

        collection_name = file.replace(".csv", "")

        # ---------- 6) 寫入 MongoDB 或輸出 CSV ----------
        if use_mongodb:
            collection = db[collection_name]
            records = df.reset_index(names="date").to_dict(orient="records")

            if not records:
                logger.warning(f"{file} 沒有有效資料，略過")
                stock_id = file.replace(".csv", "")
                if "stock_id" in stock_list_df.columns:
                    stock_list_df = stock_list_df[stock_list_df["stock_id"] != stock_id]
                    stock_list_df.to_csv(stock_list_file, index=False, encoding="utf-8-sig")
                    logger.warning(f"{stock_id} 已從股票名單移除，更新 {stock_list_file}")
                continue

            for record in records:
                # 統一日期為 YYYY-MM-DD 字串
                d = record.get("date")
                if isinstance(d, pd.Timestamp):
                    record["date"] = d.strftime("%Y-%m-%d")
                elif isinstance(d, datetime.datetime):
                    record["date"] = d.strftime("%Y-%m-%d")
                elif isinstance(d, datetime.date):
                    record["date"] = d.isoformat()
                else:
                    record["date"] = str(d)[:10]

                collection.update_one(
                    {"date": record["date"]},   # 以字串日期做 upsert key
                    {"$set": record},
                    upsert=True
                )

            logger.info(f"{file} 已寫入/更新至 MongoDB (共 {len(records)} 筆)")
        else:
            output_path = os.path.join(output_data_folder, file)
            df.to_csv(output_path, index=True)  # index.name='date' 會寫出
            logger.info(f"{file} 已存入 {output_path}")

    # 最後更新股票名單
    logger.warning(f"更新 {stock_list_file}")
    stock_list_df.to_csv(stock_list_file, index=False, encoding="utf-8-sig")



def calculate_working(df):
    df = calculate_sma(df, window=5)
    df = calculate_sma(df, window=20)
    df = calculate_sma(df, window=50)
    df = calculate_sma(df, window=60)
    df = calculate_sma(df, window=120)
    df = calculate_sma(df, window=200)
    df = calculate_ema(df, span=5)
    df = calculate_ema(df, span=20)
    df = calculate_ema(df, span=50)
    df = calculate_ema(df, span=60)
    df = calculate_ema(df, span=120)
    df = calculate_ema(df, span=200)
    df = calculate_bollinger_bands(df)

    return df.iloc[200:]


def remove_mongoDB():
    """清空 MongoDB 中的 stock_analysis 集合"""
    use_mongodb = config.get("use_mongodb", False)
    if use_mongodb:
        mongo_uri = config.get("mongo_uri", "mongodb://localhost:27017")
        client = pymongo.MongoClient(mongo_uri)
        db_name = config.get("mongo_db", "stock_db")
        client.drop_database(db_name)
        print(f"MongoDB 數據庫 {db_name} 中的所有集合內容已清空")


def update_all_stock_indicators(last_days: int = 100):
    """
    🔄 重新計算 MongoDB 所有股票的技術指標（僅寫入最後 N 天）
    - 預設 N = 100
    - 會多抓取 N + 200 天的資料用於 SMA/EMA 初始化
    - 適合每天更新最新技術指標，不需重算整個歷史
    """
    use_mongodb = config.get("use_mongodb", False)
    if not use_mongodb:
        logger.error("⚠️ config.yaml 未啟用 use_mongodb=True，無法更新資料庫")
        return

    db = get_mongo_client()
    all_collections = [col for col in db.list_collection_names() if "TW" in col]
    if not all_collections:
        logger.warning("⚠️ MongoDB 無股票資料")
        return

    from pymongo import UpdateOne

    logger.info(f"🔄 開始更新技術指標，共 {len(all_collections)} 檔股票（僅最後 {last_days} 天）")

    for stock_id in all_collections:
        collection = db[stock_id]

        # 多取200天作為緩衝區，確保EMA等指標不失真
        limit_days = last_days + 200
        cursor = collection.find({}, {"_id": 0}).sort("date", -1).limit(limit_days)
        df = pd.DataFrame(list(cursor))

        if df.empty or "date" not in df.columns:
            logger.warning(f"{stock_id} 無有效資料，略過")
            continue

        # 日期處理
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].notna()].sort_values("date")
        df.set_index("date", inplace=True)

        # 確保主要欄位存在
        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(df.columns):
            logger.warning(f"{stock_id} 缺少主要欄位，略過 ({df.columns})")
            continue

        # 重新計算技術指標
        df = calculate_working(df)

        # 僅取最後 N 天資料
        df_recent = df.tail(last_days).copy()

        # 批次 upsert 回 MongoDB
        df_reset = df_recent.reset_index()
        df_reset["date"] = df_reset["date"].dt.strftime("%Y-%m-%d")
        records = df_reset.to_dict(orient="records")

        bulk_ops = []
        for record in records:
            bulk_ops.append(
                UpdateOne(
                    {"date": record["date"]},
                    {"$set": record},
                    upsert=True
                )
            )

        if bulk_ops:
            result = collection.bulk_write(bulk_ops, ordered=False)
            logger.info(
                f"✅ {stock_id} 指標已更新 (寫入 {len(records)} 筆, matched={result.matched_count}, modified={result.modified_count}, upserted={len(result.upserted_ids)})"
            )

    logger.info("🎯 全部股票指標更新完成（僅最後 100 天）")
