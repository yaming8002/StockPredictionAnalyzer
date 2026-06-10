"""
取得股票資料（進階篇）：大量、可續傳的全史下載

逐檔下載 stock_list.csv 內所有股票的長期歷史資料。相較基礎篇，多了「大量下載」
實務上必備的幾個機制：

- 單檔下載（穩定，不像批次那麼容易被 Yahoo 限流）
- 隨機延遲（避免 anti-bot）
- 失敗自動重試
- Checkpoint 機制：寫 download_progress.json，中斷後可從上次位置續傳
- 失敗清單寫入 failed_stocks.txt
- 每 10 檔印一次進度（含剩餘時間估算）

執行：
  python _01_data/download_full_history.py
"""

import os
import sys
import time
import random
import json
import signal
from datetime import datetime

import pandas as pd

# 同目錄的基礎篇模組（共用下載與存檔邏輯）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_stock import download_stock_data, _load_stock_list


# ===== 參數區（依需求調整）=====
_HERE = os.path.dirname(os.path.abspath(__file__))

STOCK_LIST = os.path.join(_HERE, "stock_list.csv")   # 先用 fetch_stock_list.py 產生
SAVE_DIR = os.path.join(_HERE, "data")               # 輸出目錄
SAVE_FMT = "parquet"                                 # "csv" 或 "parquet"

START_DATE = "2000-01-01"
END_DATE = None        # None = 抓到今天；yfinance end 為 exclusive，函式內已處理

DELAY_MIN = 5          # 每檔之間的最小延遲（秒）
DELAY_MAX = 9          # 每檔之間的最大延遲（秒）
RETRY = 2              # 失敗重試次數
RETRY_WAIT = 30        # 重試前等待（秒）

PROGRESS_FILE = os.path.join(_HERE, "download_progress.json")
FAILED_FILE = os.path.join(_HERE, "failed_stocks.txt")
# ================================


def load_progress() -> dict:
    """讀取進度檔；不存在或損毀時回傳初始狀態。"""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"completed": [], "failed": [], "started_at": None}


def save_progress(progress: dict) -> None:
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def main():
    all_stocks = _load_stock_list(STOCK_LIST)
    total = len(all_stocks)

    # 載入進度（支援續傳）：已完成的不再下載
    progress = load_progress()
    if progress["started_at"] is None:
        progress["started_at"] = datetime.now().isoformat()
    completed_set = set(progress["completed"])
    todo = [s for s in all_stocks if s not in completed_set]

    print("=" * 70)
    print(f"完整歷史下載 {START_DATE} ~ {END_DATE or '今天'}")
    print(f"總清單  : {total} 檔")
    print(f"已完成  : {len(completed_set)}")
    print(f"待下載  : {len(todo)}")
    print(f"延遲    : {DELAY_MIN}~{DELAY_MAX} 秒/檔")
    print(f"輸出目錄: {SAVE_DIR}（{SAVE_FMT}）")
    avg_delay = (DELAY_MIN + DELAY_MAX) / 2
    print(f"預估剩餘: {len(todo) * avg_delay / 3600:.1f} 小時")
    print("=" * 70)

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Ctrl+C 處理：退出前先把進度存下來
    def handle_interrupt(signum, frame):
        print("\n收到中斷訊號，儲存進度...")
        save_progress(progress)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)

    t0 = time.time()
    n_done = len(completed_set)

    for idx, stock_id in enumerate(todo, 1):
        success = False
        last_err = None
        for attempt in range(RETRY + 1):
            try:
                result = download_stock_data(
                    stock_id,
                    start_date=START_DATE,
                    end_date=END_DATE,
                    save_path=SAVE_DIR,
                    fmt=SAVE_FMT,
                    delay_range=(0, 0),   # 延遲在本迴圈統一管理
                )
                if result:
                    success = True
                    break
                last_err = "empty data"
                break   # 無資料不重試
            except Exception as e:
                last_err = str(e)[:100]
                if attempt < RETRY:
                    print(f"  [{idx}/{len(todo)}] {stock_id} 重試 {attempt + 1}/{RETRY}: {last_err}")
                    time.sleep(RETRY_WAIT)

        if success:
            progress["completed"].append(stock_id)
        else:
            progress["failed"].append({"stock_id": stock_id, "reason": last_err})
            print(f"  [{idx}/{len(todo)}] X {stock_id} 失敗: {last_err}")

        # 每 10 檔（或最後一檔）存進度 + 印摘要
        if idx % 10 == 0 or idx == len(todo):
            save_progress(progress)
            elapsed = time.time() - t0
            avg = elapsed / idx
            remain = avg * (len(todo) - idx)
            total_completed = n_done + idx
            print(f"  [{idx}/{len(todo)}] 全進度 {total_completed}/{total} "
                  f"({total_completed / total * 100:.1f}%)  "
                  f"已耗 {elapsed / 60:.1f}m, 預估剩 {remain / 60:.1f}m  "
                  f"(失敗 {len(progress['failed'])})")

        # 每檔之間隨機延遲
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    save_progress(progress)

    # 寫失敗清單
    if progress["failed"]:
        with open(FAILED_FILE, "w", encoding="utf-8") as f:
            for entry in progress["failed"]:
                if isinstance(entry, dict):
                    f.write(f"{entry['stock_id']}\t{entry.get('reason', '')}\n")
                else:
                    f.write(f"{entry}\n")
        print(f"\n失敗清單寫入: {FAILED_FILE}")

    print(f"\n>>> 完成! 成功: {len(progress['completed'])}, 失敗: {len(progress['failed'])}")
    print(f"總耗時: {(time.time() - t0) / 3600:.2f} 小時")


if __name__ == "__main__":
    main()
