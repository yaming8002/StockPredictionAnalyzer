import logging
import os
from logging.handlers import TimedRotatingFileHandler


def setup_logger(log_file=None):
    """建立 log 記錄，顯示在 CMD 並寫入檔案，且每日分割 log 檔案"""
    if log_file is None:
        log_file = os.path.join(os.path.dirname(__file__), "logs", "stock_analysis.log")

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(log_file)  # 🔹 確保不同的 `log_file` 創建不同 logger
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        # **建立「每日」分割的 log**
        file_handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=30, encoding="utf-8")
        file_handler.suffix = "%Y-%m-%d.log"  # 🔹 讓 log 檔案以日期結尾
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

        # **建立 CMD (終端機) 輸出**
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(f"%(asctime)s - %(levelname)s - %(message)s [Log File: {log_file}]"))
        logger.addHandler(console_handler)

    return logger
