import logging
import os


def setup_logger(log_file=None):
    """建立 log 記錄，顯示在 CMD 並寫入檔案"""
    if log_file is None:
        log_file = os.path.join(os.path.dirname(__file__), "logs", "stock_analysis.log")

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger("stock_logger")

    # **避免重複添加 Handler**
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        # **建立檔案日誌記錄**
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

        # **建立 CMD (終端機) 輸出**
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(console_handler)

    return logger
