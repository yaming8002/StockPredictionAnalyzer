import logging
import os


def setup_logger(log_file=None):
    """建立 log 記錄"""
    if log_file is None:
        log_file = os.path.join(os.path.dirname(__file__), ".", "logs", "stock_analysis.log")

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger("stock_logger")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(file_handler)

    return logger
