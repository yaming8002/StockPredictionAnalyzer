import os
import re
import pandas as pd
from modules.config_loader import load_config
from modules.logger import setup_logger


logger = setup_logger("./analyze_log_folder.log")


def analyze_log_folder():
    strategy_log_folder = config = load_config()
    strategy_log_folder = config.get("strategy_log_folder", "./strategy_log")  # 正則表達式

    stock_result_pattern = re.compile(r"(?P<id>\d+\.\w+): 初始金額(?P<initial>\d+) ,最終金額 (?P<final>\d+) 獲利:(?P<profit>[-\d]+), 勝率 (?P<win_rate>[\d.]+)%")
    stock_summary_pattern = re.compile(r"(?P<id>\d+\.\w+): 總金額 (?P<total>\d+), 下注次數 (?P<count>\d+) , 獲利次數(?P<win>\d+) 勝率 (?P<win_rate>[\d.]+)%")
    overall_pattern = re.compile(r"總計:總營利(?P<total_profit>[\d.-]+), 股票數量(?P<stock_count>\d+),總下注量:(?P<total_count>\d+),每注獲利 [\d.]+, 獲勝次數(?P<total_win>\d+), 總勝率 (?P<total_win_rate>[\d.]+)%")

    all_data = {}

    for file in os.listdir(strategy_log_folder):
        if file.endswith(".log"):
            file_path = os.path.join(strategy_log_folder, file)
            filename = os.path.splitext(file)[0]
            stock_data = {}

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    stock_summary = stock_summary_pattern.search(line)
                    stock_result = stock_result_pattern.search(line)
                    overall = overall_pattern.search(line)
                    if overall:
                        logger.info(f"{filename} - {line}")

            for stock_id, data in stock_data.items():
                if stock_id not in all_data:
                    all_data[stock_id] = {}
                all_data[stock_id].update(data)

    # 整合成扁平 DataFrame
    df = pd.DataFrame.from_dict(all_data, orient="index")
    df.index.name = "股票代號"
    df.reset_index(inplace=True)

    # 匯出 CSV
    output_path = os.path.join(strategy_log_folder, "log_analysis_flat_summary.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 分析報表已成功匯出為扁平 CSV：{output_path}")


# ✅ 範例使用（請在執行時提供你的 log 資料夾路徑）
# analyze_log_flat_excel("F:/你的/log/資料夾")
