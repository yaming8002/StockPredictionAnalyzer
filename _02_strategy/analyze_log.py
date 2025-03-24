import os
import re
import pandas as pd


def extract_log_summary(strategy_log_folder: str = "./strategy_log") -> str:
    """
    讀取指定資料夾中的所有 .log 檔案，擷取每個檔案的總營利與持有天數統計資訊，
    並輸出為 Excel 檔案。

    Args:
        strategy_log_folder (str): log 檔所在資料夾

    Returns:
        str: 匯出的 Excel 路徑
    """
    # 正則表達式
    overall_pattern = re.compile(r"總計:總營利(?P<total_profit>[\d.-]+), 股票數量(?P<stock_count>\d+),總下注量:(?P<total_count>\d+),每注獲利 [\d.]+, 獲勝次數(?P<total_win>\d+), 總勝率 (?P<total_win_rate>[\d.]+)%")
    hold_pattern = re.compile(r"持有天數統計: 最大 (?P<max>\d+), 最小 (?P<min>\d+), 平均 (?P<avg>[\d.]+), 標準差 (?P<std>[\d.]+), 眾數 (?P<mode>.+)")

    summary_rows = []

    if not os.path.exists(strategy_log_folder):
        raise FileNotFoundError(f"找不到資料夾: {strategy_log_folder}")

    for file in os.listdir(strategy_log_folder):
        if file.endswith(".log"):
            file_path = os.path.join(strategy_log_folder, file)
            filename = os.path.splitext(file)[0]
            row = {"檔名": filename}

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    overall_match = overall_pattern.search(line)
                    hold_match = hold_pattern.search(line)

                    if overall_match:
                        row.update(
                            {
                                "總營利": float(overall_match.group("total_profit")),
                                "總下注量": int(overall_match.group("total_count")),
                                "總勝率": float(overall_match.group("total_win_rate")),
                                "獲勝次數": int(overall_match.group("total_win")),
                                "股票數量": int(overall_match.group("stock_count")),
                            }
                        )
                    if hold_match:
                        row.update(
                            {
                                "最大持有天數": int(hold_match.group("max")),
                                "最小持有天數": int(hold_match.group("min")),
                                "平均持有天數": float(hold_match.group("avg")),
                                "標準差": float(hold_match.group("std")),
                                "眾數": hold_match.group("mode"),
                            }
                        )
            if "總營利" in row:
                summary_rows.append(row)

    # 建成 DataFrame 並匯出
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(by="總營利", ascending=False)

    output_path = os.path.join(strategy_log_folder, "log_overall_summary.xlsx")
    summary_df.to_excel(output_path, index=False)

    return output_path
