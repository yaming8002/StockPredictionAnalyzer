import os
import math
import logging
import pandas as pd
import numpy as np
from scipy import stats
from modules.config_loader import load_config
from modules.logger import setup_logger
from modules.process_mongo import get_mongo_client, close_mongo_client
from _04_analysis.hold_days import analyze_hold_days


class StrategyRunner:
    """
    通用策略批次回測器
    - 可輸入任意策略類別 (需繼承 StockBacktest)
    - 可選擇是否輸出 log 檔
    - 可控制是否顯示每檔股票的個別回測結果
    """

    def __init__(self, strategy_cls, label=None, log_folder=None, initial_cash=100000, show_each_stock=False):
        """
        :param strategy_cls: 策略類別，例如 DowStrategy
        :param label: 回測標籤名稱，若為 None 則不產生 log 檔
        :param log_folder: log 儲存路徑，未指定則讀取 config 或不寫檔
        :param initial_cash: 初始資金
        :param show_each_stock: 是否顯示每檔股票的個別結果 (預設 True)
        """
        self.strategy_cls = strategy_cls
        self.label = label
        self.initial_cash = initial_cash
        self.show_each_stock = show_each_stock
        self.config = load_config()
        self.log_folder = log_folder or self.config.get("strategy_log_folder", "./strategy_log")
        self.log = None

        if label:
            os.makedirs(self.log_folder, exist_ok=True)
            log_path = os.path.join(self.log_folder, f"{label}.log")
            self.log = setup_logger(log_file=log_path, loglevel=logging.INFO)
        else:
            logging.basicConfig(level=logging.INFO, format="%(message)s")
            self.log = logging.getLogger("console")

    # ============================================================
    # 主執行函數
    # ============================================================
    def run(self, start_date="2015-01-01", end_date="2019-12-31"):
        db = get_mongo_client()
        collections = [col for col in db.list_collection_names() if "TW" in col]
        collections.sort()

        total_win, total_lose, total_profit = 0, 0, 0.0
        all_hold_days, all_records = [], []

        label = self.label or "no_label"

        self.log.info(f"🚀 開始回測策略: {self.strategy_cls.__name__} ({label})")
        self.log.info(f"📆 期間: {start_date} ~ {end_date}")
        self.log.info("===============================================")

        for stock_id in collections:
            try:
                bt = self.strategy_cls(
                    stock_id=stock_id,
                    start_date=start_date,
                    end_date=end_date,
                    initial_cash=self.initial_cash,
                    split_cash=5000,
                    label=label,
                )
                bt.run_backtest()
            except Exception as e:
                self.log.warning(f"⚠️ 忽略 {stock_id}，錯誤：{e}")
                continue

            buy_count = bt.win_count + bt.lose_count
            profit = bt.cash - self.initial_cash
            total_win += bt.win_count
            total_lose += bt.lose_count
            total_profit += profit
            all_hold_days.extend(bt.hold_days)
            all_records.extend(bt.trade_records)

            if self.show_each_stock:
                self.log.info(
                    f"{stock_id}: 最終資金 {bt.cash:,.0f}, "
                    f"交易 {buy_count} 次, 勝率 {bt.win_rate:.2%}, 獲利 {profit:,.0f}"
                )

        # =================== 統計 =====================
        self._analyze_results(
            collections, total_win, total_lose, total_profit, all_hold_days, all_records,
            start_date, end_date
        )

        close_mongo_client()

    # ============================================================
    # 統計輔助函數（攤開後）
    # ============================================================
    def trim_outliers(self, series, mode="auto"):
        """使用 Median Absolute Deviation (MAD) 排除極值"""
        if len(series) == 0:
            return series, None, None
        median = series.median()
        mad = (abs(series - median)).median()
        k = 3
        lower = median - k * mad
        upper = median + k * mad
        if mode == "win":
            lower = max(lower, 0)
        elif mode == "lose":
            upper = min(upper, 0)
        filtered = series[(series >= lower) & (series <= upper)]
        return filtered, lower, upper

    def get_confidence_interval(self, series):
        """取得 95% 信賴區間"""
        if len(series) < 2:
            return (np.nan, np.nan)
        mean = np.mean(series)
        std_err = stats.sem(series)
        ci = stats.t.interval(0.95, len(series) - 1, loc=mean, scale=std_err)
        return (round(ci[0], 2), round(ci[1], 2))

    # ============================================================
    # 總體績效分析
    # ============================================================
    def _analyze_results(
        self,
        collections,
        total_win,
        total_lose,
        total_profit,
        hold_days,
        trade_records,
        start_date,
        end_date,
    ):
        buy_count = total_win + total_lose
        win_rate = total_win / buy_count if buy_count > 0 else 0
        avg_profit = total_profit / buy_count if buy_count > 0 else 0

        self.log.info("===============================================")
        self.log.info(f"📊 回測績效總結 {start_date} ~ {end_date}")
        self.log.info("-----------------------------------------------")
        self.log.info(f"總股票數量：{len(collections)}")
        self.log.info(f"總交易次數：{buy_count}")
        self.log.info(f"總勝率：{win_rate:.2%}")
        self.log.info(f"總獲利金額：{total_profit:,.0f}")
        self.log.info(f"平均每筆盈虧：{avg_profit:,.2f}")
        self.log.info("-----------------------------------------------")

        if len(trade_records) == 0:
            self.log.info("❌ 無交易紀錄，結束分析")
            return

        df = pd.DataFrame(trade_records)
        df["profit_rate"] = (df["sell_price"] - df["buy_price"]) / df["buy_price"] * 100
        df["profit"] = pd.to_numeric(df["profit"], errors="coerce").fillna(0)
        df = df[df["profit"] != 0]

        win_df = df[df["profit"] > 0]
        lose_df = df[df["profit"] < 0]

        avg_win = win_df["profit"].mean() if not win_df.empty else 0
        avg_lose = lose_df["profit"].mean() if not lose_df.empty else 0
        avg_win_rate = win_df["profit_rate"].mean() if not win_df.empty else 0
        avg_lose_rate = lose_df["profit_rate"].mean() if not lose_df.empty else 0
        max_win = df["profit"].max()
        max_lose = df["profit"].min()
        avg_hold_days = df["hold_days"].mean() if "hold_days" in df else 0

        expect_value = win_rate * avg_win + (1 - win_rate) * avg_lose

        self.log.info("📈 詳細績效統計")
        self.log.info("-----------------------------------------------")
        self.log.info(f"平均獲利金額：{avg_win:,.2f}")
        self.log.info(f"平均虧損金額：{avg_lose:,.2f}")
        self.log.info(f"平均獲利報酬率：{avg_win_rate:.2f}%")
        self.log.info(f"平均虧損報酬率：{avg_lose_rate:.2f}%")
        self.log.info(f"最大單筆獲利：{max_win:,.2f}")
        self.log.info(f"最大單筆虧損：{max_lose:,.2f}")
        self.log.info(f"平均持有天數：{avg_hold_days:.1f} 天")
        self.log.info(f"期望報酬值（EV）：{expect_value:,.2f}")
        self.log.info("-----------------------------------------------")

        # =====================================================
        # 🔹 信賴區間與極值分析
        # =====================================================
        if not win_df.empty:
            win_filtered, win_low, win_high = self.trim_outliers(win_df["profit"])
            avg_win_trim = win_filtered.mean()
        else:
            avg_win_trim, win_low, win_high = avg_win, None, None

        if not lose_df.empty:
            lose_filtered, lose_low, lose_high = self.trim_outliers(lose_df["profit"])
            avg_lose_trim = lose_filtered.mean()
        else:
            avg_lose_trim, lose_low, lose_high = avg_lose, None, None

        win_ci_low, win_ci_high = self.get_confidence_interval(win_df["profit"]) if not win_df.empty else (np.nan, np.nan)
        lose_ci_low, lose_ci_high = self.get_confidence_interval(lose_df["profit"]) if not lose_df.empty else (np.nan, np.nan)

        expect_trim = win_rate * avg_win_trim + (1 - win_rate) * avg_lose_trim

        self.log.info("📊 信賴區間與極值分析")
        self.log.info("-----------------------------------------------")
        self.log.info(f"IQR 獲利區間: [{win_low:,.2f} ~ {win_high:,.2f}]")
        self.log.info(f"IQR 虧損區間: [{lose_low:,.2f} ~ {lose_high:,.2f}]")
        self.log.info(f"排除極值後平均獲利金額: {avg_win_trim:,.2f} (原本: {avg_win:,.2f})")
        self.log.info(f"排除極值後平均虧損金額: {avg_lose_trim:,.2f} (原本: {avg_lose:,.2f})")
        self.log.info(f"95% 獲利信賴區間: [{win_ci_low:,.2f}, {win_ci_high:,.2f}]")
        self.log.info(f"95% 虧損信賴區間: [{lose_ci_low:,.2f}, {lose_ci_high:,.2f}]")
        self.log.info(f"排除極值後期望報酬值 (EV,Trim): {expect_trim:,.2f}")
        self.log.info("-----------------------------------------------")

        # =====================================================
        # 💾 匯出分析結果
        # =====================================================
        if self.label:
            output_folder = self.config.get("leaning_folder", "./stock_data/leaning_label")
            os.makedirs(output_folder, exist_ok=True)
            trades_path = os.path.join(output_folder, f"{self.label}_trades.csv")
            df.to_csv(trades_path, index=False, encoding="utf-8-sig")

            summary_path = os.path.join(output_folder, f"{self.label}_summary.csv")
            summary_df = pd.DataFrame([{
                "回測起始日": start_date,
                "回測結束日": end_date,
                "股票數量": len(collections),
                "交易次數": buy_count,
                "勝率(%)": round(win_rate * 100, 2),
                "平均獲利金額": round(avg_win, 2),
                "平均虧損金額": round(avg_lose, 2),
                "平均獲利報酬率(%)": round(avg_win_rate, 2),
                "平均虧損報酬率(%)": round(avg_lose_rate, 2),
                "最大獲利": round(max_win, 2),
                "最大虧損": round(max_lose, 2),
                "平均持有天數": round(avg_hold_days, 2),
                "期望報酬值(EV)": round(expect_value, 2),
                "總獲利": round(total_profit, 2),
                "IQR獲利下限": round(win_low, 2) if win_low is not None else np.nan,
                "IQR獲利上限": round(win_high, 2) if win_high is not None else np.nan,
                "IQR虧損下限": round(lose_low, 2) if lose_low is not None else np.nan,
                "IQR虧損上限": round(lose_high, 2) if lose_high is not None else np.nan,
                "排除極值後平均獲利金額": round(avg_win_trim, 2),
                "排除極值後平均虧損金額": round(avg_lose_trim, 2),
                "排除極值後期望報酬值(EV,Trim)": round(expect_trim, 2),
                "獲利信賴區間下限(95%)": win_ci_low,
                "獲利信賴區間上限(95%)": win_ci_high,
                "虧損信賴區間下限(95%)": lose_ci_low,
                "虧損信賴區間上限(95%)": lose_ci_high,
            }])
            summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

            self.log.info(f"✅ 已輸出交易記錄：{trades_path}")
            self.log.info(f"✅ 已輸出完整統計摘要（含信賴區間）：{summary_path}")

        analyze_hold_days(hold_days, self.log)
