"""
policy_evolve.py
深度學習股票策略 — 對抗式演化 Genetic Evolution
"""

import os
import random
import numpy as np
from copy import deepcopy
from tensorflow.keras.models import clone_model

from .models import build_buy_model, build_sell_model
from _05_deeplearning.deep_strategy import DLStockBacktest


# ---------------------------------------------------------
# ✅ Policy = (Buy Model + Sell Model + Threshold)
# ---------------------------------------------------------
class TradingPolicy:
    def __init__(self, buy_model, sell_model, buy_threshold=0.6, sell_threshold=0.6):
        self.buy_model = buy_model
        self.sell_model = sell_model
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def clone(self):
        """完整深拷貝: model + weights + threshold"""
        new = TradingPolicy(
            clone_model(self.buy_model),
            clone_model(self.sell_model),
            buy_threshold=self.buy_threshold,
            sell_threshold=self.sell_threshold,
        )
        new.buy_model.set_weights(self.buy_model.get_weights())
        new.sell_model.set_weights(self.sell_model.get_weights())
        return new

    def mutate(self, mutation_rate=0.03, evolve_target="both"):
        """
        evolve_target = "sell" / "buy" / "both"
        只 mutate 目標模型，符合 Adversarial Evolution
        """

        def mutate_model(model):
            weights = model.get_weights()
            new_weights = [w + np.random.normal(0, mutation_rate, w.shape) for w in weights]
            model.set_weights(new_weights)

        if evolve_target in ("buy", "both"):
            mutate_model(self.buy_model)

        if evolve_target in ("sell", "both"):
            mutate_model(self.sell_model)

        # threshold mutation
        if evolve_target in ("buy", "both"):
            self.buy_threshold += random.uniform(-0.05, 0.05)

        if evolve_target in ("sell", "both"):
            self.sell_threshold += random.uniform(-0.05, 0.05)

        # clamp
        self.buy_threshold = min(max(self.buy_threshold, 0.40), 0.90)
        self.sell_threshold = min(max(self.sell_threshold, 0.40), 0.90)


# ---------------------------------------------------------
# ✅ 評分：回測多檔股票績效
# ---------------------------------------------------------
def evaluate_policy(policy, stock_list, start_date, end_date):
    total_cash = 0
    total_trades = 0
    win_rate_sum = 0

    for stock_id in stock_list:
        bt = DLStockBacktest(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date,
            initial_cash=200_000,
            buy_model_path=None,
            sell_model_path=None,
            window=120,
        )

        # 套用目前 policy 的模型與 threshold
        bt.buy_model = policy.buy_model
        bt.sell_model = policy.sell_model
        bt.threshold_buy = policy.buy_threshold  # <<< ✅ 重要
        bt.threshold_sell = policy.sell_threshold  # <<< ✅ 重要

        bt.run_backtest()

        total_cash += bt.cash
        total_trades += len(bt.trade_records)
        win_rate_sum += bt.win_rate

    # FITNESS (越大越好) = 最後資產 + 勝率加權 (增加穩定性)
    return total_cash + win_rate_sum * 100_000


# ---------------------------------------------------------
# ✅ 主要進化程序（支援：sell_only / buy_only / both）
# ---------------------------------------------------------
def evaluate_policy(
    policy,
    stock_list,
    start_date,
    end_date,
):
    profits = []
    losses = []
    holding_days = []

    for stock_id in stock_list:
        bt = DLStockBacktest(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date,
            initial_cash=200_000,
            buy_model_path=None,
            sell_model_path=None,
            window=120,
        )

        bt.buy_model = policy.buy_model
        bt.sell_model = policy.sell_model
        bt.threshold_buy = policy.buy_threshold
        bt.threshold_sell = policy.sell_threshold

        bt.run_backtest()

        for record in bt.trade_records:
            profit = record["profit"]
            if profit > 0:
                profits.append(profit)
            else:
                losses.append(abs(profit))

            holding_days.append(record["hold_days"])

    total_trades = len(holding_days)
    # if total_trades == 0:
    #     return -1e9  # 罰分，避免模型裝死不交易

    # --- EV (交易期望值)
    win_rate = len(profits) / total_trades
    loss_rate = len(losses) / total_trades

    avg_win = np.mean(profits) if profits else 0
    avg_loss = np.mean(losses) if losses else 0

    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

    # --- Holding Efficiency
    avg_holding = np.mean(holding_days)
    holding_eff = 45 / (avg_holding + 1)

    # --- Final Fitness Score
    fitness = expectancy * holding_eff

    return fitness


# ---------------------------------------------------------
# ✅ Run evolution (example)
# ---------------------------------------------------------
def run_deep_backtest():

    # Phase 1: evolve sell model first
    evolve_policy(
        evolve_target="sell",  # ✅ 先訓練 Sell Model
        generations=10,
        population_size=6,
        start_date="2011-01-01",
        end_date="2016-12-31",
        stock_list=["2330.TW", "2454.TW", "2603.TW", "2881.TW"],
    )

    # Phase 2: evolve buy with better sell model
    evolve_policy(
        evolve_target="buy",  # ✅ 再訓練 Buy Model
        generations=8,
        population_size=6,
        start_date="2017-01-01",
        end_date="2020-12-31",
        stock_list=["2330.TW", "2454.TW", "2603.TW", "2881.TW"],
    )
