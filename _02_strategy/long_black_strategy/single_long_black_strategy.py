"""
長黑出量後突破策略（single long black）— long_black_strategy 套件下的方案
==========================================================================

繼承 VbtSingleStrategy，交易一律由 vbt 執行（Portfolio.from_signals）；本檔只產生進出場訊號。

規則（只做多；判定日收盤 → 隔日開盤成交）：
  長黑出量（setup）：
    - 黑 K：收盤 < 開盤。
    - 長：實體 |收 − 開| 大於前 5 天每一根的實體（前 5 天紅黑不論）。
    - 出量：當日量 ≥ 前 5 天（t−1～t−5，昨日起算）平均量 × 1.5。
  進場：長黑出量之後第 1～5 個交易日內，第一次收盤 > 長黑最高價 → 判定進場。
        等待中又出現新的長黑出量，改以新的那根為準、重新計 5 天；持倉中出現的長黑不理會。
  出場（任一成立即出）：
    ① 停損：收盤 < 觸發這筆進場的長黑最低價。
    ② CHoCH（lower-high）：進場後新確認的擺動高點，低於進場以來的最高擺動高點
       （與雙底、均線交叉系列的 CHoCH 同一定義）。
    ③ 跌破 5MA：持倉中（含進場判定日）收盤曾 > MA5 之後，收盤 < MA5。
       「曾站上」先用前一根為止的狀態判定、再更新，當根不會同時啟動與觸發。
       均線天數由類別屬性 MA_EXIT 決定（註解切換，見下方優化紀錄）。

優化紀錄（註解切換：類別屬性 MA_EXIT 那幾行一次只開一行；--variant 只決定輸出資料夾，須與註解狀態一致）：
  #1 2026-11-06 ③ 改成跌破 10MA（baseline 的 5MA 平均持有不到 7 天、CHoCH 幾乎不觸發）。
     結果寫 result/single_long_black_opt1/。

擺動點：`_01_data.indicators_pattern.calculate_zigzag`（收盤價、門檻 pct）只在反轉幅度達 pct 的
確認當日才填 zigzag_turn_high，確認前是 NaN，所以 ② 不會用到未來資料。

為什麼 pct 用 CLI 參數（不用註解切換）：pct 只影響 ② CHoCH 的擺動點，是 baseline 本身要並列比較的
設定（2%／4%／6%／8%，2026-11-06 用戶指定），比照雙底系列。之後的優化一律走註解切換。
可成交門檻用 --liq：成交金額＝5 日均量 × 收盤 > 1,000 萬，判在進場判定日。

診斷（self.diag，由 driver 寫出）：setups（區間內長黑出量根數）、signals（突破判定日數）、進場數、
三種出場各自單獨觸發與同根多條觸發的次數、期末未平倉部位數與持有天數。

執行：
    python _02_strategy/long_black_strategy/single_long_black_strategy.py <資料夾> --pct 0.02 [--liq]
  輸出：result/single_long_black/p<pct>_<raw|liq1000>/
"""
import argparse
import json
import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd

from _01_data.indicators_pattern import calculate_zigzag
from _02_strategy.base.vbt import batch
from _02_strategy.base.vbt.single import VbtSingleStrategy
from _02_strategy.base.vbt.common import GLITCH, DEFAULT_START, DEFAULT_END

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
TURNOVER_MIN = 10_000_000
BODY_LOOKBACK = 5          # 長黑：實體大於前 5 天每一根
VOL_LOOKBACK, VOL_MULTIPLE = 5, 1.5   # 出量：≥ 前 5 天均量 × 1.5
BREAKOUT_WAIT = 5          # 長黑之後最多等幾個交易日突破

# 出場原因位元
R_STOP, R_CHOCH, R_MA = 1, 2, 4


class SingleLongBlackStrategy(VbtSingleStrategy):
    """長黑出量後收盤突破長黑高點進場；停損／CHoCH／跌破均線出場。訊號逐根掃描產生，成交交給 vbt。"""

    MA_EXIT = 5        # baseline：③ 跌破 5MA
    # MA_EXIT = 10     # 優化 #1：③ 改成跌破 10MA

    def __init__(self, pct: float = 0.02, liquidity: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pct = pct
        self.liquidity = liquidity
        self._trade_end = None
        self.diag = {"setups": 0, "signals": 0, "entries": 0, "exits": 0,
                     "exit_stop": 0, "exit_choch": 0, "exit_ma": 0, "exit_multi": 0,
                     "open_at_end": 0, "open_days_sum": 0, "open_days_max": 0}

    def add_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        calculate_zigzag(df, self.pct)
        close, open_ = df["close"], df["open"]
        body = (close - open_).abs()
        prev_body_max = body.shift(1).rolling(BODY_LOOKBACK).max()
        prev_vol_ma = df["volume"].shift(1).rolling(VOL_LOOKBACK).mean()
        # 前 5 天均量為 0（休市幽靈 K 棒連續出現）時任何量都會過關，故要求均量 > 0
        df["long_black"] = ((close < open_) & (body > prev_body_max)
                            & (prev_vol_ma > 0) & (df["volume"] >= prev_vol_ma * VOL_MULTIPLE)).fillna(False)
        df["ma_exit"] = close.rolling(self.MA_EXIT).mean()
        df["turnover"] = df["volume"].rolling(5).mean() * close
        return df

    def buy_signal(self, df: pd.DataFrame) -> pd.Series:
        """進場是路徑相依（要記住等待中的長黑），在 build_signals 逐根掃描；這裡只回傳 setup 日供區間閘門用。"""
        return df["long_black"].astype(bool)

    def sell_signal(self, df: pd.DataFrame) -> pd.Series:
        """出場是路徑相依，在 build_signals 逐根掃描；這裡不使用。"""
        return pd.Series(False, index=df.index)

    def window_signals(self, df: pd.DataFrame, start=None, end=None):
        self._trade_end = end                               # 診斷要知道區間終點（數期末未平倉）
        return super().window_signals(df, start, end)

    def build_signals(self, df: pd.DataFrame):
        """
        逐根掃描產生「判定日」進出場，再 shift(1) 成隔日開盤成交日交給 vbt。
        setup 取 entry_signal（已擋掉交易區間前的長黑），持倉中不接受新的 setup 與進場。
        """
        setup = self.entry_signal(df).to_numpy()
        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        ma = df["ma_exit"].to_numpy(dtype=float)
        th = df["zigzag_turn_high"].to_numpy()
        liq_ok = (df["turnover"] > TURNOVER_MIN).to_numpy() if self.liquidity else np.ones(len(df), dtype=bool)
        n = len(df)
        entries = np.zeros(n, dtype=bool)
        exits = np.zeros(n, dtype=bool)
        signals = np.zeros(n, dtype=bool)
        reason = np.zeros(n, dtype=np.int8)

        pend_high, pend_low, deadline = np.nan, np.nan, -1   # 等待中的長黑
        in_pos = False
        stop_px, peak, armed = np.nan, -np.inf, False
        for i in range(n):
            new_high = not np.isnan(th[i])
            if not in_pos:
                if i > deadline:
                    pend_high = np.nan
                # 先用「之前」的長黑判突破；當根若是新長黑，收盤必低於自身最高價，不會自己突破自己
                if not np.isnan(pend_high) and close[i] > pend_high:
                    signals[i] = True
                    if liq_ok[i]:
                        in_pos, entries[i] = True, True
                        stop_px, peak = pend_low, -np.inf
                        armed = (not np.isnan(ma[i])) and close[i] > ma[i]     # 進場判定日也算「站上」
                        pend_high, deadline = np.nan, -1
                        continue
                    pend_high, deadline = np.nan, -1         # 突破但不可成交：這根長黑用掉
                if setup[i]:
                    pend_high, pend_low, deadline = high[i], low[i], i + BREAKOUT_WAIT
                continue
            # ── 持倉中：判定出場 ──
            hit_stop = close[i] < stop_px                                          # ① 停損
            hit_choch = new_high and th[i] < peak                                  # ② lower-high
            hit_ma = armed and (not np.isnan(ma[i])) and close[i] < ma[i]          # ③ 跌破均線
            if new_high:
                peak = max(peak, th[i])
            armed = armed or ((not np.isnan(ma[i])) and close[i] > ma[i])
            if hit_stop or hit_choch or hit_ma:
                exits[i] = True
                reason[i] = (R_STOP if hit_stop else 0) + (R_CHOCH if hit_choch else 0) + (R_MA if hit_ma else 0)
                in_pos = False
                pend_high, deadline = np.nan, -1                  # 持倉中的長黑不理會

        self._count_diag(df.index, setup, signals, entries, exits, reason)
        e = pd.Series(entries, index=df.index).shift(1, fill_value=False)
        x = pd.Series(exits, index=df.index).shift(1, fill_value=False)
        return e, x

    def _count_diag(self, idx, setup, signals, entries, exits, reason):
        """只數成交日落在交易區間內的事件（判定日＋1 才成交）。"""
        lo = np.datetime64(pd.Timestamp(self._trade_start) if self._trade_start else idx[0], "ns")
        hi = np.datetime64(pd.Timestamp(self._trade_end) if self._trade_end else idx[-1], "ns")
        days = idx.values.astype("datetime64[ns]")
        exec_day = np.append(days[1:], np.datetime64("NaT", "ns"))   # 判定日 i 的成交日＝i+1
        inside = (exec_day >= lo) & (exec_day <= hi)
        self.diag["setups"] += int((setup & inside).sum())
        self.diag["signals"] += int((signals & inside).sum())
        self.diag["entries"] += int((entries & inside).sum())
        ex_in = exits & inside
        self.diag["exits"] += int(ex_in.sum())
        self.diag["exit_stop"] += int((ex_in & (reason == R_STOP)).sum())
        self.diag["exit_choch"] += int((ex_in & (reason == R_CHOCH)).sum())
        self.diag["exit_ma"] += int((ex_in & (reason == R_MA)).sum())
        self.diag["exit_multi"] += int((ex_in & (reason > 0) & ~np.isin(reason, (R_STOP, R_CHOCH, R_MA))).sum())
        # 期末未平倉：區間內最後一筆進場之後，區間內沒有出場
        ent_pos = np.flatnonzero(entries & inside)
        if len(ent_pos):
            last = ent_pos[-1]
            if not (ex_in[last + 1:]).any():
                self.diag["open_at_end"] += 1
                last_day = min(hi, days[-1])
                held = int((last_day - exec_day[last]) / np.timedelta64(1, "D"))
                self.diag["open_days_sum"] += held
                self.diag["open_days_max"] = max(self.diag["open_days_max"], held)


def run_config(folder: str, pct: float, liquidity: bool, limit=None,
               start=DEFAULT_START, end=DEFAULT_END, write_trades=True, out_root=None,
               variant=None) -> dict:
    """
    跑一組設定、寫出結果與診斷，回傳彙總 dict（driver 共用）。
    variant：只決定輸出資料夾（None＝baseline → single_long_black/；"opt1" → single_long_black_opt1/），
    須與策略檔的註解切換狀態一致。
    """
    strat = SingleLongBlackStrategy(pct=pct, liquidity=liquidity)
    result = batch.run_folder(strat, folder, start=start, end=end, limit=limit, exclude=GLITCH)
    tag = f"p{int(round(pct * 100))}_{'liq1000' if liquidity else 'raw'}"
    sub = f"single_long_black_{variant}" if variant else "single_long_black"
    out_dir = os.path.join(out_root or RESULT_DIR, sub, tag)
    batch.write_results(result, out_dir, f"single_long_black_{tag}", write_trades=write_trades)
    with open(os.path.join(out_dir, "diag.json"), "w", encoding="utf-8") as f:
        json.dump(strat.diag, f, ensure_ascii=False, indent=1)
    return {"tag": tag, **result["aggregate"], **strat.diag, "failed": len(result["failed"])}


def main(argv) -> int:
    parser = argparse.ArgumentParser(description="長黑出量後突破：資料夾批次回測（vbt）")
    parser.add_argument("folder")
    parser.add_argument("--pct", type=float, default=0.02, help="CHoCH 用的 ZigZag 門檻")
    parser.add_argument("--liq", action="store_true", help="可成交版：成交金額 > 1,000 萬")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--out", default=None, help="輸出根目錄（預設 result/）")
    parser.add_argument("--variant", default=None, help="輸出資料夾後綴（例：opt1），須與註解狀態一致")
    a = parser.parse_args(argv[1:])
    res = run_config(a.folder, a.pct, a.liq, a.limit, a.start, a.end, out_root=a.out, variant=a.variant)
    print(json.dumps({k: res[k] for k in ("tag", "交易次數", "勝率(%)", "獲利因子(PF)", "平均持有天數",
                                          "setups", "signals", "entries", "exit_stop", "exit_choch",
                                          "exit_ma", "exit_multi", "open_at_end", "failed")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
