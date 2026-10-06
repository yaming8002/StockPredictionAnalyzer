"""
MACD 變體掃描（共用執行層）
==============================
文章系列的對照表都是「同一批股票、換不同變體跑很多次」。照 `batch.run_folder` 的
做法每個變體重讀一次 parquet、重算一次指標，掃 126 組就要重做 126 次——這裡把
「讀檔 ＋ 算指標」抽出來只做一次，之後所有變體共用同一份備妥的 DataFrame。

備妥包含：`add_columns` 的向量化欄位，以及三個逐根掃描型指標（ZigZag／Supertrend／
SAR）——後三者本來是「用到才算」，但掃描時幾乎一定有變體會用到，一次算完反而省。

輸出一律走 `common.spec_row` 的規格 10 欄，並補「未平倉%」這個診斷欄：
取代型出場裡的固定停利／固定停損當唯一出場時會有大量部位抱到期末沒平倉，
沒有這一欄會讀出「勝率 99.9%、獲利因子上千」這種假表。
"""
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd

from _02_strategy.base.vbt import common
from _02_strategy.base.vbt.common import DEFAULT_END, DEFAULT_START, GLITCH
from _02_strategy.macd_strategy.macd_variants import MacdVariant

DATA = common.DATA_DIR


def prepare(folder: str = DATA, limit: int = None, start: str = DEFAULT_START,
            end: str = DEFAULT_END) -> dict:
    """讀全市場、算好所有變體會用到的欄位；回傳 {stock_id: df}。"""
    prep = MacdVariant()
    # 讀檔走 common 共用讀檔；逐檔讀、逐檔備妥，記憶體同時只壓一檔原始資料
    data = {}
    for sid, df in common.iter_market(folder, start=start, end=end, limit=limit,
                                      exclude=GLITCH, min_rows=2):
        common.ensure_columns(df)
        df = prep.add_columns(df.copy())
        prep._ensure_zigzag(df)          # 頂頂低用
        prep._ensure_supertrend(df)      # 超級趨勢出場用
        prep._ensure_psar(df)            # SAR 出場用
        data[sid] = df
    return data


def variant_trades(data: dict, base: str, filt: str, exit_name: str,
                   exit_mode: str, liquidity: bool = True):
    """單一變體跑全市場，回傳 (逐筆交易, summary)。summary 含診斷欄「未平倉%」。"""
    v = MacdVariant()
    v.BASE, v.FILTER, v.EXIT, v.EXIT_MODE, v.LIQUIDITY = (
        base, filt, exit_name, exit_mode, liquidity)
    frames, n_entry, n_trade = [], 0, 0
    for sid, df in data.items():
        res = v.run(df, sid)
        frames.append(res["trades"])
        # 未平倉率的分母＝實際開倉數。vbt 的 entries 是「所有買訊」，同一個部位
        # 沒平倉前的重複買訊不會開新倉，拿它當分母會低估未平倉率，故用狀態機重數。
        e, x = v.build_signals(df)       # df 在 prepare 已備妥欄位，不必再 add_columns
        n_entry += _count_positions(e.to_numpy(), x.to_numpy())
        n_trade += len(res["trades"])
    trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary = common.summarize_trades(trades)
    summary["未平倉%"] = (round((n_entry - n_trade) / n_entry * 100, 2)
                          if n_entry else 0.0)
    return trades, summary


def run_variant(data: dict, base: str, filt: str, exit_name: str,
                exit_mode: str, liquidity: bool = True) -> dict:
    """只要 summary 時的薄包裝（對照表 driver 用）。"""
    return variant_trades(data, base, filt, exit_name, exit_mode, liquidity)[1]


def _count_positions(entries: np.ndarray, exits: np.ndarray) -> int:
    """
    走一次狀態機數「實際開了幾個部位」：空手遇買訊才開倉，持倉中的買訊忽略。

    ⚠️ **同一根同時有買訊與賣訊 → 兩邊都不動作**（`continue`）。這條必須跟引擎一致：
    vbt `from_signals` 在同根衝突時兩邊都不執行、部位繼續留著，而這個狀態機原本會
    在那一根「平掉」，於是之後的買訊又被算成新開倉 → 開倉數被高估 → 未平倉率被高估。

    實測（200 檔）：純背離 × 跌破年線同根 664 次，未平倉率因此從真值 4.31% 被算成
    20.47%；原生出場（死叉）同根 1,184 次，1.00% 被算成 18.30%。交叉母體的黃金交叉
    與死叉幾乎不同根（12 次）、零軸 13 次，所以只差 0.2 個百分點——**這也是為什麼這個
    bug 只在背離母體現形**，看交叉母體完全發現不了。

    只影響這個診斷欄：交易筆數、獲利因子等全部來自 vbt，不經過這裡。
    """
    n, holding = 0, False
    for i in range(len(entries)):
        if entries[i] and exits[i]:
            continue
        if holding:
            if exits[i]:
                holding = False
        elif entries[i]:
            holding = True
            n += 1
    return n


def spec_rows(summaries: list) -> pd.DataFrame:
    """把 [(labels_dict, summary)] 轉成規格表（10 欄 ＋ 未平倉%）。"""
    rows = []
    for labels, s in summaries:
        row = common.spec_row(s, **labels)
        row["未平倉%"] = s.get("未平倉%", 0.0)
        rows.append(row)
    out = pd.DataFrame(rows)
    common.assert_spec_columns(out)
    return out
