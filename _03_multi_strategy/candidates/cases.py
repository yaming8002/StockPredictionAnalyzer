"""
新系列候選（海龜 vs 雙底）多股比較的組合清單 — driver／對帳／蒙地卡羅共用這一份
================================================================================
20 組，全部是可成交版（成交金額 5 日均量×股價 > 1,000 萬）：
  海龜 系統 1、系統 2；
  雙底 ZigZag 2%～10% × 出場「CHoCH 或跌破波段低點」（9 組）；
  雙底 ZigZag 2%～10% × 出場「只用 CHoCH（不設停損）」（9 組）。

每組附上單股逐筆交易檔（_02 回測段的輸出），蒙地卡羅（_04）與對帳都讀它。
"""
import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np

from _02_strategy.double_bottom_strategy.single_double_bottom_strategy import RESULT_DIR as DB_RESULT
from _02_strategy.turtle_strategy.single_turtle_strategy import RESULT_DIR as TURTLE_RESULT
from _03_multi_strategy.double_bottom.multi_double_bottom import MultiDoubleBottom, simulate_state_exit
from _03_multi_strategy.turtle.multi_turtle import MultiTurtle


def _case_turtle(system: int) -> dict:
    return {"key": f"turtle_s{system}",
            "label": f"海龜 系統 {system}（{'20 日高進／10 日低出' if system == 1 else '55 日高進／20 日低出'}）",
            "trades": os.path.join(TURTLE_RESULT, "single_turtle", f"s{system}_opt1",
                                   f"single_turtle_s{system}_trades.csv"),
            "make": lambda **kw: _with(MultiTurtle(**kw), SYSTEM=system)}


def _case_db(pct: int, stop: bool) -> dict:
    folder = "single_double_bottom" if stop else "single_double_bottom_nostop"
    tag = f"p{pct}_liq1000"
    exit_label = "CHoCH 或跌破波段低點" if stop else "只用 CHoCH（不設停損）"
    return {"key": f"db_p{pct}_{'stop' if stop else 'nostop'}",
            "label": f"雙底 {pct}%｜出場：{exit_label}",
            "trades": os.path.join(DB_RESULT, folder, tag, f"single_double_bottom_{tag}_trades.csv"),
            "make": lambda **kw: _with(MultiDoubleBottom(**kw), PCT=pct / 100, STOP_LOW=stop)}


def _with(inst, **attrs):
    """把類別層級的設定（SYSTEM／PCT／STOP_LOW）設在這個實例上。"""
    for k, v in attrs.items():
        setattr(inst, k, v)
    return inst


CASES = ([_case_turtle(1), _case_turtle(2)]
         + [_case_db(p, True) for p in range(2, 11)]
         + [_case_db(p, False) for p in range(2, 11)])


def run_with_fees(case: dict, inst, panel: dict, prio, want_equity: bool = True, fees: str = "rate") -> dict:
    """
    用 multi_double_bottom.simulate_state_exit 跑一格，費用口徑可選（"rate"／"tw"）。
    雙底直接走自己的 run_panel；海龜是狀態型出場：ths 全 NaN（不用峰值）、lowbrk＝出場面板
    （與基底 run_panel 逐筆一致，見 verify_tw_fees.py）。
    """
    if case["key"].startswith("db_"):
        return inst.run_panel(panel, prio, want_equity=want_equity, fees=fees)
    return simulate_state_exit(inst, panel, np.full(panel["price"].shape, np.nan), panel["exits"],
                               True, prio, want_equity, fees)
