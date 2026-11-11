# -*- coding: utf-8 -*-
"""
產生〈雙底（六）：出場的替換〉的表格 HTML。

替換＝出場只剩新規則那一條，原本的 ① 高點下移、② 跌破波段低點都不用（策略檔「附錄：替換」，
結果在 result/single_double_bottom_exit_replace/）。期末還沒賣出的部位不在已平倉統計裡，所以：
- 6 格中任一格未平倉率 > 10% 的出場（用戶 2026-11-03 定）不列績效，只列未平倉率＋未賣出部位的概況；
- 其餘照標準 10 欄出表、跟基本版比上底色，總表多一欄未平倉率。
- 保留組裡要加警告的出場（WARN）多一張「持有期間的回檔」表，
  看已平倉交易在賣出前曾經跌多深——勝率、獲利因子好看，可能是靠抱著帳面虧損等回來換的。

算法、位數、底色沿用 build_article4_tables／build_article2_tables；未賣出部位讀
_02_strategy/double_bottom_strategy/exit_open_positions.py 匯出的 open_positions.csv；
回檔表讀逐筆 trades＋股價 parquet 收盤（只讀資料，不跑回測）。

執行：STOCK_DATA_DIR=<股價 parquet 目錄> PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
          python _04_analysis/double_bottom/article/build_article6_tables.py
輸出：終端印各表 HTML；result/single_double_bottom_exit_replace/_article6_stats.csv（verify 用）
"""
import functools
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
for p in (_root, _here):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _02_strategy.base.vbt import common  # noqa: E402
from _02_strategy.double_bottom_strategy.exit_filter_sweep import SUB_REPLACE  # noqa: E402
from build_article2_tables import CELL_HEAD, LIQS, N_CELLS, PCTS, fmt, half_up, row_label, table  # noqa: E402
from build_article4_tables import LABELS, RESULT_DIR, stats  # noqa: E402

SRC = os.path.join(RESULT_DIR, SUB_REPLACE)
STATS_FILE = os.path.join(SRC, "_article6_stats.csv")
OPEN_MAX = 10.0                      # 任一格未平倉率超過這個 % 就不列績效
ALL_EXITS = ("base", "target", "target65", "neck", "neckfail", "bolow",
             "breakeven", "atr", "trail20", "ma10", "mtop")
OPEN_CELL = (9, True)                # 未賣出部位概況用的格子：9% 可成交
# 加警告＋回檔表的出場：逐條依當下情況判定（賣出前常要先撐過一段帳面下跌），
# 不用持有天數之類的固定門檻自動挑（用戶 2026-11-05）
WARN = ("neck", "trail20")
WARN_MARK = " ⚠️"                    # 標記放在名稱後面（用戶 2026-11-06：icon 一律放在資訊後面）


def split_exits(df: pd.DataFrame) -> tuple:
    """(保留, 移除)，各自維持 ALL_EXITS 的順序；base 不算在內。"""
    worst = df.groupby("變體")["未平倉率%"].max()
    kept = tuple(e for e in ALL_EXITS[1:] if worst[e] <= OPEN_MAX)
    dropped = tuple(e for e in ALL_EXITS[1:] if worst[e] > OPEN_MAX)
    return kept, dropped


def _cell(df, e, p, liq, col):
    return df[(df["門檻%"] == p) & (df["可成交"] == liq) & (df["變體"] == e)].iloc[0][col]


def _six_head(first: str, tail: str) -> str:
    return f"<thead><tr><th>{first}</th>{CELL_HEAD}{tail}</tr></thead>"


def open_rate_table(df: pd.DataFrame, exits) -> str:
    """移除組：各出場 6 格的期末未賣出比例（%，不上色）；基本版放第一列當對照。"""
    out = ['<div class="table-wrap"><table class="bt-table">', _six_head("出場", ""), "<tbody>"]
    for e in ("base",) + tuple(exits):
        vals = [f"{half_up(_cell(df, e, p, liq, '未平倉率%'), 1)}" for liq in LIQS for p in PCTS]
        name = f"<b>{LABELS[e]}</b>" if e == "base" else LABELS[e]
        out.append(f"<tr><td>{name}</td>{''.join(f'<td>{x}</td>' for x in vals)}</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def open_positions(e: str) -> pd.DataFrame:
    p, liq = OPEN_CELL
    return pd.read_csv(os.path.join(SRC, f"p{p}_{'liq1000' if liq else 'raw'}__{e}", "open_positions.csv"))


def open_summary(df: pd.DataFrame, exits) -> pd.DataFrame:
    """OPEN_CELL 那一格：進場筆數、未賣出筆數與比例、帳面虧損的比例、帳面報酬中位數、持有年數中位數。"""
    p, liq = OPEN_CELL
    rows = []
    for e in exits:
        op = open_positions(e)
        entries = int(_cell(df, e, p, liq, "entries"))
        rows.append({"變體": e, "進場": entries, "未賣出": len(op), "未賣出%": len(op) / entries * 100,
                     "帳面虧損%": (op["unrealized_pct"] < 0).mean() * 100,
                     "帳面報酬中位數%": op["unrealized_pct"].median(),
                     "持有年中位數": op["held_days"].median() / 365})
    return pd.DataFrame(rows)


def open_summary_table(df: pd.DataFrame, exits) -> str:
    head = ("<thead><tr><th>出場</th><th>進場筆數</th><th>期末未賣出</th><th>佔進場%</th>"
            "<th>其中帳面虧損%</th><th>帳面報酬中位數%</th><th>已持有年數中位數</th></tr></thead>")
    out = ['<div class="table-wrap"><table class="bt-table">', head, "<tbody>"]
    for _, r in open_summary(df, exits).iterrows():
        cells = [f"{r['進場']:,}", f"{r['未賣出']:,}", f"{half_up(r['未賣出%'], 1)}",
                 f"{half_up(r['帳面虧損%'], 1)}", fmt("中位數%", r["帳面報酬中位數%"]),
                 f"{half_up(r['持有年中位數'], 1)}"]
        out.append(f"<tr><td>{LABELS[r['變體']]}</td>{''.join(f'<td>{x}</td>' for x in cells)}</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def _closed_trades(e: str, p: int, liq: bool) -> pd.DataFrame:
    """該格的已平倉交易；排掉損益剛好 0 的，與主表「交易次數」同一批。"""
    tag = f"p{p}_{'liq1000' if liq else 'raw'}__{e}"
    tr = pd.read_csv(os.path.join(SRC, tag, f"single_double_bottom_{tag}_trades.csv"),
                     parse_dates=["buy_date", "sell_date"])
    tr["real_pnl"] = pd.to_numeric(tr["real_pnl"], errors="coerce").fillna(0)
    return tr[tr["real_pnl"] != 0].reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def _closes() -> dict:
    """{stock_id: (日期, 收盤)}，全市場只讀一次。"""
    return {sid: (df.index.to_numpy(), df["close"].to_numpy(dtype=float))
            for sid, df in common.iter_market(columns=["close"])}


def worst_close_pct(tr: pd.DataFrame) -> np.ndarray:
    """每筆交易持有期間最低收盤相對買進價的 %。
    持有期間＝買進日～賣出日前一天：賣出日是開盤成交，那天的收盤已經不在手上。"""
    px = _closes()
    out = np.empty(len(tr))
    for k, r in enumerate(tr.itertuples(index=False)):
        days, close = px[r.stock_id]
        a = np.searchsorted(days, r.buy_date.to_datetime64(), "left")
        b = max(np.searchsorted(days, r.sell_date.to_datetime64(), "left"), a + 1)
        out[k] = (np.nanmin(close[a:b]) / r.buy_price - 1) * 100
    return out


def drawdown(e: str) -> pd.DataFrame:
    """六格：持有超過一年的比例、持有期間曾跌 ≥10%／≥20% 的比例、賺錢交易中曾跌 ≥10% 的比例與其持有天中位數。"""
    rows = []
    for liq in LIQS:
        for p in PCTS:
            tr = _closed_trades(e, p, liq)
            dd = worst_close_pct(tr)
            held = (tr["sell_date"] - tr["buy_date"]).dt.days.to_numpy()
            win = (tr["real_pnl"] > 0).to_numpy()
            win_dd = win & (dd <= -10)
            rows.append({"門檻%": p, "可成交": liq, "變體": e, "交易次數": len(tr),
                         "超過一年%": (held > 365).mean() * 100,
                         "曾跌10%": (dd <= -10).mean() * 100, "曾跌20%": (dd <= -20).mean() * 100,
                         "賺錢曾跌10%": win_dd.sum() / win.sum() * 100,
                         "賺錢曾跌10%持有天中位數": float(np.median(held[win_dd]))})
    return pd.DataFrame(rows)


def drawdown_table(dd: pd.DataFrame) -> str:
    # 欄名補「佔比%」：數值是佔交易筆數的百分比，與「持有超過一年%」同一種寫法
    head = ("<thead><tr><th>門檻</th><th>持有超過一年%</th><th>曾跌 ≥10%<br>佔比%</th><th>曾跌 ≥20%<br>佔比%</th>"
            "<th>賺錢交易中<br>曾跌 ≥10% 佔比%</th><th>這些交易的持有天中位數</th></tr></thead>")
    out = ['<div class="table-wrap"><table class="bt-table">', head, "<tbody>"]
    for _, r in dd.iterrows():
        cells = [half_up(r["超過一年%"], 1), half_up(r["曾跌10%"], 1), half_up(r["曾跌20%"], 1),
                 half_up(r["賺錢曾跌10%"], 1), half_up(r["賺錢曾跌10%持有天中位數"], 0)]
        out.append(f"<tr><td>{row_label(int(r['門檻%']), bool(r['可成交']))}</td>"
                   f"{''.join(f'<td>{x}</td>' for x in cells)}</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def summary_table(df: pd.DataFrame, exits, warn=()) -> str:
    """總表：保留組 6 格的獲利因子＋比基本版高的格數＋未平倉率範圍（不上色）；warn 裡的出場名稱前加警告符號。"""
    out = ['<div class="table-wrap"><table class="bt-table">',
           _six_head("出場", "<th>比基本版高</th><th>期末未賣出%</th>"), "<tbody>"]
    for e in ("base",) + tuple(exits):
        pfs, ups = [], 0
        for liq in LIQS:
            for p in PCTS:
                v, b = _cell(df, e, p, liq, "獲利因子"), _cell(df, "base", p, liq, "獲利因子")
                pfs.append(fmt("獲利因子", v))
                ups += v > b
        sub = df[df["變體"] == e]["未平倉率%"]
        rng = f"{half_up(sub.min(), 1)}～{half_up(sub.max(), 1)}"
        tail = f"<td>—</td><td>{rng}</td>" if e == "base" else f"<td>{ups}／{N_CELLS}</td><td>{rng}</td>"
        name = f"<b>{LABELS[e]}</b>" if e == "base" else LABELS[e] + (WARN_MARK if e in warn else "")
        out.append(f"<tr><td>{name}</td>{''.join(f'<td>{x}</td>' for x in pfs)}{tail}</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def all_tables(df: pd.DataFrame) -> list:
    """文章裡依序出現的全部表格：基本版、保留組各一張（加警告的緊接回檔表）、總表，
    最後是移除組的未平倉率與未賣出概況。"""
    kept, dropped = split_exits(df)
    warn = tuple(e for e in kept if e in WARN)
    out = [("base", table(df, "base"))]
    for e in kept:
        out.append((e, table(df, e)))
        if e in warn:
            out.append((f"回檔:{e}", drawdown_table(drawdown(e))))
    return out + [("總表", summary_table(df, kept, warn)),
                  ("未賣出比例", open_rate_table(df, dropped)), ("未賣出概況", open_summary_table(df, dropped))]


def main() -> int:
    df = stats(ALL_EXITS, src=SRC)
    df.to_csv(STATS_FILE, index=False, encoding="utf-8-sig")
    kept, dropped = split_exits(df)
    print(f"保留：{[LABELS[e] for e in kept]}｜移除：{[LABELS[e] for e in dropped]}"
          f"｜警告：{[LABELS[e] for e in WARN]}")
    for name, html in all_tables(df):
        print(f"=== {LABELS.get(name, name)} ===")
        print(html)
    print(f"→ {STATS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
