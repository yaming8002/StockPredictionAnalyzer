"""
MACD（七）（八）（九）：出場規則 × 母體的三張表
===================================================
重建 2026-08-27／09-06／09-07 三輪遺失的 scratchpad driver，一支檔三種模式：

  --mode parallel → `_exit_parallel.csv`（（七）篇，**2026-09-06 附加版定義**）
      趨勢類五條（跌破 MA200／Supertrend 翻空／SAR 翻空／頂頂低／跌破 20 日低）× 4 母體，
      **附加**：原生出場 OR 新規則，先觸發者算。
      ⚠️ 舊的 `_exit_parallel.csv` 是 09-02 那輪（全部走逐根掃描）；（七）篇用的是 09-06 重跑，
      四條向量化規則走 `native | rule`，與本 driver 相同（見「定義與口徑」）。
  --mode sweep → `_exit_sweep.csv`（（七）（八）篇的 40 格）
      A 取代：趨勢類五條 × 4 母體，原生出場整條換掉（mix 在此組與交叉必然逐字相同，當恆等檢查）；
      B 疊加：風控類五條（吊燈 3ATR／回落 10%／停損 2ATR／停利 +20%／抱滿 60 天）× 4 母體。
  --mode pure → `_exit_pure_risk.csv`（（九）篇表一）
      風控類五條當**唯一出場**（純取代）× 3 基礎；停利／停損兩條在這個接法下口徑失效
      （只等一個固定價位、沒碰到就無限期抱著），表上的「進場筆數／已平倉／未平倉%」就是排除依據。
  --mode all（預設）→ 三張都跑，共用同一份 prepare。

母體：交叉（黃金交叉進×死叉出）／零軸（上穿 0 進×下穿 0 出）／背離（底背離進×死叉出）／
      交叉進×零軸出（mix）。可成交門檻（成交金額 > 1,000 萬）只 gate 進場。

【診斷欄】（memory backtest-diagnostic-columns：疊加規則必量生效率，否則沒觸發會長得像沒差異）
  分母一律是已平倉部位數，用與引擎一致的狀態機重數（同根買賣訊同時成立 → 兩邊都不動作，見
  macd_sweep._count_positions）；重數的平倉數若與 vbt 的逐筆交易數對不上會印出警告。
  - 新規則先觸發%（parallel）：出場那一根新規則成立、**原生出場沒成立**＝新規則搶在前面。
  - 風控生效%（sweep 的 B 組）：出場那一根新規則成立，**含與原生出場同一根成立**
    （08-27 紀錄的定義：「含與原出場同日觸發」）。
  兩欄定義不同是沿用舊表：以 30 檔冒煙對舊 `_exit_parallel_smoke.csv`，四條向量化規則的
  舊值與「搶先」逐位吻合（例：交叉＋SAR 38.11 vs 38.14、零軸＋超級趨勢 61.03 vs 61.03），
  「含同根」則高出 5～20 個百分點；舊表的頂頂低那欄卻是「含同根」（2.62 vs 2.63）——
  09-02 driver 兩種規則用了不同算法。本 driver 統一：先觸發%一律「搶先」（頂頂低因此與舊表
  不同口徑，冒煙 2.63→1.97）、生效%一律「含同根」。
  最後一欄「未平倉%」所有模式都附（取代型必量）；pure 模式沿用舊表把它放在前段。

【定義與口徑】
  - 跌破 MA200：`close < ma_long & close.shift(1) >= ma_long.shift(1)`（昨收比**昨日** MA200），
    與 single_macd_strategy.sell_signal 的註解行、macd_variants 同一份。舊紀錄提過另一版
    「昨收比今日 MA200」，兩版差約萬分之二的獲利因子；本 driver 一律用策略檔現行這版。
  - 頂頂低與五條風控是路徑相依，走 single_macd_strategy._scan_path_exits 逐根掃描；
    附加＝use_native=True、取代＝False（由 MacdVariant.EXIT_MODE 決定，不看 _REPLACE_RULES）。
  - 向量化附加（四條趨勢規則）交給 vbt：同一根同時有買訊與賣訊時兩邊都不動作；
    逐根掃描則是「持倉中遇賣訊就平、當根的買訊不接」——兩條路徑在同根衝突的處理不同，
    背離母體（底背離與死叉常同根）受影響最大。這是引擎現況，本 driver 不改。

執行：
    python _02_strategy/macd_strategy/macd_exit_sweep.py [--mode all] [--limit 30] [--out 目錄]
輸出：預設 result/single_macd/
"""
import argparse
import os
import sys
import time

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numba import njit  # noqa: E402

from _02_strategy.base.vbt.common import DEFAULT_END, DEFAULT_START  # noqa: E402
from _02_strategy.macd_strategy.macd_sweep import (  # noqa: E402
    legacy_row, make_variant, prepare, variant_trades, write_csv)
from _02_strategy.macd_strategy.macd_variants import _PATH_EXITS  # noqa: E402
from _02_strategy.macd_strategy.single_macd_strategy import (  # noqa: E402
    ATR_PERIOD, EXIT_LOWER_HIGH, RESULT_DIR)

OUT = os.path.join(RESULT_DIR, "single_macd")

POPULATIONS = [("cross", "交叉"), ("zero", "零軸"), ("div", "背離"), ("mix", "交叉進×零軸出")]
BASES = POPULATIONS[:3]

# 出場顯示名稱沿用舊表（與 macd_variants.NAME_EXIT 的文章用語不同，下游讀舊表欄值）
EXIT_LABEL = {
    "ma200": "跌破MA200", "supertrend": "Supertrend翻空", "psar": "SAR翻空",
    "lowerhigh": "頂頂低", "donchian": "跌破20日低", "chandelier": "吊燈3ATR",
    "trail10": "回落10%", "atrstop": "停損2ATR", "takeprofit": "停利+20%", "time60": "抱滿60天",
}
# 各表的出場順序沿用舊表
PARALLEL_EXITS = ["ma200", "supertrend", "psar", "lowerhigh", "donchian"]
TREND_EXITS = ["ma200", "lowerhigh", "donchian", "supertrend", "psar"]
RISK_EXITS = ["chandelier", "trail10", "atrstop", "takeprofit", "time60"]
PURE_EXITS = ["atrstop", "chandelier", "trail10", "takeprofit", "time60"]


# ── 診斷：出場那一根「新規則本身有沒有成立」───────────────────────────
@njit(cache=True)
def _scan_fired(entry_raw, native_exit, open_, high_, close_, atr, turn_high,
                rule, param, use_native):
    """
    與 single_macd_strategy._scan_path_exits **逐行相同**的掃描，多回傳 fired（判定日）：
    出場那一根本規則有成立（不論原生出場是否同根成立）。
    只用來算生效率；進出場訊號仍以策略檔那支為準——verify_macd_sweeps 會逐檔比對兩支的
    entries／exits 完全相同，改了 _scan_path_exits 這裡要跟著改。
    """
    n = len(close_)
    entries = np.zeros(n, np.bool_)
    exits = np.zeros(n, np.bool_)
    fired_at = np.zeros(n, np.bool_)
    in_pos = False
    entry_i = -1
    entry_px = 0.0
    atr_at_entry = 0.0
    peak_px = -1e18
    peak_th = -1e18
    for i in range(n):
        if not in_pos:
            if entry_raw[i] and i + 1 < n:
                in_pos = True
                entries[i] = True
                entry_i = i
                entry_px = open_[i + 1]
                atr_at_entry = atr[i]
                peak_px = -1e18
                peak_th = -1e18
            continue
        if high_[i] > peak_px:
            peak_px = high_[i]
        fired = False
        if rule == 1:
            t = turn_high[i]
            if not np.isnan(t):
                if t < peak_th:
                    fired = True
                else:
                    peak_th = t
        elif rule == 2:
            a = atr[i]
            if not np.isnan(a) and close_[i] < peak_px - param * a:
                fired = True
        elif rule == 3:
            if close_[i] < peak_px * (1.0 - param):
                fired = True
        elif rule == 4:
            if not np.isnan(atr_at_entry) and close_[i] < entry_px - param * atr_at_entry:
                fired = True
        elif rule == 5:
            if close_[i] >= entry_px * (1.0 + param):
                fired = True
        elif rule == 6:
            if (i - entry_i) >= param:
                fired = True
        if fired or (use_native and native_exit[i]):
            exits[i] = True
            fired_at[i] = fired
            in_pos = False
    return entries, exits, fired_at


def scan_with_fired(v, df: pd.DataFrame):
    """用 v 的設定跑 _scan_fired，回傳判定日的 (entries, exits, fired)；呼叫前須已設好交易區間。"""
    rule = _PATH_EXITS[v.EXIT]
    if rule == EXIT_LOWER_HIGH:
        turn_high = df["zigzag_turn_high"].to_numpy(dtype=np.float64)
    else:
        turn_high = np.full(len(df), np.nan)
    return _scan_fired(
        v.entry_signal(df).to_numpy(), v.sell_signal(df).to_numpy(),
        df["open"].to_numpy(dtype=np.float64), df["high"].to_numpy(dtype=np.float64),
        df["close"].to_numpy(dtype=np.float64),
        df[f"atr_{ATR_PERIOD}"].to_numpy(dtype=np.float64),
        turn_high, rule, v._RULE_PARAM[rule], v.EXIT_MODE == "append")


def _exit_flag_count(entries: np.ndarray, exits: np.ndarray, flag: np.ndarray,
                     native: np.ndarray):
    """
    狀態機重數（成交日）：回傳 (平倉數, 平倉那一根 flag 成立的數, flag 成立且原生出場不成立的數)。
    同根買賣訊同時成立 → 兩邊都不動作，與 vbt 及 macd_sweep._count_positions 一致。
    """
    n_exit, n_flag, n_only, holding = 0, 0, 0, False
    for i in range(len(entries)):
        if entries[i] and exits[i]:
            continue
        if holding:
            if exits[i]:
                holding = False
                n_exit += 1
                n_flag += int(flag[i])
                n_only += int(flag[i] and not native[i])
        elif entries[i]:
            holding = True
    return n_exit, n_flag, n_only


def rule_share(data: dict, base: str, exit_name: str, exit_mode: str,
               start: str = DEFAULT_START, end: str = DEFAULT_END):
    """
    全市場出場歸因，回傳 (含同根%, 搶先%, 重數的平倉數)：
      含同根%＝出場那一根新規則有成立（不論原生出場是否同根成立）；
      搶先% ＝新規則成立、原生出場**沒有**成立（新規則單獨造成這次出場）。
    向量化規則：規則本身＝同設定改成取代時的 sell_signal；路徑相依：_scan_fired 的 fired。
    """
    v = make_variant(base, "none", exit_name, exit_mode)
    alone = make_variant(base, "none", exit_name, "replace")      # 只取規則本身
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    tot_exit, tot_flag, tot_only = 0, 0, 0
    for sid, df in data.items():
        _, e, x = v.window_signals(df, start, end)                 # 成交日、已裁到區間
        keep = (df.index >= lo) & (df.index <= hi)
        if exit_name in _PATH_EXITS:
            se, sx, fired = scan_with_fired(v, df)
            # 自我檢查：診斷用掃描必須與策略檔的掃描產出相同訊號，否則比例不可信
            se_s = pd.Series(se, index=df.index).shift(1, fill_value=False)[keep]
            sx_s = pd.Series(sx, index=df.index).shift(1, fill_value=False)[keep]
            if not (se_s.equals(e) and sx_s.equals(x)):
                raise AssertionError(f"{sid}：_scan_fired 與 _scan_path_exits 訊號不一致")
            flag = pd.Series(fired, index=df.index).shift(1, fill_value=False)[keep]
        else:
            flag = alone.sell_signal(df).shift(1, fill_value=False)[keep]
        # 原生出場（取代型沒有原生出場可搶先，一律視為不成立）
        if exit_mode == "append":
            native = v._native_exit(df).fillna(False).astype(bool)
            native = native.shift(1, fill_value=False)[keep].to_numpy()
        else:
            native = np.zeros(int(keep.sum()), np.bool_)
        n_exit, n_flag, n_only = _exit_flag_count(e.to_numpy(), x.to_numpy(),
                                                  flag.to_numpy(), native)
        tot_exit += n_exit
        tot_flag += n_flag
        tot_only += n_only
    if not tot_exit:
        return 0.0, 0.0, 0
    return (round(tot_flag / tot_exit * 100, 2), round(tot_only / tot_exit * 100, 2), tot_exit)


def _run(data: dict, base: str, exit_name: str, mode: str, diag_col: str = None,
         strict: bool = False):
    """
    跑一格；diag_col 有給就加算出場歸因。回傳 (summary, 診斷欄 dict)。
    strict=True 取「搶先%」（新規則成立、原生出場沒成立），False 取「含同根%」。
    """
    t = time.time()
    _, s = variant_trades(data, base, "none", exit_name, mode)
    diag = {}
    if diag_col:
        incl, only, n_exit = rule_share(data, base, exit_name, mode)
        diag[diag_col] = only if strict else incl
        if n_exit != s["已平倉數"]:
            print(f"  ⚠ 生效率分母 {n_exit:,} ≠ vbt 平倉數 {s['已平倉數']:,}"
                  f"（{base}×{exit_name}×{mode}），該格生效率不可信", flush=True)
    extra = f"｜{diag_col} {diag[diag_col]}" if diag_col else ""
    print(f"  {base} × {exit_name}（{mode}）：{s['交易次數']:,} 筆｜PF {s['獲利因子(PF)']}｜"
          f"未平倉 {s['未平倉%']}%{extra}｜{time.time() - t:.0f} 秒", flush=True)
    return s, diag


def run_parallel(data: dict) -> list:
    """（七）篇：趨勢類五條 × 4 母體，附加。"""
    rows = []
    for base, pop in POPULATIONS:
        for ex in PARALLEL_EXITS:
            s, diag = _run(data, base, ex, "append", "新規則先觸發%", strict=True)
            rows.append(legacy_row({"組": "A並行", "母體": pop, "出場": EXIT_LABEL[ex]},
                                   s, len(data), diag))
    return rows


def run_sweep(data: dict) -> list:
    """（七）（八）篇 40 格：A 取代（趨勢五條）＋ B 疊加（風控五條）× 4 母體。"""
    rows, a_group = [], {}
    for base, pop in POPULATIONS:
        for ex in TREND_EXITS:
            s, _ = _run(data, base, ex, "replace")
            a_group[(base, ex)] = common_key(s)
            rows.append(legacy_row({"組": "A取代", "母體": pop, "出場": EXIT_LABEL[ex]},
                                   s, len(data), {"風控生效%": None}))
        for ex in RISK_EXITS:
            s, diag = _run(data, base, ex, "append", "風控生效%")
            rows.append(legacy_row({"組": "B疊加", "母體": pop, "出場": EXIT_LABEL[ex]},
                                   s, len(data), diag))
    # 恆等檢查：A 組把出場整條換掉後 mix 與交叉是同一個策略，數字必須逐字相同
    same = all(a_group[("mix", ex)] == a_group[("cross", ex)] for ex in TREND_EXITS)
    print(f"  【恆等檢查】A 取代組 mix ≡ 交叉：{'OK' if same else 'NG（引擎或設定有問題）'}",
          flush=True)
    return rows


def common_key(s: dict) -> tuple:
    """恆等檢查用：交易次數＋獲利因子＋總獲利。"""
    return s["交易次數"], s["獲利因子(PF)"], s["總獲利"]


def run_pure(data: dict) -> list:
    """（九）篇表一：風控五條當唯一出場 × 3 基礎，附進場筆數／已平倉／未平倉%。"""
    rows = []
    for base, name in BASES:
        for ex in PURE_EXITS:
            s, _ = _run(data, base, ex, "replace")
            diag = {"進場筆數": s["開倉數"], "已平倉": s["已平倉數"], "未平倉%": s["未平倉%"]}
            rows.append(legacy_row({"基礎": name, "出場": EXIT_LABEL[ex]}, s, len(data),
                                   diag, trailing_unclosed=False))
    return rows


MODES = {
    "parallel": (run_parallel, "_exit_parallel.csv"),
    "sweep": (run_sweep, "_exit_sweep.csv"),
    "pure": (run_pure, "_exit_pure_risk.csv"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description="MACD 出場規則 × 母體（附加／取代／純取代）")
    ap.add_argument("--mode", choices=[*MODES, "all"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 檔（冒煙用）")
    ap.add_argument("--out", default=OUT, help=f"輸出目錄（預設 {OUT}）")
    a = ap.parse_args()

    t0 = time.time()
    data = prepare(limit=a.limit)
    print(f"載入並備妥 {len(data)} 檔｜{time.time() - t0:.0f} 秒", flush=True)
    for mode in (MODES if a.mode == "all" else [a.mode]):
        fn, name = MODES[mode]
        print(f"【{mode} → {name}】", flush=True)
        path = write_csv(fn(data), a.out, name)
        print(f"  → {path}｜累計 {time.time() - t0:.0f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
