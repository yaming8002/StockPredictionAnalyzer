"""
MACD 單股全部對照表：正式結果全市場重現 ＋ 公開教學檔逐筆對帳（vbt 單股引擎）
==============================================================================
文章（一）～（十）用的對照表都是 macd_sweep.variant_trades（MacdVariant）跑出來、只存彙總列的 CSV。
這支在全市場 2,258 檔、標準區間把**每一張表的每一列**重算一次：

一、正式結果重現：重算的規格欄（交易次數、勝率、持有天、獲利／虧損平均、中位數、期望值、獲利因子、
    總獲利）＋未平倉%／參與股票數，必須與存檔 CSV 逐欄相同（數值容許 1e-9 相對誤差）。
    涵蓋 _baselines、_matrix_3x3、_entry_sweep_v2、_exit_parallel、_exit_sweep、_exit_pure_risk、
    macd_exit_replace、macd_combo_6x6。生效率類診斷欄（新規則先觸發%、風控生效%）不重算
    （由 verify_macd_sweeps 檢查四驗邏輯）。
二、公開教學檔逐筆對帳：設定在 single_macd_strategy.py 有對應註解切換狀態的（verify_macd_sweeps 的
    toggle_cases，共 119 種），另把策略檔照文章切換後 exec，全市場逐筆交易必須與 MacdVariant 完全相同。

執行：python _02_strategy/macd_strategy/verify_macd_full.py [--workers 4] [--limit N] [--list]
    預設全市場；--limit N 只跑前 N 檔（只驗流程，存檔是全市場、無從比對，會略過檢查一）。
    每個進程各自備妥全市場資料（約 4 GB），--workers 依記憶體調整。
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _02_strategy.base.vbt import common  # noqa: E402
from _02_strategy.macd_strategy import (macd_combo, macd_entry_sweep, macd_exit_replace,  # noqa: E402
                                        macd_exit_sweep, macd_matrix_sweep)
from _02_strategy.macd_strategy.macd_sweep import prepare, variant_trades  # noqa: E402
from _02_strategy.macd_strategy.macd_variants import NAME_BASE, NAME_EXIT, NAME_FILTER  # noqa: E402
from _02_strategy.macd_strategy.single_macd_strategy import RESULT_DIR  # noqa: E402
from _02_strategy.macd_strategy.verify_macd_sweeps import (_same, _trades, load_toggled,  # noqa: E402
                                                            toggle_cases, toggled_source)

SINGLE = os.path.join(RESULT_DIR, "single_macd")
REL_TOL = 1e-9


def stored_rows() -> list:
    """[(CSV 路徑, 標籤欄 dict, 設定 (base, filt, exit, mode, liq))]，依各 driver 的常數展開。"""
    rows = []
    f = os.path.join(SINGLE, "_baselines.csv")
    rows += [(f, {"變體": lb}, (b, "none", "native", "replace", False)) for lb, b in macd_matrix_sweep.BASELINES]
    f = os.path.join(SINGLE, "_matrix_3x3.csv")
    rows += [(f, {"組合": f"{en} × {xn}"}, (b, "none", ex, "replace", True))
             for b, en in macd_matrix_sweep.ENTRIES for ex, xn in macd_matrix_sweep.EXITS]
    f = os.path.join(SINGLE, "_entry_sweep_v2.csv")
    rows += [(f, {"母體": pop, "濾網": NAME_FILTER[fl]}, (b, fl, "native", "replace", True))
             for b, pop in macd_entry_sweep.POPULATIONS for fl in macd_entry_sweep.FILTERS]
    xs, lab = macd_exit_sweep, macd_exit_sweep.EXIT_LABEL
    f = os.path.join(SINGLE, "_exit_parallel.csv")
    rows += [(f, {"組": "A並行", "母體": pop, "出場": lab[ex]}, (b, "none", ex, "append", True))
             for b, pop in xs.POPULATIONS for ex in xs.PARALLEL_EXITS]
    f = os.path.join(SINGLE, "_exit_sweep.csv")
    for b, pop in xs.POPULATIONS:
        rows += [(f, {"組": "A取代", "母體": pop, "出場": lab[ex]}, (b, "none", ex, "replace", True))
                 for ex in xs.TREND_EXITS]
        rows += [(f, {"組": "B疊加", "母體": pop, "出場": lab[ex]}, (b, "none", ex, "append", True))
                 for ex in xs.RISK_EXITS]
    f = os.path.join(SINGLE, "_exit_pure_risk.csv")
    rows += [(f, {"基礎": name, "出場": lab[ex]}, (b, "none", ex, "replace", True))
             for b, name in xs.BASES for ex in xs.PURE_EXITS]
    f = os.path.join(macd_exit_replace.OUT, "macd_exit_replace.csv")
    rows += [(f, {"母體": NAME_BASE[b], "出場": NAME_EXIT[ex]}, (b, "none", ex, "replace", True))
             for b in macd_exit_replace.BASES for ex in macd_exit_replace.EXITS]
    f = os.path.join(macd_combo.OUT, "macd_combo_6x6.csv")
    rows += [(f, {"母體": macd_combo.POP_NAME[b], "進場濾網": NAME_FILTER[fl], "出場": macd_combo.EXIT_NAME[ex]},
              (b, fl, ex, "replace", True))
             for b in macd_combo.BASES for fl in macd_combo.GRID_FILTERS for ex in macd_combo.GRID_EXITS]
    return rows


def teaching_map() -> dict:
    """設定 → (說明, toggled_source 參數)。"""
    return {cfg: (desc, kw) for desc, kw, cfg in toggle_cases()}


def _close(want, got) -> bool:
    try:
        w, g = float(want), float(got)
    except (TypeError, ValueError):
        return str(want) == str(got)
    if np.isnan(w) and np.isnan(g):
        return True
    return bool(np.isclose(w, g, rtol=REL_TOL, atol=1e-9))


def compare_row(stored: pd.DataFrame, labels: dict, s: dict, n_stock: int) -> str:
    """存檔列 vs 重算 summary；回傳問題說明（空字串＝一致）。"""
    hit = stored
    for k, v in labels.items():
        hit = hit[hit[k].astype(str) == str(v)]
    if len(hit) != 1:
        return f"存檔找到 {len(hit)} 列（應為 1）"
    row = hit.iloc[0]
    want = dict(common.spec_row(s))
    want.update({"未平倉%": s.get("未平倉%"), "參與股票數": n_stock,
                 "進場筆數": s.get("開倉數"), "已平倉": s.get("已平倉數")})
    diffs = [f"{k}: 存檔 {row[k]} vs 重算 {v}" for k, v in want.items()
             if k in row.index and not _close(row[k], v)]
    return "；".join(diffs[:6])


def run_group(jobs: list, limit) -> list:
    """一組設定：備妥資料一次，逐設定重算；回傳 [(設定, 說明, 問題清單)]。"""
    data = prepare(limit=limit)
    csv_cache, teach = {}, teaching_map()
    out = []
    for cfg, row_specs in jobs:
        t0 = time.time()
        base, filt, ex, mode, liq = cfg
        trades, s = variant_trades(data, base, filt, ex, mode, liq)
        problems = []
        if limit is None:
            for path, labels in row_specs:
                if path not in csv_cache:
                    csv_cache[path] = pd.read_csv(path)
                p = compare_row(csv_cache[path], labels, s, len(data))
                if p:
                    problems.append(f"{os.path.basename(path)} {labels}：{p}")
        how = "MacdVariant"
        if cfg in teach:
            desc, kw = teach[cfg]
            mod = load_toggled(toggled_source(**kw), f"full_{abs(hash(cfg))}")
            t = _trades(mod.SingleMacdStrategy(), data)
            how = f"MacdVariant＋策略檔「{desc}」"
            if not _same(t, trades):
                problems.append(f"策略檔逐筆不同：策略檔 {len(t):,} 筆 vs MacdVariant {len(trades):,} 筆")
        out.append((cfg, how, len(trades), len(data), problems, time.time() - t0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 檔（略過檢查一）；省略或 0＝全市場")
    ap.add_argument("--list", action="store_true", help="只列出要跑的設定數，不執行")
    ap.add_argument("--files", nargs="+", default=None,
                    help="只驗這幾張存檔表（檔名，例 _baselines.csv）；給了就不另跑表外的策略檔註解狀態")
    a = ap.parse_args()
    limit = a.limit or None

    by_cfg = {}
    for path, labels, cfg in stored_rows():
        if a.files and os.path.basename(path) not in a.files:
            continue
        by_cfg.setdefault(cfg, []).append((path, labels))
    if not a.files:
        for cfg in teaching_map():
            by_cfg.setdefault(cfg, [])
    jobs = list(by_cfg.items())
    n_rows = sum(len(v) for v in by_cfg.values())
    n_teach = sum(cfg in teaching_map() for cfg in by_cfg)
    print(f"{len(jobs)} 種設定（存檔 {n_rows} 列、策略檔註解狀態 {n_teach} 種）｜"
          f"{'全市場' if limit is None else f'前 {limit} 檔（略過檢查一）'}｜{a.workers} 進程", flush=True)
    if a.list:
        return 0

    t0 = time.time()
    groups = [jobs[i::a.workers] for i in range(a.workers)]
    bad = 0
    with ProcessPoolExecutor(max_workers=len(groups)) as pool:
        futs = [pool.submit(run_group, g, limit) for g in groups if g]
        for f in futs:
            for cfg, how, n_tr, n_stock, problems, sec in f.result():
                bad += bool(problems)
                print(f"[{'NG' if problems else 'OK'}] {cfg}｜{how}｜{n_tr:,} 筆｜{n_stock} 檔｜{sec:.0f} 秒",
                      flush=True)
                for p in problems:
                    print("    " + p, flush=True)
    print(f"不一致 {bad} 種設定｜{time.time() - t0:.0f} 秒")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
