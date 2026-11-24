"""
底部島型進場優化掃描（vbt 單股）：策略檔 single_island_strategy.py 的進場 #14～#33 全部跑一遍。
================================================================================================
出場固定用出場掃描（排除資料異常後、裸版）PF 前五名的單獨規則（用戶 2026-11-17：「先依照出場的五個規則，
改成只針對入場的優化」）：
  持有 20 根／替換 #6 回落 10%／替換 #9 ATR 3 倍／替換 #12 向下跳空／替換 #5 回落 5%
組合：(基準＋20 條進場條件) × 5 出場 × 裸版／可成交 1,000 萬＝210 組。基準同批次重跑，當比較基準與錨點。
--exits h10 topisland：補跑〈島型反轉（一）〉的另外兩種出場（持有 10 根、替換 #13 頂部島型），
  讓進場篇沿用基本篇的三種出場（用戶 2026-11-18 選定）；彙總另存 --summary-name 指定的檔名。
母體：data_anomaly.island_exclude()（2026-11-23 起 2,143 檔；之前是排除 GLITCH 與資料異常 346 檔的 1,912 檔）。

診斷：生效率（%）＝1 − 該條件 signals ÷ 同口徑基準 signals（被擋掉的進場判定比例；signals 與出場無關）；
      未平倉率（%）＝期末仍持有 ÷ 進場數。

執行：python _02_strategy/island_strategy/island_entry_sweep.py [--workers 10] [--limit N] [--out 目錄] [--only 子字串]
      [--exits h20 trail10 ...] [--summary-name _sweep_summary.csv]
輸出：result/single_island_entry/<tag>/、result/single_island_entry/_sweep_summary.csv
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd

from _02_strategy.base.vbt import common
from _02_strategy.island_strategy.data_anomaly import island_exclude
from _02_strategy.island_strategy.single_island_strategy import (
    ENTRY_FILTERS, RESULT_DIR, run_config, variant_tag)

SUBDIR = "single_island_entry"
# 出場代號 → (持有根數, 主出場, 是否替換)；替換版不看持有根數（填 20 只為了 tag 一致）
EXITS = {"h20": (20, None, False), "trail10": (20, "trail10", True), "atr": (20, "atr", True),
         "gapdown": (20, "gapdown", True), "trail5": (20, "trail5", True),    # 出場掃描 clean 裸版單獨規則 PF 前五
         "h10": (10, None, False), "topisland": (20, "topisland", True)}      # 〈島型反轉（一）〉另外兩種出場
DEFAULT_EXITS = ("h20", "trail10", "atr", "gapdown", "trail5")


def build_jobs(exit_names=DEFAULT_EXITS) -> list:
    """回傳 (進場條件, 出場代號, 持有根數, 主出場, 替換, 可成交) 清單。"""
    return [(f, name, *EXITS[name], liq) for f in (None, *ENTRY_FILTERS)
            for name in exit_names for liq in (False, True)]


def _job(args):
    (entry, exit_name, hold, exit_rule, replace, liq), limit, out_root, extra = args
    t0 = time.time()
    res = run_config(common.DATA_DIR, hold, liq, limit=limit, out_root=out_root,
                     exit_rule=exit_rule, replace=replace, subdir=SUBDIR,
                     extra_exclude=extra, entry_filter=entry)
    ent = res["entries"]
    res.update({"entry": entry or "base", "exit_name": exit_name, "liq": liq,
                "未平倉率(%)": round(100 * res["open_at_end"] / ent, 2) if ent else 0.0,
                "秒": round(time.time() - t0)})
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="輸出根目錄（預設 result/；冒煙測試請指到暫存資料夾）")
    ap.add_argument("--only", default=None, help="只跑 tag 含此子字串的組（冒煙／補跑用）")
    ap.add_argument("--exits", nargs="+", default=list(DEFAULT_EXITS), choices=list(EXITS))
    ap.add_argument("--summary-name", default="_sweep_summary.csv")
    a = ap.parse_args()
    extra = island_exclude()
    print(f"島型母體排除 {len(extra)} 檔", flush=True)
    jobs = [(j, a.limit, a.out, extra) for j in build_jobs(a.exits)]
    if a.only:
        jobs = [j for j in jobs
                if a.only in variant_tag(j[0][2], j[0][5], None, j[0][3], j[0][4], j[0][0])]
    print(f"共 {len(jobs)} 組", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(_job, j) for j in jobs]
        for f in as_completed(futs):
            r = f.result()
            rows.append(r)
            print(f"{r['tag']}｜{r['交易次數']:,} 筆｜PF {r['獲利因子(PF)']}｜未平倉 {r['未平倉率(%)']}%｜{r['秒']} 秒",
                  flush=True)
    df = pd.DataFrame(rows)
    base = df[df["entry"] == "base"].set_index(["exit_name", "liq"])["signals"]
    key = pd.MultiIndex.from_arrays([df["exit_name"], df["liq"]])
    base_sig = base.reindex(key).to_numpy()
    df["生效率(%)"] = (100 * (1 - df["signals"] / base_sig)).round(2)
    out = os.path.join(a.out or RESULT_DIR, SUBDIR, a.summary_name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.sort_values("tag").to_csv(out, index=False, encoding="utf-8-sig")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
