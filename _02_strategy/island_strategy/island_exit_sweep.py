"""
底部島型出場優化全組合掃描（vbt 單股）：策略檔 single_island_strategy.py 的 #1～#13 全部跑一遍。

組合（每組 × 裸版／可成交 1,000 萬）：
  基準：持有 10／20／40 根（40 根是模式 3 測過的對照，順便重現 10／20 根錨點）。
  附加：持有 10／20 根 ＋ 停損 #1～#4 或主出場 #5～#13 其中一條，先觸發者算。
  附錄（替換）：拿掉持有 N 根，停損（無／#1～#4）× 主出場（#5～#10、#12、#13），
               含「只有停損」4 組（模式 3 的停損 × 移動停利組合也在這裡）。
剔除規則（用戶 2026-11-17：「賣不到的情況不可以超過總交易的10%」）：未平倉率＝期末仍持有 ÷ 進場數，
  同一條規則兩個口徑（裸版／可成交）任一 > 10% 就標記剔除（照雙底替換版 2026-11-03 的判準）；剔除的仍完整記錄。

每組一個進程（彼此獨立），結果寫 result/single_island_exit/<tag>/，彙總寫 result/single_island_exit/_sweep_summary.csv。

--clean：排除島型母體外的股票（2026-11-17～11-22 是資料異常 346 檔；2026-11-23 起改用 data_anomaly.island_exclude()），
  輸出改到 result/single_island_exit_clean/。

執行：python _02_strategy/island_strategy/island_exit_sweep.py [--workers 10] [--limit N] [--out 目錄] [--only 子字串] [--clean]
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
    EXIT_RULES, RESULT_DIR, STOP_RULES, run_config, variant_tag)

SUBDIR = "single_island_exit"
OPEN_MAX = 0.10          # 未平倉率上限（用戶 2026-11-17）


def build_jobs() -> list:
    """回傳 (hold, stop_rule, exit_rule, replace) 清單。"""
    jobs = [(h, None, None, False) for h in (10, 20, 40)]
    for h in (10, 20):
        jobs += [(h, s, None, False) for s in STOP_RULES]
        jobs += [(h, None, x, False) for x in EXIT_RULES]
    rep_main = [x for x in EXIT_RULES if x != "ma10switch"]
    jobs += [(20, s, None, True) for s in STOP_RULES]
    jobs += [(20, s, x, True) for s in (None, *STOP_RULES) for x in rep_main]
    return jobs


def _job(args):
    (hold, stop, exit_rule, replace), liq, limit, out_root, subdir, extra = args
    t0 = time.time()
    res = run_config(common.DATA_DIR, hold, liq, limit=limit, out_root=out_root,
                     stop_rule=stop, exit_rule=exit_rule, replace=replace, subdir=subdir,
                     extra_exclude=extra)
    res.update({"hold": "—" if replace else hold, "stop": stop or "none", "exit": exit_rule or "none",
                "mode": "替換" if replace else ("基準" if not (stop or exit_rule) else "附加"),
                "liq": liq, "秒": round(time.time() - t0)})
    ent = res["entries"]
    res["未平倉率(%)"] = round(100 * res["open_at_end"] / ent, 2) if ent else 0.0
    n_exit = res["exits"]
    res["停損生效率(%)"] = round(100 * res["exit_stop"] / n_exit, 2) if n_exit else 0.0
    res["主出場生效率(%)"] = round(100 * res["exit_main"] / n_exit, 2) if n_exit else 0.0
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="輸出根目錄（預設 result/；冒煙測試請指到暫存資料夾）")
    ap.add_argument("--only", default=None, help="只跑 tag 含此子字串的組（冒煙／補跑用）")
    ap.add_argument("--clean", action="store_true",
                    help="只跑島型母體（data_anomaly.island_exclude()），輸出到 single_island_exit_clean/")
    a = ap.parse_args()
    extra = island_exclude() if a.clean else frozenset()
    subdir = SUBDIR + ("_clean" if a.clean else "")
    if a.clean:
        print(f"島型母體排除 {len(extra)} 檔", flush=True)
    jobs = [(j, liq, a.limit, a.out, subdir, extra) for j in build_jobs() for liq in (False, True)]
    if a.only:
        jobs = [j for j in jobs if a.only in variant_tag(j[0][0], j[1], j[0][1], j[0][2], j[0][3])]
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
    key = df["mode"] + "|" + df["hold"].astype(str) + "|" + df["stop"] + "|" + df["exit"]
    worst = df.groupby(key)["未平倉率(%)"].transform("max")
    df["剔除(未平倉>10%)"] = worst > OPEN_MAX * 100
    out = os.path.join(a.out or RESULT_DIR, subdir, "_sweep_summary.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.sort_values("tag").to_csv(out, index=False, encoding="utf-8-sig")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
