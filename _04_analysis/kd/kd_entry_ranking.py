"""
KD 交叉：進場 PF 排名重算（分析段，選「PF 前 6」用）
=====================================================
KD 系列（五）～（八）的六個進場（創250／創120／跳空／低檔＋紅K／創60／底背離）是從
2026-07-20 reference「進場端·獨立條件」那張排名取前 6：**每個進場條件各自疊在黃金交叉上、
出場固定一般死叉（隔離出場的影響）、可成交門檻 1,000 萬，依獲利因子排序**。
那張表是 2001 起點、舊 scratchpad driver 跑的（driver 已遺失）；這裡改讀單股掃描段
`_02_strategy/kd_strategy/kd_sweep.py` 以標準區間 2002～2025 重產的彙總，照同一規則重排。

**這支只報告、不改選集**：新排名前 6 若與舊選集不同，要不要換由用戶決定
（換了會連動（五）矩陣、（六）（七）多股、（八）結論整條線）。

候選集合＝kd_sweep 的 d0720_entries（07-20 表列出的進場條件，含基準「純黃金交叉」）。
07-26 另做的 50 中線、交叉強度等條件不在 07-20 的候選表裡，另列「參考」區、不參與排名。

執行（先跑單股掃描段）：
    python _02_strategy/kd_strategy/kd_sweep.py --set entries
    python _04_analysis/kd/kd_entry_ranking.py
輸出：_02_strategy/kd_strategy/result/kd_ranking/kd_entry_ranking.csv ＋ 終端表。
"""
import argparse
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd  # noqa: E402

from _02_strategy.base.vbt import common  # noqa: E402
from _02_strategy.kd_strategy import kd_sweep  # noqa: E402
from _02_strategy.kd_strategy.kd_variants import NAME_ENTRY  # noqa: E402
from _02_strategy.kd_strategy.single_kd_strategy import RESULT_DIR  # noqa: E402

SWEEP_DIR = os.path.join(RESULT_DIR, "single_kd")
OUT = common.result_dir("kd_strategy", "kd_ranking")
_SRC = [SWEEP_DIR]               # 讀取來源（--sweep-dir 可改；用 list 免 global）
TOP_N = 6

# 舊選集（文章（五）～（八）現行的六個進場，依舊排名序）
OLD_TOP6 = ("breakout250", "breakout120", "gap", "low_redk", "breakout60", "divergence")
# 2026-07-20 reference 表上的舊 PF（2001 起點）；K 線品質三種只記了區間 0.768~0.773，個別值不明
OLD_PF = {
    "breakout250": 0.916, "breakout120": 0.870, "gap": 0.827, "low_redk": 0.822,
    "breakout60": 0.819, "divergence": 0.807, "low_zone": 0.802, "bull_align_5_20_60": 0.799,
    "breakout20": 0.797, "bull_align_20_60_120": 0.793, "above_ma20": 0.778, "obv_up_1d": 0.777,
    "ma120_gt_ma200": 0.775, "above_ma60": 0.758, "above_ma120": 0.760, "above_ma200": 0.760,
    "vol_above_ma5": 0.752, "vol_x2": 0.748, "vol_x1_5": 0.744, "confirm2": 0.739,
    "cmf_pos": 0.728, (): 0.763,
}


def _entry_of(name: str):
    """變體名 → 單一進場條件名（純黃金交叉回傳 ()）；多條件 AND 的變體回傳 None。"""
    entry = kd_sweep.VARIANTS[name]["entry"]
    if len(entry) == 0:
        return ()
    return entry[0] if len(entry) == 1 else None


def _load(name: str):
    """讀某變體的單股彙總；沒跑過回傳 None。"""
    path = os.path.join(_SRC[0], name, "single_kd_aggregate.csv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path, encoding="utf-8-sig").iloc[0]


def _row(name: str, agg, group: str) -> dict:
    entry = _entry_of(name)
    hd_name = kd_sweep._v("amt", entry, "high_death")          # 同進場 × 高檔死叉（參考）
    hd = _load(hd_name)
    old = OLD_PF.get(entry)
    pf = round(float(agg["獲利因子(PF)"]), 3)
    return {"分組": group, "進場": NAME_ENTRY.get(entry, "黃金交叉（基準）") if entry else "黃金交叉（基準）",
            "代號": "+".join(entry) if isinstance(entry, tuple) else entry,
            "變體": name, "交易次數": int(agg["交易次數"]), "勝率%": agg["勝率(%)"],
            "獲利因子": pf, "總獲利(萬)": round(float(agg["總獲利"]) / 10000, 1),
            "舊PF(07-20)": old, "PF差(新−舊)": round(pf - old, 3) if old is not None else None,
            "×高檔死叉 PF": round(float(hd["獲利因子(PF)"]), 3) if hd is not None else None}


def main() -> int:
    ap = argparse.ArgumentParser(description="KD 進場 PF 排名重算（× 死叉、可成交 1,000 萬）")
    ap.add_argument("--sweep-dir", default=None, help="單股掃描結果根目錄（冒煙可改暫存）")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.sweep_dir:
        _SRC[0] = a.sweep_dir

    cand = kd_sweep.SETS["d0720_entries"]
    # 07-20 表以外、同為「單一條件 × 死叉 × 可成交」的進場（07-26 等），只列參考
    extra = [n for n in kd_sweep.SETS["entries"]
             if n not in cand and n.startswith("amt__") and n.endswith("__death")
             and _entry_of(n) not in (None, ())]

    rows, missing = [], []
    for group, names in (("候選(07-20)", cand), ("參考(非候選)", extra)):
        for n in names:
            agg = _load(n)
            if agg is None:
                missing.append(n)
                continue
            rows.append(_row(n, agg, group))
    if missing:
        print(f"⚠️ 缺 {len(missing)} 個變體的彙總（先跑 kd_sweep.py --set entries）：{missing}")
    if not rows:
        raise SystemExit("沒有任何彙總可排")

    df = pd.DataFrame(rows)
    df = df.sort_values(["分組", "獲利因子"], ascending=[True, False], kind="mergesort")
    is_cand = (df["分組"] == "候選(07-20)") & (df["代號"] != "")
    df["新排名"] = None
    df.loc[is_cand, "新排名"] = range(1, int(is_cand.sum()) + 1)
    old_rank = {e: i + 1 for i, e in enumerate(
        sorted((e for e in OLD_PF if e != ()), key=lambda e: -OLD_PF[e]))}
    df["舊排名"] = df["代號"].map(lambda c: old_rank.get(c))

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "kd_entry_ranking.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(df.drop(columns=["變體"]).to_string(index=False))

    ranked = df[is_cand].reset_index(drop=True)
    new_top = list(ranked["代號"].head(TOP_N))
    print(f"\n新前 {TOP_N}：{new_top}")
    print(f"舊前 {TOP_N}：{list(OLD_TOP6)}")
    print(f"新進：{[e for e in new_top if e not in OLD_TOP6] or '無'}｜"
          f"掉出：{[e for e in OLD_TOP6 if e not in new_top] or '無'}｜"
          f"順序相同：{new_top == list(OLD_TOP6)}")
    if len(ranked) > TOP_N:
        cut = ranked.iloc[TOP_N - 1]["獲利因子"] - ranked.iloc[TOP_N]["獲利因子"]
        print(f"第 {TOP_N} 名與第 {TOP_N + 1} 名的 PF 差：{cut:.3f}"
              f"（{ranked.iloc[TOP_N - 1]['進場']} vs {ranked.iloc[TOP_N]['進場']}）")
    print(f"→ {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
