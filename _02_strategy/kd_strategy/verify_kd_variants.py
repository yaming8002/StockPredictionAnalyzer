# -*- coding: utf-8 -*-
"""
驗 kd_variants／kd_sweep 的參數化條件與教學主體逐筆一致。

為什麼要驗：single_kd_strategy.py 的優化是「註解切換」，kd_variants.py 把同一批條件改成
類別屬性選。兩份寫法只要有一個字走鐘（門檻 >= 寫成 >、欄位取錯），掃描表就會跟文章
裡貼的程式碼對不上，而且看數字看不出來。所以這裡直接拿策略檔原始碼來比：

檢查一：程式化切換策略檔的註解行（讀原始碼 → 取消指定那幾行的註解 → exec 成一份臨時模組），
        每個切換狀態（baseline、opt1、baseline_lot、baseline_amt、opt1_amt、opt3~opt9）跑抽樣股票，
        再用 kd_sweep 的同一條路徑（prepare 備欄一次 → KdVariantShared.run）跑對應變體，
        逐筆交易 DataFrame 必須完全相同。
檢查二：多股檔 multi_kd.py 的 6 個進場濾網（＋2 個錨點進場）與高檔死叉出場，和 KdVariant 在同一批股票上
        add_columns 後的布林訊號逐根相同。

交易區間一律標準區間 DEFAULT_START～DEFAULT_END，df 傳全史（指標暖身）。
日後動到 single_kd_strategy.py、multi_kd.py 或 kd_variants.py 的條件，前後都該重跑。

執行：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python -W ignore \
        _02_strategy/kd_strategy/verify_kd_variants.py [--n 40]
"""
import argparse
import glob
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd  # noqa: E402

from _02_strategy.base.vbt import common  # noqa: E402
from _02_strategy.base.vbt.common import DEFAULT_END, DEFAULT_START, GLITCH  # noqa: E402
from _02_strategy.kd_strategy import kd_sweep  # noqa: E402
from _03_multi_strategy.kd.multi_kd import ANCHORS as MULTI_ANCHORS  # noqa: E402
from _03_multi_strategy.kd.multi_kd import ENTRIES as MULTI_ENTRIES, MultiKD  # noqa: E402

STRATEGY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "single_kd_strategy.py")

# 切換狀態 → 要取消註解的行（以該行的註解標記辨識）
_OPT1 = ["# 優化 #1：低檔黃金交叉", "# 優化 #1：高檔死亡交叉"]
TOGGLES = {
    "baseline": [],
    "opt1": _OPT1,
    "baseline_lot": ["# 優化 #2a："],
    "baseline_amt": ["# 優化 #2b："],
    "opt1_amt": _OPT1 + ["# 優化 #2b："],
    "opt3": ["# 優化 #3："],
    "opt4": ["# 優化 #4："],
    "opt5": ["# 優化 #5："],
    "opt6": ["# 出場優化 #6："],
    "opt7": ["# 出場優化 #7："],
    "opt8": ["# 出場優化 #8："],
    "opt9": ["# 出場優化 #9："],
}
# 切換狀態 → kd_sweep 的對應變體
EQUIV = {
    "baseline": "raw__golden__death",
    "opt1": "raw__low_zone__high_death",
    "baseline_lot": "lot__golden__death",
    "baseline_amt": "amt__golden__death",
    "opt1_amt": "amt__low_zone__high_death",
    "opt3": "raw__cmf_pos__death",
    "opt4": "raw__divergence__death",
    "opt5": "raw__ma120_gt_ma200__death",
    "opt6": "raw__golden__high_death",
    "opt7": "raw__golden__k_down50",
    "opt8": "raw__golden__climax",
    "opt9": "raw__golden__top_div",
}


def toggled_class(markers: list):
    """讀策略檔原始碼、取消指定註解行，exec 成臨時模組後回傳其 SingleKDStrategy。"""
    with open(STRATEGY_FILE, encoding="utf-8") as fh:
        lines = fh.read().splitlines(keepends=True)
    for mk in markers:
        hits = [i for i, ln in enumerate(lines)
                if mk in ln and ln.lstrip().startswith("# signal")]
        if len(hits) != 1:
            raise RuntimeError(f"註解標記 {mk!r} 應恰好對到 1 行，實際 {len(hits)} 行")
        i = hits[0]
        lines[i] = lines[i].replace("# signal", "signal", 1)
    ns = {"__name__": "single_kd_toggled", "__file__": STRATEGY_FILE}
    exec(compile("".join(lines), STRATEGY_FILE, "exec"), ns)
    return ns["SingleKDStrategy"]


def sample_stocks(n: int) -> list:
    """全市場等距抽 n 檔（排除 GLITCH），避免只抽到檔名排序最前面的 ETF。"""
    paths = sorted(glob.glob(os.path.join(common.DATA_DIR, "*.parquet")))
    sids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    sids = [s for s in sids if s not in GLITCH]
    step = max(1, len(sids) // n)
    return sids[::step][:n]


def load(sid: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(common.DATA_DIR, f"{sid}.parquet")).sort_index()


def check_toggles(data: dict) -> bool:
    print("── 檢查一：策略檔註解切換 vs kd_sweep 變體（逐筆交易）──")
    ok_all = True
    for state, markers in TOGGLES.items():
        cls = toggled_class(markers)
        v = kd_sweep.make_variant(EQUIV[state])
        n_trades, bad = 0, []
        for sid, (raw, prepared) in data.items():
            a = cls().run(raw, sid, start=DEFAULT_START, end=DEFAULT_END)["trades"]
            b = v.run(prepared, sid, start=DEFAULT_START, end=DEFAULT_END)["trades"]
            try:
                pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
            except AssertionError as exc:
                bad.append((sid, len(a), len(b), str(exc).splitlines()[0]))
            n_trades += len(a)
        status = "OK" if not bad else f"不一致 {len(bad)} 檔"
        print(f"  {state:<13}↔ {EQUIV[state]:<30} 交易 {n_trades:>6} 筆｜{status}")
        for row in bad[:5]:
            print(f"      {row}")
        ok_all &= not bad
    return ok_all


def check_multi(data: dict) -> bool:
    print("── 檢查二：multi_kd 進場濾網／出場 vs KdVariant（布林訊號逐根）──")
    ok_all = True
    for e in MULTI_ENTRIES + MULTI_ANCHORS:
        m = MultiKD()
        m.ENTRY = e
        # 錨點 golden ＝ KdVariant 的純黃金交叉（ENTRY=()）
        v = kd_sweep.make_variant(kd_sweep._v("amt", () if e == "golden" else e, "high_death"))
        n_buy, n_sell, bad = 0, 0, []
        for sid, (raw, prepared) in data.items():
            dm = m.add_columns(raw.copy())
            mb = m.buy_signal(dm).fillna(False).astype(bool)
            ms = m.sell_signal(dm).fillna(False).astype(bool)
            vb, vs = v.buy_signal(prepared), v.sell_signal(prepared)
            if not (mb.equals(vb) and ms.equals(vs)):
                bad.append((sid, int(mb.sum()), int(vb.sum()), int(ms.sum()), int(vs.sum())))
            n_buy += int(mb.sum())
            n_sell += int(ms.sum())
        status = "OK" if not bad else f"不一致 {len(bad)} 檔"
        print(f"  {e:<12} 買訊 {n_buy:>6}｜賣訊 {n_sell:>6}｜{status}")
        for row in bad[:5]:
            print(f"      {row}")
        ok_all &= not bad
    return ok_all


def main() -> int:
    ap = argparse.ArgumentParser(description="KD 變體 vs 策略檔註解切換／多股檔 一致性驗證")
    ap.add_argument("--n", type=int, default=40, help="抽樣股票數")
    a = ap.parse_args()

    data = {}
    for sid in sample_stocks(a.n):
        raw = load(sid)
        if len(raw.loc[DEFAULT_START:DEFAULT_END]) < 2:
            continue
        data[sid] = (raw, kd_sweep.prepare(raw))
    print(f"抽樣 {len(data)} 檔｜{DEFAULT_START}~{DEFAULT_END}")
    ok1 = check_toggles(data)
    ok2 = check_multi(data)
    print("全部一致" if ok1 and ok2 else "⚠️ 有不一致，見上方")
    return 0 if ok1 and ok2 else 1


if __name__ == "__main__":
    sys.exit(main())
