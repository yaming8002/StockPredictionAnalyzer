"""
KD 交叉（六）（七）：多股共用資金 × 6 交易策略 × 兩種投法 × 四種買入排序（多股回測段）
=====================================================================================
把「PF 前 6 進場 × 高檔死叉」（`multi_kd.MultiKD`）搬進單一本金、多檔共用現金的組合回測。
引擎用 `_03_multi_strategy/base/fast_multi.py`（與 vbt 多股引擎逐筆對帳一致，KD 版見
verify_kd_fast.py），因為「隨機買入順序 × 1,000 次」用 vbt 版要跑好幾天。

**份數**（資金防線 floor=0.80；兩投法公式不同、份數絕不可共用）：
  S＝單股蒙地卡羅「連敗 P95」，讀分析段 `_04_analysis/kd/kd_montecarlo.py` 的輸出 CSV。
  定額（fixed）       ：份數＝round(S/0.2)，每筆＝100 萬／份數（不複利）。
  比例（percent_floor）：份數＝round(1/(1−0.8^(1/S)))，每筆＝已實現權益／份數，**下限 1 萬**。
    下限取 1 萬（不是 100 萬／份數）：照 2026-07-31 reference「比例＝每筆＝已實現權益×(1/份數)
    （percent_floor，下限 1 萬）」，與 MACD 多股的 PCT_MIN_INVEST 同口徑。
    下限也是擋單門檻（現金 < 下限就不買），取 100 萬／份數會讓後期現金不足 1.5 萬就停買，
    行為與文章（七）的數字不同。

**排序**：低價／流動性（5 日均量×收盤）／高價 三種固定排序 ＋ 隨機排序 × N 次
（每次每天每檔抽一個 [0,1) 均勻亂數當優先序，種子 SEED0＋第 k 次；亂數無平手）。
隨機的代表列＝**N 次裡「最終權益」中位數那一次的完整列**——文章（六）（七）表註
「隨機列＝1000 次隨機順序取中位那次」、07-31 reference「代表列＝1000 次中最終權益中位那次
的完整規格」；另附資金倍數中位（P5／P95）、「已實現權益曾跌破本金 8 成」次數比例。
同時另存一份**逐欄中位數**（MACD 多股 macd_multi_random 的做法）供對照，不進文章表。

交易區間＝標準區間 2002～2025，指標吃全史暖身（build_panel 先算再切）。
每個交易策略跑完就落地（fixed_<進場>.csv、random_<進場>.csv），中斷後重跑會跳過已完成的。

輸出（result/ 不進版控；--out 可改，冒煙用）：_02_strategy/kd_strategy/result/kd_multi/
  units.csv               份數表（交易策略｜S｜定額份數｜定額每筆｜比例份數｜比例每筆）
  fixed_<進場>.csv / random_<進場>.csv / random_colmed_<進場>.csv   逐策略（可續跑）
  orderings.csv           全部固定排序列
  random_median_run.csv   全部隨機代表列（最終權益中位那次）
  random_colmedian.csv    全部隨機逐欄中位
  kd_multi_result.csv     文章表：固定排序＋隨機代表列，依投法分段、策略區塊依低價總獲利排、區塊內依總獲利排
  各列另附「<欄名>_精確」＝未四捨五入值（exact_stats；隨機代表列的資金倍數中位／P5／P95 也有），
  文章出表從精確值一次四捨五入，避免存檔兩位小數再取一位的兩次進位。
執行：
  python _03_multi_strategy/kd/kd_multi_driver.py [--runs 1000] [--workers 6]
        [--entries breakout250 ...] [--limit N --out <暫存目錄> --mc-csv <MC 表>]
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
from _02_strategy.base.vbt.common import DEFAULT_END, DEFAULT_START, GLITCH  # noqa: E402
from _02_strategy.kd_strategy.kd_variants import NAME_ENTRY  # noqa: E402
from _03_multi_strategy.base.fast_multi import run_panel_fast  # noqa: E402
from _03_multi_strategy.kd.multi_kd import ANCHORS, ENTRIES, MultiKD  # noqa: E402

INIT_CASH = 1_000_000.0
FLOOR = 0.80                   # 資金防線：連敗 S 筆後權益仍 ≥ 8 成
PCT_MIN_INVEST = 10_000.0      # 比例投入的每筆下限（07-31 reference；同 MACD）
SEED0 = 20261009               # 隨機排序的種子起點；固定住才能重現同一批抽樣
WANT = ["open", "high", "low", "close", "volume"]
OUT = common.result_dir("kd_strategy", "kd_multi")
# 單股蒙地卡羅表（_04_analysis/kd/kd_montecarlo.py 的輸出）。回測層不往上 import 分析層，
# 所以這裡自己拼路徑；兩邊都用 result_dir("kd_strategy", "kd_mc")/kd_montecarlo.csv。
MC_CSV = os.path.join(common.result_dir("kd_strategy", "kd_mc"), "kd_montecarlo.csv")
ORDERS = ("低價", "流動性", "高價")

_DATA = None


def entry_name(e: str) -> str:
    """表格用的交易策略名（皆 × 高檔死叉）。"""
    return "黃金交叉" if e == "golden" else NAME_ENTRY[e]


def units(s: int):
    """S → (定額份數, 比例份數)。兩投法公式不同、份數不可共用。"""
    n_fixed = int(round(s / (1.0 - FLOOR)))                    # 線性
    n_pct = int(round(1.0 / (1.0 - FLOOR ** (1.0 / s))))       # 幾何
    if n_fixed == n_pct:
        raise AssertionError(f"S={s} 兩投法份數相同，公式寫錯了")
    return n_fixed, n_pct


def sizings(s: int) -> list:
    """回傳 [(投法, sizing_mode, invest_ratio, min_invest, 份數)]。"""
    n_fixed, n_pct = units(s)
    return [("定額", "fixed", 1.0, INIT_CASH / n_fixed, n_fixed),
            ("比例", "percent_floor", 1.0 / n_pct, PCT_MIN_INVEST, n_pct)]


def read_s_table(path: str) -> dict:
    """MC 表 → {進場代號: S}（只取 × 高檔死叉 的列）。"""
    if not os.path.isfile(path):
        raise SystemExit(f"找不到 {path}\n先跑：python _04_analysis/kd/kd_montecarlo.py")
    mc = pd.read_csv(path, encoding="utf-8-sig")
    mc = mc[mc["出場"] == "high_death"]
    return dict(zip(mc["進場"], mc["連敗 P95"].astype(int)))


def load_all(folder: str, limit=None) -> dict:
    """讀全市場全史（指標暖身），只篩掉標準區間內不到 2 根的檔。"""
    data = common.load_market(folder, columns=WANT, limit=limit, exclude=GLITCH, min_rows=2)
    return {sid: df for sid, df in data.items()
            if len(df.loc[DEFAULT_START:DEFAULT_END]) >= 2}


def _init(folder: str, limit):
    global _DATA
    _DATA = load_all(folder, limit)


def prio_panels(data: dict, close: pd.DataFrame) -> dict:
    """低價／流動性／高價三種優先序，直接由原始資料對齊出來，不必重掃訊號。"""
    turn = pd.DataFrame({sid: data[sid]["volume"].rolling(5).mean() * data[sid]["close"]
                         for sid in close.columns}).reindex(index=close.index, columns=close.columns)
    c = close.to_numpy(np.float64)
    return {"低價": -c, "流動性": turn.to_numpy(np.float64), "高價": c}


EXACT = "_精確"    # 未四捨五入欄的後綴（文章出表要從精確值一次進位，見 exact_stats）


def exact_stats(trades: pd.DataFrame) -> dict:
    """
    規格 9 欄的**未四捨五入**值（定義同 common.summarize_trades：排除淨損益 0、報酬率用毛報酬）。
    summarize_trades 存兩位小數，文章再取一位時若剛好落在 .x5 會變成兩次進位、方向可能錯，
    所以另存精確值給出表端用；不改 summarize_trades 本身（其他系列共用）。
    """
    t = trades[pd.to_numeric(trades["real_pnl"], errors="coerce").fillna(0.0) != 0]
    n = len(t)
    if n == 0:
        return {}
    pnl = t["real_pnl"].astype(float)
    rate = (t["sell_price"] - t["buy_price"]) / t["buy_price"] * 100
    days = (pd.to_datetime(t["sell_date"]) - pd.to_datetime(t["buy_date"])).dt.days
    win, lose = pnl > 0, pnl < 0
    loss_sum = abs(float(pnl[lose].sum()))
    vals = {"勝率%": win.sum() / n * 100, "平均持有天": float(days.mean()),
            "獲利平均%": float(rate[win].mean()) if win.any() else 0.0,
            "虧損平均%": float(rate[lose].mean()) if lose.any() else 0.0,
            "中位數%": float(rate.median()), "期望值/筆": float(pnl.sum()) / n,
            "獲利因子": float(pnl[win].sum()) / loss_sum if loss_sum > 0 else float("inf"),
            "總獲利(萬)": float(pnl.sum()) / 10_000}
    return {k + EXACT: v for k, v in vals.items()}


def row_of(e: str, mname: str, n_units: int, per: float, order: str, res: dict) -> dict:
    """規格 9 欄 ＋ 擋單／份數／已實現權益最低／最終權益／資金倍數／最大回撤 ＋ 各欄精確值。"""
    s = res["summary"]
    return common.spec_row(
        s, 交易策略=entry_name(e), 進場=e, 投法=mname, 份數=n_units, 每筆=round(per),
        排序=order, 擋單=res["blocked_orders"]) | {
        "已實現權益最低%": s["已實現權益最低(%)"],
        "本金曾<80%": s["已實現權益最低(%)"] < FLOOR * 100,
        "最終權益": s["最終權益"], "資金倍數": round(s["最終權益"] / INIT_CASH, 2),
        "最大回撤%": s["最大回撤(%)"]} | exact_stats(res["trades"])


def random_summary(rows: list) -> tuple:
    """
    N 次隨機 → (代表列, 逐欄中位列)。
    代表列＝最終權益中位那一次的完整列（N 為偶數時取排序後第 N//2 名，偏上中位，同均線交叉 driver），
    加掛資金倍數中位／P5／P95、本金曾<80% 的比例、擋單中位。
    """
    df = pd.DataFrame(rows)
    eq = df["最終權益"].to_numpy()
    rep = df.iloc[int(np.argsort(eq, kind="mergesort")[len(eq) // 2])].to_dict()
    mult = eq / INIT_CASH
    rep.update({"排序": "隨機", "隨機次數": len(df),
                "資金倍數_中位": round(float(np.median(mult)), 2),
                "資金倍數_P5": round(float(np.percentile(mult, 5)), 2),
                "資金倍數_P95": round(float(np.percentile(mult, 95)), 2),
                "資金倍數_中位" + EXACT: float(np.median(mult)),
                "資金倍數_P5" + EXACT: float(np.percentile(mult, 5)),
                "資金倍數_P95" + EXACT: float(np.percentile(mult, 95)),
                "本金曾<80%_比例%": round(float(df["本金曾<80%"].mean() * 100), 1),
                "擋單_中位": int(df["擋單"].median())})
    fixed_cols = {"份數", "每筆"}                      # 標籤欄（每次都一樣），不取中位
    med = {c: (round(float(df[c].median()), 4)
               if df[c].dtype.kind in "fi" and c not in fixed_cols else df[c].iloc[0])
           for c in df.columns if not c.endswith(EXACT)}     # 精確欄只給代表列用
    med.update({"排序": "隨機(逐欄中位)", "隨機次數": len(df),
                "總獲利(萬)P5": round(float(df["總獲利(萬)"].quantile(0.05)), 1),
                "總獲利(萬)P95": round(float(df["總獲利(萬)"].quantile(0.95)), 1)})
    return rep, med


def run_entry(e: str, s: int, runs: int, out_dir: str) -> str:
    """一個交易策略：面板建一次，兩投法 × 三固定排序 ＋ 隨機 × runs，跑完落地。"""
    t0 = time.time()
    m = MultiKD()
    m.ENTRY, m.PRIO = e, "low_price"
    m.initial_cash = INIT_CASH
    panel = m.build_panel(_DATA, DEFAULT_START, DEFAULT_END)
    prios = prio_panels(_DATA, panel["close"])
    det, rnd, colmed = [], [], []
    buf = np.empty(panel["price"].shape, dtype=np.float64)   # 隨機優先序緩衝區重用（全市場一份 ~100 MB）
    for mname, mode, ratio, floor, n_units in sizings(s):
        m.sizing_mode, m.invest_ratio, m.min_invest = mode, ratio, floor
        for order in ORDERS:
            det.append(row_of(e, mname, n_units, floor, order,
                              run_panel_fast(m, panel, prio=prios[order])))
        if runs <= 0:
            continue
        rows = []
        for k in range(runs):
            np.random.default_rng(SEED0 + k).random(out=buf)
            rows.append(row_of(e, mname, n_units, floor, "隨機", run_panel_fast(m, panel, prio=buf)))
        rep, med = random_summary(rows)
        rnd.append(rep)
        colmed.append(med)
    pd.DataFrame(det).to_csv(os.path.join(out_dir, f"fixed_{e}.csv"), index=False, encoding="utf-8-sig")
    if rnd:
        pd.DataFrame(rnd).to_csv(os.path.join(out_dir, f"random_{e}.csv"), index=False,
                                 encoding="utf-8-sig")
        pd.DataFrame(colmed).to_csv(os.path.join(out_dir, f"random_colmed_{e}.csv"), index=False,
                                    encoding="utf-8-sig")
    return f"{entry_name(e)}｜S={s}｜{len(det)} 固定＋{len(rnd)} 隨機｜{time.time() - t0:.0f} 秒"


def article_table(det: pd.DataFrame, rnd: pd.DataFrame) -> pd.DataFrame:
    """
    文章（六）（七）表的列序：依投法分段；策略區塊依「低價」總獲利高到低、區塊內依總獲利高到低。
    """
    both = pd.concat([det, rnd], ignore_index=True) if len(rnd) else det.copy()
    out = []
    for mname in ("定額", "比例"):
        part = both[both["投法"] == mname]
        low = part[part["排序"] == "低價"].set_index("進場")["總獲利(萬)"]
        for e in low.sort_values(ascending=False, kind="mergesort").index:
            out.append(part[part["進場"] == e].sort_values("總獲利(萬)", ascending=False,
                                                          kind="mergesort"))
    return pd.concat(out, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="KD 多股：6 策略 × 兩投法 × 四排序")
    ap.add_argument("--runs", type=int, default=1000, help="隨機排序次數（0＝不跑隨機）")
    ap.add_argument("--workers", type=int, default=6, help="平行進程數（每進程各載一份全市場資料）")
    ap.add_argument("--entries", nargs="+", default=ENTRIES, choices=ENTRIES + ANCHORS)
    ap.add_argument("--folder", default=common.DATA_DIR)
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 檔（冒煙用，請搭配 --out）")
    ap.add_argument("--out", default=OUT, help="輸出目錄（冒煙請改暫存目錄）")
    ap.add_argument("--mc-csv", default=MC_CSV, help="單股蒙地卡羅表（讀 S）")
    a = ap.parse_args()

    s_table = read_s_table(a.mc_csv)
    missing = [e for e in a.entries if e not in s_table]
    if missing:
        raise SystemExit(f"MC 表缺這些進場 × 高檔死叉 的 S：{missing}（{a.mc_csv}）")
    os.makedirs(a.out, exist_ok=True)

    urows = []
    for e in a.entries:
        (_, _, _, fixed_per, n_fixed), (_, _, ratio, _, n_pct) = sizings(s_table[e])
        urows.append({"交易策略": entry_name(e), "進場": e, "S": s_table[e],
                      "定額份數": n_fixed, "定額每筆": round(fixed_per),
                      "比例份數": n_pct, "比例每筆": f"1/{n_pct}（下限 {PCT_MIN_INVEST:,.0f}）"})
    units_df = pd.DataFrame(urows)
    units_df.to_csv(os.path.join(a.out, "units.csv"), index=False, encoding="utf-8-sig")
    print(units_df.to_string(index=False))

    def done(e):
        ok = os.path.isfile(os.path.join(a.out, f"fixed_{e}.csv"))
        return ok and (a.runs <= 0 or os.path.isfile(os.path.join(a.out, f"random_{e}.csv")))

    todo = [e for e in a.entries if not done(e)]
    print(f"{len(todo)}/{len(a.entries)} 策略待跑｜隨機 {a.runs} 次｜{a.workers} 進程｜"
          f"{DEFAULT_START}~{DEFAULT_END}", flush=True)
    if todo:
        with ProcessPoolExecutor(max_workers=min(a.workers, len(todo)), initializer=_init,
                                 initargs=(a.folder, a.limit)) as pool:
            futs = [pool.submit(run_entry, e, s_table[e], a.runs, a.out) for e in todo]
            for f in futs:
                print(f.result(), flush=True)

    det = pd.concat([pd.read_csv(os.path.join(a.out, f"fixed_{e}.csv")) for e in a.entries])
    common.assert_spec_columns(det)
    det.to_csv(os.path.join(a.out, "orderings.csv"), index=False, encoding="utf-8-sig")
    rnd_files = [os.path.join(a.out, f"random_{e}.csv") for e in a.entries]
    rnd = pd.DataFrame()
    if all(os.path.isfile(p) for p in rnd_files):
        rnd = pd.concat([pd.read_csv(p) for p in rnd_files])
        rnd.to_csv(os.path.join(a.out, "random_median_run.csv"), index=False, encoding="utf-8-sig")
        pd.concat([pd.read_csv(os.path.join(a.out, f"random_colmed_{e}.csv")) for e in a.entries]).to_csv(
            os.path.join(a.out, "random_colmedian.csv"), index=False, encoding="utf-8-sig")
    table = article_table(det, rnd)
    table.to_csv(os.path.join(a.out, "kd_multi_result.csv"), index=False, encoding="utf-8-sig")
    show = ["交易策略", "投法", "份數", "排序", "交易次數", "擋單", "勝率%", "獲利因子",
            "總獲利(萬)", "資金倍數", "最大回撤%", "已實現權益最低%"]
    print(table[show].to_string(index=False))
    print(f"→ {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
