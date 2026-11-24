"""
用證交所／櫃買中心官方資料，修正 yfinance 股價裡有問題的股票

流程（三步，前兩步要連網、第三步不用）：
  1. 找出有問題的股票與月份：掃 2002–2025，三種異常任一條超過門檻就列入
     （門檻唯一定義＝`_02_strategy/island_strategy/data_anomaly.py` 的 ANOMALY_RULES）：
       - 整段缺漏：相鄰兩筆資料隔了 > 20 天
       - 沒成交卻有價格變動：量 0 但收盤和前一天不同的天數 ≥ 20
       - 長期沒成交：連續量 0 > 60 列
     要查的月份＝有量 0 的月份＋缺漏橫跨的月份；上市股 2010 前證交所查不到，記成「查不到」。
  2. 下載（`download`）：
       - 逐月查上面那些月份。
       - 比對後發現本機是「還原價」的股票（同一天官方÷本機偏離 > 1% 的天數超過 5%），
         補抓整段歷史，之後整段換成官方原始價——只換一部分會在交界多出假跳空。
       - 上市櫃前（興櫃）沒有官方資料：二分搜尋官方第一個有成交的月份，它之前的資料刪掉。
  3. 修正（`plan` 只算、`apply` 寫檔）：
       - 還原價的股票：整段換成官方原始價（上市股 2010 前刪除），官方最後一天之後的本機資料接在後面。
       - 其他股票：只換查過的月份，並逐月把官方價換算成本機的價格基準（避免交界假跳空）；
         官方整月沒資料（未上市、興櫃、停牌）就刪掉本機該月。
       - 一律只留官方「有成交」的日子。
       - 中斷超過 90 天的（例如代號換給另一家公司），只留中斷之後的資料。
       - 修完仍異常、或 2002–2025 剩不到 250 列 → 列入「回測不納入」清單：
           資料不足／興櫃（官方至今查無）／無法驗證（異常落在 2010 前的上市股 → 刪掉 2010 前再判一次）；
         缺漏期間官方也沒有成交＝真的停止交易過（多半是減資換股），資料正確，照常回測。

官方價格是原始價：配股、減資、分割那天的價格跳動會原樣留著（少數情況，這裡不另外處理）。

執行：
  python _01_data/official_repair.py download --data-dir <股價資料夾>
  python _01_data/official_repair.py plan     --data-dir <股價資料夾>
  python _01_data/official_repair.py apply    --data-dir <股價資料夾> --out-dir <輸出資料夾>
輸出：result/official_raw/repair_plan.csv（逐檔修法與修前修後統計）、apply 時每檔一個 parquet（OHLCV＋指標）
"""
import argparse
import json
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_HERE, ".."))
for p in (_HERE, _project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from official_download import (RAW_DIR, TWSE_MIN_MONTH, download_months, download_transfer_months,  # noqa: E402
                               first_listed_month, load_official, month_has_data)
from stock_technical import add_all_indicators  # noqa: E402
from _02_strategy.island_strategy.data_anomaly import ANOMALY_RULES, longest_run  # noqa: E402

START, END = "2002-01-01", "2025-12-31"
OHLCV = ["open", "high", "low", "close", "volume"]
ADJUSTED_DEV, ADJUSTED_SHARE = 0.01, 0.05   # 官方÷本機偏離 > 1% 的天數 > 5% ＝本機是還原價
SCALE_TOL = 0.01                            # 只換部分月份時，官方÷本機偏離 > 1% 才換算
LONG_GAP_DAYS = 90                          # 減資換股停牌約 3～5 週，不可當成換公司
MIN_ROWS = 250                              # 不到約一年資料，指標（MA200）與回測都不成立


def load_local(sid: str, data_dir: str) -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(data_dir, f"{sid}.parquet"))
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def list_stocks(data_dir: str) -> list:
    return sorted(n[:-8] for n in os.listdir(data_dir)
                  if n.endswith(".parquet") and (n.endswith(".TW.parquet") or n.endswith(".TWO.parquet")))


def anomaly_stats(df: pd.DataFrame) -> dict:
    """2002–2025 區間的三項統計（和 data_anomaly.scan_one 同一套算法）。"""
    seg = df.loc[START:END]
    if len(seg) < 2:
        return {k: None for k in ANOMALY_RULES}
    zero = (seg["volume"] == 0).to_numpy()
    moved = zero & (seg["close"] != seg["close"].shift(1)).to_numpy()
    return {"max_gap_days": int(seg.index.to_series().diff().dt.days.max()),
            "zero_vol_moved": int(moved.sum()), "max_zero_run": longest_run(zero)}


def is_abnormal(stats: dict) -> bool:
    return any(stats[k] is not None and stats[k] > th for k, th in ANOMALY_RULES.items())


def anomaly_months(seg: pd.DataFrame) -> list:
    """要查的月份＝含量 0 的月份＋缺漏（間隔 > 20 天）橫跨的月份。"""
    months = set(seg.index[seg["volume"] == 0].to_period("M"))
    gaps = seg.index.to_series().diff().dt.days
    for day, gap in gaps[gaps > ANOMALY_RULES["max_gap_days"]].items():
        months |= set(pd.period_range((day - pd.Timedelta(days=int(gap))).to_period("M"), day.to_period("M")))
    return sorted(months)


def build_plan(data_dir: str) -> dict:
    """步驟 1：全市場找出有問題的股票，列出要查／查不到的月份。"""
    plan = {}
    for sid in list_stocks(data_dir):
        seg = load_local(sid, data_dir).loc[START:END]
        if len(seg) < 2 or not is_abnormal(anomaly_stats(seg)):
            continue
        months = anomaly_months(seg)
        too_early = [m for m in months if sid.endswith(".TW") and m < TWSE_MIN_MONTH]
        plan[sid] = {"query": [str(m) for m in months if m not in too_early],
                     "unverifiable": [str(m) for m in too_early]}
    return plan


def price_adjusted(sid: str, local: pd.DataFrame, months: list, raw_dir: str) -> bool:
    """本機是不是還原價：查過的月份裡，同一天官方÷本機偏離 > 1% 的天數超過 5%。

    只看本身市場的官方列（上櫃轉上市補查的櫃買列不算）：轉市場前後的價格基準可能不同，
    那幾個月由 to_local_scale 逐月換算即可，不必因此整段換掉。
    """
    off = load_official(sid, raw_dir, own_market_only=True)
    off = off[off.index.to_period("M").isin({pd.Period(m, "M") for m in months})]
    loc = local[local["close"] > 0]
    common = off.index.intersection(loc.index)
    if len(common) == 0:
        return False
    ratio = off.loc[common, "close"] / loc.loc[common, "close"]
    return bool(((ratio - 1).abs() > ADJUSTED_DEV).mean() > ADJUSTED_SHARE)


def full_months(sid: str, local: pd.DataFrame, end_month: pd.Period) -> list:
    """還原價股票要補抓的整段月份：本機 2002 起第一個月（上市股不早於 2010-01）到 end_month。"""
    first = local.loc[START:].index.min().to_period("M")
    if sid.endswith(".TW"):
        first = max(first, TWSE_MIN_MONTH)
    return [str(m) for m in pd.period_range(first, end_month)]


def months_with_data(sid: str, months: list, raw_dir: str) -> dict:
    """已查月份 → 官方是否有成交資料。"""
    raw = pd.read_csv(os.path.join(raw_dir, f"{sid}.csv"), parse_dates=["date"])
    with_data = set(raw[(raw["volume"] > 0) & raw["date"].notna()]["month"].astype(str))
    return {m: m in with_data for m in months}


def transfer_months(sid: str, months: list, raw_dir: str) -> list:
    """上市股在第一個有資料的月份之前、證交所查無資料的月份：可能是上櫃轉上市前，要改查櫃買。"""
    if not sid.endswith(".TW"):
        return []
    has = months_with_data(sid, months, raw_dir)
    data_months = sorted(m for m, ok in has.items() if ok)
    return [m for m, ok in has.items() if not ok and (not data_months or m < data_months[0])]


def probe_range(has: dict):
    """第一個有資料的月份之前有「官方沒資料」的月份，而且兩者不相鄰 → 回傳要二分搜尋的 (lo, hi)。"""
    data_months = sorted(m for m, ok in has.items() if ok)
    hi = data_months[0] if data_months else str(pd.Timestamp.now().to_period("M") + 1)
    empty_before = [m for m, ok in has.items() if not ok and m < hi]
    if not empty_before:
        return None
    lo = max(empty_before)
    return (lo, hi) if (pd.Period(hi, "M") - pd.Period(lo, "M")).n > 1 else None


def adjacent_first(has: dict):
    """最後一個沒資料的月份和第一個有資料的月份相鄰 → 第一個有資料的月份就是上市櫃起始月。"""
    data_months = sorted(m for m, ok in has.items() if ok)
    if data_months and any(not ok and m < data_months[0] for m, ok in has.items()):
        return data_months[0]
    return None


def probe_first_listed(sid: str, months: list, local: pd.DataFrame, raw_dir: str):
    """查上市櫃起始月（要連網）。回傳 {"first": 月份或 None, "note": 說明}；不需要搜尋回傳 None。

    先抽查「本機最早可查月份～lo」中間一個月：那個月官方有資料，代表 lo 只是上市後的短暫停牌，
    不可以把之前的歷史刪掉。
    """
    rng = probe_range(months_with_data(sid, months, raw_dir))
    if rng is None:
        return None   # 不需要搜尋：repair 時改用 adjacent_first 判斷
    lo, hi = pd.Period(rng[0], "M"), pd.Period(rng[1], "M")
    earliest = local.index.min().to_period("M")
    if sid.endswith(".TW"):
        earliest = max(earliest, TWSE_MIN_MONTH)
    if (lo - earliest).n >= 2:
        mid = earliest + (lo - earliest).n // 2
        if str(mid) not in months and month_has_data(sid, mid, raw_dir):
            return {"first": None, "note": f"{mid} 官方有資料 → {lo} 為上市後停牌，不刪前段"}
    first = first_listed_month(sid, lo, hi, raw_dir)
    future = first > pd.Timestamp.now().to_period("M")
    return {"first": str(first), "note": "至今官方查無資料（興櫃）" if future else f"官方第一個有資料的月份 {first}"}


def state_path(raw_dir: str) -> str:
    return os.path.join(raw_dir, "_repair_state.json")


def save_state(state: dict, raw_dir: str) -> None:
    os.makedirs(raw_dir, exist_ok=True)
    with open(state_path(raw_dir), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)


def load_state(raw_dir: str) -> dict:
    with open(state_path(raw_dir), encoding="utf-8") as f:
        return json.load(f)


def cmd_download(data_dir: str, raw_dir: str, end_month: pd.Period) -> int:
    """步驟 2：下載異常月份 → 判斷還原價、補抓整段 → 查上市櫃起始月；狀態存 _repair_state.json，可中斷續跑。"""
    plan = build_plan(data_dir)
    state = {"plan": plan, "adjusted": {}, "first_listed": {}}
    if os.path.exists(state_path(raw_dir)):
        old = load_state(raw_dir)
        state["adjusted"], state["first_listed"] = old.get("adjusted", {}), old.get("first_listed", {})
    print(f"有問題的股票 {len(plan)} 檔", flush=True)
    for i, (sid, item) in enumerate(plan.items(), 1):
        local = load_local(sid, data_dir)
        download_months(sid, item["query"], raw_dir)
        if sid not in state["adjusted"]:
            state["adjusted"][sid] = price_adjusted(sid, local, item["query"], raw_dir)
        transfer = transfer_months(sid, item["query"], raw_dir)
        if transfer:   # 判斷完還原價才補查櫃買（判斷只看本身市場）
            download_transfer_months(sid, transfer, raw_dir)
        if state["adjusted"][sid]:
            download_months(sid, full_months(sid, local, end_month), raw_dir)
        elif sid not in state["first_listed"]:
            state["first_listed"][sid] = probe_first_listed(sid, item["query"], local, raw_dir)
        save_state(state, raw_dir)
        print(f"[{i}/{len(plan)}] {sid} 完成（還原價：{state['adjusted'][sid]}）", flush=True)
    return 0


def to_local_scale(add: pd.DataFrame, loc: pd.DataFrame) -> pd.DataFrame:
    """只換部分月份時，逐月用「同日收盤 官方÷本機 的中位數」把官方價換回本機基準（量反向換算）。"""
    parts = []
    for _, off_m in add.groupby(add.index.to_period("M")):
        common = off_m.index.intersection(loc.index[loc["volume"] > 0])
        ratio = (off_m.loc[common, "close"] / loc.loc[common, "close"]).median() if len(common) else 1.0
        if abs(ratio - 1) > SCALE_TOL:
            off_m = off_m.copy()
            off_m[["open", "high", "low", "close"]] /= ratio
            off_m["volume"] *= ratio
        parts.append(off_m)
    return pd.concat(parts) if parts else add


def rebuild_ohlcv(sid: str, months: list, adjusted: bool, local: pd.DataFrame, first_listed,
                  raw_dir: str) -> pd.DataFrame:
    """還原價股票整段換；其他股票只換查過的月份。上市櫃起始月之前全部刪除。"""
    off = load_official(sid, raw_dir)
    loc = local[OHLCV]
    if adjusted:
        start = pd.Period(min(months), "M").start_time
        tail = loc[loc.index > off.index.max()] if len(off) else loc.iloc[0:0]
        out = pd.concat([off[off.index >= start], tail])
    else:
        periods = {pd.Period(m, "M") for m in months}
        keep = ~loc.index.to_period("M").isin(periods)
        add = to_local_scale(off[off.index.to_period("M").isin(periods)], loc)
        out = pd.concat([loc[keep], add])
    if first_listed:
        out = out[out.index >= pd.Period(first_listed, "M").start_time]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index.name = "date"
    return out


def cut_long_gap(df: pd.DataFrame) -> tuple:
    """最後一個 > 90 天的中斷之前全部刪除；回傳（新序列, 刪掉的起訖說明）。"""
    gaps = df.index.to_series().diff().dt.days
    long = gaps[gaps > LONG_GAP_DAYS]
    if long.empty:
        return df, ""
    cut_at = long.index[-1]
    return df[df.index >= cut_at], (f"{df.index.min():%Y-%m-%d}~{df.index[df.index < cut_at].max():%Y-%m-%d}"
                                     f"（間隔 {int(long.iloc[-1])} 天）")


def untestable_reason(df: pd.DataFrame, stats: dict, months: list, adjusted: bool) -> tuple:
    """回傳（類別, 說明）；類別空字串＝可回測。類別：資料不足／無法驗證／官方確認停牌。"""
    seg = df.loc[START:END]
    if len(seg) < MIN_ROWS:
        return "資料不足", f"修完 2002–2025 只剩 {len(seg)} 列（< {MIN_ROWS}）"
    bad = [f"{k}={stats[k]}>{th}" for k, th in ANOMALY_RULES.items() if stats[k] is not None and stats[k] > th]
    if not bad:
        return "", ""
    checked = set(months)
    gaps = seg.index.to_series().diff().dt.days
    unchecked = []
    for day, gap in gaps[gaps > ANOMALY_RULES["max_gap_days"]].items():
        span = pd.period_range((day - pd.Timedelta(days=int(gap))).to_period("M"), day.to_period("M"))
        if not adjusted and not all(str(m) in checked for m in span):
            unchecked.append(f"{day - pd.Timedelta(days=int(gap)):%Y-%m-%d}~{day:%Y-%m-%d}")
    zero_left = int((seg["volume"] == 0).sum())   # 查過的月份只留有成交的日子，剩下的量 0 都在查不到的月份
    if unchecked or zero_left:
        return "無法驗證", "、".join(bad) + f"；查不到的缺漏 {unchecked[:3]}、量 0 列 {zero_left}"
    return "官方確認停牌", "、".join(bad) + "（缺漏期間官方也沒有成交）"


def classify(sid: str, ohlcv: pd.DataFrame, months: list, adjusted: bool) -> tuple:
    """修完之後判定能不能回測；回傳（序列, 類別, 說明, 備註）。"""
    category, reason = untestable_reason(ohlcv, anomaly_stats(ohlcv), months, adjusted)
    note = ""
    if category == "無法驗證" and sid.endswith(".TW"):
        # 異常落在證交所查不到的 2010 前 → 刪掉 2010 前再判一次
        ohlcv = ohlcv[ohlcv.index >= TWSE_MIN_MONTH.start_time]
        note = f"2010 前刪除（原判無法驗證：{reason}）"
        category, reason = untestable_reason(ohlcv, anomaly_stats(ohlcv), months, adjusted)
    if category == "官方確認停牌":   # 停牌缺漏是真的，資料正確，照常回測
        note, category, reason = f"官方確認停牌，照常回測：{reason}", "", ""
    if ohlcv.empty:
        category, reason = "興櫃", "至今官方上市櫃資料查無（興櫃），整檔移除"
    return ohlcv, category, reason, note


def repair_one(sid: str, item: dict, state: dict, data_dir: str, raw_dir: str) -> tuple:
    """修一檔：回傳（repair_plan 一列, 修好的 OHLCV）。"""
    local = load_local(sid, data_dir)
    adjusted = state["adjusted"][sid]
    months = sorted(set(item["query"]) | (set(full_months(sid, local, end_month_of(state))) if adjusted else set()))
    entry = state["first_listed"].get(sid)
    if entry:   # 有查過起始月（含「上市後停牌、不刪前段」的 first=None）
        first = entry["first"]
    else:       # 還原價股票整段只用官方資料，自然從第一個有資料的月份開始
        first = None if adjusted else adjacent_first(months_with_data(sid, item["query"], raw_dir))
    ohlcv, cut_note = cut_long_gap(rebuild_ohlcv(sid, months, adjusted, local, first, raw_dir))
    ohlcv, category, reason, note = classify(sid, ohlcv, item["query"] if not adjusted else months, adjusted)
    row = {"stock_id": sid, "adjusted": adjusted, "local_rows": len(local), "new_rows": len(ohlcv),
           "new_first": f"{ohlcv.index.min():%Y-%m-%d}" if len(ohlcv) else "", "first_listed": first or "",
           "long_gap_cut": cut_note, "unverifiable_months": len(item["unverifiable"]),
           **{f"before_{k}": v for k, v in anomaly_stats(local).items()},
           **{f"after_{k}": v for k, v in anomaly_stats(ohlcv).items()},
           "category": category, "reason": reason, "note": note}
    return row, ohlcv


def end_month_of(state: dict) -> pd.Period:
    """還原價股票整段補抓的結束月份（download 當時的月份，存在狀態檔）。"""
    return pd.Period(state.get("end_month", str(pd.Timestamp.now().to_period("M"))), "M")


def write_parquet(path: str, ohlcv: pd.DataFrame) -> None:
    """OHLCV＋指標寫成 parquet；換算過的量會有小數，四捨五入成整數股。"""
    ohlcv = ohlcv.assign(volume=ohlcv["volume"].round().astype("int64"))
    out = add_all_indicators(ohlcv.copy())
    out.index.name = "date"
    out.to_parquet(path, index=True)


def cmd_repair(data_dir: str, raw_dir: str, out_dir: str = None) -> int:
    """步驟 3：plan（out_dir=None，只算）／apply（寫到 out_dir，原資料夾不動）。"""
    state = load_state(raw_dir)
    rows = []
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    for sid, item in state["plan"].items():
        row, ohlcv = repair_one(sid, item, state, data_dir, raw_dir)
        rows.append(row)
        if out_dir and len(ohlcv):
            write_parquet(os.path.join(out_dir, f"{sid}.parquet"), ohlcv)
    plan = pd.DataFrame(rows)
    path = os.path.join(raw_dir, "repair_plan.csv")
    plan.to_csv(path, index=False, encoding="utf-8-sig")
    counts = plan["category"].replace("", "可回測").value_counts().to_dict()
    print(f"{len(plan)} 檔（還原價整段換 {int(plan['adjusted'].sum())} 檔）→ {path}\n  {counts}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="用證交所／櫃買官方資料修正股價資料")
    ap.add_argument("cmd", choices=["download", "plan", "apply"])
    ap.add_argument("--data-dir", required=True, help="原本的股價資料夾（一檔一個 parquet）")
    ap.add_argument("--raw-dir", default=RAW_DIR, help="官方原始資料快取")
    ap.add_argument("--out-dir", help="apply 的輸出資料夾（不覆蓋原資料）")
    ap.add_argument("--end-month", default=str(pd.Timestamp.now().to_period("M")),
                    help="還原價股票整段補抓到哪個月（預設本月）")
    a = ap.parse_args()
    if a.cmd == "download":
        rc = cmd_download(a.data_dir, a.raw_dir, pd.Period(a.end_month, "M"))
        state = load_state(a.raw_dir)
        state["end_month"] = a.end_month
        save_state(state, a.raw_dir)
        return rc
    if a.cmd == "apply" and not a.out_dir:
        ap.error("apply 需要 --out-dir")
    return cmd_repair(a.data_dir, a.raw_dir, a.out_dir if a.cmd == "apply" else None)


if __name__ == "__main__":
    sys.exit(main())
