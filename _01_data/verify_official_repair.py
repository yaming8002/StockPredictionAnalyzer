"""
驗證 official_repair.py：拿同一批官方原始資料，修出來的 OHLCV 要和「期望的修正後資料」逐檔一致。

用途：官方資料全部重抓約 43 小時；已經有一份抓好的快取與修正結果時，用這支離線比對，
確認修正規則（換整段／換月份＋換算基準／只留有成交日／起始月／長中斷／不納入清單）都重現得出來。

輸入：
  --orig-dir      修正前的股價資料夾
  --raw-dir       官方原始資料快取（official_download 的格式，一檔一個 CSV）
  --expected-dir  期望的修正後股價資料夾（比對 OHLCV；不在這個資料夾的股票＝期望被移除）
  --first-listed  上市櫃起始月的查詢結果 JSON（{代號: {"first": "YYYY-MM"|null, "note": ...} 或 null}）
  --end-month     還原價股票整段補抓的結束月份
輸出：終端印比對摘要；不一致的股票列在 <raw-dir>/verify_mismatch.csv

執行：
  python _01_data/verify_official_repair.py --orig-dir ... --raw-dir ... --expected-dir ... --first-listed ... --end-month 2026-11
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from official_repair import OHLCV, build_plan, price_adjusted, load_local, repair_one  # noqa: E402


def same_ohlcv(got: pd.DataFrame, exp: pd.DataFrame) -> str:
    """回傳不一致說明；一致回傳空字串。量四捨五入成整數股後比（寫檔時同樣處理）。"""
    exp = exp[OHLCV]
    if not got.index.equals(exp.index):
        return f"日期不同：修出 {len(got)} 列、期望 {len(exp)} 列"
    got = got.assign(volume=got["volume"].round())
    diff = ~np.isclose(got.to_numpy(dtype=float), exp.to_numpy(dtype=float), rtol=1e-9, atol=1e-6, equal_nan=True)
    if diff.any():
        r, c = np.argwhere(diff)[0]
        return f"{got.index[r]:%Y-%m-%d} {OHLCV[c]} 修出 {got.iat[r, c]} 期望 {exp.iat[r, c]}"
    return ""


def build_state(a) -> dict:
    """依修正前資料建清單、判斷還原價；起始月用給定的查詢結果。"""
    plan = build_plan(a.orig_dir)
    adjusted = {sid: price_adjusted(sid, load_local(sid, a.orig_dir), item["query"], a.raw_dir)
                for sid, item in plan.items()}
    with open(a.first_listed, encoding="utf-8") as f:
        first_listed = json.load(f)
    return {"plan": plan, "adjusted": adjusted, "first_listed": first_listed, "end_month": a.end_month}


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("--orig-dir", "--raw-dir", "--expected-dir", "--first-listed", "--end-month"):
        ap.add_argument(name, required=True)
    a = ap.parse_args()
    state = build_state(a)
    print(f"有問題的股票 {len(state['plan'])} 檔，其中還原價 {sum(state['adjusted'].values())} 檔", flush=True)
    rows, mismatch = [], []
    for sid, item in state["plan"].items():
        row, ohlcv = repair_one(sid, item, state, a.orig_dir, a.raw_dir)
        rows.append(row)
        exp_path = os.path.join(a.expected_dir, f"{sid}.parquet")
        if not os.path.exists(exp_path):
            msg = "" if len(ohlcv) == 0 else f"期望移除，修出 {len(ohlcv)} 列"
        else:
            msg = same_ohlcv(ohlcv, load_local(sid, a.expected_dir))
        if msg:
            mismatch.append({"stock_id": sid, "problem": msg})
    plan = pd.DataFrame(rows)
    plan.to_csv(os.path.join(a.raw_dir, "verify_repair_plan.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(mismatch, columns=["stock_id", "problem"]).to_csv(
        os.path.join(a.raw_dir, "verify_mismatch.csv"), index=False, encoding="utf-8-sig")
    print(f"OHLCV 一致 {len(plan) - len(mismatch)} 檔、不一致 {len(mismatch)} 檔")
    print("分類：", plan["category"].replace("", "可回測").value_counts().to_dict())
    print("備註：", plan["note"].str.split("，|（").str[0].replace("", "（無）").value_counts().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
