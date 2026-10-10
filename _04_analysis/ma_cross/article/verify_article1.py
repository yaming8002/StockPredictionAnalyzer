# -*- coding: utf-8 -*-
"""
驗〈均線交叉（一）：黃金交叉與死亡交叉〉（ma-cross-golden-cross）：兩張表每一格＋正文統計句。

  主表＝無門檻 baseline（result/ma_cross/baseline），10 欄，正值紅 pos／負值綠 neg。
  流動性表＝5 日均量>1,000 張（result/ma_cross/liq1000），9 欄（無中位數）。
  正文的每個數字、比較與計數都由逐筆交易重算，再確認文章裡真的出現那段字。

執行（BLOG_DIR 指向 blog 專案根目錄）：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python _04_analysis/ma_cross/article/verify_article1.py
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from article_common import (PAIRS, Checker, all_metrics, common, fmt, post_text,  # noqa: E402
                            section, tables, trades, verify_value_table)

SLUG = "ma-cross-golden-cross"


def tail_share(variant: str, pair: str) -> float:
    """尾部集中度：淨損益最大的前 1% 交易佔總淨獲利的比例（%）。"""
    p = trades(variant, pair)["real_pnl"]
    p = p[p != 0].sort_values(ascending=False)
    k = max(1, int(len(p) * 0.01))
    return float(p.iloc[:k].sum() / p.sum() * 100)


def verify_claims(ck: Checker, text: str) -> None:
    base, liq = all_metrics("baseline"), all_metrics("liq1000")
    b, lq = base["50/200"], liq["50/200"]

    ck.check(all(base[p]["stocks"] == 2258 for p in PAIRS) or
             max(base[p]["stocks"] for p in PAIRS) <= 2258, "股票數超過 2,258")
    ck.phrase(text, "全台股 2,258 檔")
    # 中位數 21 組全負、勝率全部不到五成、期望值只有 5/10 為負
    ck.check(all(base[p]["med"] < 0 for p in PAIRS), "baseline 中位數不是 21 組全負")
    ck.phrase(text, "「中位數%」這一欄 21 組全是負的")
    ck.check(all(base[p]["wr"] < 50 for p in PAIRS), "baseline 勝率有組 ≥ 五成")
    ck.phrase(text, "全部不到五成")
    neg_ev = [p for p in PAIRS if base[p]["ev"] < 0]
    ck.check(neg_ev == ["5/10"], f"baseline 期望值為負的組：{neg_ev}（文：幾乎全是正的，只有 5/10）")
    ck.phrase(text, f"每筆交易的期望值高達 **{fmt(base['60/200']['ev'], 0, sign=True)} 元**")
    # 單筆平均虧損最深的組
    deepest = min(PAIRS, key=lambda p: base[p]["al"])
    ck.check(deepest == "120/200", f"平均虧損最深的是 {deepest}")
    ck.phrase(text, f"一筆平均也才賠掉投入的 {fmt(-base['120/200']['al'], 1)}%")
    ck.phrase(text, f"平均一筆可以賺到 **{fmt(base['120/200']['aw'], 1, sign=True)}%**")
    # 5703.TWO：baseline 50/200 那筆抱 469 天、幾乎打平
    t = trades("baseline", "50/200")
    hold = (t["sell_date"] - t["buy_date"]).dt.days
    x = t[(t["stock_id"] == "5703.TWO") & (hold == 469)]
    ck.check(len(x) == 1 and abs(x["sell_price"].iloc[0] / x["buy_price"].iloc[0] - 1) < 0.01,
             "找不到 5703.TWO 抱 469 天、幾乎打平的那筆")
    ck.phrase(text, "抱了 469 天、幾乎原地踏步")
    # 加流動性門檻：筆數砍六到七成
    cut = {p: 1 - liq[p]["n"] / base[p]["n"] for p in PAIRS}
    ck.check(all(0.6 <= c < 0.75 for c in cut.values()),
             f"流動性砍掉比例超出六到七成：{ {p: round(c, 3) for p, c in cut.items()} }")
    ck.phrase(text, f"50/200 從 {fmt(b['n'])} 筆掉到 {fmt(lq['n'])} 筆")
    ck.phrase(text, f"120/200 從 {fmt(base['120/200']['n'])} 筆掉到 {fmt(liq['120/200']['n'])} 筆")
    m6b, m6l = base["60/200"], liq["60/200"]
    ck.phrase(text, f"獲利因子從 {fmt(m6b['pf'], 2)} 降到 {fmt(m6l['pf'], 2)}、"
                    f"總獲利從 {fmt(m6b['tot'])} 萬掉到 {fmt(m6l['tot'])} 萬")
    pos = [p for p in PAIRS if liq[p]["ev"] > 0 and liq[p]["pf"] > 1 and liq[p]["tot"] > 0]
    ck.check(len(pos) == 20, f"流動性版期望值／獲利因子／總獲利皆正的組數 {len(pos)}（文：大多仍是正的）")
    ck.check(all(liq[p]["wr"] < 50 for p in PAIRS), "流動性版勝率有組 ≥ 五成（文：沒過半）")
    in34 = sum(30 <= liq[p]["wr"] < 40 for p in PAIRS)
    ck.check(in34 > 10, f"流動性版勝率落在三到四成的組數 {in34}（文：多落在三到四成）")
    in_pf = sum(1.2 <= round(liq[p]["pf"], 2) <= 2.0 for p in PAIRS)
    ck.check(in_pf > 10, f"流動性版獲利因子 1.2～2.0 的組數 {in_pf}（文：多在 1.2～2.0）")
    # 50/200 baseline 那一段
    ck.phrase(text, f"勝率 {fmt(b['wr'], 1)}%、平均賺 {fmt(b['aw'], 1)}% / 平均賠 {fmt(-b['al'], 1)}%、"
                    f"獲利因子 {fmt(b['pf'], 2)}、每筆期望值 {fmt(b['ev'], 0, sign=True)} 元、"
                    f"平均抱 {fmt(b['hold'], 0)} 天")
    ck.phrase(text, f"50/200 的獲利因子就從 {fmt(b['pf'], 2)} 掉到 {fmt(lq['pf'], 2)}、"
                    f"總獲利從 {fmt(b['tot'])} 萬縮到 {fmt(lq['tot'])} 萬")
    ck.check(fmt(b["wr"], 0) == "36", f"50/200 勝率 {b['wr']:.2f}（文：三成六）")
    ck.phrase(text, "為什麼勝率三成六還能賺")
    ck.check(10 <= -b["al"] < 13 and 50 <= b["aw"] < 60, "50/200 平均賠／賺不是「一成出頭／五成多」")
    # 尾部集中度：每個短均線裡，長均線用 200 的那組前 1% 交易佔比最低
    for s in (5, 10, 20, 50, 60):
        # 總淨獲利為負（5/10）時佔比沒有意義，排除
        grp = [p for p in PAIRS if p.startswith(f"{s}/") and base[p]["tot"] > 0]
        low = min(grp, key=lambda p: tail_share("baseline", p))
        ck.check(low == f"{s}/200", f"短均 {s}：尾部集中度最低的是 {low}（文：長均線用 200 那群最不靠尾巴）")
    # 散戶那段（流動性版 50/200）
    ck.phrase(text, f"每筆平均要抱 **{fmt(lq['hold'], 0)} 天（含假日）≈ 8 個月**")
    ck.check(round(lq["hold"] / 30.44) == 8, "流動性版 50/200 持有天換算不是約 8 個月")
    ck.phrase(text, f"**中位數是 {fmt(lq['med'], 1)}%**——甚至比沒篩流動性時的 {fmt(b['med'], 1)}% 還更深")
    ck.check(lq["med"] < b["med"], "流動性版中位數沒有比 baseline 更深")
    # 2617.TW 那筆：baseline 50/200
    t2 = t[(t["stock_id"] == "2617.TW") & (t["buy_date"] == "2020-09-03")]
    if ck.check(len(t2) == 1, "找不到 2617.TW 2020-09-03 那筆"):
        r = t2.iloc[0]
        ret = (r["sell_price"] / r["buy_price"] - 1) * 100
        ck.phrase(text, f"{fmt(r['sell_price'], 2)} 元出場，**{fmt(ret, 0, sign=True)}%**")
        ck.phrase(text, f"抱了 {(r['sell_date'] - r['buy_date']).days} 天")
        ck.check(round(r["buy_price"]) == 17, f"2617.TW 進場價 {r['buy_price']}（文：17 元附近）")
        px = pd.read_parquet(os.path.join(common.DATA_DIR, "2617.TW.parquet"))
        hi = float(px.loc[r["buy_date"]:r["sell_date"], "high"].max())
        ck.check(round(hi) == 78, f"2617.TW 持有期間最高 {hi}（文：衝到 78 元）")
    # 價格 +0.3% 卻因費稅淨賠的那筆
    rate = (t["sell_price"] / t["buy_price"] - 1) * 100
    ck.check(bool(((rate.round(1) == 0.3) & (t["real_pnl"] < 0)).any()),
             "找不到價格 +0.3% 卻淨賠的交易")
    # 勝率排序：兩條均線都在 50 日以上的 6 組正好是勝率前 6 名
    six = ["50/60", "50/120", "50/200", "60/120", "60/200", "120/200"]
    top6 = sorted(PAIRS, key=lambda p: -base[p]["wr"])[:6]
    ck.check(set(top6) == set(six), f"baseline 勝率前 6 名：{top6}")
    lo, hi = min(base[p]["wr"] for p in six), max(base[p]["wr"] for p in six)
    ck.phrase(text, f"勝率是 21 組裡最高的前 6 名（{fmt(lo, 1)}%～{fmt(hi, 1)}%）")
    ck.check(all(base[p]["ev"] > 0 for p in six), "兩條都在 50 日以上的 6 組有期望值不為正")


def main() -> int:
    ck = Checker("均線交叉（一）")
    text = post_text(SLUG)
    tb = tables(section(text, "## 回測結果（21 組）", "## 客觀觀察"))
    ck.check(len(tb) == 1, f"主表段應有 1 張表，實際 {len(tb)}")
    verify_value_table(ck, tb[0], "baseline", "主表", color="sign")
    tl = tables(section(text, "## 補一道現實濾網：流動性", "## 重點整理"))
    ck.check(len(tl) == 1, f"流動性段應有 1 張表，實際 {len(tl)}")
    verify_value_table(ck, tl[0], "liq1000", "流動性表", color="sign")
    verify_claims(ck, text)
    return ck.done()


if __name__ == "__main__":
    raise SystemExit(main())
