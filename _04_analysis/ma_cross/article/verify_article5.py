# -*- coding: utf-8 -*-
"""
驗〈均線交叉（五）：多股回測（上）〉（multi-ma-cross-ordering-fixed，定額）：7 張表每一格＋正文統計句。

資料＝_03_multi_strategy/ma_cross/ma_cross_multi_driver.py --task angle_adx 的輸出。
表格驗法見 verify_multi_tables.py，統計句的計算見 verify_multi_claims.py。

執行（BLOG_DIR 指向 blog 專案根目錄）：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python _04_analysis/ma_cross/article/verify_article5.py
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from article_common import TRADING_DAYS, Checker, common, fmt, mc, multi, post_text  # noqa: E402
from verify_multi_claims import order_stats, single_ev, sizing_stats  # noqa: E402
from verify_multi_tables import verify_tables  # noqa: E402

SLUG = "multi-ma-cross-ordering-fixed"
MODE = "定額"


def verify_claims(ck: Checker, text: str, df) -> None:
    z = sizing_stats(df, MODE)
    o = order_stats(df, MODE)
    n_hi, n_ruin, n_r90 = len(z["e_hi"]), len(z["e_ruin"]), len(z["e_ruin90"])
    f_ok = 21 - len(z["f_ruin"])
    ck.check(z["f_ruin"] == ["5/10", "5/50", "10/20", "10/50"], f"公式破產的組：{z['f_ruin']}（文：5/10、5/50、10/20、10/50）")
    ck.phrase(text, f"報酬多數更高（{n_hi}/21 組），但有 {n_ruin} 組中途曾跌破八成底線（其中 {n_r90} 組機率在九成以上）；"
                    f"公式讓出報酬，換到 {f_ok}/21 組守住底線")
    ck.phrase(text, f"在多數組合都贏公式（21 組裡 {n_hi} 組）")
    ck.phrase(text, f"5/120 就差到 {fmt(z['E']['5/120']['總獲利(萬)_中位'])} 萬對 {fmt(z['F']['5/120']['總獲利(萬)_中位'])} 萬")
    ck.phrase(text, f"等分 20 份有 {n_ruin} 組會在交易途中把本金打到剩不到**八成**——其中 {n_r90} 組的破產率高達九成以上")
    ck.phrase(text, f"另外 {len(z['e_zero'])} 組等分的破產率是 0（{'、'.join(z['e_zero'])}），其中 {len(z['e_zero_hi'])} 組報酬也高於公式")
    ck.phrase(text, f"只有 {len(z['f_ruin'])} 組破產（原因不一樣，下面會講），其餘 {f_ok} 組的破產率是 0")
    ck.check(9 <= n_ruin <= 12, f"等分破產組 {n_ruin}（文：一半左右的組合）")
    ck.check(min(single_ev(p) for p in mc().index if p != "5/10") > 0 and single_ev("5/10") < 0,
             "單股每筆期望：5/10 不是唯一為負")
    three = ("5/50", "10/50", "10/20")
    pct = lambda p: fmt(z["F"][p]["本金曾<80%_比例%"], 1).replace(".0", "")  # noqa: E731
    ev_txt = "、".join(f"{fmt(single_ev(p), 2, sign=True)}%" for p in three)
    pct_txt = "、".join(f"{pct(p)}%" for p in three)
    ck.phrase(text, f"這三組都不是期望為負（單股每筆期望 {ev_txt}），1,000 次裡卻分別有 {pct_txt} 曾跌破八成")
    lows = [z["F"][p]["已實現權益最低%"] for p in three]
    ck.phrase(text, f"取中位那次的低點約 {fmt(min(lows))}%～{fmt(max(lows))}%")
    ck.check(z["E"]["5/10"]["本金曾<80%_比例%"] == 100 and z["F"]["5/10"]["本金曾<80%_比例%"] == 100,
             "5/10 不是兩種分批都 100% 破產")
    # 10/50 的最低手續費段落：每筆金額低於「最低手續費÷費率」就按 20 元收
    fee_floor_amt = common.MIN_COMMISSION / common.COMMISSION
    per_1050 = z["F"]["10/50"]["每筆"]
    ck.phrase(text, f"公式分批每筆約 {fmt(per_1050)} 元，按 {common.COMMISSION * 100:g}% 的手續費率只要約 "
                    f"{fmt(per_1050 * common.COMMISSION, 1)} 元，但單筆最低收 {fmt(common.MIN_COMMISSION)} 元")
    ck.check(abs(common.MIN_COMMISSION / per_1050 * 100 - 0.3) < 0.05, "10/50 公式每筆的實際手續費率不是約 0.3%")
    ck.check(z["E"]["10/50"]["每筆"] == 50_000 and z["E"]["10/50"]["每筆"] > fee_floor_amt,
             "等分 20 份每筆不是 5 萬、或會碰到最低手續費")
    n_below = sum(z["F"][p]["每筆"] < fee_floor_amt for p in z["F"])
    ck.phrase(text, f"21 組裡有 {n_below} 組低於約 {fmt(fee_floor_amt / 10_000, 1)} 萬元")
    ck.check(z["E"]["10/50"]["本金曾<80%_比例%"] == 0 and z["F"]["10/50"]["本金曾<80%_比例%"] == 100,
             "10/50 不是等分 0%、公式 100%")
    # 交易量門檻＝交易日數
    ck.phrase(text, f"一個乾淨的門檻是 **{fmt(TRADING_DAYS)} 筆**")
    low = [p for p in mc().index if mc().loc[p, "交易數"] < TRADING_DAYS]
    ck.check(all(p in low for p in ["10/200", "20/200", "50/200", "60/200", "120/200"]),
             f"低於門檻的組：{low}")
    ck.phrase(text, f"**低於 {fmt(TRADING_DAYS)} 的多半是長均線組合**")
    # 排序
    same, top = o["same"], o["top"]
    ck.phrase(text, f"**21 組裡有 {len(same)} 組，四種排序一模一樣**")
    blk_same = [o["blk"][p] for p in same]
    ck.phrase(text, f"這些組合**擋單只有 {min(blk_same)}～{max(blk_same)} 筆**")
    ck.phrase(text, f"**其餘 {21 - len(same)} 組擋單較多**")
    big = max(o["spread"], key=o["spread"].get)
    ck.phrase(text, f"四種排序的總獲利最多只差 {fmt(o['spread'][big])} 萬（{big}）")
    ck.phrase(text, f"其中**低價優先在 {len(top['低價'])} 組最高**")
    ck.phrase(text, f"（{' 與 '.join(top['隨機'])} 是隨機最高、{' 與 '.join(top['高價'])} 是高價最高、"
                    f"{' 與 '.join(top['流動性'])} 是流動性最高）")
    sh = o["shown"]["50/60"]
    ck.phrase(text, f"50/60：低價 {fmt(sh['低價'])} 萬 vs 高價 {fmt(sh['高價'])} 萬")
    ck.check(all(v < 0 for v in o["tot"]["5/10"].values()), "5/10 四排序不是全部虧損")
    ck.phrase(text, f"四種排序有 {len(same)} 組完全一樣、其餘 {21 - len(same)} 組最多也只差 {fmt(o['spread'][big])} 萬")
    ck.phrase(text, f"低價在 {len(top['低價'])} 組最高")
    ck.phrase(text, f"報酬多數更高（{n_hi}/21），但有 {n_ruin} 組的**破產率（本金<80%）大於 0、其中 {n_r90} 組在九成以上**")
    ck.phrase(text, f"卻在 {f_ok}/21 組**守住八成底線**")
    ck.phrase(text, f"**5/50、10/50、10/20 期望為正**，1,000 次裡仍有 {pct_txt} 破底線")


def main() -> int:
    ck = Checker("均線交叉（五）")
    text = post_text(SLUG)
    df = multi("angle_adx")
    verify_tables(ck, text, df, MODE)
    verify_claims(ck, text, df)
    return ck.done()


if __name__ == "__main__":
    raise SystemExit(main())
