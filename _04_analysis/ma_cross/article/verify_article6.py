# -*- coding: utf-8 -*-
"""
驗〈均線交叉（六）：多股回測（下）〉（multi-ma-cross-ordering-dynamic，比例）：7 張表每一格＋正文統計句。

正文有不少句子是拿本篇（比例）跟（五）（定額）比，所以兩種投法都會算。
表格驗法見 verify_multi_tables.py，統計句的計算見 verify_multi_claims.py。

執行（BLOG_DIR 指向 blog 專案根目錄）：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python _04_analysis/ma_cross/article/verify_article6.py
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from article_common import TRADING_DAYS, Checker, PAIRS, fmt, half_up, multi, post_text  # noqa: E402
from verify_multi_claims import order_stats, single_ev, sizing_stats  # noqa: E402
from verify_multi_tables import verify_tables  # noqa: E402

SLUG = "multi-ma-cross-ordering-dynamic"
MODE = "比例"


def verify_claims(ck: Checker, text: str, df) -> None:
    z, zf = sizing_stats(df, MODE), sizing_stats(df, "定額")
    o, of = order_stats(df, MODE), order_stats(df, "定額")
    n_ruin, n_r90 = len(z["e_ruin"]), len(z["e_ruin90"])
    pct = lambda zz, p: fmt(zz["F"][p]["本金曾<80%_比例%"], 1)  # noqa: E731
    pct0 = lambda zz, p: pct(zz, p).replace(".0", "")  # noqa: E731
    # 摘要與破產率段
    ck.check(z["e_ruin"] == zf["e_ruin"], "等分踩破的組，比例與定額不是同一批")
    ck.check(z["f_ruin"] == zf["f_ruin"], "公式踩破的組，比例與定額不是同一批")
    ck.phrase(text, f"等分這邊的風險故事跟固定投入幾乎一樣（同樣 {n_ruin} 組踩破八成），公式這邊破產的仍是同樣 "
                    f"{len(z['f_ruin'])} 組，但破產率被推高（5/50、10/20 換成複利後從 {pct0(zf, '5/50')}%、"
                    f"{pct0(zf, '10/20')}% 升到 {pct0(z, '5/50')}%、{pct0(z, '10/20')}%）")
    ck.check(z["f_ruin"] == ["5/10", "5/50", "10/20", "10/50"], f"公式破產的組：{z['f_ruin']}")
    up = sum(z["med"](z["F"][p]) > zf["med"](zf["F"][p]) for p in PAIRS)
    ck.phrase(text, f"公式的報酬在多數組合（21 組裡 {up} 組）被放大了")
    ck.phrase(text, f"光公式的 5/50 就從先前那篇固定投入的 {fmt(zf['F']['5/50']['總獲利(萬)_中位'])} 萬，"
                    f"一路複利到 {fmt(z['F']['5/50']['總獲利(萬)_中位'])} 萬")
    ck.phrase(text, f"21 組裡等分贏 {len(z['e_hi'])} 組、公式贏 {21 - len(z['e_hi'])} 組")
    ck.phrase(text, f"等分 20 份有 {n_ruin} 組會在交易途中把本金打到剩不到**八成**——其中 {n_r90} 組的破產率高達九成以上")
    ck.phrase(text, f"另外 {len(z['e_zero'])} 組等分的破產率是 0，其中 {len(z['e_zero_hi'])} 組報酬也高於公式")
    ck.phrase(text, f"有 {len(z['f_ruin'])} 組破產（原因不一樣，下面會講），其餘 {21 - len(z['f_ruin'])} 組的破產率是 0")
    ck.phrase(text, f"5/50、10/20 在固定投入時破產率是 {pct0(zf, '5/50')}%、{pct0(zf, '10/20')}%，"
                    f"換成會複利的動態投入後變成 {pct0(z, '5/50')}%、{pct0(z, '10/20')}%（10/50 兩種投入都是 "
                    f"{pct0(z, '10/50')}%）")
    ck.check(pct0(z, "10/50") == pct0(zf, "10/50"), "10/50 兩種投入破產率不同")
    ck.phrase(text, f"（單股每筆期望 {fmt(single_ev('5/50'), 2, sign=True)}%、{fmt(single_ev('10/50'), 2, sign=True)}%、"
                    f"{fmt(single_ev('10/20'), 2, sign=True)}%），1,000 次裡卻分別有 {pct0(z, '5/50')}%、"
                    f"{pct0(z, '10/50')}%、{pct0(z, '10/20')}% 曾跌破八成；同樣這三組在固定投入時是 "
                    f"{pct0(zf, '5/50')}%、{pct0(zf, '10/50')}%、{pct0(zf, '10/20')}%")
    ck.check(z["F"]["5/10"]["本金曾<80%_比例%"] == 100 == z["E"]["5/10"]["本金曾<80%_比例%"], "5/10 不是 100% 破產")
    ck.phrase(text, f"一個乾淨的門檻是 **{fmt(TRADING_DAYS)} 筆**")
    ck.phrase(text, f"**低於 {fmt(TRADING_DAYS)} 的多半是長均線組合**")
    # 排序
    ck.check(o["colored"] > of["colored"], f"上色格數 比例 {o['colored']} 不多於 定額 {of['colored']}")
    big_f = max(of["spread"], key=of["spread"].get)
    ck.phrase(text, f"固定投入「四排序最多只差 {fmt(of['spread'][big_f])} 萬」")
    big = max(o["spread"], key=o["spread"].get)
    ck.check(big == "10/50", f"比例四排序差距最大的是 {big}")
    ck.check(130 < o["spread"]["10/50"] < 140, f"10/50 差距 {o['spread']['10/50']:.1f}（文：超過 130 萬）")
    sh = o["shown"]["10/50"]
    ck.check(max(o["tot"]["10/50"], key=o["tot"]["10/50"].get) == "低價"
             and min(o["tot"]["10/50"], key=o["tot"]["10/50"].get) == "高價", "10/50 最高／最低不是低價／高價")
    ck.phrase(text, f"（低價 {fmt(sh['低價'])} 萬 vs 高價 {fmt(sh['高價'])} 萬），5/50 也差 80 多萬")
    ck.check(80 <= o["spread"]["5/50"] < 90, f"5/50 差距 {o['spread']['5/50']:.1f}（文：80 多萬）")
    ck.phrase(text, f"如 5/50：{fmt(of['blk']['5/50'])} → {fmt(o['blk']['5/50'])}、10/50："
                    f"{fmt(of['blk']['10/50'])} → {fmt(o['blk']['10/50'])}")
    ck.phrase(text, f"（動態投入 {len(o['top']['低價'])} 組最高，固定投入 {len(of['top']['低價'])} 組）")
    ck.check(all(v < 0 for v in o["tot"]["5/10"].values()), "5/10 四排序不是全部虧損")
    # 重點整理
    ck.phrase(text, f"但有 {n_ruin} 組的**破產率（本金<80%）大於 0、其中 {n_r90} 組在九成以上**；公式（用最壞連敗反推）在 "
                    f"{21 - len(z['f_ruin'])}/21 組**守住八成底線**（固定投入同樣是 {21 - len(zf['f_ruin'])}/21）——"
                    f"但破產的那幾組換成複利後更難守住，5/50 由 {pct0(zf, '5/50')}% 升到 {pct0(z, '5/50')}%、"
                    f"10/20 由 {pct0(zf, '10/20')}% 升到 {pct0(z, '10/20')}%")
    ck.check(o["spread"]["10/50"] > 130, "四排序差距沒有超過 130 萬")


def main() -> int:
    ck = Checker("均線交叉（六）")
    text = post_text(SLUG)
    df = multi("angle_adx")
    verify_tables(ck, text, df, MODE)
    verify_claims(ck, text, df)
    return ck.done()


if __name__ == "__main__":
    raise SystemExit(main())
