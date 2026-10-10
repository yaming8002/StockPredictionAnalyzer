# -*- coding: utf-8 -*-
"""
產生均線交叉（一）（二）（三）三篇單股回測表的 HTML 列（只排版，數字全由逐筆交易重算）。

  （一）黃金交叉：無門檻 baseline 主表（10 欄，正負上色）＋ 5 日均量>1,000 張表（9 欄，無中位數）
  （二）調整測試（上）：強 K／雙重確認／盤整 ADX<25，「基準→調整」表
  （三）調整測試（下）：多頭排列／夾角>20°／跌破短均出場，「基準→調整」表

「基準→調整」表的底色：勝率、獲利平均、中位數、期望值、獲利因子五欄，
調整後（未捨入真值）高於基準＝粉紅 up、低於＝淺綠 down；其餘欄不上色。

執行：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python \
        _04_analysis/ma_cross/article/build_single_tables.py --article 1|2|3
"""
import argparse
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from article_common import PAIRS, fmt, metrics  # noqa: E402

BASE = "liq1000"
# 篇 → [(段落名稱, variant)]；段落名稱只是印出來分段用
ADJUST = {
    2: [("強 K", "strongk_liq1000"), ("雙重確認", "confirm_liq1000"),
        ("盤整 ADX<25", "vol_adx25_liq1000")],
    3: [("多頭排列", "align_liq1000"), ("夾角>20度", "angle20_liq1000"),
        ("跌破短均線就賣", "exit_below_short_liq1000")],
}
PAINT = ("wr", "aw", "med", "ev", "pf")     # 有方向的五欄


def sign_cls(v) -> str:
    return "pos" if v > 0 else "neg"


def d1_row(pair: str, m: dict, with_median: bool) -> str:
    """（一）的表：正值紅（pos）、負值綠（neg）；勝率、持有天、獲利因子不上色。"""
    cells = [pair, fmt(m["n"]), fmt(m["wr"], 1), fmt(m["hold"], 0)]
    tds = [f"<td>{c}</td>" for c in cells]
    tds.append(f'<td class="{sign_cls(m["aw"])}">{fmt(m["aw"], 1)}</td>')
    tds.append(f'<td class="{sign_cls(m["al"])}">{fmt(m["al"], 1)}</td>')
    if with_median:
        tds.append(f'<td class="{sign_cls(m["med"])}">{fmt(m["med"], 1)}</td>')
    tds.append(f'<td class="{sign_cls(m["ev"])}">{fmt(m["ev"], 0)}</td>')
    tds.append(f"<td>{fmt(m['pf'], 2)}</td>")
    tds.append(f'<td class="{sign_cls(m["tot"])}">{fmt(m["tot"], 0)}</td>')
    return "<tr>" + "".join(tds) + "</tr>"


def arrow_row(pair: str, b: dict, a: dict) -> str:
    """「基準→調整」一列；期望值不加千分位（表格窄），交易次數加。"""
    spec = [("n", 0, True), ("wr", 1, False), ("hold", 0, False), ("aw", 1, False),
            ("al", 1, False), ("med", 1, False), ("ev", 0, False), ("pf", 2, False),
            ("tot", 0, True)]
    tds = [f"<td>{pair}</td>"]
    for key, nd, comma in spec:
        txt = f"{fmt(b[key], nd, comma)}→{fmt(a[key], nd, comma)}"
        cls = ""
        if key in PAINT:
            cls = ' class="up"' if a[key] > b[key] else ' class="down"'
        tds.append(f"<td{cls}>{txt}</td>")
    return "<tr>" + "".join(tds) + "</tr>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--article", type=int, choices=[1, 2, 3], required=True)
    a = ap.parse_args()
    if a.article == 1:
        print("=== （一）無門檻 baseline ===")
        print("\n".join(d1_row(p, metrics("baseline", p), True) for p in PAIRS))
        print("\n=== （一）5 日均量>1,000 張 ===")
        print("\n".join(d1_row(p, metrics(BASE, p), False) for p in PAIRS))
        return 0
    for name, variant in ADJUST[a.article]:
        print(f"=== （{'二' if a.article == 2 else '三'}）{name}（{variant}）===")
        print("\n".join(arrow_row(p, metrics(BASE, p), metrics(variant, p)) for p in PAIRS))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
