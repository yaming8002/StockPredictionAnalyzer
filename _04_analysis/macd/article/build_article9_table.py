# -*- coding: utf-8 -*-
"""
產第九篇第二張表的 24 列 HTML ＋ 正文要用的統計，避免手抄。

欄位來源分兩種，附註裡有交代：
- 「原本的出場」與「附加」兩欄＝引用（四）（七）（八）篇**已發佈**的數字，
  讀者翻回去看得到同樣的值。
- 「替換」與「未平倉」兩欄＝本篇同一批跑出來的（macd_exit_replace_all.csv）。

執行：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 F:/stock-analyzer/.venv/Scripts/python.exe \
        _04_analysis/macd/article/build_article9_table.py
"""

import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
from _02_strategy.base.vbt import common  # noqa: E402
import sys

import pandas as pd

CSV = os.path.join(common.require_blog_dir(), "reference", "macd", "data", "macd_exit_replace_all.csv")
# 母體 → (文章顯示名, 基準線獲利因子, 基準線持有天)；取自（四）篇已發佈數字
POPS = [("交叉", "交叉", 0.9867, 19), ("零軸", "零軸", 1.2387, 43),
        ("背離", "背離", 1.0289, 51)]
# (CSV 的出場名, 文章顯示名, 出處, 三母體的「附加」獲利因子)
RULES = [
    ("跌破MA200", "跌破年線", "七", {"交叉": 0.9769, "零軸": 1.2073, "背離": 0.9758}),
    ("Supertrend翻空", "超級趨勢", "七", {"交叉": 0.9850, "零軸": 1.1606, "背離": 1.0105}),
    ("SAR翻空", "拋物線 SAR", "七", {"交叉": 0.9379, "零軸": 0.9645, "背離": 0.9507}),
    ("頂頂低", "波段高點走低", "七", {"交叉": 0.9804, "零軸": 1.1634, "背離": 0.9380}),
    ("跌破20日低", "跌破二十日低", "七", {"交叉": 0.9906, "零軸": 1.2029, "背離": 0.8650}),
    ("吊燈3ATR", "吊燈 3×ATR", "八", {"交叉": 0.9740, "零軸": 1.1117, "背離": 0.8934}),
    ("自最高點回落10%", "自最高點回落 10%", "八",
     {"交叉": 0.9515, "零軸": 1.1096, "背離": 0.8628}),
    ("抱滿60天", "抱滿 60 天", "八", {"交叉": 0.9864, "零軸": 1.1892, "背離": 0.9970}),
]


def main():
    d = pd.read_csv(CSV)
    rows, stats = [], []
    for pop, shown, base_pf, base_days in POPS:
        for key, name, src, add in RULES:
            r = d[(d["母體"] == pop) & (d["出場"] == key)]
            if not len(r):
                sys.exit(f"CSV 查無 {pop}/{key}")
            r = r.iloc[0]
            rep, unc, days = r["獲利因子"], r["未平倉%"], r["平均持有天"]
            cls = "up" if rep > add[pop] else "down"    # 底色比的是同一列的「附加」
            rows.append(
                f'<tr><td>{shown}</td><td>{name}</td><td>{unc:.2f}%</td>'
                f'<td>{base_pf:.4f}</td><td>{add[pop]:.4f}</td>'
                f'<td class="{cls}">{rep:.4f}</td>'
                f'<td>{base_days} → {days:.0f}</td></tr>')
            stats.append({"母體": pop, "出場": name, "出處": src, "基準線": base_pf,
                          "附加": add[pop], "替換": round(rep, 4),
                          "勝替換": rep > add[pop], "勝基準線": rep > base_pf,
                          "未平倉%": round(unc, 2), "持有天": round(days, 1)})
    print("\n".join(rows))
    s = pd.DataFrame(stats)
    print("\n" + "=" * 70)
    print(f"替換勝過附加：{s['勝替換'].sum()} / {len(s)}")
    print(f"替換勝過基準線：{s['勝基準線'].sum()} / {len(s)}")
    for pop, _, _, _ in POPS:
        x = s[s["母體"] == pop]
        print(f"  {pop}：勝附加 {x['勝替換'].sum()}/{len(x)}、"
              f"勝基準線 {x['勝基準線'].sum()}/{len(x)}")
    print(f"未平倉% 範圍：{s['未平倉%'].min():.2f} ~ {s['未平倉%'].max():.2f}"
          f"（最高＝{s.loc[s['未平倉%'].idxmax(), '母體']}×"
          f"{s.loc[s['未平倉%'].idxmax(), '出場']}）")
    print("\n替換輸給附加的格子：")
    print(s[~s["勝替換"]][["母體", "出場", "附加", "替換"]].to_string(index=False))
    print("\n替換也勝過基準線的格子：")
    print(s[s["勝基準線"]][["母體", "出場", "基準線", "替換", "持有天"]]
          .to_string(index=False))
    print("\n各出場勝基準線的母體數（跨母體一致性）：")
    print(s.groupby("出場", sort=False)["勝基準線"].sum().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
