# -*- coding: utf-8 -*-
"""
產第九篇第二張表的 24 列 HTML ＋ 正文要用的統計，避免手抄。

欄位來源（全部讀 SPA 回測段的產出，2026-10-08 起不再寫死常數、也不讀 blog 那份舊副本）：
- 「替換」「未平倉」「持有天」＝ result/macd_exit_replace/macd_exit_replace.csv
  （_02_strategy/macd_strategy/macd_exit_replace.py）。
- 「原本的出場」＝同一份 CSV 的「原生出場」列（同一批跑出來，與（四）篇矩陣對角線同值）。
- 「附加」＝（七）篇趨勢出場讀 result/single_macd/_exit_parallel.csv 的 A並行組、
  （八）篇風控出場讀 _exit_sweep.csv 的 B疊加組（_02_strategy/macd_strategy/macd_exit_sweep.py）。
文章是否正確引用（四）（七）（八）篇已發佈的數字，由 verify_article9.py 對 md 檢查。

執行：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python \
        _04_analysis/macd/article/build_article9_table.py
"""

import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd  # noqa: E402

from _02_strategy.base.vbt import common  # noqa: E402

REPLACE_CSV = os.path.join(common.result_dir("macd_strategy", "macd_exit_replace"),
                           "macd_exit_replace.csv")
SWEEP_DIR = common.result_dir("macd_strategy", "single_macd")
# replace CSV 的母體名（NAME_BASE）→ 文章／舊表用的短名
POP_SHORT = {"黃金交叉": "交叉", "零軸上穿": "零軸", "純背離": "背離"}
POPS = ["交叉", "零軸", "背離"]
# (replace CSV 的出場名, 文章顯示名, 出處篇, 附加數字的來源檔, 該檔的組, 該檔的出場名)
RULES = [
    ("跌破年線", "跌破年線", "七", "_exit_parallel.csv", "A並行", "跌破MA200"),
    ("超級趨勢翻空", "超級趨勢", "七", "_exit_parallel.csv", "A並行", "Supertrend翻空"),
    ("SAR翻空", "拋物線 SAR", "七", "_exit_parallel.csv", "A並行", "SAR翻空"),
    ("波段高點走低", "波段高點走低", "七", "_exit_parallel.csv", "A並行", "頂頂低"),
    ("跌破二十日低", "跌破二十日低", "七", "_exit_parallel.csv", "A並行", "跌破20日低"),
    ("吊燈3ATR", "吊燈 3×ATR", "八", "_exit_sweep.csv", "B疊加", "吊燈3ATR"),
    ("自最高點回落10%", "自最高點回落 10%", "八", "_exit_sweep.csv", "B疊加", "回落10%"),
    ("抱滿60天", "抱滿 60 天", "八", "_exit_sweep.csv", "B疊加", "抱滿60天"),
]


def load_replace() -> pd.DataFrame:
    """讀替換版結果，母體換成短名（verify_article9 共用）。"""
    d = pd.read_csv(REPLACE_CSV)
    d["母體"] = d["母體"].map(POP_SHORT)
    return d


def appended_pf() -> dict:
    """{(文章出場名, 母體): 附加版獲利因子}。"""
    cache, out = {}, {}
    for _, name, _, fname, grp, key in RULES:
        if fname not in cache:
            cache[fname] = pd.read_csv(os.path.join(SWEEP_DIR, fname))
        d = cache[fname]
        for pop in POPS:
            r = d[(d["組"] == grp) & (d["母體"] == pop) & (d["出場"] == key)]
            if len(r) != 1:
                sys.exit(f"{fname} 查無（或重複）{grp}/{pop}/{key}")
            out[(name, pop)] = float(r.iloc[0]["獲利因子"])
    return out


def main():
    d = load_replace()
    add = appended_pf()
    rows, stats = [], []
    for pop in POPS:
        base = d[(d["母體"] == pop) & (d["出場"] == "原生出場")].iloc[0]
        base_pf, base_days = base["獲利因子"], base["平均持有天"]
        for key, name, src, *_ in RULES:
            r = d[(d["母體"] == pop) & (d["出場"] == key)]
            if not len(r):
                sys.exit(f"CSV 查無 {pop}/{key}")
            r = r.iloc[0]
            rep, unc, days = r["獲利因子"], r["未平倉%"], r["平均持有天"]
            a = add[(name, pop)]
            cls = "up" if rep > a else "down"    # 底色比的是同一列的「附加」
            rows.append(
                f'<tr><td>{pop}</td><td>{name}</td><td>{unc:.2f}%</td>'
                f'<td>{base_pf:.4f}</td><td>{a:.4f}</td>'
                f'<td class="{cls}">{rep:.4f}</td>'
                f'<td>{base_days:.0f} → {days:.0f}</td></tr>')
            stats.append({"母體": pop, "出場": name, "出處": src, "基準線": base_pf,
                          "附加": a, "替換": round(rep, 4),
                          "勝替換": rep > a, "勝基準線": rep > base_pf,
                          "未平倉%": round(unc, 2), "持有天": round(days, 1)})
    print("\n".join(rows))
    s = pd.DataFrame(stats)
    print("\n" + "=" * 70)
    print(f"替換勝過附加：{s['勝替換'].sum()} / {len(s)}")
    print(f"替換勝過基準線：{s['勝基準線'].sum()} / {len(s)}")
    for pop in POPS:
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
