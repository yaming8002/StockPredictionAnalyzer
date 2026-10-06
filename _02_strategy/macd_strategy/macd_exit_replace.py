"""
MACD（九）：十條出場全部改成「當唯一出場」的替換版全表
=========================================================
矩陣：母體 3（黃金交叉／零軸上穿／純背離，各配自己的原生出場當基準線）
      × 出場 11（原生出場 ＋（七）（八）兩篇的十條）＝ 33 組。

十條依（七）（八）的分組：
  （七）趨勢結束五條：跌破 MA200／Supertrend 翻空／SAR 翻空／波段高點走低／跌破 20 日低
  （八）風控五條：吊燈 3×ATR／自最高點回落 10%／固定停損 2×ATR／固定停利 +20%／抱滿 60 天

**替換是附錄、不是主軸**：出場優化一律以「附加」為主軸（原出場留著、新規則疊上去），
量到的才是「這條規則有沒有優化原策略」；整條換掉量到的是「換一套技術分析」，是另一
個問題。這張表只回答後者。

「未平倉%」一定要一起看：**固定停利+20% 與固定停損 2×ATR 當唯一出場就是口徑失效**
——只等一個固定價位、沒觸到就無限期抱著，會讀出「勝率 99.9%、獲利因子上千」這種假表
（未平倉率 9~19%，其餘九條最高只有 3.82%）。這兩條照跑、但不納入比較，文章的替換版
因此是**八條**不是十條；跑它們的目的是留下未平倉率當排除依據。

執行：python _02_strategy/macd_strategy/macd_exit_replace.py [--limit 300]
"""
import argparse
import os
import sys
import time

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from _02_strategy.base.vbt import common
from _02_strategy.macd_strategy.macd_variants import NAME_BASE, NAME_EXIT
from _02_strategy.macd_strategy.macd_sweep import prepare, run_variant, spec_rows

OUT = common.result_dir("macd_strategy", "macd_exit_replace")
BASES = ["cross", "zero", "div"]
# 原生出場擺第一個當基準線
EXITS = ["native", "ma200", "supertrend", "psar", "lowerhigh", "donchian",
         "chandelier", "trail10", "atrstop", "takeprofit", "time60"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 檔（冒煙用）")
    a = ap.parse_args()

    t0 = time.time()
    data = prepare(limit=a.limit)
    print(f"載入並備妥 {len(data)} 檔｜{len(BASES) * len(EXITS)} 組｜"
          f"{time.time() - t0:.0f} 秒", flush=True)

    out = []
    for base in BASES:
        for ex in EXITS:
            t = time.time()
            s = run_variant(data, base, "none", ex, "replace")
            out.append(({"母體": NAME_BASE[base], "出場": NAME_EXIT[ex]}, s))
            print(f"  {NAME_BASE[base]} × {NAME_EXIT[ex]}：{s['交易次數']:,} 筆｜"
                  f"PF {s['獲利因子(PF)']}｜未平倉 {s['未平倉%']}%｜"
                  f"{time.time() - t:.0f} 秒", flush=True)

    df = spec_rows(out)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "macd_exit_replace.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n耗時 {time.time() - t0:.0f} 秒｜{len(df)} 列 → {path}")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
