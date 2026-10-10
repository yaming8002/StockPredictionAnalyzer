# -*- coding: utf-8 -*-
"""
驗 KD 交叉（二）kd-cross-zone：三張結果表逐格＋正文所有數字句（含台積電 2008–09 那筆交易）。

台積電那筆：從「低檔黃金交叉買、高檔死叉賣」的逐筆交易取 2008–09 唯一一筆，驗進出價、報酬、天數；
「中途 20 次一般死亡交叉」照配圖腳本 _04_analysis/kd/charts/_draw_kd_zone_trade.py 的定義數
（買進日 < 訊號日 < 賣出成交日）；這個區間的最後一次就是觸發出場的高檔死叉，
所以「被忽略的中途回檔」＝總數 − 1。

執行：
    BLOG_DIR=<blog 專案根目錄> PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python \
        _04_analysis/kd/article/verify_kd_article2.py
"""
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd  # noqa: E402

from _01_data.indicators_momentum_volume import calculate_kd  # noqa: E402
from _02_strategy.base.vbt import common  # noqa: E402
from _04_analysis.kd.article.build_kd_single_tables import ZONE, article2  # noqa: E402
from _04_analysis.kd.article.kd_article_common import (SWEEP, Checker, fmt,  # noqa: E402
                                                       read_post, single_metrics)

BASE = "amt__golden__death"


def wan(n: int) -> str:
    """筆數寫成「x.x 萬」（文章口語：1.2 萬、3.3 萬）。"""
    return fmt(n / 10_000, 1)


def tsmc_trade(ck: Checker, text: str):
    t = pd.read_parquet(os.path.join(SWEEP, ZONE, "single_kd_trades.parquet"))
    w = t[(t["stock_id"] == "2330.TW") & (t["buy_date"] >= "2008-01-01") & (t["buy_date"] <= "2009-12-31")]
    if not ck.check(len(w) == 1, f"台積電 2008–09 應只有一筆，實際 {len(w)}"):
        return
    r = w.iloc[0]
    ret = (r["sell_price"] - r["buy_price"]) / r["buy_price"] * 100
    days = (r["sell_date"] - r["buy_date"]).days
    ck.contains(text, f"在這裡買進，約 {fmt(r['buy_price'], 0)} 元", "台積電買價")
    ck.contains(text, f"才賣出，約 {fmt(r['sell_price'], 0)} 元", "台積電賣價")
    ck.contains(text, f"這一趟 **＋{fmt(ret, 1)}%、抱了 {days} 天**", "台積電報酬／天數")
    ck.contains(text, f"從 {fmt(r['buy_price'], 0)} 到 {fmt(r['sell_price'], 0)} 的波段", "台積電波段價位")

    df = pd.read_parquet(os.path.join(common.DATA_DIR, "2330.TW.parquet")).sort_index()
    calculate_kd(df, 9, 3, 3)
    k, d = df["k"], df["d"]
    death = (k < d) & (k.shift(1) >= d.shift(1))
    span = death[(death.index > r["buy_date"]) & (death.index < r["sell_date"])]
    n_chart = int(span.sum())
    last = span[span].index[-1]
    exit_is_high = bool(k[last] > 80 and d[last] > 80)
    ck.check(exit_is_high, f"區間最後一次死叉（{last.date()}）應是觸發出場的高檔死叉")
    ck.contains(text, f"一共出現了 **{n_chart} 次一般死亡交叉**（最後一次就是觸發出場的那個高檔死叉）",
                "區間內死叉次數（配圖灰色 ▽ 數）")
    ck.contains(text, f"把前面 **{n_chart - 1} 次中途的回檔**全部當成雜訊忽略掉", "被忽略的中途死叉次數")


def main() -> int:
    ck = Checker("KD 交叉（二）")
    text = read_post("kd-cross-zone")
    ck.tables(text, article2())
    b, z = single_metrics(BASE), single_metrics(ZONE)
    ent, ext = single_metrics("amt__low_zone__death"), single_metrics("amt__golden__high_death")
    loose, strict = single_metrics(ZONE + "__z30_70"), single_metrics(ZONE + "__z10_90")

    ck.contains(text, f"獲利因子從 {fmt(b['獲利因子'], 2)} 到 {fmt(z['獲利因子'], 2)}、每筆期望值從 "
                      f"{fmt(b['期望值/筆'], 0)} 元變 {fmt(z['期望值/筆'], 0, True)} 元、中位數也從負轉正"
                      f"（{fmt(z['中位數%'], 1, True)}%）、總損益從 {fmt(b['總獲利(萬)'], 0)} 萬變 "
                      f"{fmt(z['總獲利(萬)'], 0, True)} 萬", "限定前後對照")
    ck.check(b["中位數%"] < 0 < z["中位數%"], "中位數由負轉正")
    ck.contains(text, f"交易次數從 {b['交易次數'] // 10_000} 萬筆掉到 {wan(z['交易次數'])} 萬筆", "筆數")
    ck.contains(text, f"平均持有天數從 {fmt(b['平均持有天'], 0)} 天暴增到 **{fmt(z['平均持有天'], 0)} 天**",
                "持有天數")
    ck.contains(text, f"勝率也上到 {fmt(z['勝率%'], 1)}%", "勝率")
    tsmc_trade(ck, text)

    ck.contains(text, f"（獲利因子 {fmt(ent['獲利因子'], 2)}、期望值 {fmt(ent['期望值/筆'], 0)} 元、"
                      f"總損益 {fmt(ent['總獲利(萬)'], 0)} 萬）", "只有進場那半")
    ck.check(ent["獲利因子"] < 1, "只有進場那半仍是負的")
    ck.contains(text, f"獲利因子 {fmt(ext['獲利因子'], 2)}、總損益 {fmt(ext['總獲利(萬)'], 0, True)} 萬**，"
                      "反而是三個版本裡最高的", "只有出場那半")
    three = [ent, ext, z]
    ck.check(ext["獲利因子"] == max(m["獲利因子"] for m in three)
             and ext["總獲利(萬)"] == max(m["總獲利(萬)"] for m in three), "出場那半 PF、總損益皆三者最高")
    ck.contains(text, f"完整版（{fmt(z['獲利因子'], 2)}），其實比「只有出場那半」（{fmt(ext['獲利因子'], 2)}）"
                      "還低一些", "完整版 vs 出場那半")
    ck.contains(text, f"交易從 {wan(ext['交易次數'])} 萬筆被砍到 {wan(z['交易次數'])} 萬筆", "筆數砍掉")

    ck.contains(text, f"獲利因子掉到 {fmt(loose['獲利因子'], 2)}、總損益只剩 {fmt(loose['總獲利(萬)'], 0, True)} 萬",
                "30/70")
    ck.check(loose["交易次數"] > z["交易次數"], "30/70 交易變多")
    ck.contains(text, f"獲利因子衝到 {fmt(strict['獲利因子'], 2)}、每筆期望值 {fmt(strict['期望值/筆'], 0, True)} 元",
                "10/90")
    ck.contains(text, f"只湊出 {fmt(strict['交易次數'], 0)} 筆交易、平均一筆要抱 "
                      f"{fmt(strict['平均持有天'], 0)} 天（約 {fmt(strict['平均持有天'] / 365, 1)} 年）", "10/90 樣本")
    ck.contains(text, f"20／80 版獲利因子 {fmt(z['獲利因子'], 2)}、總損益 {fmt(z['總獲利(萬)'], 0, True)} 萬",
                "重點整理")
    return ck.done()


if __name__ == "__main__":
    raise SystemExit(main())
