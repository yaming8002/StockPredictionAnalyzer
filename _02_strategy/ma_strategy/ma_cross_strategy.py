"""
雙均線交叉策略（MA cross）— ma_strategy 套件下的方案（vbt 框架）

繼承 VbtSingleStrategy，只記錄「判定日」買賣條件（隔日開盤成交、費稅、tick 由基底處理）：
  進場：short MA 上穿 long MA（黃金交叉）→ 買（baseline）
  出場：short MA 下穿 long MA（死亡交叉）→ 賣
  成交：判定日「隔日開盤」（基底統一，無 look-ahead）
【對應文章】均線交叉系列：https://stockanalyzer.sailforthlab.dev/archives/?subcategory=%E5%9D%87%E7%B7%9A%E4%BA%A4%E5%8F%89


優化紀錄（#N = 改動序號，依測試先後；旗標設定切換，最優版可再凍結成註解）：
  數字為 2002–2025 重跑結果（PF＝獲利因子、EV＝期望值/筆(元)；完整見 reference/ma_cross/_full_aggregate_all_variants.csv）。
  baseline 60/200：交易 22,886、PF 2.47、EV 1,206、持有 249 天；50/200：PF 2.42、EV 1,151、持有 236 天。
  #1 MIN_VOL_ZHANG  流動性底線：5 日均量 > N 張（=N×1000 股）。✅ 勝率不掉、可成交：60/200 PF 2.47→2.13（>300 張）／1.96（>1000 張）
  #2 VOL_ADX>0      波動率：交叉前一根 ADX(14) < VOL_ADX（盤整時才進）。△ PF 小升、量大減：60/200 PF 2.47→2.52（<25）／2.61（<20），
                    交易 22,886→8,519／5,010
  #3 CONFIRM        雙重確認（連 2 日 short>long 才進）。❌ 對雙均線無感：60/200 PF 2.47→2.48、交易幾乎不變
  #4 ANGLE_DAYS>0   嚴格夾角：short>long 且 ratio(短/長) 連 N 日遞增（--angle-days N 或 --angle3=3）。○ 幾乎同 baseline：
                    50/200 PF 2.42→2.43；2 日與 3 日無實質差別
  #5 STRONGK        交叉當日強 K（長紅開→收≥5% 或 跳空過短均線）。✅ 長組合每筆期望值升（60/200 EV 1,206→1,352、
                    120/200 1,146→1,337）、PF 大致持平（60/200 2.47→2.45、50/200 2.42→2.27），交易砍到約 4~5 成
                    〔跳空基準 2026-06-26 由長均改短均，訊號主由長紅貢獻，跳空僅補位〕
  #6 ALIGN          多頭排列（新定義）：120日線>200日線 且 收盤>120日線（長線仍多頭格局）才進，固定120/200。
                    1000 張基準下 PF 升（60/200 1.96→2.29、120/200 1.91→1.97），但交易大減（60/200 8,593→1,470）
  #7 VOL_BOTH       量能 BOTH（lab 冠軍量能）：當日量 > 20日均量×1.5 且 連 3 日量 ≥ 100萬股。❌ 過濾過度：交易砍到約 1/10、
                    60/200 PF 2.47→1.66、總獲利 2,760→148 萬
  #8 CHOCH          CHoCH 早期出場（ZigZag 2% 進場以來 lower-high 出場；死叉 或 CHoCH）。❌ 長組合有害：勝率小升但砍趨勢利潤
                    （60/200 EV 1,206→184、PF 2.47→1.42、持有 249→48 天）
  #9 EXIT_BELOW_SHORT 出場加嚴：收盤由上跌破短均線也出場（除死亡交叉外，--exit-below-short）。❌ 全21組有害：黃金交叉後價格
                    本就會回測短均，等於「一拉回就跑」，持有天全崩(50/200 236→42)、砍掉趨勢利潤（60/200 PF 2.47→1.54；5/10 0.95→0.72）。
  #10 DIVERGE       夾角擴大進場（短均N日%斜率>長均%斜率）/ 收斂或死叉出場（--diverge [--diverge-margin X] [--slope-win N]）。
                    ❌ 同 #9：收斂出場太敏感，持有天 236→42~53、交易×2.8、長均200那3組 PF 2.4 上下→1.35~1.41。問題在「出場」不在進場。
                    純進場版（--diverge-entry-only，只夾角擴大進場+死叉出場）= ○ 幾乎同 baseline（50/200、60/200 PF 不變、
                    120/200 2.38→2.40）：交叉當下短均本就比長均爬得快，「夾角擴大」與交叉高度重疊。
  #11 ANGLE_DEG      黃金交叉夾角(短均-長均度數)>N度才進（--angle-deg N；每日%斜率當正切）。夾角>20度：交易砍至約 1/3 以下，
                    每筆期望值升（50/200 EV 1,151→1,465、60/200 1,206→1,633、60/120 804→1,474），PF 60/120 2.15→2.46、
                    50/200 2.42→2.29；120/200 PF 3.76 但僅 1,703 筆、樣本小不穩。+流動性 300 張：60/120 PF 2.40、EV 1,430，
                    高於強K+300張（60/120 1.97／840、60/200 2.26／1,270），但總獲利較低（635 vs 870 萬）。

執行（全市場、掃資料夾、彙總；輸出 result/ma_cross/<variant>/，格式同 single_ma）：
  python _02_strategy/ma_strategy/ma_cross_strategy.py <資料夾> --short 50 --long 200 [--confirm] [--angle3]
         [--vol-adx 25] [--strongk] [--min-vol-zhang 1000] --variant <名稱>
"""
import argparse
import os
import sys

# 直接執行此檔時，把專案根目錄加進 sys.path（讓 _02_strategy.* 點號 import 可解析）
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd

from _01_data.indicators_trend import calculate_sma, calculate_adx
from _01_data.indicators_pattern import calculate_zigzag
from _02_strategy.base.vbt import batch
from _02_strategy.base.vbt.single import VbtSingleStrategy
# 標準回測區間 DEFAULT_START/END 與資料品質排除集 GLITCH：跨策略共用，統一由 base/vbt/common 取用（單一定義）。
from _02_strategy.base.vbt.common import DEFAULT_START, DEFAULT_END, GLITCH

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")


def _ma(df: pd.DataFrame, window: int) -> pd.Series:
    """取 sma_{window}：優先用 parquet 既有欄位，無則用 _01_data 的 calculate_sma 補算（不在策略內自算）。"""
    col = f"sma_{window}"
    if col not in df.columns:
        calculate_sma(df, window)
    return df[col]


class MACrossStrategy(VbtSingleStrategy):
    """雙均線交叉。設定 SHORT_MA/LONG_MA（短<長）+ 各優化旗標。只描述判定日訊號。"""

    SHORT_MA = 50
    LONG_MA = 200
    # 優化旗標（預設全關 = baseline 純交叉）
    CONFIRM = False         # 連 2 日 short>long 才進
    ANGLE_DAYS = 0          # >0：short>long 且 ratio 連 N 日遞增（夾角擴大 N 日驗證）
    VOL_ADX = 0.0           # >0：交叉前一根 ADX(14) < VOL_ADX 才進
    STRONGK = False         # 交叉當日 長紅 或 跳空過短均線
    MIN_VOL_ZHANG = 0       # >0：5 日均量 > N×1000 股
    ALIGN = False           # 多頭排列（新定義）：120日線>200日線 且 收盤>120日線（長線仍多頭）才進，固定120/200
    VOL_BOTH = False        # 量能 BOTH（lab 冠軍）：當日量 > 20日均量×1.5 且 連 3 日量 ≥ 100萬股
    CHOCH = False           # CHoCH 早期出場（ZigZag 2% lower-high 出場；路徑相依、覆寫 build_signals）
    EXIT_BELOW_SHORT = False  # #9 出場：收盤由上跌破短均線也出場（除死亡交叉外）
    DIVERGE = False           # #10 夾角擴大進場 / 收斂或死叉出場（短均%斜率 vs 長均%斜率）
    DIVERGE_MARGIN = 0.0      # #10 進場門檻：短均N日%斜率 − 長均N日%斜率 須 > 此值（0=只要短均爬得快）
    SLOPE_WIN = 5             # #10 斜率視窗：以 MA 的 N 日 %變化當斜率（scale-invariant）
    DIVERGE_EXIT = True       # #10 DIVERGE 時是否加「收斂出場」（False=只用死叉出場，純測進場品質）
    ANGLE_DEG = 0.0           # #11 黃金交叉「夾角(短均-長均)度數 > 此值」才進（filter，疊在交叉上；每日%斜率當正切）
    MIN_VOL_SHARES = 0        # 進場「當日成交量 > N 股」才進（filter；與 MIN_VOL_ZHANG 的5日均量不同，這是當日量）

    def add_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """短/長均線(sma)、ADX、zigzag 引用 _01_data 指標；量能均線/強K 無對應 indicators 故依需求補。"""
        df["ma_short"] = _ma(df, self.SHORT_MA)
        df["ma_long"] = _ma(df, self.LONG_MA)
        df["ratio"] = df["ma_short"] / df["ma_long"]          # 夾角代理（>1 表短在長之上）
        w = self.SLOPE_WIN                                    # #10 各均線「N 日 %斜率」（scale-invariant）
        df["sl_short"] = df["ma_short"] / df["ma_short"].shift(w) - 1
        df["sl_long"] = df["ma_long"] / df["ma_long"].shift(w) - 1
        # #10b 夾角度數：每日%斜率(百分點)當正切 → arctan 換度數；夾角 = 短均角度 − 長均角度（長均≈平→夾角≈短均角度）
        df["angle_short"] = np.degrees(np.arctan(df["sl_short"] / w * 100))
        df["angle_long"] = np.degrees(np.arctan(df["sl_long"] / w * 100))
        df["angle_gap"] = df["angle_short"] - df["angle_long"]
        df["vol_ma5"] = df["volume"].rolling(5).mean()
        df["vol_ma20"] = df["volume"].rolling(20).mean()      # 量能 BOTH 相對放量基準
        # 強 K：長紅(開→收≥5%) 或 跳空開在短均之上（跳空基準用短均，非長均）
        df["long_red"] = df["close"] >= df["open"] * 1.05
        df["gap_over_short"] = (df["open"] > df["ma_short"]) & (df["open"] > df["close"].shift(1))
        # ADX(14) Wilder，引用 _01_data 的 calculate_adx（寫入 df["adx"]）
        calculate_adx(df, 14)
        # ALIGN（新定義）：長線仍處多頭格局才進＝120日線 > 200日線 且 收盤 > 120日線。
        # 固定用 120/200（不隨交叉的短/長變動），代表「整體長期趨勢還在多頭」這道大方向濾網。
        ma120 = _ma(df, 120)
        ma200 = _ma(df, 200)
        df["bull_align"] = (ma120 > ma200) & (df["close"] > ma120)
        if self.CHOCH:
            calculate_zigzag(df, 0.02)   # 型態指標：擺動高低點（CHoCH 用），引用自 _01_data.indicators_pattern
        return df

    def buy_signal(self, df: pd.DataFrame) -> pd.Series:
        """進場「判定日」：依旗標組合。基底延到隔日開盤成交。"""
        above = df["ma_short"] > df["ma_long"]
        if self.DIVERGE:
            div = above & (df["sl_short"] > df["sl_long"] + self.DIVERGE_MARGIN)  # #10 夾角擴大：短均%斜率 > 長均%斜率
            sig = div & ~div.shift(1, fill_value=False)
        elif self.ANGLE_DAYS and self.ANGLE_DAYS > 0:
            # 嚴格：短在長之上 且 夾角(ratio) 連 N 日遞增（N 個遞增步）
            r = df["ratio"]
            sig = above.copy()
            for i in range(self.ANGLE_DAYS):
                sig = sig & (r.shift(i) > r.shift(i + 1, fill_value=0.0))
        elif self.CONFIRM:
            cross = above & ~above.shift(1, fill_value=False)
            sig = above & cross.shift(1, fill_value=False)   # 連 2 日 short>long
        else:
            sig = above & ~above.shift(1, fill_value=False)  # baseline 黃金交叉

        if self.VOL_ADX and self.VOL_ADX > 0:
            sig = sig & (df["adx"].shift(1) < self.VOL_ADX)
        if self.STRONGK:
            sig = sig & (df["long_red"] | df["gap_over_short"])
        if self.ANGLE_DEG and self.ANGLE_DEG > 0:
            sig = sig & (df["angle_gap"] > self.ANGLE_DEG)   # #11 夾角(短均-長均)度數 > 門檻才進
        if self.MIN_VOL_SHARES and self.MIN_VOL_SHARES > 0:
            sig = sig & (df["volume"] > self.MIN_VOL_SHARES)  # 當日成交量 > N 股 才進
        if self.MIN_VOL_ZHANG and self.MIN_VOL_ZHANG > 0:
            sig = sig & (df["vol_ma5"] > self.MIN_VOL_ZHANG * 1000)
        if self.ALIGN:
            sig = sig & df["bull_align"]
        if self.VOL_BOTH:
            v = df["volume"]
            rel = v > df["vol_ma20"] * 1.5                                  # 相對放量 1.5 倍
            absol = (v >= 1_000_000) & (v.shift(1) >= 1_000_000) & (v.shift(2) >= 1_000_000)  # 連 3 日絕對 ≥100萬
            sig = sig & rel & absol
        return sig.fillna(False).astype(bool)

    def sell_signal(self, df: pd.DataFrame) -> pd.Series:
        """出場「判定日」：short 下穿 long（死亡交叉）；#9 開啟時，收盤由上跌破短均線也算出場。基底延到隔日開盤成交。"""
        below = df["ma_short"] < df["ma_long"]
        sig = below & ~below.shift(1, fill_value=False)        # 死亡交叉
        if self.EXIT_BELOW_SHORT:
            pb = df["close"] < df["ma_short"]
            sig = sig | (pb & ~pb.shift(1, fill_value=False))  # 收盤由上跌破短均線
        if self.DIVERGE and self.DIVERGE_EXIT:
            conv = df["sl_short"] <= df["sl_long"]             # #10 夾角收斂：短均不再爬得比長均快
            sig = sig | (conv & ~conv.shift(1, fill_value=False))
        return sig.fillna(False).astype(bool)

    def build_signals(self, df: pd.DataFrame):
        """
        CHOCH=False → 沿用基底（向量化、自動位移隔日成交）。
        CHOCH=True  → 路徑相依：單檔逐根掃描，出場 = 死亡交叉 或 CHoCH（進場以來擺動高點
                      出現 lower-high）。產出「判定日」訊號後自行 shift(1) 成隔日開盤成交。
        """
        if not self.CHOCH:
            return super().build_signals(df)

        entry_raw = self.entry_signal(df).to_numpy()        # 判定日進場（cross + 其他進場旗標；區間前不算）
        death = self.sell_signal(df).to_numpy()             # 判定日死亡交叉
        th = df["zigzag_turn_high"].to_numpy()
        n = len(df)
        entries = np.zeros(n, dtype=bool)
        exits = np.zeros(n, dtype=bool)
        in_pos = False
        peak = -np.inf                                      # 進場以來最高擺動高點
        for i in range(n):
            if not in_pos:
                if entry_raw[i]:
                    in_pos = True
                    entries[i] = True
                    peak = -np.inf
            else:
                if not np.isnan(th[i]):
                    if th[i] < peak:                        # 新擺動高點低於進場以來峰值 → CHoCH
                        exits[i] = True
                        in_pos = False
                        continue
                    peak = max(peak, th[i])
                if death[i]:                                # 死亡交叉也出場
                    exits[i] = True
                    in_pos = False
        # 判定日 → 隔日開盤成交（與基底慣例一致）
        e = pd.Series(entries, index=df.index).shift(1, fill_value=False)
        x = pd.Series(exits, index=df.index).shift(1, fill_value=False)
        return e, x


def main(argv) -> int:
    parser = argparse.ArgumentParser(description="雙均線交叉：資料夾批次回測（vbt）")
    parser.add_argument("folder")
    parser.add_argument("--short", type=int, default=50)
    parser.add_argument("--long", type=int, default=200)
    parser.add_argument("--variant", default="baseline", help="輸出子資料夾名")
    parser.add_argument("--confirm", action="store_true", help="雙重確認(連2日)")
    parser.add_argument("--angle3", action="store_true", help="嚴格:夾角連3日擴大（= --angle-days 3）")
    parser.add_argument("--angle-days", type=int, default=0, help="嚴格:夾角連 N 日遞增（0=關）")
    parser.add_argument("--vol-adx", type=float, default=0.0, help="交叉前一根 ADX<此值才進")
    parser.add_argument("--strongk", action="store_true", help="交叉當日強K")
    parser.add_argument("--min-vol-zhang", type=int, default=0, help="5日均量>N張(=N*1000股)")
    parser.add_argument("--align", action="store_true", help="多頭排列:比long更長的均線多頭排列才進(併入自ma_align)")
    parser.add_argument("--vol-both", action="store_true", help="量能BOTH:當日量>20日均量×1.5 且 連3日≥100萬股(lab冠軍量能)")
    parser.add_argument("--choch", action="store_true", help="CHoCH早期出場:ZigZag2%% lower-high 出場(路徑相依)")
    parser.add_argument("--exit-below-short", action="store_true", help="#9 出場:收盤由上跌破短均線也出場")
    parser.add_argument("--diverge", action="store_true", help="#10 夾角擴大進場/收斂或死叉出場")
    parser.add_argument("--diverge-margin", type=float, default=0.0, help="#10 進場門檻:短均N日%%斜率-長均N日%%斜率 > 此值")
    parser.add_argument("--slope-win", type=int, default=5, help="#10 斜率視窗(MA N日%%變化)")
    parser.add_argument("--diverge-entry-only", action="store_true", help="#10 只用夾角擴大進場+死叉出場(拿掉收斂出場)")
    parser.add_argument("--angle-deg", type=float, default=0.0, help="#11 黃金交叉夾角(短均-長均)度數 > 此值才進(每日%%斜率當正切)")
    parser.add_argument("--min-vol-shares", type=int, default=0, help="進場當日成交量 > N 股 才進(當日量,非5日均量)")
    parser.add_argument("--trades", action="store_true")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv[1:])

    if args.short >= args.long:
        print(f"錯誤：short({args.short}) 必須 < long({args.long})")
        return 1

    strat = MACrossStrategy()
    strat.SHORT_MA, strat.LONG_MA = args.short, args.long
    strat.CONFIRM = args.confirm
    strat.ANGLE_DAYS = 3 if args.angle3 else args.angle_days
    strat.VOL_ADX = args.vol_adx
    strat.STRONGK = args.strongk
    strat.MIN_VOL_ZHANG = args.min_vol_zhang
    strat.ALIGN = args.align
    strat.VOL_BOTH = args.vol_both
    strat.CHOCH = args.choch
    strat.EXIT_BELOW_SHORT = args.exit_below_short
    strat.DIVERGE = args.diverge
    strat.DIVERGE_MARGIN = args.diverge_margin
    strat.SLOPE_WIN = args.slope_win
    strat.DIVERGE_EXIT = not args.diverge_entry_only
    strat.ANGLE_DEG = args.angle_deg
    strat.MIN_VOL_SHARES = args.min_vol_shares

    result = batch.run_folder(strat, args.folder,
                              start=args.start, end=args.end, limit=args.limit,
                              exclude=GLITCH)
    out_dir = os.path.join(RESULT_DIR, "ma_cross", args.variant)
    label = f"ma_cross_{args.short}_{args.long}"
    written = batch.write_results(result, out_dir, label, write_trades=args.trades)

    agg = result["aggregate"]
    print(f"=== 雙均線交叉 {args.short}/{args.long}（variant={args.variant}）===")
    print(f"參與股票數: {agg['參與股票數']}（失敗 {agg['失敗檔數']} 檔）")
    print(f"交易次數: {agg['交易次數']} | 勝率(%): {agg['勝率(%)']} | EV: {agg['期望報酬值(EV)']} | 總獲利: {agg['總獲利']:.0f}")
    print(f"輸出: {out_dir}")
    if result["failed"]:
        print(f"失敗檔（前 5）: {result['failed'][:5]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
