"""
雙均線交叉策略（MA cross）— ma_strategy 套件下的方案（vbt 框架）

繼承 VbtSingleStrategy，只記錄「判定日」買賣條件（隔日開盤成交、費稅、tick 由基底處理）：
  進場：short MA 上穿 long MA（黃金交叉）→ 買（baseline）
  出場：short MA 下穿 long MA（死亡交叉）→ 賣
  成交：判定日「隔日開盤」（基底統一，無 look-ahead）

優化紀錄（#N = 改動序號，依測試先後；旗標設定切換，最優版可再凍結成註解）：
  #1 MIN_VOL_ZHANG  流動性底線：5 日均量 > N 張（=N×1000 股）。✅ 勝率不掉、可成交
  #2 VOL_ADX>0      波動率：交叉前一根 ADX(14) < VOL_ADX（盤整時才進）。△ PF 小升、量大減
  #3 CONFIRM        雙重確認（連 2 日 short>long 才進）。❌ 對雙均線無效
  #4 ANGLE_DAYS>0   嚴格夾角：short>long 且 ratio(短/長) 連 N 日遞增（--angle-days N 或 --angle3=3）。○ 長組合輕微正向
  #5 STRONGK        交叉當日強 K（長紅開→收≥5% 或 跳空過長均線）。✅ 長組合 EV(Trim) 大升（最佳）
  #6 ALIGN          多頭排列：比 long 更長的均線呈多頭排列才進（併入自原 ma_align）。❌ 不優於 baseline

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

import pandas as pd

from _02_strategy.base.vbt import batch
from _02_strategy.base.vbt.single import VbtSingleStrategy

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
DEFAULT_START = "2001-01-01"
DEFAULT_END = "2025-12-31"
ALL_MAS = (5, 10, 20, 50, 60, 120, 200)   # 多頭排列判斷用的標準均線集合（ALIGN 旗標）


class MACrossStrategy(VbtSingleStrategy):
    """雙均線交叉。設定 SHORT_MA/LONG_MA（短<長）+ 各優化旗標。只描述判定日訊號。"""

    SHORT_MA = 50
    LONG_MA = 200
    # 優化旗標（預設全關 = baseline 純交叉）
    CONFIRM = False         # 連 2 日 short>long 才進
    ANGLE_DAYS = 0          # >0：short>long 且 ratio 連 N 日遞增（夾角擴大 N 日驗證）
    VOL_ADX = 0.0           # >0：交叉前一根 ADX(14) < VOL_ADX 才進
    STRONGK = False         # 交叉當日 長紅 或 跳空過長均線
    MIN_VOL_ZHANG = 0       # >0：5 日均量 > N×1000 股
    ALIGN = False           # 多頭排列：比 long 更長的均線呈多頭排列才進（併入自 ma_align 多頭排列突破）

    def add_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """加短/長均線、比例、ADX、量能、強 K 欄位（依旗標需求；一律算，開銷小）。"""
        df["ma_short"] = df["close"].rolling(self.SHORT_MA).mean()
        df["ma_long"] = df["close"].rolling(self.LONG_MA).mean()
        df["ratio"] = df["ma_short"] / df["ma_long"]          # 夾角代理（>1 表短在長之上）
        df["vol_ma5"] = df["volume"].rolling(5).mean()
        # 強 K：長紅(開→收≥5%) 或 跳空開在長均之上
        df["long_red"] = df["close"] >= df["open"] * 1.05
        df["gap_over_long"] = (df["open"] > df["ma_long"]) & (df["open"] > df["close"].shift(1))
        # ADX(14) Wilder（自算）
        high, low, close = df["high"], df["low"], df["close"]
        prev_c = close.shift(1)
        up = high.diff(); dn = -low.diff()
        plus_dm = up.where((up > dn) & (up > 0), 0.0)
        minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
        tr = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
        p = 14
        atr = tr.ewm(alpha=1 / p, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1 / p, adjust=False).mean() / atr
        minus_di = 100 * minus_dm.ewm(alpha=1 / p, adjust=False).mean() / atr
        dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).fillna(0.0)
        df["adx"] = dx.ewm(alpha=1 / p, adjust=False).mean()
        # ALIGN：比 long 更長的標準均線是否多頭排列（短>長逐一成立）；併入自 ma_align
        longer = [m for m in ALL_MAS if m > self.LONG_MA]
        if len(longer) >= 2:
            for m in longer:
                df[f"ma_{m}"] = df["close"].rolling(m).mean()
            align = pd.Series(True, index=df.index)
            for a, b in zip(longer[:-1], longer[1:]):
                align &= df[f"ma_{a}"] > df[f"ma_{b}"]
            df["bull_align"] = align
        else:
            df["bull_align"] = True   # 無 2 條以上更長均線 → 不設限制
        return df

    def buy_signal(self, df: pd.DataFrame) -> pd.Series:
        """進場「判定日」：依旗標組合。基底延到隔日開盤成交。"""
        above = df["ma_short"] > df["ma_long"]
        if self.ANGLE_DAYS and self.ANGLE_DAYS > 0:
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
            sig = sig & (df["long_red"] | df["gap_over_long"])
        if self.MIN_VOL_ZHANG and self.MIN_VOL_ZHANG > 0:
            sig = sig & (df["vol_ma5"] > self.MIN_VOL_ZHANG * 1000)
        if self.ALIGN:
            sig = sig & df["bull_align"]
        return sig.fillna(False).astype(bool)

    def sell_signal(self, df: pd.DataFrame) -> pd.Series:
        """出場「判定日」：short 下穿 long（死亡交叉）。基底延到隔日開盤成交。"""
        below = df["ma_short"] < df["ma_long"]
        return below & ~below.shift(1, fill_value=False)


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

    result = batch.run_folder(strat, args.folder,
                              start=args.start, end=args.end, limit=args.limit)
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
