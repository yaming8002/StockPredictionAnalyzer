"""
單一均線突破策略（single MA breakout）— ma_strategy 套件下的方案

繼承 VbtSingleStrategy，只記錄「判定日」買賣條件（隔日開盤成交、費稅、tick 由基底處理）：
  進場：突破單一 MA。**目前啟用 = opt11（最優）= 優化 #7（強 K 突破）+ 優化 #10（突破前 ADX<20 盤整）**。
        其餘優化以註解保留可重啟；把進場優化全註解掉 → 回 baseline（突破當日判定）。
  出場：收盤「下穿」單一 MA（跌破）。
  成交：判定日的「隔日開盤」（基底統一，有訊號一律隔日成交、不在訊號當日收盤）。

附分析：對資料中所有 MA 期數各跑一次，列出比較。
  期數來源：自動偵測資料欄位 sma_<N>；偵測不到才用 DEFAULT_PERIODS。

優化紀錄（#N = 改動序號，依測試先後；切換靠 buy_signal 內「# 優化 #N」註解，各 #N 獨立可疊加；
         結果以 MA=200「EV/筆 → / PF →」，baseline = 194 / 1.62。完整見 reference/single_ma/）：
  #1 雙日確認        → 221 / 1.60   小幅，後被 #7 超越
  #2 量能濾網        → 168 / 1.36（量增）、156 / 1.37（只門檻）  ❌ 砍低流動肥尾
  #3 CMF>0.1         → 224 / 1.52   △ 長均線 EV 升、PF 混、輸 #7
  #4 距離出場        → −34 / 0.87   ❌ 失敗，砍贏家全 MA 由賺轉賠（別砍贏家）
  #5 整理+帶量       → 104 / 1.27   ❌ 輸 baseline
  #6 在MA下方+帶量   → 127 / 1.31   ❌ 扣分元兇＝帶量
  #7 強 K 突破       → 282 / 1.73   ✅ 順動能、全 MA 提升【採用】
  #8 #7+雙日確認     → 240 / 1.55   △ 不如 #7 單獨
  #9 #7+回檔確認     → 176 / 1.48   ❌ 濾掉不回頭飆股
  #10 盤整後 ADX     → 207 / 1.69（<25）、249 / 1.85（<20）  ○ 順動能補位
  #11 #7+#10(ADX<20) → 438 / 2.23   ✅✅ 全測試最佳【採用＝目前啟用】
  #12 #11+流動性門檻 → 87 / 1.23（>1000張）、136 / 1.36（>300張）
  （另：多頭排列突破已併入 ma_cross_strategy.py 的 ALIGN 旗標，測試 ❌ 不優於 baseline）

⚠️ 重大但書（流動性體檢）：#11 賣出日成交量中位僅 ~210 張、約 30% < 50 張（賣不掉）。加可成交
   底線(5日均量>300張)後 edge 大幅蒸發（MA200 PF 2.23→1.36）→ 帳面績效大半來自賣不掉的低流動
   小型股肥尾；可成交真實 edge 薄（長均線 PF~1.3），不宜當大資金主力。後續測試一律帶「5日均量>300張」
   底線。完整彙整見 blog/reference/single_ma/single-ma_master-reference.md。

執行（全市場單一 MA，掃整個資料夾、彙總；結果寫策略同目錄 ./result）：
  優化 #1（預設，# 優化 #1 行不註解）：
    python _02_strategy/ma_strategy/single_ma_strategy.py <資料夾> --ma 20 --variant opt1
  baseline（把 # 優化 #1 行註解掉後）：
    python _02_strategy/ma_strategy/single_ma_strategy.py <資料夾> --ma 20 --variant baseline
  （--variant 只決定輸出資料夾 result/single_ma/<variant>/；行為切換一律靠註解，兩者請一致）
"""
import argparse
import os
import re
import sys

# 直接執行此檔時，把專案根目錄加進 sys.path（讓 _02_strategy.* 點號 import 可解析）
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd

from _01_data.indicators_trend import calculate_sma
from _01_data.indicators_momentum_volume import calculate_cmf
from _01_data.indicators_volatility import calculate_atr_pct
from _02_strategy.base.vbt import batch
from _02_strategy.base.vbt.single import VbtSingleStrategy


def _ma(df: pd.DataFrame, window: int) -> pd.Series:
    """取 sma_{window}：優先用 parquet 既有欄位，無則用 _01_data 的 calculate_sma 補（不在策略內自算）。"""
    col = f"sma_{window}"
    if col not in df.columns:
        calculate_sma(df, window)
    return df[col]


def _cmf(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """取 cmf_{window}：優先用 parquet 既有欄位，無則用 _01_data 的 calculate_cmf 補。"""
    col = f"cmf_{window}"
    if col not in df.columns:
        calculate_cmf(df, window)
    return df[col]


def _atr_pct(df: pd.DataFrame, window: int) -> pd.Series:
    """取 atr_pct_{window}：優先用 parquet 既有欄位，無則用 _01_data 的 calculate_atr_pct 補。"""
    col = f"atr_pct_{window}"
    if col not in df.columns:
        calculate_atr_pct(df, window)
    return df[col]


# 回測結果輸出目錄（策略同目錄底下 ./result，已於 .gitignore 排除）
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")

# 偵測不到資料欄位時的預設 MA 期數
DEFAULT_PERIODS = (5, 10, 20, 50, 60, 120, 200)

# 標準回測區間：後續測試一律以此為主（掐掉 2000 殘月與 2026 未滿年，資料較穩定）；可用 --start/--end 覆蓋
DEFAULT_START = "2001-01-01"
DEFAULT_END = "2025-12-31"


class SingleMAStrategy(VbtSingleStrategy):
    """
    單一均線突破。設定 MA_PERIOD 後即可用。只描述「判定日」訊號，
    「隔日開盤成交 + 台股費用 / 稅 / tick」全由基底 VbtSingleStrategy 處理。

    優化以「註解切換」管理（見 buy_signal）：各「# 優化 #N」獨立一行可疊加，
      取消註解＝啟用該優化、註解＝關閉；全部註解 = baseline。
    """

    MA_PERIOD = 20

    def add_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """均線/CMF/ATR% 引用 _01_data 指標；量能均線/K線/ADX 無對應 indicators 故自算。"""
        n = self.MA_PERIOD
        df["ma"] = _ma(df, n)                                   # 引用 sma（_01_data）
        df["vol_ma5"] = df["volume"].rolling(5).mean()         # 量能均線無對應 indicators，保留自算
        df["vol_ma20"] = df["volume"].rolling(20).mean()
        df["cmf"] = _cmf(df, 20)                                # 引用 cmf_20（_01_data）；opt3 用
        # opt5：N 日均量（帶量比較基準，N=MA 期數）+ 突破前整理旗標
        df["vol_man"] = df["volume"].rolling(n).mean()
        m = min((n + 1) // 2, 10)   # 需要的收斂日數 = ceil(N/2)，上限 10（避免長均線門檻過高）
        # 波動率 = ATR%，窗口隨 MA「等比放大」(14:20)：MA<=20 用 14、MA>20 用 round(MA*14/20)。引用 calculate_atr_pct。
        atr_win = 14 if n <= 20 else round(n * 14 / 20)
        vol = _atr_pct(df, atr_win)                            # ATR%（窗口隨 MA 放大；_01_data）
        contract_below = (vol < vol.shift(1)) & (df["close"] < df["ma"])   # 當日波動較昨日收斂 且 收盤在 MA 下方
        # 累計（非連續）：突破前 N 日內，contract_below 累計達 m 日即算「整理過」
        df["consolidated"] = (contract_below.rolling(n).sum().shift(1) >= m).fillna(False)
        # opt6：不要求波動收斂，只看「突破前在 MA 下方累計 >= ceil(N/2) 日」（長期在均線下）
        df["below_enough"] = ((df["close"] < df["ma"]).rolling(n).sum().shift(1) >= (n + 1) // 2)
        # opt7：突破那根 K 線的強度（二選一）—— 長紅(開→收>=5%) 或 跳空開在均線之上
        df["long_red"] = df["close"] >= df["open"] * 1.05
        df["gap_over_ma"] = (df["open"] > df["ma"]) & (df["open"] > df["close"].shift(1))
        # opt10：ADX(14)（Wilder）—— 自算不靠預存欄位。ADX<25 表趨勢弱/盤整
        high, low, close = df["high"], df["low"], df["close"]
        prev_c = close.shift(1)
        up = high.diff(); dn = -low.diff()
        plus_dm = up.where((up > dn) & (up > 0), 0.0)
        minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
        tr14 = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
        p = 14
        atr_w = tr14.ewm(alpha=1 / p, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1 / p, adjust=False).mean() / atr_w
        minus_di = 100 * minus_dm.ewm(alpha=1 / p, adjust=False).mean() / atr_w
        dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).fillna(0.0)
        df["adx"] = dx.ewm(alpha=1 / p, adjust=False).mean()
        return df

    def buy_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        進場「判定日」訊號（基底會自動延到隔日開盤成交）。

        baseline：當日收盤上穿 MA（突破當天判定）。
        優化 #1（雙日確認）：突破隔日收盤仍站上 MA 才判定（多等一天，過濾一日假突破）。
        優化 #2（量能濾網，測試後不採用、預設註解）：取消註解＝只留流動性門檻 5 日均量 > 100 萬股。
        優化 #3（CMF 資金流向，測試後不採用、預設註解）：取消註解＝20 日 CMF > 0.1。
        優化 #5（正統反轉突破，目前測試中、預設啟用）：突破 + 突破前整理 + 帶量。
            整理 = 突破前 N(=MA 期數) 日內，累計 min(ceil(N/2),10) 日「波動(ATR%,窗口隨 MA 放大)收斂
                  且 收盤在 MA 下方」；帶量 = 突破當日 volume > N 日均量。（opt5 啟用時 opt1 暫關）

        ▶ 切換：每個「# 優化 #N」獨立一行，可各自註解／取消註解疊加；全註解 = baseline。
        """
        above = df["close"] > df["ma"]
        signal = above & ~above.shift(1, fill_value=False)   # baseline：突破當日（上穿 MA）
        signal = signal & (df["long_red"] | df["gap_over_ma"])  # 優化 #7：突破那根 K 線需 長紅(開→收>=5%) 或 跳空過均線
        signal = signal & (df["adx"].shift(1) < 20)  # 優化 #10：突破前一根 K 的 ADX(14) < 20（盤整/弱趨勢後突破）。與 #7 同開＝opt11 組合
        # signal = signal & (df["vol_ma5"] > 300_000)  # 優化 #12：5 日均量 > 30 萬股(=300張) 流動性門檻（測試用，暫關回 opt11）
        # signal = above & signal.shift(1, fill_value=False)   # 優化 #1：雙日確認（疊 opt7 後反不如 opt7 單獨，預設關。取消註解＝強突破隔日仍站上才買 = opt8）
        # signal = signal.shift(1, fill_value=False) & (df["open"] > df["close"]) & above   # 優化 #9：opt7 隔天收黑但沒跌破 MA → 再隔一天買（測試後不採用，預設關）
        # 以下為測試後不採用的進場濾網（皆輸 baseline，保留可重啟）：
        # signal = signal & (df["vol_ma5"] > 1_000_000)        # 優化 #2：量能濾網（不採用，見 reference 2026-06-15 opt2）
        # signal = signal & (df["cmf"] > 0.1)                  # 優化 #3：CMF（不採用，見 reference 2026-06-15 opt3）
        # signal = signal & df["consolidated"] & (df["volume"] > df["vol_man"])  # 優化 #5：整理+帶量（不採用）
        # signal = signal & df["below_enough"] & (df["volume"] > df["vol_man"])  # 優化 #6：在 MA 下方+帶量（不採用）
        return signal

    def sell_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        出場「判定日」訊號（基底會自動延到隔日開盤成交）。

        baseline：今天收盤下穿 MA（昨收 >= MA、今收 < MA）。
        優化 #4（保留下穿出場 + 距離連 3 日縮水也賣）：原下穿出場保留，另加「與 MA 距離
            dist=close−ma 連續 3 日縮水（dist 一日比一日小）」也賣——鈍化版，避免初版「一停就跑」。

        ▶ 切換：把標「# 優化 #4」那一行【取消註解】→ 疊加 3 日縮水出場；【註解】→ 只用下穿出場。
        """
        below = df["close"] < df["ma"]
        signal = below & ~below.shift(1, fill_value=False)   # baseline 出場：下穿 MA（保留）
        # 優化 #4（測試後不採用，預設註解；取消註解＝下穿 或 距離連 3 日縮水也賣。詳見 reference 2026-06-15 opt4）
        # dist = df["close"] - df["ma"]
        # shrink = dist < dist.shift(1)
        # signal = signal | (shrink & shrink.shift(1, fill_value=False) & shrink.shift(2, fill_value=False))  # 優化 #4：距離連 3 日縮水也賣
        return signal


def detect_ma_periods(df: pd.DataFrame) -> list:
    """從資料欄位 sma_<N> 萃取 MA 期數（排序）；無則回 DEFAULT_PERIODS。"""
    periods = []
    for col in df.columns:
        match = re.fullmatch(r"sma_(\d+)", str(col))
        if match:
            periods.append(int(match.group(1)))
    return sorted(periods) or list(DEFAULT_PERIODS)


def run_all_periods(df: pd.DataFrame, periods=None,
                    initial_cash: float = 100_000, split_cash: float = 10_000) -> dict:
    """
    對每個 MA 期數跑單一均線突破，回傳 {period: summary}。
    periods 不給 → 自動偵測資料中的 MA 期數。
    """
    periods = periods or detect_ma_periods(df)
    results = {}
    for period in periods:
        strat = SingleMAStrategy(initial_cash=initial_cash, split_cash=split_cash)
        strat.MA_PERIOD = period
        results[period] = strat.run(df)["summary"]
    return results


def main(argv) -> int:
    """CLI：指定資料夾，固定單一 MA 期數，掃全部股票各自獨立回測並彙總。"""
    parser = argparse.ArgumentParser(description="單一均線突破：資料夾批次回測")
    parser.add_argument("folder", help="OHLCV parquet 資料夾路徑")
    parser.add_argument("--ma", type=int, default=20, help="單一 MA 期數（預設 20）")
    parser.add_argument("--variant", choices=("baseline", "opt1", "opt2", "opt2b", "opt3", "opt4", "opt5", "opt6", "opt7", "opt8", "opt9", "opt10", "opt10b", "opt11", "opt12", "opt12b"), default="opt1",
                        help="輸出資料夾分流（result/single_ma/<variant>/）；行為切換靠 buy_signal "
                             "內『# 優化 #1』那行的註解，--variant 只決定寫去哪，兩者請保持一致")
    parser.add_argument("--trades", action="store_true",
                        help="另存逐筆交易紀錄（預設不存，只出彙總）")
    parser.add_argument("--start", default=DEFAULT_START,
                        help=f"起始日 YYYY-MM-DD（預設標準區間 {DEFAULT_START}）")
    parser.add_argument("--end", default=DEFAULT_END,
                        help=f"結束日 YYYY-MM-DD（預設標準區間 {DEFAULT_END}）")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 檔（測試用）")
    args = parser.parse_args(argv[1:])

    strat = SingleMAStrategy()
    strat.MA_PERIOD = args.ma

    result = batch.run_folder(strat, args.folder,
                              start=args.start, end=args.end, limit=args.limit)
    # 結果分流：baseline / opt1 各自獨立子資料夾（對照用、互不覆蓋）；
    # 注意 variant 只是輸出位置，真正行為由 buy_signal 的「# 優化 #1」註解決定，務必一致
    out_dir = os.path.join(RESULT_DIR, "single_ma", args.variant)
    label = f"single_ma_{args.ma}"
    written = batch.write_results(result, out_dir, label, write_trades=args.trades)

    agg = result["aggregate"]
    print(f"=== 全市場單一均線突破 MA={args.ma}（variant={args.variant}）===")
    print(f"參與股票數: {agg['參與股票數']}（失敗 {agg['失敗檔數']} 檔）")
    print(f"交易次數: {agg['交易次數']}")
    print(f"勝率(%): {agg['勝率(%)']}")
    print(f"期望報酬值(EV): {agg['期望報酬值(EV)']}")
    print(f"總獲利: {agg['總獲利']:.0f}")
    print(f"輸出目錄: {out_dir}")
    for path in written:
        print(f"  - {os.path.basename(path)}")
    if result["failed"]:
        print(f"失敗檔（前 10）: {result['failed'][:10]}")
    print(f"⚠️ 行為由 buy_signal 內『# 優化 #1』那行的註解決定；請確認與 --variant={args.variant} 一致")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
