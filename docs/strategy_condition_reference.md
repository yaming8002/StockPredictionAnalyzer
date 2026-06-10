# 交易策略條件統計歸納

> 目的：統整所有現有策略的買賣條件與參數變化，作為開發新策略的參考基礎。
> 更新日期：2026-04-30

---

## 一、條件模組命名表

所有條件拆解為獨立的「模組」，每個模組有名稱與說明，方便組合與討論。

### 買入條件模組

| 模組名稱 | 說明 | 常用參數 |
|----------|------|----------|
| **TREND_ALIGN** | 多頭排列：close > SMA_50 且 SMA_120 > SMA_200 | — |
| **TREND_ALIGN_STRONG** | 強多頭排列：SMA_50 > SMA_120 > SMA_200 | — |
| **SMA_CROSS** | 均線黃金交叉：SMA_20 由下往上穿越 SMA_50 | 期數：20/50 |
| **SMA_ANGLE** | 均線角度向上：arctan(slope) > 門檻 | 門檻：0°、20° |
| **SMA_PULLBACK** | 股價回踩均線後回升：4 日內距均線 < 3% | 距離：3% |
| **CMF_BUY** | 資金淨流入：CMF > 門檻 | 門檻：0.05、0.1 |
| **ZIGZAG_BREAK** | 突破 ZigZag 高點雙確認：close >= turn_high 且前日 close >= turn_high | 回撤：2%、3% |
| **PIVOT3_BREAK** | 3-Bar Pivot 突破確認 | 門檻：1.5% |
| **UPPER_SHADOW** | 上影線過濾：上影線 <= 實體 × 2 | 倍數：2 |
| **VOL_LIQUIDITY** | 流動性門檻：volume_ma5 >= 最小值 | 台股：1M；美股：500K |
| **VOL_AMPLIFY** | 放量確認：成交量 > ma5 × 倍數 | 倍數：1.5、2.0 |
| **CHOP_FILTER** | 排除盤整：Choppiness < 門檻 | 門檻：31 |
| **ADX_TREND** | 趨勢強度確認：ADX > 門檻 | 門檻：20 |
| **RSI_MOMENTUM** | RSI 動能確認：RSI > 門檻 | 門檻：50 |
| **ATR_RR** | 風報比篩選：估算報酬 >= 風險 × 倍數（以 ATR 估算） | 倍數：1.5；ATR 目標：4x |
| **MA_EXTENSION** | 均線乖離過濾：close / SMA_50 <= 上限（排除追高） | 上限：1.12 |
| **VCP_FILTER** | 波動率收縮確認：ATR_5 / ATR_20 < 門檻 | 門檻：0.7 |
| **PRICE_CAP** | 股價上限（排除高價股） | 台股：60 元 |
| **BOLLINGER_LOW** | 布林帶下緣突破：前日收盤 < 下軌，當日反彈 | — |
| **DONCHIAN_HIGH** | 唐奇安通道突破：創 N 日新高 | 窗口：20 日 |
| **VOLUME_CLIMAX_ENTRY** | 大量下跌後量縮走平進場 | 量倍：2x；走平期：3日；觀察期：10日 |

---

### 賣出條件模組

| 模組名稱 | 說明 | 常用參數 |
|----------|------|----------|
| **STOP_ZIGZAG_LOW** | 跌破買入時的 ZigZag turn_low（硬停損） | — |
| **STOP_CURRENT_LOW** | 跌破當前 ZigZag turn_low（結構轉弱） | — |
| **STOP_ATR** | ATR 動態止損：entry - ATR × 倍數 | 倍數：2.5 |
| **STOP_ATR_TRAIL** | ATR 移動止損：最高收盤 - ATR × 倍數 | 倍數：2.5 |
| **CMF_HOT** | CMF 過熱出場：CMF > 門檻 | 門檻：0.9 |
| **CMF_OUTFLOW** | CMF 資金流出：CMF < 門檻 | 門檻：-0.05、-0.1 |
| **CLIMAX_TOP** | 量價同步創 N 日新高（頂部訊號） | 窗口：60 日；持有：>= 5 日 |
| **CLIMAX_CONFIRM** | Climax 隔日確認：收盤 < Climax 日低點 | — |
| **CLIMAX_TIGHT_STOP** | Climax 後止損上移至當日最低（緊止損） | — |
| **TRAILING_PROFIT** | 漸進追蹤停利（依獲利比例分級） | L1:10%/92%；L2:20%/95%；Climax:97% |
| **RSI_REVERSAL** | RSI 動能反轉：曾 >= 60 後跌至 <= 40 | — |
| **TIME_STOP** | 時間止損：持有 > N 日 + 動能弱 | N=45；RSI<50；CMF<0 |
| **VOL_SHRINK** | 連續 N 日極少量出場：量 < MA20 × 比例 | N=5；比例：40% |
| **VOL_DIVERGE** | 量價背離出場：新高點成交量 < 前高 × 比例 | 比例：0.7；持有：>= 5 日 |
| **VOL_EXTREME_LOW** | 絕對極少量出場：單日量 < 門檻 | 門檻：30,000 股 |
| **SMA_DEATH_CROSS** | 均線死亡交叉：SMA_20 < SMA_50 | — |
| **BOLLINGER_REVERSAL** | 布林帶：前日收盤 > 今日收盤 | — |
| **DONCHIAN_LOW** | 唐奇安通道跌破：創 N 日新低 | 窗口：10 日 |
| **SMA_BREAK** | 跌破均線：close < SMA_N | — |

---

## 二、策略條件組合矩陣

各策略使用的買賣條件組合一覽。✓ = 使用，— = 未使用，`*` = 修改版本。

### 買入條件矩陣

| 策略 | TREND_ALIGN | CMF_BUY | ZIGZAG_BREAK | UPPER_SHADOW | VOL_LIQUIDITY | VOL_AMPLIFY | ADX | VCP | MA_EXT | RSI | PRICE_CAP |
|------|:-----------:|:-------:|:------------:|:------------:|:-------------:|:-----------:|:---:|:---:|:------:|:---:|:---------:|
| dow_strategy_CMF_2026_new | ✓ | ✓ 0.1 | ✓ 2% | ✓ | ✓ 1M | — | — | — | — | — | ✓ 60 |
| dow_no_cmf | ✓ | — | ✓ 2% | ✓ | ✓ 1M | — | — | — | — | — | ✓ 60 |
| dow_pro | ✓ strong | ✓ 0.05 | ✓ 3% | ✓ | ✓ 1M | — | ✓ 20 | — | — | — | — |
| dow_ma_extension | ✓ | ✓ 0.1 | ✓ 2% | ✓ | ✓ 1M | — | — | — | ✓ 1.12 | — | ✓ 60 |
| dow_vol_contraction | ✓ | ✓ 0.1 | ✓ 2% | ✓ | ✓ 1M | — | — | ✓ 0.7 | — | — | ✓ 60 |
| dow_momentum | ✓ | ✓ 0.1 | ✓ 2% | ✓ | ✓ 1M | — | — | — | — | ✓ 50 | ✓ 60 |
| dow_atr_stop | ✓ | ✓ 0.1 | ✓ 2% | ✓ | ✓ 1M | — | — | — | — | — | ✓ 60 |
| dow_3bar_pivot | ✓ | ✓ 0.1 | ✓ pivot | ✓ | ✓ 1M | — | — | — | — | — | ✓ 60 |
| ma_strategy | — | — | — | — | — | — | — | — | — | — | — |
| single_ma | — | — | — | — | ✓ 1M | ✓ 1.5 | — | — | — | — | — |
| turtle | — | — | — | — | — | — | — | — | — | — | — |
| bollinger | — | — | — | — | — | — | — | — | — | — | — |
| volue | — | — | — | — | — | ✓ 2.0 | — | — | — | — | — |
| us_dow | ✓ | ✓ 0.1 | ✓ 2% | ✓ | ✓ 500K | — | — | — | — | — | — |

### 賣出條件矩陣

| 策略 | STOP_ZZ_LOW | STOP_CURRENT | CMF_HOT | CLIMAX | TRAILING | ATR_STOP | RSI_REV | TIME | VOL_SHRINK | VOL_DIV |
|------|:-----------:|:------------:|:-------:|:------:|:--------:|:--------:|:-------:|:----:|:----------:|:-------:|
| dow_CMF_2026_new | ✓ | ✓ | ✓ 0.9 | — | — | — | — | — | — | — |
| dow_climax | ✓ | ✓ | ✓ 0.9 | ✓ 60d | — | — | — | — | — | — |
| dow_climax_confirm | ✓ | ✓ | ✓ 0.9 | ✓+確認 | — | — | — | — | — | — |
| dow_climax_confirm_low | ✓ | ✓ | ✓ 0.9 | ✓ 隔日低 | — | — | — | — | — | — |
| dow_climax_tightstop | ✓ | ✓ | ✓ 0.9 | ✓ 緊止損 | — | — | — | — | — | — |
| dow_trailing_profit | ✓ | ✓ | ✓ 0.9 | ✓ | ✓ 分級 | — | — | — | — | — |
| dow_atr_stop | ✓ | ✓ | — | — | — | ✓ 2.5 | — | — | — | — |
| dow_pro | ✓ | — | — | — | — | ✓ trail | — | — | — | — |
| dow_momentum | ✓ | ✓ | ✓ 0.9 | — | — | — | ✓ | ✓ 45d | — | — |
| dow_volshrink5 | ✓ | ✓ | ✓ 0.9 | — | — | — | — | — | ✓ 5d/40% | — |
| dow_vol_diverge | ✓ | ✓ | ✓ 0.9 | — | — | — | — | — | — | ✓ 0.7 |
| dow_CMF_strong_sell | ✓ | ✓ | ✓ 0.9 | — | — | — | — | — | ✓ 30K | — |
| ma_strategy | — | — | — | — | — | — | — | — | — | — |
| turtle | — | — | — | — | — | — | — | — | — | — |

---

## 三、技術指標使用頻率統計

| 指標 | 使用策略數 | 典型參數 | 用途 |
|------|:----------:|----------|------|
| ZigZag | ~26 | 2%（主流）、3%（Pro）、1.5%（Pivot） | 轉折點偵測、買賣訊號基礎 |
| SMA | ~25 | 20、50、120、200 | 趨勢判斷、均線排列 |
| CMF | ~23 | 窗口 20；買 0.1、賣 0.9 | 資金流入/流出 |
| Volume / Vol_MA5 | ~24 | 1M（台股）、500K（美股） | 流動性、放量確認 |
| ATR | ~7 | 5、14（標準） | 止損距離、波動率比較 |
| RSI | ~3 | 窗口 14；門檻 40/50/60 | 動能確認、出場 |
| ADX | 1 | 窗口 14；門檻 20 | 趨勢強度 |
| OBV | 1 | — | 量能平衡（次要） |
| Choppiness | 2 | 窗口 14；門檻 31 | 排除盤整行情 |
| Bollinger Band | 1 | 標準 20/2 | 極值反彈進場 |
| Donchian Channel | 1 | 高：20 日；低：10 日 | 通道突破 |

---

## 四、參數變化範圍總覽

### ZigZag 回撤門檻
| 數值 | 使用策略 |
|------|----------|
| 1.5% | 3-Bar Pivot |
| **2%** | 絕大多數 DOW 策略（主流） |
| 3% | DOW Pro |

### CMF 閾值
| 情境 | 門檻 | 策略 |
|------|------|------|
| 買入 | 0.05 | DOW Pro |
| 買入 | **0.1** | 主流 DOW 策略 |
| 賣出（流入過熱） | **0.9** | 主流 DOW 策略 |
| 賣出（流出確認） | -0.05 | DOW Pro |
| 賣出（流出確認） | -0.1 | DOW ATR Stop |
| 無 CMF | — | dow_no_cmf 系列 |

### SMA 組合
| 組合 | 策略 |
|------|------|
| SMA_20 + SMA_50 | MA 策略 |
| SMA_20 + SMA_50 + SMA_60 + SMA_120 + SMA_200 | Single MA |
| close > SMA_50 且 SMA_120 > SMA_200 | DOW CMF 2026（主流） |
| SMA_50 > SMA_120 > SMA_200 | DOW Pro、部分 DOW 系列 |

### 流動性門檻
| 市場 | volume_ma5 最小值 |
|------|-------------------|
| 台股（主流） | 1,000,000 |
| 美股 | 500,000 |
| 極少量出場觸發 | < 30,000 股（dow_strong_sell） |
| 極少量比例觸發 | < MA20 × 40%，連續 5 日（dow_volshrink5） |

### ATR 止損倍數
| 類型 | 倍數 | 策略 |
|------|------|------|
| 固定止損 | 2.5x | DOW ATR Stop |
| 移動止損 | 2.5x（從最高收盤往下） | DOW Pro |
| 估算目標 | 4x（估算潛在報酬） | DOW Pro |

### Climax Top 參數
| 參數 | 常見值 |
|------|--------|
| 回望窗口 | 60 日 |
| 最少持有 | 5 日（標準）、60 日（60d 版） |
| 確認方式 | 即時出場、隔日確認、緊止損、追蹤停利 |

### 追蹤停利分級
| 獲利達到 | 追蹤因子 | 回撤出場 |
|----------|----------|----------|
| < 10% | 無追蹤 | — |
| >= 10% | 92% | 8% 回撤 |
| >= 20% | 95% | 5% 回撤 |
| Climax 後 | 97% | 3% 回撤 |

---

## 五、DOW 系列演化樹

```
dow_strategy_CMF_2026_new（基準）
│
├── 賣出增強
│   ├── dow_climax_strategy               (+Climax Top 即時出場)
│   ├── dow_climax_confirm_strategy       (+Climax 隔日確認)
│   ├── dow_climax_confirm_low_strategy   (+Climax 跌破當日低點確認)
│   ├── dow_climax_tightstop_strategy     (+Climax 止損上移)
│   ├── dow_trailing_profit_strategy      (+漸進追蹤停利)
│   ├── dow_atr_stop_strategy             (+ATR 動態止損，移除 CMF_HOT)
│   ├── dow_momentum_strategy             (+RSI 動能反轉, 時間止損)
│   ├── dow_volshrink5_strategy           (+5 日極少量出場)
│   ├── dow_vol_diverge_strategy          (+量價背離出場)
│   └── dow_strategy_CMF_strong_sell      (+絕對極少量強制出場)
│
├── 買入增強
│   ├── dow_pro_strategy                  (+ADX, 移動止損, 風報比)
│   ├── dow_ma_extension_strategy         (+均線乖離上限 1.12)
│   └── dow_vol_contraction_strategy      (+VCP 波動率收縮確認)
│
├── 移除 CMF
│   ├── dow_no_cmf_strategy               (移除 CMF 過濾)
│   └── dow_no_cmf_strategy_60d           (移除 CMF + Climax 60d)
│
├── 轉折方法替換
│   ├── dow_strategy_3bar_pivot           (ZigZag → 3-Bar Pivot)
│   └── dow_strategy_zigzag_v2            (ZigZag Bug 修正版)
│
└── 市場擴展
    └── us_dow_strategy                   (台股 → 美股)
```

---

## 六、非 DOW 策略獨立條件

| 策略 | 核心邏輯 | 特色條件 |
|------|----------|----------|
| **MA 策略** | SMA_20/50 黃金/死亡交叉 | Choppiness < 31 |
| **Single MA** | 均線角度 + 回踩確認 | 角度 arctan > 20°；距離 < 3% |
| **Turtle** | 唐奇安通道突破 | 創 20 日新高買；跌破 10 日新低賣 |
| **Bollinger** | 布林帶下緣反彈 | 無量能或均線條件 |
| **Volume** | 大量下跌後量縮走平 | 觀察期機制（10 日窗口） |

---

## 七、未被使用但可探索的條件組合

根據現有模組，以下組合尚未嘗試，可作為新策略開發的方向：

| 想法 | 涉及模組 | 潛在優勢 |
|------|----------|----------|
| DOW + CHOP_FILTER | ZIGZAG_BREAK + CHOP_FILTER | 避免在橫盤中反覆假突破 |
| DOW + SMA_ANGLE | ZIGZAG_BREAK + SMA_ANGLE | 確保均線仍在加速向上 |
| MA_CROSS + CMF_BUY | SMA_CROSS + CMF_BUY | 均線交叉時需資金流入確認 |
| BOLLINGER + VOL_AMPLIFY | BOLLINGER_LOW + VOL_AMPLIFY | 布林帶反彈需有量 |
| DONCHIAN + TREND_ALIGN | 突破 + 多頭排列 | 趨勢中的通道突破更可靠 |
| TRAILING_PROFIT + VCP | 持倉管理 + 低波動進場 | 進場時波動小，停利空間更清晰 |
| 均線策略 + ATR_RR | SMA_CROSS + ATR_RR | 均線交叉策略加入風報比篩選 |

---

## 八、新策略開發建議框架

開發新策略時，建議從以下三層結構各選 1-3 個模組組合：

```
【進場條件】（決定何時買）
  趨勢確認層：TREND_ALIGN / TREND_ALIGN_STRONG / SMA_ANGLE
  訊號觸發層：ZIGZAG_BREAK / SMA_CROSS / DONCHIAN_HIGH / BOLLINGER_LOW
  過濾確認層：CMF_BUY / ADX_TREND / VCP_FILTER / CHOP_FILTER / RSI_MOMENTUM
  量能確認層：VOL_LIQUIDITY / VOL_AMPLIFY
  品質過濾層：UPPER_SHADOW / MA_EXTENSION / PRICE_CAP / ATR_RR

【持倉管理】（決定持多久）
  基礎止損：STOP_ZIGZAG_LOW / STOP_ATR
  動態止損：STOP_ATR_TRAIL / TRAILING_PROFIT
  頂部偵測：CLIMAX_TOP / CLIMAX_CONFIRM / VOL_DIVERGE

【出場條件】（決定何時賣）
  資金訊號：CMF_HOT / CMF_OUTFLOW
  動能衰退：RSI_REVERSAL / TIME_STOP
  量能萎縮：VOL_SHRINK / VOL_EXTREME_LOW
  結構轉弱：STOP_CURRENT_LOW / SMA_BREAK
```
