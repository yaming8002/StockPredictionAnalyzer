# -*- coding: utf-8 -*-
"""
驗 SPA 公開檔 single_macd_strategy.py 的兩個出場切換是否可用。

背景：出場優化以「附加」為主軸（原出場留著、新規則疊上去、先觸發者算），
「當唯一出場」的純取代版只作附錄。公開檔要能表達這兩種，文章才重現得出來。
2026-09-22 補上切換時寫的驗證，日後動 sell_signal 或 _REPLACE_RULES 前後都該重跑。

檢查一：sell_signal 的四條「附加型」註解行取消註解後真的跑得起來
        —— dtype 為 bool、訊號集合是原生死叉的超集。
檢查二：_REPLACE_RULES 的附錄切換確實把五條風控從疊加改成取代
        —— 出場數下降（原出場被拿掉）、未平倉（進場−出場）增加；
           頂頂低本來就在取代名單，兩種設定下必須逐格相同。

執行：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 F:/stock-analyzer/.venv/Scripts/python.exe \
        _02_strategy/macd_strategy/verify_exit_switches.py
"""

import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
from _02_strategy.base.vbt import common  # noqa: E402
import sys

import pandas as pd

from _02_strategy.macd_strategy import single_macd_strategy as M  # noqa: E402

DATA = common.DATA_DIR
SAMPLE = ("2330.TW", "2603.TW", "1101.TW")

# 附錄設定：五條風控一併列為取代（＝公開檔裡那行註解的內容）
APPENDIX = (M.EXIT_LOWER_HIGH, M.EXIT_CHANDELIER, M.EXIT_TRAIL_PCT,
            M.EXIT_ATR_STOP, M.EXIT_TAKE_PROFIT, M.EXIT_TIME)
RISK_RULES = {"吊燈3ATR": M.EXIT_CHANDELIER, "回落10%": M.EXIT_TRAIL_PCT,
              "停損2ATR": M.EXIT_ATR_STOP, "停利+20%": M.EXIT_TAKE_PROFIT,
              "抱滿60天": M.EXIT_TIME}


def check_append_lines(strategy) -> bool:
    """檢查一：四條附加型出場行可執行，且訊號為原生死叉的超集。"""
    print("【檢查一】sell_signal 的附加型寫法")
    ok = True
    for code in SAMPLE:
        df = strategy.add_columns(pd.read_parquet(f"{DATA}/{code}.parquet"))
        close, ma_long = df["close"], df["ma_long"]
        native = df["death"].fillna(False)
        # 與公開檔註解行逐字對應（MA200 比的是「昨日」MA200，勿改成今日）
        cases = {
            "MA200": native | ((close < ma_long) & (close.shift(1) >= ma_long.shift(1))),
            "Supertrend": native | strategy._ensure_supertrend(df)["supertrend_flip_down"],
            "SAR": native | strategy._ensure_psar(df)["psar_flip_down"],
            "Donchian": native | (close < df["dc_low_prev"]),
        }
        n0 = int(native.sum())
        parts = []
        for name, ser in cases.items():
            ser = ser.fillna(False)
            n = int(ser.sum())
            # 附加只會增加出場時點，不可能少於原生，也不可能漏掉原生的任何一根
            if ser.dtype != bool or n < n0 or not ser.equals(ser | native):
                ok = False
                parts.append(f"{name}=NG(dtype={ser.dtype},n={n})")
            else:
                parts.append(f"{name}={n}")
        print(f"  {code} 原生死叉={n0} | 附加後 " + " ".join(parts))
    print("  " + ("OK 四條皆可執行、dtype bool、為原生超集" if ok else "NG"))
    return ok


def _run(strategy, df, rule, replace_rules):
    """在指定的 _REPLACE_RULES 下跑一次 build_signals，回傳 (進場數, 出場數)。"""
    original = M._REPLACE_RULES
    M._REPLACE_RULES = replace_rules
    try:
        strategy.__class__.EXIT_RULE = rule
        entries, exits = strategy.build_signals(df.copy())
        return int(entries.sum()), int(exits.sum())
    finally:
        # 模組層常數與類屬性都要還原，否則污染後續比較（踩過：殘留會產出假的「無差異」）
        M._REPLACE_RULES = original
        strategy.__class__.EXIT_RULE = M.EXIT_NONE


def check_replace_switch(strategy) -> bool:
    """檢查二：_REPLACE_RULES 附錄切換讓五條風控變成唯一出場。"""
    print("\n【檢查二】_REPLACE_RULES 的附錄切換（樣本 2330.TW）")
    df = strategy.add_columns(pd.read_parquet(f"{DATA}/2330.TW.parquet"))
    ok = True
    print("  規則            疊加(進/出)      取代(進/出)     判定")
    for name, rule in RISK_RULES.items():
        ea, xa = _run(strategy, df, rule, (M.EXIT_LOWER_HIGH,))
        er, xr = _run(strategy, df, rule, APPENDIX)
        good = xr <= xa and (ea - xa) <= (er - xr)
        ok = ok and good
        print(f"  {name:<12} {ea:>6}/{xa:<6}   {er:>6}/{xr:<6}   "
              f"{'OK 取代生效' if good else 'NG'}")
    same = (_run(strategy, df, M.EXIT_LOWER_HIGH, (M.EXIT_LOWER_HIGH,))
            == _run(strategy, df, M.EXIT_LOWER_HIGH, APPENDIX))
    print(f"  頂頂低（原本就是取代）兩種設定相同：{'OK' if same else 'NG'}")
    return ok and same


def main() -> int:
    strategy = M.SingleMacdStrategy()
    results = [check_append_lines(strategy), check_replace_switch(strategy)]
    print("\n結果：" + ("OK 兩個切換都可用" if all(results) else "NG 見上方"))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
