"""
驗證 MACD 掃描 driver 的設定層（macd_variants.MacdVariant）
=============================================================
重建（一）～（九）篇的掃描 driver 時，MacdVariant 多了 mix 母體與 death／zero_down 兩條出場。
這支檢查五件事，**任何一項不過就回傳非 0**：

一、教學主體對照：把 single_macd_strategy.py 的註解切換行**用程式改寫原始碼**（取消註解／加註解），
    exec 成一個獨立模組，跑出的逐筆交易必須與對應的 MacdVariant 設定逐筆相同。
    涵蓋：矩陣 9 格、無門檻基準線 3 格、mix、九條濾網 × 4 母體、四條向量化出場的附加／取代、
    五條風控的疊加／純取代（EXIT_RULE ＋ _REPLACE_RULES 附錄行）、頂頂低（取代）。
    ⚠️ 頂頂低「附加」在策略檔沒有對應的註解狀態（_REPLACE_RULES 預設把它列為取代），無從對照。
二、既有設定回歸：macd_combo／macd_exit_replace／macd_mc_trades／_03 verify_multi_macd 用到的
    每一組設定，改版前後的 MacdVariant 逐筆交易必須相同。改版前＝`git show <ref>:macd_variants.py`
    （預設 HEAD），並把那一行 `self.buy_signal(df)` 補成 `self.entry_signal(df)`——HEAD 還沒有
    2026-10-08 的交易區間改動，不補的話差異來自引擎改版、不是本次改動。
三、新設定的恆等式：(交叉, death 取代)≡(交叉, 原生)、(零軸, zero_down 取代)≡(零軸, 原生)、
    (背離, death 取代)≡(背離, 原生)、(mix, 原生)≡(交叉, zero_down 取代)。
四、生效率診斷：macd_exit_sweep 的 _scan_fired 與策略檔 _scan_path_exits 訊號逐檔相同
    （rule_share 內建檢查，不同就 raise），且狀態機重數的平倉數＝vbt 平倉數。
五、既有的 verify_exit_switches（策略檔兩個出場切換可用）。

執行：
    python _02_strategy/macd_strategy/verify_macd_sweeps.py [--limit 40] [--ref HEAD]
"""
import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import time
import types

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd  # noqa: E402

from _02_strategy.base.vbt.common import DEFAULT_END, DEFAULT_START  # noqa: E402
from _02_strategy.macd_strategy import single_macd_strategy as M  # noqa: E402
from _02_strategy.macd_strategy import verify_exit_switches  # noqa: E402
from _02_strategy.macd_strategy.macd_exit_sweep import rule_share  # noqa: E402
from _02_strategy.macd_strategy.macd_sweep import (  # noqa: E402
    make_variant, prepare, variant_trades)
from _02_strategy.macd_strategy.macd_variants import MacdVariant  # noqa: E402

STRAT_PATH = M.__file__
HERE = os.path.dirname(os.path.abspath(__file__))

# ── 註解切換行的定位字串（每個都必須在原始碼裡剛好出現一次）─────────────
BUY_LINE = {"cross": "# 基礎 A 交叉：黃金交叉", "zero": "# 基礎 B 零軸：DIF 上穿 0",
            "div": "# 基礎 C 背離：純背離（不靠交叉）"}
FILTER_LINE = {"adx25": "①趨勢強度 ADX>25", "rsi": "②RSI<50 且上升", "volume": "③放量 1.5 倍",
               "hist_rising": "④柱狀圖連兩根遞增", "cmf": "⑤量能 CMF>0",
               "ma200": "⑥趨勢過濾 收盤>MA200", "align": "⑦均線多頭排列 5>20>60",
               "high250": "⑧創 250 日新高", "gap": "⑨跳空（開盤>昨日最高）"}
LIQ_LINE = "# 優化：流動性（成交金額 > 1,000 萬）"
SELL_LINE = {"death": "# 基礎 A 交叉 / C 背離：死亡交叉", "zero_down": "# 基礎 B 零軸：DIF 下穿 0",
             "beardiv": "# 矩陣用：頂背離出場"}
APPEND_LINE = {"ma200": "# ＋跌破 MA200", "supertrend": "# ＋Supertrend(10,3) 翻空",
               "psar": "# ＋拋物線 SAR 翻空", "donchian": "# ＋跌破前 20 日最低"}
REPLACE_LINE = {"ma200": "# 取代：跌破 MA200", "supertrend": "# 取代：Supertrend(10,3) 翻空",
                "psar": "# 取代：拋物線 SAR 翻空", "donchian": "# 取代：跌破前 20 日最低"}
RULE_LINE = {"lowerhigh": "# EXIT_RULE = EXIT_LOWER_HIGH", "chandelier": "# EXIT_RULE = EXIT_CHANDELIER",
             "trail10": "# EXIT_RULE = EXIT_TRAIL_PCT", "atrstop": "# EXIT_RULE = EXIT_ATR_STOP",
             "takeprofit": "# EXIT_RULE = EXIT_TAKE_PROFIT", "time60": "# EXIT_RULE = EXIT_TIME"}
RULE_DEFAULT = "    EXIT_RULE = EXIT_NONE"
REPLACE_DEFAULT = "_REPLACE_RULES = (EXIT_LOWER_HIGH,)"
REPLACE_APPENDIX = ("# _REPLACE_RULES = (EXIT_LOWER_HIGH, EXIT_CHANDELIER, EXIT_TRAIL_PCT,",
                    "#                   EXIT_ATR_STOP, EXIT_TAKE_PROFIT, EXIT_TIME)")
# 各母體在策略檔裡的「進場基礎行 ＋ 原生出場行」
POP_LINES = {"cross": ("cross", "death"), "zero": ("zero", "zero_down"),
             "div": ("div", "death"), "mix": ("cross", "zero_down")}


# ── 原始碼改寫 ─────────────────────────────────────────────────────
def _find(lines: list, key: str) -> int:
    hits = [i for i, ln in enumerate(lines) if key in ln]
    if len(hits) != 1:
        raise ValueError(f"定位字串「{key}」在策略檔出現 {len(hits)} 次（必須剛好 1 次）")
    return hits[0]


def _uncomment(lines: list, key: str) -> None:
    i = _find(lines, key)
    body = lines[i].lstrip()
    if not body.startswith("# "):
        raise ValueError(f"「{key}」那行原本就不是註解：{lines[i]!r}")
    lines[i] = lines[i][:len(lines[i]) - len(body)] + body[2:]


def _comment(lines: list, key: str) -> None:
    i = _find(lines, key)
    body = lines[i].lstrip()
    if body.startswith("#"):
        raise ValueError(f"「{key}」那行原本就是註解：{lines[i]!r}")
    lines[i] = lines[i][:len(lines[i]) - len(body)] + "# " + body


def toggled_source(pop="cross", filt="none", liq=True, vec_exit=None, vec_mode=None,
                   rule=None, appendix=False, sell=None) -> str:
    """
    依參數改寫 single_macd_strategy.py 的註解切換行，回傳改寫後的原始碼。
    pop 決定進場基礎行與原生出場行；sell 有給就改用該出場行（矩陣的非原生配對）。
    """
    with open(STRAT_PATH, encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    buy, native = POP_LINES[pop]
    sell = sell or native
    if buy != "cross":
        _comment(lines, BUY_LINE["cross"])
        _uncomment(lines, BUY_LINE[buy])
    if filt != "none":
        _uncomment(lines, FILTER_LINE[filt])
    if liq:
        _uncomment(lines, LIQ_LINE)
    if sell != "death":
        _comment(lines, SELL_LINE["death"])
        _uncomment(lines, SELL_LINE[sell])
    if vec_exit:
        _uncomment(lines, (APPEND_LINE if vec_mode == "append" else REPLACE_LINE)[vec_exit])
    if rule:
        _comment(lines, RULE_DEFAULT)
        _uncomment(lines, RULE_LINE[rule])
    if appendix:
        _comment(lines, REPLACE_DEFAULT)
        for key in REPLACE_APPENDIX:
            _uncomment(lines, key)
    return "\n".join(lines)


def load_toggled(src: str, tag: str):
    """
    exec 改寫後的原始碼成獨立模組（不進 sys.modules）。
    numba 的 cache=True 需要實體檔案定位快取，exec 的程式碼沒有，故改成 @njit，再把掃描函式
    換回原模組已編譯好的那支——函式本體沒被改寫（verify 只動註解切換行），換回不影響對照，
    又省掉每個模組各編譯一次。
    """
    mod = types.ModuleType(f"_toggled_{tag}")
    mod.__file__ = STRAT_PATH
    exec(compile(src.replace("@njit(cache=True)", "@njit"), f"<toggled {tag}>", "exec"),
         mod.__dict__)
    mod._scan_path_exits = M._scan_path_exits
    return mod


def _trades(strategy, data: dict) -> pd.DataFrame:
    frames = [strategy.run(df, sid, start=DEFAULT_START, end=DEFAULT_END)["trades"]
              for sid, df in data.items()]
    return pd.concat(frames, ignore_index=True)


def _same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return a.reset_index(drop=True).equals(b.reset_index(drop=True))


def _variant(cls, base, filt, exit_name, mode, liquidity=True):
    v = cls()
    v.BASE, v.FILTER, v.EXIT, v.EXIT_MODE, v.LIQUIDITY = base, filt, exit_name, mode, liquidity
    return v


# ── 檢查一 ───────────────────────────────────────────────────────
def toggle_cases() -> list:
    """(說明, toggled_source 參數, MacdVariant 設定 (base, filt, exit, mode, liq))"""
    cases = []
    for base in ("cross", "zero", "div"):                     # 矩陣 9 格
        for ex in ("death", "zero_down", "beardiv"):
            cases.append((f"矩陣 {base}×{ex}", {"pop": base, "sell": ex},
                          (base, "none", ex, "replace", True)))
    for base in ("cross", "zero", "div"):                     # 無門檻基準線
        cases.append((f"基準線無門檻 {base}", {"pop": base, "liq": False},
                      (base, "none", "native", "replace", False)))
    for pop in ("cross", "zero", "div", "mix"):
        cases.append((f"原生 {pop}", {"pop": pop}, (pop, "none", "native", "replace", True)))
        for f in FILTER_LINE:                                 # 九濾網 × 四母體
            cases.append((f"濾網 {pop}×{f}", {"pop": pop, "filt": f},
                          (pop, f, "native", "replace", True)))
        for ex in APPEND_LINE:                                # 向量化出場：附加
            cases.append((f"附加 {pop}×{ex}", {"pop": pop, "vec_exit": ex, "vec_mode": "append"},
                          (pop, "none", ex, "append", True)))
        for ex in ("chandelier", "trail10", "atrstop", "takeprofit", "time60"):  # 風控疊加
            cases.append((f"疊加 {pop}×{ex}", {"pop": pop, "rule": ex},
                          (pop, "none", ex, "append", True)))
        cases.append((f"取代 {pop}×lowerhigh", {"pop": pop, "rule": "lowerhigh"},
                      (pop, "none", "lowerhigh", "replace", True)))
    for base in ("cross", "zero", "div"):
        for ex in REPLACE_LINE:                               # 向量化出場：取代
            cases.append((f"取代 {base}×{ex}", {"pop": base, "vec_exit": ex, "vec_mode": "replace"},
                          (base, "none", ex, "replace", True)))
        for ex in ("chandelier", "trail10", "atrstop", "takeprofit", "time60"):  # 純取代
            cases.append((f"純取代 {base}×{ex}", {"pop": base, "rule": ex, "appendix": True},
                          (base, "none", ex, "replace", True)))
    return cases


def check_toggles(data: dict) -> bool:
    print("【檢查一】策略檔註解切換 ≡ MacdVariant 設定")
    ok, n = True, 0
    for i, (desc, kw, cfg) in enumerate(toggle_cases()):
        mod = load_toggled(toggled_source(**kw), str(i))
        a = _trades(mod.SingleMacdStrategy(), data)
        b = _trades(make_variant(*cfg), data)
        good = _same(a, b)
        ok &= good
        n += 1
        if not good:
            print(f"  NG {desc}：策略檔 {len(a)} 筆 vs MacdVariant {len(b)} 筆")
    print(f"  {'OK' if ok else 'NG'}｜{n} 種註解狀態")
    return ok


# ── 檢查二 ───────────────────────────────────────────────────────
def legacy_configs() -> list:
    """既有 driver 用到的 (base, filt, exit, mode)；import 各 driver 的常數，不手抄。"""
    from _02_strategy.macd_strategy import macd_combo, macd_exit_replace, macd_mc_trades
    cfgs = set()
    for b in macd_combo.BASES:
        for f in macd_combo.FILTERS:
            for ex in macd_combo.EXITS:
                cfgs.add((b, f, ex, "replace"))
    for b in macd_exit_replace.BASES:
        for ex in macd_exit_replace.EXITS:
            cfgs.add((b, "none", ex, "replace"))
    for _, b, f, ex in macd_mc_trades.CASES:
        cfgs.add((b, f, ex, "replace"))
    # _03 verify_multi_macd：variant_trades(base, entry, "ma200", "replace")，組合已含在 combo／mc 內
    return sorted(cfgs)


def load_before(ref: str):
    """改版前的 macd_variants（git ref 版 ＋ entry_signal 那一行），存成暫存檔再 import。"""
    rel = "_02_strategy/macd_strategy/macd_variants.py"
    src = subprocess.run(["git", "show", f"{ref}:{rel}"], cwd=_root, capture_output=True,
                         check=True).stdout.decode("utf-8")
    old_line = "self.buy_signal(df).to_numpy(),"
    if old_line in src:
        src = src.replace(old_line, "self.entry_signal(df).to_numpy(),")
    tmp = tempfile.mkdtemp(prefix="macd_variants_before_")
    path = os.path.join(tmp, "macd_variants_before.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location("macd_variants_before", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, tmp


def check_legacy(data: dict, ref: str) -> bool:
    print(f"\n【檢查二】既有設定改版前（{ref}＋entry_signal）後逐筆相同")
    before, tmp = load_before(ref)
    ok, cfgs = True, legacy_configs()
    try:
        for cfg in cfgs:
            a = _trades(_variant(before.MacdVariant, *cfg), data)
            b = _trades(_variant(MacdVariant, *cfg), data)
            if not _same(a, b):
                ok = False
                print(f"  NG {cfg}：改版前 {len(a)} 筆 vs 改版後 {len(b)} 筆")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"  {'OK' if ok else 'NG'}｜{len(cfgs)} 組設定")
    return ok


# ── 檢查三 ───────────────────────────────────────────────────────
def check_identities(data: dict) -> bool:
    print("\n【檢查三】新設定的恆等式")
    pairs = [(("cross", "none", "death", "replace"), ("cross", "none", "native", "replace")),
             (("zero", "none", "zero_down", "replace"), ("zero", "none", "native", "replace")),
             (("div", "none", "death", "replace"), ("div", "none", "native", "replace")),
             (("mix", "none", "native", "replace"), ("cross", "none", "zero_down", "replace"))]
    ok = True
    for a_cfg, b_cfg in pairs:
        good = _same(_trades(make_variant(*a_cfg), data), _trades(make_variant(*b_cfg), data))
        ok &= good
        print(f"  {'OK' if good else 'NG'} {a_cfg[0]}×{a_cfg[2]} ≡ {b_cfg[0]}×{b_cfg[2]}")
    return ok


# ── 檢查四 ───────────────────────────────────────────────────────
def check_diagnostics(data: dict) -> bool:
    print("\n【檢查四】生效率診斷（掃描一致＋平倉數對得上）")
    ok = True
    cases = [(p, ex, "append") for p in ("cross", "zero", "div", "mix")
             for ex in ("ma200", "supertrend", "psar", "donchian", "lowerhigh",
                        "chandelier", "trail10", "atrstop", "takeprofit", "time60")]
    cases += [(p, ex, "replace") for p in ("cross", "div")
              for ex in ("lowerhigh", "atrstop", "takeprofit")]
    for base, ex, mode in cases:
        pct, only, n_exit = rule_share(data, base, ex, mode)   # 掃描不一致會直接 raise
        if only > pct:                                      # 搶先是含同根的子集
            ok = False
            print(f"  NG {base}×{ex}×{mode}：搶先 {only}% > 含同根 {pct}%")
        _, s = variant_trades(data, base, "none", ex, mode)
        if n_exit != s["已平倉數"]:
            ok = False
            print(f"  NG {base}×{ex}×{mode}：重數 {n_exit} vs vbt {s['已平倉數']}")
        # 取代型沒有原生出場可搶先，每一筆平倉都必須是規則本身觸發
        if mode == "replace" and n_exit and pct != 100.0:
            ok = False
            print(f"  NG {base}×{ex}×{mode}：取代型生效率 {pct}% ≠ 100%")
    print(f"  {'OK' if ok else 'NG'}｜{len(cases)} 格")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--ref", default="HEAD", help="改版前 macd_variants 的 git ref")
    a = ap.parse_args()
    t0 = time.time()
    data = prepare(limit=a.limit)
    print(f"樣本 {len(data)} 檔｜{time.time() - t0:.0f} 秒\n")
    results = [check_toggles(data), check_legacy(data, a.ref), check_identities(data),
               check_diagnostics(data)]
    print("\n【檢查五】verify_exit_switches")
    results.append(verify_exit_switches.main() == 0)
    print(f"\n耗時 {time.time() - t0:.0f} 秒｜結果："
          + ("OK 全部通過" if all(results) else "NG 見上方"))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
