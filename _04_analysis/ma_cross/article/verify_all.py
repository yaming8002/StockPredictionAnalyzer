# -*- coding: utf-8 -*-
"""
一次跑完均線交叉（一）～（七）七篇的驗證；任何一篇有錯就回傳非 0。

執行（BLOG_DIR 指向 blog 專案根目錄；（七）要 DIVIDEND_FILE 算 0050 含息）：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python _04_analysis/ma_cross/article/verify_all.py
"""
import importlib
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

ARTICLES = [f"verify_article{i}" for i in range(1, 8)]


def main() -> int:
    bad = []
    for name in ARTICLES:
        print(f"── {name}")
        if importlib.import_module(name).main() != 0:
            bad.append(name)
    print("全部通過" if not bad else f"有錯：{bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
