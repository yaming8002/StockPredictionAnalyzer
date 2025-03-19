import backtrader as bt


class CustomPandasData(bt.feeds.PandasData):
    lines = ("sma5", "sma10", "sma20", "sma50", "sma60", "sma100", "sma120", "sma200", "ema5", "ema10", "ema20", "ema50", "ema60", "ema100", "ema120", "ema200")
    params = (
        ("sma5", -1),
        ("sma10", -1),
        ("sma20", -1),
        ("sma50", -1),
        ("sma60", -1),
        ("sma100", -1),
        ("sma120", -1),
        ("sma200", -1),
        ("ema5", -1),
        ("ema10", -1),
        ("ema20", -1),
        ("ema50", -1),
        ("ema60", -1),
        ("ema100", -1),
        ("ema120", -1),
        ("ema200", -1),
    )
