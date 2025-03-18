import json
import os


def load_config(config_file=None):
    """從 config.json 讀取設定"""
    if config_file is None:
        config_file = os.path.join(os.path.dirname(__file__), "..", "config.json")

    with open(config_file, "r") as file:
        config = json.load(file)
    return config
