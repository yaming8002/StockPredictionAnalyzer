# models.py
import tensorflow as tf
from tensorflow.keras import layers, models, Input


# -----------------------------------------------------
# BUY model — CNN 單輸入（用來決定是否買入）
# -----------------------------------------------------
def build_buy_model(window=120, n_features=15):
    """
    BUY 模型 — 單輸入（120天的時序資料）
    用於【進場判斷】，輸入的資料來源是 120 天的股票資料（無買點資訊）。
    """
    inputs = Input(shape=(window, n_features), name="input_buy_window")

    x = layers.Conv1D(64, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="BUY_MODEL")
    model.compile(optimizer="adam", loss="binary_crossentropy")

    return model


# -----------------------------------------------------
# SELL model — CNN + 靜態特徵（雙輸入）
# -----------------------------------------------------
def build_sell_model(window=120, n_features=15, buy_features_count=15):
    """
    SELL 模型 — 雙輸入（持倉期間才會使用）
    Input A: 120 天的時序資料
    Input B: 買入當下的資料 + 買入價
    """
    # --------------------
    # Input A：120天時序特徵 (CNN branch)
    # --------------------
    inputs_current = Input(shape=(window, n_features), name="input_current")

    x = layers.Conv1D(64, 5, padding="same", activation="relu")(inputs_current)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)  # -> (batch, 128)
    x = layers.Dense(64, activation="relu")(x)

    # --------------------
    # Input B：買入當天特徵 + 買入價格（靜態特徵 branch）
    # --------------------
    inputs_buy_day = Input(shape=(buy_features_count,), name="input_buy_day")
    inputs_buy_price = Input(shape=(1,), name="input_buy_price")  # scalar

    y = layers.Concatenate(name="concat_buy_features")([inputs_buy_day, inputs_buy_price])
    y = layers.Dense(32, activation="relu")(y)
    y = layers.Dense(16, activation="relu")(y)

    # --------------------
    # Merge CNN + static features
    # --------------------
    merged = layers.Concatenate(name="merge_current_buyinfo")([x, y])
    merged = layers.Dense(64, activation="relu")(merged)
    merged = layers.Dropout(0.3)(merged)

    output = layers.Dense(1, activation="sigmoid")(merged)

    model = models.Model(inputs=[inputs_current, inputs_buy_day, inputs_buy_price], outputs=output, name="SELL_MODEL")
    model.compile(optimizer="adam", loss="binary_crossentropy")

    return model
