# from random import random
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, BatchNormalization
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.callbacks import EarlyStopping
# import numpy as np
# import tensorflow as tf
# from sklearn.metrics import confusion_matrix, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns
# from _03_deeplearning.unit import build_training_data_gen, build_training_data_gen_from_df


# def build_model(input_shape, learning_rate=0.001, dropout_rate=0.5, conv_filters=64, dense_units=64):
#     """
#     建立 CNN 模型，用於二元分類（輸出為 sigmoid）

#     參數:
#         input_shape (tuple): 輸入資料形狀，例如 (30, 19)
#         learning_rate (float): 優化器學習率
#         dropout_rate (float): Dropout 比例，用來避免 overfitting
#         conv_filters (int): Conv1D 過濾器數量
#         dense_units (int): Dense 層神經元數量

#     回傳:
#         model (tf.keras.Model): 編譯好的模型
#     """
#     model = Sequential(
#         [
#             Input(shape=input_shape, name="Input_Layer"),
#             Conv1D(filters=conv_filters, kernel_size=3, activation="relu", kernel_regularizer=l2(0.001), name="Conv1D"),
#             BatchNormalization(name="BN_Conv"),
#             MaxPooling1D(pool_size=2, name="MaxPool"),
#             Flatten(name="Flatten"),
#             Dense(dense_units, activation="relu", kernel_regularizer=l2(0.001), name="Dense"),
#             BatchNormalization(name="BN_Dense"),
#             Dropout(dropout_rate, name="Dropout"),
#             Dense(1, activation="sigmoid", name="Output"),
#         ]
#     )

#     model.compile(optimizer=Adam(learning_rate=learning_rate), loss="binary_crossentropy", metrics=["accuracy"])

#     return model


# def train(X, y, epochs=10, batch_size=32):
#     model = build_model(input_shape=(X.shape[1], X.shape[2]))
#     model.summary()
#     model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
#     model.save("model_stock_cnn.h5")


# def generator_wrapper(file_path):
#     return lambda: build_training_data_gen(file_path)


# def test_04_train_model(file_path="./stock_data/leaning_label/sma_120_sma_200_trades.csv", model_path="model_stream_cnn.h5"):
#     # 🔍 一次把所有資料先讀出來（靜態模式）
#     X_all, y_all = [], []
#     trades = pd.read_csv(file_path, parse_dates=["buy_date"])
#     trades = trades.sample(frac=1).reset_index(drop=True)
#     # ✅ 切分 train / val（80% / 20%）
#     split_index = int(len(trades) * 0.8)
#     train_trades = trades[:split_index]
#     val_trades = trades[split_index:]

#     # ✅ 從訓練資料 generator 取得 input_shape
#     sample_X, _ = next(build_training_data_gen_from_df(train_trades))
#     input_shape = sample_X.shape

#     # ✅ 建訓練資料集
#     train_dataset = (
#         tf.data.Dataset.from_generator(
#             lambda: build_training_data_gen_from_df(train_trades),
#             output_signature=(
#                 tf.TensorSpec(shape=input_shape, dtype=tf.float32),
#                 tf.TensorSpec(shape=(), dtype=tf.int32),
#             ),
#         )
#         .batch(64)
#         .prefetch(tf.data.AUTOTUNE)
#     )

#     # ✅ 驗證資料集
#     val_dataset = (
#         tf.data.Dataset.from_generator(
#             lambda: build_training_data_gen_from_df(val_trades),
#             output_signature=(
#                 tf.TensorSpec(shape=input_shape, dtype=tf.float32),
#                 tf.TensorSpec(shape=(), dtype=tf.int32),
#             ),
#         )
#         .batch(64)
#         .prefetch(tf.data.AUTOTUNE)
#     )

#     # ✅ 建模型
#     model = build_model(input_shape=input_shape)

#     # ✅ class_weight 設定
#     win_trades = train_trades[train_trades["profit"] > 0]
#     win_per = len(win_trades) / len(train_trades) * 1.2
#     class_weight = {0: 1.0, 1: win_per}

#     # ✅ EarlyStopping
#     early_stop = EarlyStopping(patience=5, restore_best_weights=True)

#     # ✅ 訓練模型
#     model.fit(train_dataset, validation_data=val_dataset, epochs=100, class_weight=class_weight, callbacks=[early_stop])

#     # ✅ 儲存模型
#     model.save(model_path)
#     print("✅ 模型訓練完成並儲存")

#     # ✅ 測試資料隨機抽樣
#     print("🎯 隨機抽樣 100 筆資料做測試")
#     X_test, y_test = [], []
#     for X, y in build_training_data_gen_from_df(trades):  # 🔄 使用同一份資料來源
#         if random() < 0.1:
#             X_test.append(X)
#             y_test.append(y)
#         if len(X_test) >= 100:
#             break
#     X_test = np.array(X_test)
#     y_test = np.array(y_test)

#     # ✅ 評估
#     loss, acc = model.evaluate(X_test, y_test)
#     print(f"📊 測試集準確率: {acc:.4f}, 損失: {loss:.4f}")
#     print(f"📊 訓練集：賺錢筆數 {len(train_trades[train_trades['profit'] > 0])} / 總筆數 {len(train_trades)}")
#     print(f"📊 測試集：賺錢筆數 {len(val_trades[val_trades['profit'] > 0])} / 總筆數 {len(val_trades)}")


# def test_05_predict_from_model(file_path="./stock_data/leaning_label/sma_120_sma_200_trades.csv", model_path="model_stream_cnn.h5"):
#     # ✅ 載入模型
#     print("📥 載入模型中...")
#     model = tf.keras.models.load_model(model_path)

#     # ✅ 收集所有資料
#     print("📦 準備測試資料中...")
#     X_test, y_test = [], []
#     count_line = 0
#     for X, y in build_training_data_gen(file_path):
#         X_test.append(X)
#         y_test.append(y)
#         count_line += 1
#         if count_line > 1000:
#             break

#     X_test = np.array(X_test)
#     y_test = np.array(y_test)

#     print(f"✅ 總資料數量：{len(X_test)}")

#     # ✅ 模型評估
#     loss, acc = model.evaluate(X_test, y_test)
#     print(f"📊 準確率: {acc:.4f}, 損失: {loss:.4f}")

#     # ✅ 預測與處理結果
#     y_prob = model.predict(X_test)
#     y_pred = (y_prob > 0.5).astype(int).reshape(-1)

#     # ✅ 混淆矩陣
#     cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

#     # 包裝成 DataFrame 加上標題
#     cm_df = pd.DataFrame(cm, index=["實際：賠錢（0）", "實際：賺錢（1）"], columns=["預測：賠錢（0）", "預測：賺錢（1）"])

#     print("\n🧾 混淆矩陣（含欄位標籤）：")
#     print(cm_df)

#     # ✅ 精確率/召回率/F1
#     print("\n📈 分類評估報告：")
#     print(classification_report(y_test, y_pred, digits=4))

#     # ✅ 畫圖
#     plt.figure(figsize=(6, 5))
#     sns.heatmap(
#         cm,
#         annot=True,
#         fmt="d",
#         cmap="Blues",
#         xticklabels=["預測：賠", "預測：賺"],
#         yticklabels=["實際：賠", "實際：賺"],
#     )
#     plt.xlabel("預測結果")
#     plt.ylabel("實際標籤")
#     plt.title("🎯 混淆矩陣")
#     plt.tight_layout()
#     plt.show()
