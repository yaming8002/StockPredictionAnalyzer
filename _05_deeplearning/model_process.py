import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from tensorflow.python.data.ops.dataset_ops import DatasetV2

from _04_deeplearning.unit import build_training_data_gen_from_df, build_training_data_gen_one_hot_from_df


class DpTrainerBisis(ABC):

    def __init__(
        self,
        file_path,
        save_path,
        input_shape,
        output_shape,
        optimizer=None,
        is_onehot=False,
        batch_size=64,
        epochs=100,
        patience=5,
    ):
        """
        初始化 LSTM Trainer
        """
        self.file_path = file_path
        self.save_path = save_path
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.optimizer = optimizer if optimizer else Adam(learning_rate=0.001)
        self.patience = patience
        self.batch_size = batch_size
        self.is_onehot = is_onehot
        self.epochs = epochs
        self.model = self.build_model()
        if self.is_onehot:
            self.loss_function = "categorical_crossentropy"
        else:
            self.loss_function = "binary_crossentropy"

        self.model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=["accuracy"])

        # ✅ 初始化歷史紀錄
        self.history = {"accuracy": [], "loss": [], "val_accuracy": [], "val_loss": []}
        print("✅ 模型結構:")
        self.model.summary()

    @abstractmethod
    def build_model(self):
        """
        抽象方法：需要覆寫此方法來自定義 LSTM 模型
        """
        pass

    def tranging_model(self):
        """
        訓練並評估模型
        """
        print(f"📥 載入訓練資料: {self.file_path}")
        trades = pd.read_csv(self.file_path, parse_dates=["buy_date"])
        split_index = int(len(trades) * 0.95)
        train_trades = trades[:split_index]
        val_trades = trades[split_index:]

        trades = trades.sample(frac=1).reset_index(drop=True)
        # ✅ class_weight 設定
        win_trades = train_trades[train_trades["profit"] > 0]
        win_per = len(win_trades) / len(train_trades)
        class_weight = {0: 1.0, 1: 1 / win_per}

        self.fit(train_trades, val_trades, class_weight)
        self.evaluate(val_trades)
        self.save_model()
        self.plot_history()

    def fit(self, train_df, val_df, class_weight):
        """
        訓練模型
        """
        print("🚀 開始訓練模型...")

        steps_per_epoch = max(len(train_df) // self.batch_size, 1)
        validation_steps = max(len(val_df) // self.batch_size, 1)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            print(f"\n🔄 Epoch {epoch + 1} 開始，打亂資料...")
            train_df = train_df.sample(frac=1).reset_index(drop=True)
            print(f"🔄 資料生成器已建立, 批次大小: {self.batch_size}, 使用 One-Hot: {self.is_onehot}")
            train_dataset = self.create_data_generator(train_df)
            val_dataset = self.create_data_generator(val_df)

            # 單次 Epoch 訓練
            history_epoch = self.model.fit(train_dataset, steps_per_epoch=steps_per_epoch, class_weight=class_weight, verbose=1)
            val_loss, val_accuracy = self.model.evaluate(val_dataset, steps=validation_steps)

            self.history["accuracy"].append(history_epoch.history["accuracy"][0])
            self.history["loss"].append(history_epoch.history["loss"][0])
            self.history["val_accuracy"].append(val_accuracy)
            self.history["val_loss"].append(val_loss)
            print(f" 🔸 訓練集 - Loss: {history_epoch.history['loss'][0]}, Accuracy: {history_epoch.history['accuracy'][0]}")
            print(f" 🔹 驗證集 - Loss: {val_loss}, Accuracy: {val_accuracy}")

            # Early Stopping 檢查
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"🔎 Early Stopping 計數: {patience_counter}/{self.patience}")

            if patience_counter >= self.patience:
                print("🛑 觸發 Early Stopping!")
                break

    def create_data_generator(self, dataframe) -> DatasetV2:
        """
        建立資料生成器
        """
        if self.is_onehot:
            dataset = tf.data.Dataset.from_generator(
                lambda: build_training_data_gen_one_hot_from_df(dataframe),
                output_signature=(
                    tf.TensorSpec(shape=(30, 19), dtype=tf.float32),
                    tf.TensorSpec(shape=(2,), dtype=tf.int32),
                ),
            )
        else:
            dataset = tf.data.Dataset.from_generator(
                lambda: build_training_data_gen_from_df(dataframe),
                output_signature=(
                    tf.TensorSpec(shape=(30, 19), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.int32),
                ),
            )

        # 🚀 自動調整 buffer size 避免超出範圍
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def evaluate(self, val_df):
        """
        評估模型
        """
        print("📊 開始模型評估...")
        val_dataset = self.create_data_generator(val_df)
        loss, acc = self.model.evaluate(val_dataset)
        print(f"📊 測試集準確率: {acc:.4f}, 損失: {loss:.4f}")
        return loss, acc

    def save_model(self):
        """
        儲存模型
        """
        self.model.save(self.save_path, include_optimizer=True)
        print(f"✅ 模型已儲存至: {self.save_path}")

    def plot_history(self):
        """
        繪製訓練過程
        """
        plt.figure(figsize=(12, 5))
        plt.plot(self.history["accuracy"], label="Train Accuracy")
        plt.plot(self.history["val_accuracy"], label="Validation Accuracy")
        plt.plot(self.history["loss"], label="Train Loss")
        plt.plot(self.history["val_loss"], label="Validation Loss")
        plt.legend()
        plt.title("訓練過程")
        plt.show()
