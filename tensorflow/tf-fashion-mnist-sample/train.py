#!/usr/bin/python
# -*- coding: utf-8 -*-

# if use Nvidia GPU
# pip install tensorflow[and-cuda]

# without gpu, default
# pip install tensorflow

from zipfile import ZipFile
import tensorflow as tf
import numpy as np
import os
import requests
from io import BytesIO
import zipfile
import gzip
from tensorflow import keras

DATASET_PATH = "/home/jovyan/data/fashion"
DATA_ONLINE_URL = "https://docs.daocloud.io/fashion-mnist-data.zip"
MODEL_PATH = "/home/jovyan/model"
MODEL_VERSION = "1"

is_output_checkpoint = (
    os.getenv("OUTPUT_CHECKPOINT").strip() == "1"
    if os.getenv("OUTPUT_CHECKPOINT")
    else False
)


class DatasetManager:
    def __init__(self, data_path, model_path, online_url):
        self.dataset_path = data_path
        self.model_path = model_path
        self.data_online_url = online_url

    def ensure_dataset(self):
        if self.check_dataset():
            print(f"数据集已存在于 {self.dataset_path}，将直接使用。")
        else:
            print(f"数据集不存在于 {self.dataset_path}，开始在线下载。")
            self.download_data()
        return self.dataset_path

    def check_dataset(self):
        if os.path.exists(self.dataset_path):
            files = os.listdir(self.dataset_path)
            if files:
                print(f"数据集目录 {self.dataset_path} 存在并包含文件。")
                return True
            else:
                print(f"警告：数据集目录 {self.dataset_path} 存在但为空。")
        else:
            print(f"数据集目录 {self.dataset_path} 不存在。")
        return False

    def download_data(self):
        try:
            print(f"正在从 {self.data_online_url} 下载数据...")
            response = requests.get(self.data_online_url)
            response.raise_for_status()

            with ZipFile(BytesIO(response.content)) as zip_file:
                file_names = zip_file.namelist()
                if file_names:
                    dataset_folder = file_names[0].split("/")[0]
                    extract_path = os.path.dirname(os.getcwd())
                    zip_file.extractall(extract_path)

                    # 更新数据集路径为解压后的目录
                    self.dataset_path = os.path.join(extract_path, dataset_folder)
                    print(
                        f"数据下载并解压成功。数据集路径已更新为: {self.dataset_path}"
                    )
                else:
                    print("错误：ZIP 文件似乎是空的。")
        except requests.exceptions.RequestException as e:
            print(f"下载数据时出错: {e}")
        except zipfile.BadZipFile:
            print("错误：下载的文件不是有效的 ZIP 文件。")

    def ensure_model(self):
        if os.path.exists(self.model_path):
            print(f"模型路径 {self.model_path} 已存在。")
        else:
            print(
                f"模型路径 {self.model_path} 不存在，将在当前目录创建 'model' 文件夹。"
            )
            self.model_path = os.path.join(os.getcwd(), "model")
            os.makedirs(self.model_path, exist_ok=True)
            print(f"新的模型路径已创建：{self.model_path}")
        return self.model_path

    def get_path(self):
        return self.dataset_path


def load_data(data_folder):
    """
    Loads the dataset from the specified folder.

    Args:
        data_folder (str): The path to the folder containing the dataset files.

    Returns:
        tuple: A tuple containing the training and test datasets.
    """

    files = [
        "train-labels-idx1-ubyte.gz",
        "train-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
    ]

    paths = []

    for fname in files:
        paths.append(os.path.join(data_folder, fname))
    with gzip.open(paths[0], "rb") as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(paths[1], "rb") as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(
            len(y_train), 28, 28
        )
    with gzip.open(paths[2], "rb") as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(paths[3], "rb") as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(
            len(y_test), 28, 28
        )

    return (x_train, y_train), (x_test, y_test)


def train():
    """
    Trains a neural network model on the Fashion MNIST dataset.

    This function performs the following steps:
    1. Loads the training and test datasets.
    2. Scales the image pixel values to the range [0.0, 1.0].
    3. Reshapes the training images for input into the model.
    4. Defines and compiles a neural network model.
    5. Trains the model on the training data.
    6. Evaluates the model on the test data.
    7. Saves the trained model and logs.

    Returns:
        None
    """

    # load data
    data_manager = DatasetManager(DATASET_PATH, MODEL_PATH, DATA_ONLINE_URL)
    final_dataset_path = data_manager.ensure_dataset()
    (train_images, train_labels), (test_images, test_labels) = load_data(
        final_dataset_path
    )

    # scale the values to 0.0 to 1.0
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # reshape for feeding into the model
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    # build the model and train
    model = keras.Sequential(
        [
            keras.layers.Conv2D(
                input_shape=(28, 28, 1),
                filters=8,
                kernel_size=3,
                strides=2,
                activation="relu",
                name="Conv1",
            ),
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation=tf.nn.softmax, name="Softmax"),
        ]
    )

    model.summary()

    epochs = 10

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    final_model_path = data_manager.ensure_model()

    export_path = os.path.join(final_model_path, "model.keras")
    log_path = final_model_path + "/train/logs"

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)
    checkpoint_path = export_path + ".weights.h5"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, verbose=1
    )

    callbacks = [tensorboard_callback]

    if is_output_checkpoint:
        callbacks.append(cp_callback)

    model.fit(train_images, train_labels, epochs=epochs, callbacks=callbacks)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_acc}\nTest loss: {test_loss}")

    tf.keras.models.save_model(model, export_path, overwrite=True)

    print(f"Model exported to: {export_path}")


if __name__ == "__main__":
    train()
