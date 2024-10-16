#!/usr/bin/python
# -*- coding: utf-8 -*-

# if use Nvidia GPU
# pip install tensorflow[and-cuda]

# without gpu, default
# pip install tensorflow

import tensorflow as tf
import numpy as np
import os
import gzip
from tensorflow import keras

dataset_path = "/home/jovyan/data/fashion"
model_path = "/home/jovyan/model"
model_version = "1"

is_output_checkpoint = (
    os.getenv("OUTPUT_CHECKPOINT").strip() == "1"
    if os.getenv("OUTPUT_CHECKPOINT")
    else False
)


def load_data(data_folder):
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
    # load data
    (train_images, train_labels), (test_images, test_labels) = load_data(
        "/home/jovyan/data/data/fashion"
    )

    # scale the values to 0.0 to 1.0
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # reshape for feeding into the model
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

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

    testing = False
    epochs = 5

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    export_path = os.path.join("/home/jovyan/model", "model.keras")

    logdir = "/home/jovyan/model/train/logs/"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    checkpoint_path = export_path + ".weights.h5"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, verbose=1
    )

    callbacks = [tensorboard_callback]

    print("is_output_model: {}".format(os.environ.get("is_output_model")))

    if is_output_checkpoint:
        callbacks.append(cp_callback)

    model.fit(train_images, train_labels, epochs=epochs, callbacks=callbacks)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_acc}\nTest loss: {test_loss}")

    tf.keras.models.save_model(model, export_path, overwrite=True)

    print(f"Model exported to: {export_path}")


if __name__ == "__main__":
    train()
