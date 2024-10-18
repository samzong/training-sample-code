# 导入必要的库
import os
import tensorflow as tf
import horovod.tensorflow.keras as hvd  # 注意这里的导入方式
import numpy as np
import gzip
import urllib.request
import datetime
import logging
import sys

# 设置日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化Horovod
hvd.init()

# 处理 macOS 上 Open MPI 的共享内存问题
# 如果您在 macOS 上运行，可以设置以下环境变量
if sys.platform == "darwin":
    os.environ["OMPI_MCA_btl_vader_single_copy_mechanism"] = "none"

# 配置GPU设置
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    local_rank = hvd.local_rank()
    if local_rank < len(gpus):
        gpu = gpus[local_rank]
        tf.config.set_visible_devices(gpu, "GPU")
        tf.config.experimental.set_memory_growth(gpu, True)
    else:
        # 如果本地GPU数量不足，使用CPU
        tf.config.set_visible_devices([], "GPU")
else:
    # 如果没有GPU可用，使用CPU
    tf.config.set_visible_devices([], "GPU")


# 加载Fashion MNIST数据集函数
def load_fashion_mnist(data_dir="./data/fashion_mnist"):
    base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    def download_file(filename):
        url = base_url + filename
        local_path = os.path.join(data_dir, filename)
        if not os.path.exists(local_path):
            if hvd.rank() == 0:
                os.makedirs(data_dir, exist_ok=True)
                logger.info(f"正在下载 {filename}...")
                try:
                    urllib.request.urlretrieve(url, local_path)
                    logger.info(f"{filename} 下载完成。")
                except Exception as e:
                    logger.error(f"下载 {filename} 时出错: {e}")
                    raise
            # 所有进程等待数据下载完成
            logger.info(f"进程 {hvd.rank()} 正在等待数据下载完成...")
        # 确保所有进程同步
        hvd.broadcast(tf.convert_to_tensor(0), root_rank=0)
        return local_path

    def load_images(filename):
        with gzip.open(filename, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            return data.reshape(-1, 28, 28)

    def load_labels(filename):
        with gzip.open(filename, "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=8)

    # 下载并加载数据
    try:
        train_images = load_images(download_file(files["train_images"]))
        train_labels = load_labels(download_file(files["train_labels"]))
        test_images = load_images(download_file(files["test_images"]))
        test_labels = load_labels(download_file(files["test_labels"]))
    except Exception as e:
        logger.error(f"加载数据时出错: {e}")
        raise

    return (train_images, train_labels), (test_images, test_labels)


# 加载数据集
try:
    (train_images, train_labels), (test_images, test_labels) = load_fashion_mnist()
except Exception as e:
    logger.error(f"程序退出，原因：{e}")
    exit(1)

# 数据归一化
train_images = train_images / 255.0
test_images = test_images / 255.0

# 创建 tf.data.Dataset
batch_size = 128

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shard(num_shards=hvd.size(), index=hvd.rank())
train_dataset = train_dataset.shuffle(10000, seed=42).batch(batch_size)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = test_dataset.shard(num_shards=hvd.size(), index=hvd.rank())
test_dataset = test_dataset.batch(batch_size)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)


# 定义模型
def create_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    return model


model = create_model()

# 定义损失函数、优化器和指标
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
initial_lr = 0.001

# 针对 M1/M2 Mac，使用 legacy 优化器
if sys.platform == "darwin":
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=initial_lr * hvd.size())
else:
    opt = tf.keras.optimizers.Adam(learning_rate=initial_lr * hvd.size())

# 将优化器封装为Horovod分布式优化器
opt = hvd.DistributedOptimizer(opt)

# 编译模型
model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])

# 设置回调函数
callbacks = [
    # 将初始变量从根节点广播到其他节点
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    # 平均化指标
    hvd.callbacks.MetricAverageCallback(),
    # 学习率温暖启动
    hvd.callbacks.LearningRateWarmupCallback(
        initial_lr=initial_lr * hvd.size(), warmup_epochs=3, verbose=1
    ),
]

if hvd.rank() == 0:
    # 仅在根节点记录TensorBoard日志和保存模型检查点
    log_dir = os.path.join(
        "./logs/fit/", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath="./checkpoints/ckpt-{epoch}", save_weights_only=True
        )
    )

# 训练模型
try:
    model.fit(
        train_dataset,
        epochs=10,
        validation_data=test_dataset,
        callbacks=callbacks,
        verbose=1 if hvd.rank() == 0 else 0,
    )
except Exception as e:
    logger.error(f"训练过程中出错: {e}")
    exit(1)

# 保存最终模型
# 保存最终模型
if hvd.rank() == 0:
    try:
        # 定义模型保存路径，符合 Triton 的模型仓库结构
        model_name = "fashion_mnist_model"
        model_version = "1"
        model_repository = "./model_repository"
        export_path = os.path.join(
            model_repository, model_name, model_version, "model.savedmodel"
        )

        # 创建目录
        os.makedirs(export_path, exist_ok=True)

        # 保存模型为 TensorFlow SavedModel 格式
        model.save(export_path)
        logger.info(f"模型已成功保存到 {export_path}，可用于 Triton 推理。")

        # （可选）创建 Triton 配置文件 config.pbtxt
        config_path = os.path.join(model_repository, model_name, "config.pbtxt")
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                f.write(f"""
name: "{model_name}"
platform: "tensorflow_savedmodel"
max_batch_size: 128
input [
  {{
    name: "flatten_input"
    data_type: TYPE_FP32
    format: FORMAT_NONE
    dims: [28, 28]
  }}
]
output [
  {{
    name: "dense_1"
    data_type: TYPE_FP32
    dims: [10]
  }}
]
""")
            logger.info(f"已创建 Triton 配置文件 {config_path}。")
    except Exception as e:
        logger.error(f"保存模型时出错: {e}")
        exit(1)
