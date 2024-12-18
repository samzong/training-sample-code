import os
import torch
import deepspeed
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from llama_factory import LlamaFactory

# 设置训练参数
epochs = 3
train_batch_size = 4
learning_rate = 5e-5
weight_decay = 1e-2
max_grad_norm = 1.0
local_model_path = "/mnt/model"  # 模型挂载路径
local_data_path = "/mnt/data"  # 数据挂载路径
checkpoint_path = "/mnt/checkpoints"  # Checkpoint 保存挂载路径
final_model_path = "/mnt/final_model"  # 最终模型保存挂载路径

# 加载数据集
dataset = load_dataset(local_data_path)  # 使用挂载的本地数据路径
train_dataset = dataset["train"]
val_dataset = dataset["validation"]

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(local_model_path)


# 数据预处理函数
def preprocess_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True)


train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# 创建 DataLoader
train_sampler = DistributedSampler(train_dataset, shuffle=True)
train_loader = DataLoader(
    train_dataset, batch_size=train_batch_size, sampler=train_sampler
)
val_loader = DataLoader(val_dataset, batch_size=train_batch_size)

# 加载本地模型
model = AutoModelForCausalLM.from_pretrained(local_model_path)

# DeepSpeed 配置
ds_config = {
    "train_batch_size": train_batch_size,
    "gradient_accumulation_steps": 8,
    "zero_optimization": {"stage": 2, "offload_param": True, "offload_optimizer": True},
    "fp16": {"enabled": True},
}

# 初始化 DeepSpeed
model, optimizer, _, _ = deepspeed.initialize(
    args=None, model=model, optimizer=None, config_params=ds_config
)

# 使用 LLaMA-Factory 配置优化器和调度器
llama = LlamaFactory(model)

llama_config = {
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "max_grad_norm": max_grad_norm,
}

llama.setup_optimizer_and_scheduler(optimizer, **llama_config)

# 训练循环
for epoch in range(epochs):
    model.train()
    for step, batch in enumerate(train_loader):
        # 准备输入数据
        inputs = tokenizer(
            batch["text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        # 前向传播
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

        # 反向传播
        model.backward(loss)

        # 使用 LLaMA-Factory 进行优化步骤
        llama.step()

        # 输出训练进度
        if step % 10 == 0:
            print(f"Epoch {epoch}/{epochs} - Step {step}, Loss: {loss.item()}")

    # 保存 checkpoint
    if epoch % 1 == 0:  # 每个 epoch 保存一次
        checkpoint_epoch_path = os.path.join(
            checkpoint_path, f"checkpoint_epoch_{epoch}"
        )
        os.makedirs(checkpoint_epoch_path, exist_ok=True)
        model.save_checkpoint(checkpoint_epoch_path, epoch)
        print(f"Checkpoint saved at {checkpoint_epoch_path}")

# 验证阶段
model.eval()
with torch.no_grad():
    for batch in val_loader:
        inputs = tokenizer(
            batch["text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        outputs = model(**inputs, labels=inputs["input_ids"])
        val_loss = outputs.loss
        print(f"Validation Loss: {val_loss.item()}")

# 保存最终模型
os.makedirs(final_model_path, exist_ok=True)
model.save_checkpoint(final_model_path, model.global_step)
print(f"Final model saved at {final_model_path}")
