# （过期）Qwen-7B 模型分布式微调操作文档

> 注意：本操作文档已过期，请参考 [qwen.py](qwen.py) 进行操作。

## 1. 需求概述

本操作文档介绍了如何使用 **DeepSpeed** 和 **LLaMA-Factory** 对 **Qwen-7B** 模型进行分布式微调训练，并结合 **Kubernetes** 和 **Kubeflow** 通过 **PyTorchJob** 管理训练作业。训练数据和代码将通过挂载数据卷的方式提供，训练将在标准的 PyTorch Docker 镜像中进行。

### 主要组件：

- **DeepSpeed**：用于优化训练，支持大规模并行训练和高效的内存管理。
- **LLaMA-Factory**：提供定制化的大模型训练配置和优化策略，配合 DeepSpeed 提升训练效率。
- **PyTorchJob**：通过 Kubeflow 管理和调度 PyTorch 作业，提供分布式训练的集群支持。

## 2. 系统需求

- **Kubernetes 集群**：用于集群管理和作业调度。
- **Kubeflow**：用于运行和管理 PyTorchJob。
- **PyTorch 和 DeepSpeed**：进行分布式训练和优化。
- **LLaMA-Factory**：用于模型的定制化训练配置。

## 3. 安装和配置

### 3.1 安装 PyTorch 和 DeepSpeed

首先，我们需要安装 `PyTorch` 和 `DeepSpeed`。你可以使用以下命令来安装它们：

```bash
pip install torch deepspeed
```

### 3.2 安装 LLaMA-Factory

`LLaMA-Factory` 是一个高效的定制化训练库，可以通过以下命令安装：

```bash
pip install llama-factory
```

以上，构建为基础镜像。

### 3.3 配置 Kubernetes 和 Kubeflow

- **Kubernetes**：请确保你的集群已经配置完毕，并能够支持多节点训练。
- **Kubeflow**：安装并配置 Kubeflow，在集群中运行 PyTorch 作业。

详细的 Kubeflow 和 PyTorchJob 配置可以参考官方文档 [Kubeflow PyTorchJob](https://www.kubeflow.org/docs/components/training/pytorch/).

## 4. 数据集准备

需要提前准备好数据集和路径，这里默认不提供。

> 以下为代码分解介绍，详细请查看 [ft_qwen-7b.py](ft_qwen-7b.py).

### 4.1 加载数据集

这里我们假设你有一个文本数据集。使用 `datasets` 库加载并进行预处理。以下是一个加载数据集并进行 tokenization 的示例：

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# 加载数据集
dataset = load_dataset("your_dataset")  # 请替换为你的数据集路径
train_dataset = dataset['train']
val_dataset = dataset['validation']

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen-7B")

# 数据预处理
def preprocess_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True)

train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)
```

### 4.2 DataLoader 配置

```python
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

train_sampler = DistributedSampler(train_dataset, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=4, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=4)
```

## 5. 模型配置与初始化

### 5.1 加载 Qwen-7B 模型

使用 Hugging Face Transformers 库加载预训练模型：

```python
from transformers import AutoModelForCausalLM

# 加载 Qwen-7B 模型
model_name = "Qwen-7B"
model = AutoModelForCausalLM.from_pretrained(model_name)
```

### 5.2 DeepSpeed 配置

使用 `DeepSpeed` 配置文件来进行优化，特别是启用混合精度训练和 Zero Redundancy Optimizer (ZeRO) 优化策略。

```python
ds_config = {
    "train_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "zero_optimization": {
        "stage": 2,
        "offload_param": True,
        "offload_optimizer": True
    },
    "fp16": {
        "enabled": True
    }
}
```

### 5.3 DeepSpeed 初始化

通过 `deepspeed.initialize` 来初始化 DeepSpeed 配置，并将模型和优化器与 DeepSpeed 配合。

```python
import deepspeed

# 初始化 DeepSpeed
model, optimizer, _, _ = deepspeed.initialize(args=None, model=model, optimizer=None, config_params=ds_config)
```

### 5.4 LLaMA-Factory 初始化

通过 `LLaMA-Factory` 对训练过程进行优化和定制。以下是如何使用 `LLaMA-Factory` 配置训练优化器和超参数：

```python
from llama_factory import LlamaFactory

# 初始化 LLaMA-Factory
llama = LlamaFactory(model)

# 设置优化器和调度器
llama_config = {
    "learning_rate": 5e-5,
    "weight_decay": 1e-2,
    "max_grad_norm": 1.0
}

# 配置优化器
llama.setup_optimizer_and_scheduler(optimizer, **llama_config)
```

## 6. 训练循环

以下是完整的训练循环，包括前向传播、反向传播、优化步骤以及 checkpoint 保存。

```python
epochs = 3

# 分布式训练设置
if torch.distributed.is_initialized():
    torch.distributed.barrier()

# 训练循环
for epoch in range(epochs):
    model.train()
    for step, batch in enumerate(train_loader):
        inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        # 前向传播
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss

        # 反向传播
        model.backward(loss)

        # 使用 LLaMA-Factory 优化器步骤
        llama.step()

        # 输出信息
        if step % 10 == 0:
            print(f"Epoch {epoch}/{epochs} - Step {step}, Loss: {loss.item()}")

    # 保存 checkpoint
    if model.global_step % 1000 == 0:
        model.save_checkpoint("checkpoints", model.global_step)
```

## 7. 验证与模型保存

在训练完成后，我们执行验证步骤，并保存最终模型：

```python
# 验证阶段
model.eval()
with torch.no_grad():
    for batch in val_loader:
        inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        outputs = model(**inputs, labels=inputs['input_ids'])
        val_loss = outputs.loss
        print(f"Validation Loss: {val_loss.item()}")

# 保存最终模型
model.save_checkpoint("checkpoints", model.global_step)
```

## 8. 集群部署与训练

创建 pytorch job
