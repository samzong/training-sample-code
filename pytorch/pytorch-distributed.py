import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


def train():
    # 打印环境信息
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    print(f"Rank: {rank}, World Size: {world_size}")

    # 初始化分布式环境
    try:
        if world_size > 1:
            dist.init_process_group("nccl")
            print("Distributed process group initialized successfully")
        else:
            print("Running in non-distributed mode")
    except Exception as e:
        print(f"Error initializing process group: {e}")
        return

    # 设置设备
    try:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
            print(f"Using CUDA device: {device}")
        else:
            device = torch.device("cpu")
            print("CUDA not available, using CPU")
    except Exception as e:
        print(f"Error setting device: {e}")
        device = torch.device("cpu")
        print("Falling back to CPU")

    try:
        model = SimpleModel().to(device)
        print("Model moved to device successfully")
    except Exception as e:
        print(f"Error moving model to device: {e}")
        return

    try:
        if world_size > 1:
            ddp_model = DDP(
                model,
                device_ids=[rank % torch.cuda.device_count()]
                if torch.cuda.is_available()
                else None,
            )
            print("DDP model created successfully")
        else:
            ddp_model = model
            print("Using non-distributed model")
    except Exception as e:
        print(f"Error creating DDP model: {e}")
        return

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # 生成一些随机数据
    try:
        data = torch.randn(100, 10, device=device)
        labels = torch.randn(100, 1, device=device)
        print("Data generated and moved to device successfully")
    except Exception as e:
        print(f"Error generating or moving data to device: {e}")
        return

    for epoch in range(10):
        try:
            ddp_model.train()
            outputs = ddp_model(data)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if rank == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        except Exception as e:
            print(f"Error during training epoch {epoch}: {e}")
            break

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
