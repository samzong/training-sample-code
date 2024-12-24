import os
import torch
import deepspeed
import json
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from datasets import Dataset, DatasetDict
import logging
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 从环境变量获取配置
MODEL_PATH = os.getenv('MODEL_PATH', 'Qwen/Qwen2-0.5B')  # 模型路径
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './output')  # 输出目录
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '2'))  # 批次大小
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', '5'))  # 训练轮次
LEARNING_RATE = float(os.getenv('LEARNING_RATE', '5e-5'))  # 学习率
TRAIN_FILE = os.getenv('TRAIN_FILE', '')  # 训练数据文件路径
VAL_FILE = os.getenv('VAL_FILE', '')  # 验证数据文件路径

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data_from_file(file_path):
    """从文件加载数据"""
    if not file_path or not os.path.exists(file_path):
        return None
    
    logger.info(f"Loading data from file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        return {'text': texts}
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        return None

def generate_sample_data():
    """生成或加载训练数据"""
    # 尝试从文件加载训练数据
    train_data = load_data_from_file(TRAIN_FILE)
    val_data = load_data_from_file(VAL_FILE)
    
    # 如果没有指定文件或加载失败，使用默认数据
    if train_data is None:
        logger.info("Using default training data")
        train_data = {
            'text': [
                # 基础对话
                "用户: 你好\n助手: 你好！很高兴见到你。有什么我可以帮你的吗？",
                "用户: 请介绍一下自己\n助手: 我是一个AI助手，我可以帮助回答问题、编写代码、进行分析等任务。",
                "用户: 今天天气怎么样？\n助手: 作为AI助手，我无法直接获取实时天气信息。建议您查看天气预报或看看窗外。",
                
                # 知识问答
                "用户: 什么是人工智能？\n助手: 人工智能是让计算机模拟人类智能的科学技术，包括机器学习、深度学习等领域。它能够实现语言理解、图像识别、决策制定等功能。",
                "用户: 解释下什么是机器学习\n助手: 机器学习是人工智能的一个重要分支，它通过算法让计算机从数据中学习规律和模式，而不是通过明确的编程指令。常见的应用包括图像识别、语音识别和推荐系统等。",
                
                # 编程相关
                "用户: 写一个Python函数计算斐波那契数列\n助手: 好的，这是一个计算斐波那契数列的函数：\n\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
                "用户: 如何在Python中处理异常？\n助手: 在Python中，我们使用try-except语句处理异常：\n\ntry:\n    # 可能发生异常的代码\n    result = 10 / 0\nexcept ZeroDivisionError:\n    # 处理特定异常\n    print('除数不能为0')\nexcept Exception as e:\n    # 处理其他异常\n    print(f'发生错误：{e}')",
            ]
        }
    
    if val_data is None:
        logger.info("Using default validation data")
        val_data = {
            'text': [
                "用户: 2+2等于多少?\n助手: 2+2等于4。这是基础的加法运算。",
                "用户: 介绍下Python\n助手: Python是一种高级编程语言，以其简洁的语法和丰富的库而闻名。它被广泛应用于web开发、数据分析、人工智能等领域。",
                "用户: 如何学习编程？\n助手: 学习编程的建议：\n1. 选择一门适合入门的语言如Python\n2. 掌握基础语法\n3. 多动手练习\n4. 做项目积累经验\n5. 阅读他人代码\n6. 持续学习新技术",
            ]
        }
    
    # 打印数据集大小信息
    logger.info(f"Training data size: {len(train_data['text'])} samples")
    logger.info(f"Validation data size: {len(val_data['text'])} samples")
    
    return DatasetDict({
        'train': Dataset.from_dict(train_data),
        'validation': Dataset.from_dict(val_data)
    })

class TextDataset(Dataset):
    """数据集类"""
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

    def __getitem__(self, idx):
        item = {key: val[idx].clone() for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

def is_distributed():
    """判断是否为分布式训练环境"""
    return bool(int(os.environ.get('WORLD_SIZE', 0)) > 1)

def setup_training_environment():
    """设置训练环境（单机或分布式）"""
    if is_distributed():
        logger.info("Initializing distributed training environment...")
        deepspeed.init_distributed()
        local_rank = int(os.getenv('LOCAL_RANK', '0'))
    else:
        logger.info("Running in single-machine mode...")
        local_rank = 0
    
    return local_rank

def is_local_path(path):
    """判断是否为本地路径"""
    return os.path.exists(path) and os.path.isdir(path)

def load_model_and_tokenizer(model_path):
    """加载模型和分词器"""
    logger.info(f"Loading model from: {model_path}")
    
    # 判断是否为本地路径
    if is_local_path(model_path):
        logger.info("Loading from local directory...")
        kwargs = {
            "local_files_only": True,
            "trust_remote_code": True
        }
    else:
        logger.info("Loading from Hugging Face hub...")
        kwargs = {
            "trust_remote_code": True
        }
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        return tokenizer
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        raise

def train():
    # 设置训练环境
    local_rank = setup_training_environment()
    
    # 清理内存
    torch.cuda.empty_cache()
    
    # 初始化模型和tokenizer
    tokenizer = load_model_and_tokenizer(MODEL_PATH)
    
    # 根据环境选择模型配置
    model_kwargs = {
        "trust_remote_code": True,
        "local_files_only": is_local_path(MODEL_PATH)
    }
    
    if is_distributed():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,  # 分布式训练使用 FP16
            **model_kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float32,  # 单机训练使用 FP32
            device_map="auto",
            **model_kwargs
        )

    # 准备数据
    dataset = generate_sample_data()
    train_dataset = TextDataset(dataset['train']['text'], tokenizer)
    val_dataset = TextDataset(dataset['validation']['text'], tokenizer)
    del dataset

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=not is_distributed(),  # 分布式模式下关闭 shuffle
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,
        pin_memory=True
    )

    if is_distributed():
        # 分布式模式：使用 DeepSpeed
        logger.info("Initializing DeepSpeed engine...")
        with open('ds_config.json') as f:
            ds_config = json.load(f)
        
        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            config=ds_config,
            model_parameters=model.parameters(),
        )
    else:
        # 单机模式：使用普通的优化器
        logger.info("Initializing single-machine training...")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=0.01,
            eps=1e-8
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=50,
            num_training_steps=len(train_loader) * NUM_EPOCHS
        )
        model_engine = model  # 为了统一接口

    # 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        model_engine.train()
        total_loss = 0
        
        # 只在主进程显示进度条
        if local_rank == 0:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        else:
            progress_bar = train_loader
        
        for batch in progress_bar:
            # 将数据移到正确的设备上
            if is_distributed():
                batch = {k: v.to(model_engine.device) for k, v in batch.items()}
            else:
                batch = {k: v.to(model.device) for k, v in batch.items()}
            
            if is_distributed():
                # DeepSpeed 方式
                loss = model_engine(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                ).loss
                model_engine.backward(loss)
                model_engine.step()
            else:
                # 普通训练方式
                optimizer.zero_grad()
                outputs = model_engine(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            # 只在主进程或单机模式下更新进度条
            if local_rank == 0:
                total_loss += loss.item()
                if not is_distributed():
                    progress_bar.set_postfix({'loss': loss.item()})

        # 在主进程或单机模式下进行验证和保存
        if local_rank == 0:
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

            # 验证
            model_engine.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    if is_distributed():
                        batch = {k: v.to(model_engine.device) for k, v in batch.items()}
                        outputs = model_engine(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels']
                        )
                    else:
                        batch = {k: v.to(model.device) for k, v in batch.items()}
                        outputs = model_engine(**batch)
                    val_loss += outputs.loss.item()

            avg_val_loss = val_loss / len(val_loader)
            logger.info(f"Validation loss: {avg_val_loss:.4f}")

            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = os.path.join(OUTPUT_DIR, 'best_model')
                if is_distributed():
                    model_engine.save_checkpoint(best_model_path)
                else:
                    model.save_pretrained(best_model_path)
                tokenizer.save_pretrained(best_model_path)
                logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")

    # 在主进程或单机模式下保存最终模型
    if local_rank == 0:
        final_model_path = os.path.join(OUTPUT_DIR, 'final_model')
        if is_distributed():
            model_engine.save_checkpoint(final_model_path)
        else:
            model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        logger.info("Training completed and model saved!")

if __name__ == "__main__":
    train()
