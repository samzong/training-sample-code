# Draft

```bash
guidellm benchmark \
  --processor "/Users/x/.lmstudio/models/lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit" \ # 目录里要有 tokenizer.json 等文件
  --target "http://127.0.0.1:1234" \
  --rate-type "synchronous" \
  --max-seconds 10 \
  --max-requests 10 \
  --model "qwen/qwen3-4b-2507" \
  --data "prompt_tokens=128,output_tokens=128"


OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES guidellm benchmark \
  --target "http://localhost:1234" \
  --rate-type sweep \
  --max-seconds 30 \
  --rate 40 \
  --max-requests 10 \
  --data "prompt_tokens=32,output_tokens=32,samples=1" \
  --model "qwen/qwen3-4b-2507" \
  --processor "/Users/x/.lmstudio/models/lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit"


inference-benchmarker \
  --tokenizer-name "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit" \
  --max-vus 10 \
  --duration 120s \
  --url http://localhost:1234 \
  --warmup 30s \
  --benchmark-kind rate \
  --rates 1.0 \
  --prompt-options "num_tokens=200,max_tokens=220,min_tokens=180,variance=10" \
  --decode-options "num_tokens=200,max_tokens=220,min_tokens=180,variance=10"
```

测试 rate， 固定并发测试 TTFT 

```bash
inference-benchmarker \
  --tokenizer-name "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit" \
  --max-vus 10 \
  --duration 120s \
  --url http://localhost:1234 \
  --warmup 30s \
  --benchmark-kind rate \
  --rates 1.0 \
  --prompt-options "num_tokens=200,max_tokens=220,min_tokens=180,variance=10" \
  --decode-options "num_tokens=200,max_tokens=220,min_tokens=180,variance=10"
```

测试 QPS

```bash
HF_HUB_OFFLINE=1 inference-benchmarker \
  --tokenizer-name "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit" \
  --max-vus 5 \
  --duration 120s \
  --url http://localhost:1234 \
  --warmup 30s \
  --benchmark-kind sweep \
  --prompt-options "num_tokens=64,max_tokens=64,min_tokens=32,variance=16" \
  --decode-options "num_tokens=64,max_tokens=64,min_tokens=32,variance=16"
```

参数介绍

| 参数                 | 含义                           | 说明                                                                                                                                                     |
| ------------------ | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--tokenizer-name` | 模型对应的 **tokenizer 名称**       | 用于计算输入 token 数量。这里指定了一个 LM Studio 导出的量化模型 `"lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit"`。                                                      |
| `--max-vus`        | 最大虚拟用户数 (VUs, Virtual Users) | 模拟同时发起请求的最大并发数，100 表示最多同时有 100 个虚拟请求线程在跑。                                                                                                              |
| `--duration`       | 压测持续时间                       | 整个 benchmark 持续 120 秒。                                                                                                                                 |
| `--url`            | 被测服务器地址                      | 即 LLM 推理服务的 OpenAI-兼容接口，这里是 `http://localhost:1234`。                                                                                                   |
| `--warmup`         | 预热时间                         | 压测开始前先跑 30 秒预热，不计入统计，用于让模型和 GPU 稳定。                                                                                                                    |
| `--benchmark-kind` | 压测模式                         | `rate` 模式表示固定请求速率（open-loop），按设定的 QPS 发请求，而不是固定并发数。其它模式有：`sweep`（扫 QPS）、`throughput`（固定并发）。                                                            |
| `--rates`          | 请求速率 (QPS)                   | 每秒发起的请求数，这里设置为 1、5、10（可多次指定）。每个数值表示一个阶段的速率。                                                                                                            |
| `--prompt-options` | 输入提示词配置                      | 控制输入 prompt 的 token 长度分布。参数为：<br>• `num_tokens=200` 平均 200 tokens<br>• `max_tokens=220` 最大 220<br>• `min_tokens=180` 最小 180<br>• `variance=10` 随机波动 10 |
| `--decode-options` | 输出解码配置                       | 控制模型生成输出的 token 数量分布，与上类似。用于模拟真实生成长度的方差。                                                                                                               |
