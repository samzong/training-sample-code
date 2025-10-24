# Benchmarks

Currently, we use guidellm as a benchmark tool.

Visit https://github.com/vllm-project/guidellm for more details.

## How to Install

The best way is to clone the repository and install it locally.

> **Note:** Requires Python 3.13

```bash
uv venv .venv --python 3.13
source .venv/bin/activate

# Install from repository root
uv pip install -e .
```

## Mock Server

With guidellm, you can easily start an OpenAI-compatible API server like this.

```bash
# For more options, run: guidellm mock-server --help
guidellm mock-server --port 8008 --model gpt-oss
```

## Benchmark Modes

| Mode        | Purpose                         | Use Cases                           | Frequency            | When to Use                        |
| :---------- | :------------------------------ | :---------------------------------- | :------------------- | :--------------------------------- |
| sweep       | Auto-scan for system capacity   | Initial baseline, rate range test   | High: default mode   | New service or configuration       |
| constant    | Fixed rate to measure response  | Near capacity limits, SLA testing   | High: real traffic    | Setting SLA or on-call checks      |
| throughput  | Max concurrent to peak TPS      | Capacity planning, hardware margin  | Medium-High: with sweep | Evaluating scaling or bottlenecks  |
| synchronous | Single-threaded latency test    | New model test, major changes      | Medium: per major update | Pre-deployment stability check     |
| poisson     | Random traffic to test resilience| Simulate natural request arrival   | Medium: complement constant | Evaluating queuing tolerance       |
| concurrent  | Fixed concurrency latency curve | Compare different configs/versions | Low-Medium: tuning   | Reproducing production patterns    |

### Synchronous Mode: rate-type "synchronous"

Sends one request at a time, used to measure single-request performance.

```bash
guidellm benchmark \
  --target "http://127.0.0.1:1234" \
  --processor "/Users/x/.lmstudio/models/lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit" \
  --rate-type "synchronous" \
  --max-seconds 120 \
  --max-requests 100 \
  --model "qwen/qwen3-4b-2507" \
  --data "prompt_tokens=256,output_tokens=256,samples=10"
```

| **Parameter**                                           | **Description**                                              |
| ------------------------------------------------------- | ------------------------------------------------------------ |
| --target http://127.0.0.1:1234                          | Target inference server, typically vLLM's OpenAI-compatible API endpoint |
| --processor ... Qwen3-4B-Instruct-2507-MLX-4bit         | Tokenizer/processor for token calculation, must match the model |
| --rate-type synchronous                                 | **Synchronous mode**: sends one request at a time, measures single-request performance |
| --max-seconds 120                                       | Maximum duration for benchmark run: 120 seconds              |
| --max-requests 100                                      | Maximum number of requests to send: 100                      |
| --data prompt_tokens=256,output_tokens=256,samples=1000 | Synthetic data: average prompt length 256 tokens, generate 256 tokens, 1000 samples total |

### Concurrent Mode: rate-type "concurrent"

Simulates a fixed number of concurrent requests (e.g., 8 clients requesting simultaneously).

```bash
guidellm benchmark run \
  --target "http://127.0.0.1:1234" \
  --processor "/Users/x/.lmstudio/models/lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit" \
  --rate-type concurrent \
  --rate 8 \
  --data "prompt_tokens=256,output_tokens=256,samples=10"
```

### Constant Rate Mode: rate-type "constant"

Simulates fixed-rate requests (constant-rate asynchronous load).

```bash
guidellm benchmark run \
  --target "http://127.0.0.1:1234" \
  --processor "/Users/x/.lmstudio/models/lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit" \
  --rate-type constant \
  --rate 8 \
  --max-seconds 120 \
  --max-requests 100 \
  --data "prompt_tokens=256,output_tokens=256,samples=10"
```

### Sweep Mode: rate-type "sweep"

Auto-scans from minimum to maximum rate to find system capacity limits.

```bash
guidellm benchmark run \
  --target "http://127.0.0.1:1234" \
  --processor "/Users/x/.lmstudio/models/lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit" \
  --rate-type sweep \
  --max-seconds 120 \
  --data "prompt_tokens=256,output_tokens=256,samples=10"
```

### Poisson Distribution Mode: rate-type "poisson"

Simulates real production traffic (Poisson distribution), load pattern closer to user request fluctuations.

```bash
guidellm benchmark run \
  --target "http://127.0.0.1:1234" \
  --processor "/Users/x/.lmstudio/models/lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit" \
  --rate-type poisson \
  --rate 8 \
  --max-seconds 120 \
  --max-requests 100 \
  --data "prompt_tokens=256,output_tokens=256,samples=10"
```

### Throughput Mode: rate-type "throughput"

Initiates all requests in parallel to measure server maximum throughput (max TPS / Token/s).

```bash
guidellm benchmark run \
  --target "http://127.0.0.1:1234" \
  --processor "/Users/x/.lmstudio/models/lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit" \
  --rate-type throughput \
  --max-seconds 120 \
  --max-requests 100 \
  --data "prompt_tokens=256,output_tokens=256,samples=10"
```

## Similar Projects

Some projects similar to guidellm, for reference only, and have not been extensively tested.

- [Inference Benchmarker](https://github.com/huggingface/inference-benchmarker)