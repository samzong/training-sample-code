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

## Benchmark

```bash
guidellm benchmark \
  --target "http://127.0.0.1:1234" \
  --processor "Qwen/Qwen3-0.6B" \
  --rate-type "synchronous" \
  --max-seconds 10 \
  --max-requests 10 \
  --data "prompt_tokens=128,output_tokens=128"
```

## Similar Projects

Some projects similar to guidellm, for reference only, and have not been extensively tested.

- [Inference Benchmarker](https://github.com/huggingface/inference-benchmarker)