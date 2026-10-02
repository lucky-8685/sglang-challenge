# HW2-珠姆吉

本目录整理 SGLang 前缀缓存实验（任务一）和 `/generate` 请求主流程追踪（任务二）。实验基于 SGLang v0.5.14、Qwen3-0.6B 与 RadixCache。

## 目录与报告数据对应

- `report.pdf`：任务一实验设计、两轮结果与分析；任务二流程图、关键函数和不超过一页的说明。
- `AI 使用说明情况（第二次挑战）.pdf`：AI 使用说明。
- `作业感受.pdf`：本次挑战的完成过程、困难与体会。
- `src/target1/run_target1.py`：任务一流式请求、缓存清理、共享前缀预热、并发测量和汇总脚本。
- `src/target1/data/shared_prefix/`：共享前缀组 32 条请求及独立预热请求。
- `src/target1/data/dispersed_prefix/`：分散前缀组 32 条请求。
- `results/target1/shared_prefix/run-1/`、`results/target1/dispersed_prefix/run-1/`：第一轮逐请求 CSV 和汇总 JSON；对应报告表格中的 run-1。
- `results/target1/shared_prefix/run-2/`、`results/target1/dispersed_prefix/run-2/`：第二轮逐请求 CSV 和汇总 JSON；对应报告表格中的 run-2。

CSV 保存逐请求记录；同目录 `summary.json` 保存成功率、吞吐、缓存命中、实际 Prefill token 数及 TTFT/TPOT/E2E 的 p50、p95。报告数字由这些汇总文件整理。

## 实验环境

- 操作系统：Windows 11 + WSL Ubuntu（实验在 WSL 中运行）
- CPU：Intel(R) Core(TM) i9-14900HX
- GPU（CUDA 推理设备）：NVIDIA GeForce RTX 5060 Laptop GPU（由实验环境中的 `torch.cuda.get_device_name(0)` 确认）。机器另有 Intel(R) UHD Graphics 集成显卡；它不是本次 CUDA 推理设备。
- SGLang：0.5.14
- PyTorch：2.11.0+cu130
- CUDA：PyTorch CUDA 可用（True）；PyTorch 构建版本为 `2.11.0+cu130`；设备为 NVIDIA GeForce RTX 5060 Laptop GPU
- Ray：2.56.0
- 模型：本地目录 `/home/elaine/models/Qwen3-0.6B`（Qwen3-0.6B）
- Python：3.10.12

## 安装与启动

安装方式：在 WSL Ubuntu 中使用 Python 虚拟环境 `.venv` 管理依赖。原始安装命令未保留；实验记录确认该环境包含 SGLang 0.5.14、PyTorch 2.11.0+cu130、Ray 2.56.0 及 `requests`。本作业包不包含虚拟环境本身。复用已配置的 `.venv` 后，进入原实验仓库并启动模型服务：

```bash
cd ~/sglang-challenge
source .venv/bin/activate
python -m sglang.launch_server --model-path /home/elaine/models/Qwen3-0.6B --host 0.0.0.0 --port 30000
```

保持服务终端运行，再开一个 WSL 终端运行实验。将本作业目录放在 `~/sglang-challenge/HW2-珠姆吉/` 后：

```bash
cd ~/sglang-challenge/HW2-珠姆吉
source ../.venv/bin/activate
python src/target1/run_target1.py --run-id run-3
```

依赖：Python 包 `requests`；模型服务须已在 `127.0.0.1:30000` 启动。脚本通过 `/v1/models` 检查服务，先清空缓存并预热共享前缀组，再测共享组；之后再次清空缓存并测分散组。每组 32 条请求、最大并发 8、`max_new_tokens=16`、`temperature=0`、`ignore_eos=true`、`sampling_seed=2026`，请求以流式方式发送。新结果写到 `results/target1/<group>/run-3/`，不会覆盖 run-1、run-2。
