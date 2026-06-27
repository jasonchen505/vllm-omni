# vLLM-Omni 全流程复现计划

> 基于 8x RTX 4090 (24GB) 硬件环境

---

## 一、硬件资源评估

### 1.1 硬件规格

| 资源 | 规格 | 备注 |
|------|------|------|
| GPU | 8x RTX 4090 | 24GB 显存, SM 89, Ada Lovelace |
| 总显存 | 192GB | 8 × 24GB |
| CUDA 能力 | 8.9 | 满足最低要求 7.0 |
| FP8 支持 | CUTLASS FP8 | 不支持 quack (Blackwell 专用) |

### 1.2 可运行模型矩阵

| 类别 | 模型 | 参数量 | 显存需求 | GPU 数量 | 优先级 |
|------|------|--------|----------|----------|--------|
| **TTS** | MOSS-TTS-Nano | 0.1B | ~10 GiB | 1 | P0 |
| **TTS** | Qwen3-TTS 0.6B | 0.6B | ~13.5 GiB | 1 | P0 |
| **TTS** | VoxCPM2 | 2B | ~22 GiB | 1 | P0 |
| **TTS** | CosyVoice3 | 0.5B | ~15 GiB | 1 | P1 |
| **图像** | SD3.5 Medium | ~2B | ~20 GiB | 1 | P0 |
| **图像** | OmniGen2 | ~3B | ~20 GiB | 1 | P1 |
| **全模态** | Qwen2.5-Omni-3B | 3B | ~18 GiB | 1 | P1 |
| **全模态** | MiniCPM-o 4.5 | 8B | ~160 GiB | 8 (TP=4) | P2 |
| **全模态** | Qwen3-Omni-30B | 30B (3B激活) | ~180 GiB | 8 (TP=4+阶段分离) | P2 |

---

## 二、复现阶段规划

### 阶段 0: 环境搭建 (Day 1)

**目标**: 搭建开发环境，验证基础功能

**步骤**:

```bash
# 1. 克隆项目
cd /data/home/yizhou
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni

# 2. 创建虚拟环境
conda create -n vllm-omni python=3.10 -y
conda activate vllm-omni

# 3. 安装依赖
pip install -e .
pip install -r requirements/cuda.txt

# 4. 验证安装
python -c "import vllm_omni; print(vllm_omni.__version__)"

# 5. 检查 GPU
nvidia-smi
python -c "import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
```

**验证标准**:
- [ ] 项目成功安装
- [ ] 8 张 4090 识别正常
- [ ] CUDA 版本兼容

---

### 阶段 1: 单卡 TTS 模型复现 (Day 2-3)

**目标**: 在单卡 4090 上运行最轻量的 TTS 模型

#### 1.1 MOSS-TTS-Nano (最轻量, ~10 GiB)

```bash
# 离线推理
python examples/offline_inference/text_to_speech/moss_tts_nano/end2end.py \
    --text "Hello, this is a test of MOSS TTS Nano." \
    --ref-audio /path/to/reference.wav \
    --output-dir ./output/moss_tts_nano

# 在线服务
vllm serve OpenMOSS-Team/MOSS-TTS-Nano \
    --omni --port 8091 --host 0.0.0.0

# 测试 API
curl http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input": "Hello world", "model": "MOSS-TTS-Nano"}' \
    --output output.wav
```

#### 1.2 Qwen3-TTS 0.6B (官方验证 4090)

```bash
# 使用官方配置
vllm serve Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
    --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
    --omni --port 8092 --host 0.0.0.0

# 离线测试
python examples/offline_inference/text_to_speech/qwen3_tts/end2end.py \
    --query-type CustomVoice \
    --text "Hello, this is Qwen3 TTS." \
    --output-dir ./output/qwen3_tts
```

#### 1.3 VoxCPM2 (2B, 已验证 4090, ~22 GiB)

```bash
# 离线推理
python examples/offline_inference/text_to_speech/voxcpm2/end2end.py \
    --text "Hello, this is VoxCPM2 speaking." \
    --output-dir ./output/voxcpm2

# 在线服务
vllm serve openbmb/VoxCPM2 \
    --omni --port 8093 --host 0.0.0.0
```

**学习要点**:
- TTS 模型的多阶段架构 (Talker + Code2Wav)
- 音频 codec (RVQ) 的工作原理
- 流式音频输出的实现

**验证标准**:
- [ ] 生成音频文件可正常播放
- [ ] 理解 TTS 流水线的各阶段
- [ ] 掌握部署配置参数

---

### 阶段 2: 单卡图像生成模型复现 (Day 4-5)

**目标**: 运行图像生成模型，理解 Diffusion 架构

#### 2.1 Stable Diffusion 3.5 Medium (~20 GiB)

```bash
# 离线推理
python examples/offline_inference/text_to_image/text_to_image.py \
    --model stabilityai/stable-diffusion-3.5-medium \
    --prompt "a beautiful sunset over the ocean, cinematic lighting" \
    --height 1024 --width 1024 \
    --num-inference-steps 28 \
    --output ./output/sd35_sunset.png

# 在线服务
vllm serve stabilityai/stable-diffusion-3.5-medium \
    --omni --port 8094 --host 0.0.0.0
```

#### 2.2 Z-Image-Turbo (需降低分辨率)

```bash
# 降低分辨率避免 OOM
python examples/offline_inference/text_to_image/text_to_image.py \
    --model Tongyi-MAI/Z-Image-Turbo \
    --prompt "a cup of coffee on a wooden table" \
    --height 512 --width 512 \
    --num-inference-steps 9 \
    --guidance-scale 0.0 \
    --output ./output/z_image_coffee.png
```

#### 2.3 TeaCache 加速验证

```bash
# 不使用 TeaCache
python examples/offline_inference/text_to_image/text_to_image.py \
    --model stabilityai/stable-diffusion-3.5-medium \
    --prompt "a cat sitting on a chair" \
    --output ./output/no_teacache.png

# 使用 TeaCache
python examples/offline_inference/text_to_image/text_to_image.py \
    --model stabilityai/stable-diffusion-3.5-medium \
    --prompt "a cat sitting on a chair" \
    --cache-backend tea_cache \
    --tea-cache-rel-l1-thresh 0.2 \
    --output ./output/with_teacache.png

# 对比生成时间和质量
```

**学习要点**:
- Diffusion 模型的前向/反向过程
- Flow-Matching vs DDPM 的区别
- TeaCache 的工作原理和效果

**验证标准**:
- [ ] 生成图像质量正常
- [ ] TeaCache 加速效果明显
- [ ] 理解 Diffusion 采样过程

---

### 阶段 3: 多卡并行策略复现 (Day 6-7)

**目标**: 掌握 Tensor Parallelism 和多阶段部署

#### 3.1 双卡 Tensor Parallelism

```bash
# ERNIE-Image 双卡部署
vllm serve baidu/ERNIE-Image \
    --omni \
    --tensor-parallel-size 2 \
    --enable-cpu-offload \
    --port 8095 --host 0.0.0.0

# 测试
python examples/offline_inference/text_to_image/text_to_image.py \
    --model baidu/ERNIE-Image \
    --prompt "a futuristic city" \
    --output ./output/ernie_city.png
```

#### 3.2 四卡 MiniCPM-o 4.5 (TP=4)

```bash
# 使用专用 8x4090 配置 (实际使用 4 卡)
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve openbmb/MiniCPM-o-4_5 \
    --deploy-config vllm_omni/deploy/minicpmo_4_5_8x4090.yaml \
    --trust-remote-code \
    --port 8096 --host 0.0.0.0
```

#### 3.3 八卡全模态模型

```bash
# Qwen2.5-Omni-7B (如果显存允许)
vllm serve Qwen/Qwen2.5-Omni-7B \
    --omni \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.85 \
    --port 8097 --host 0.0.0.0
```

**学习要点**:
- Tensor Parallelism 的原理和实现
- 多阶段模型的 GPU 分配策略
- KV Cache 在多卡间的传输

**验证标准**:
- [ ] 多卡部署成功
- [ ] 理解 TP 切分原理
- [ ] 掌握 GPU 资源分配

---

### 阶段 4: Async Chunk 流水线复现 (Day 8-9)

**目标**: 验证 Async Chunk 对 TTFP 的优化效果

#### 4.1 对比实验

```bash
# 不使用 Async Chunk
vllm serve Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
    --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
    --omni --port 8098

# 使用 Async Chunk (修改配置)
# 编辑 vllm_omni/deploy/qwen3_tts.yaml，设置 async_chunk: true
vllm serve Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
    --deploy-config vllm_omni/deploy/qwen3_tts_async_chunk.yaml \
    --omni --port 8099

# 运行基准测试
python benchmarks/tts/benchmark_serving.py \
    --base-url http://localhost:8098 \
    --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
    --num-prompts 10 \
    --output results_no_async.json

python benchmarks/tts/benchmark_serving.py \
    --base-url http://localhost:8099 \
    --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
    --num-prompts 10 \
    --output results_async.json

# 对比 TTFP 和 E2E 延迟
```

**学习要点**:
- Async Chunk 的流水线并行原理
- TTFP vs E2E 的权衡
- 阶段预热和增量传输

**验证标准**:
- [ ] TTFP 显著降低
- [ ] 理解流水线并行原理
- [ ] 掌握性能调优方法

---

### 阶段 5: KV Cache 传输机制复现 (Day 10-11)

**目标**: 理解跨阶段 KV Cache 传输

#### 5.1 多阶段 TTS 部署

```bash
# Qwen3-TTS 完整流水线 (Thinker + Talker + Code2Wav)
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --omni --port 8100 --host 0.0.0.0

# 查看日志中的 KV 传输信息
# 关注 "KV transfer OK" 日志
```

#### 5.2 监控 KV 传输指标

```bash
# 启用统计日志
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --omni --log-stats --port 8100

# 查看 Prometheus 指标
curl http://localhost:8100/metrics | grep transfer

# 关键指标：
# vllm:omni_transfer_size_bytes
# vllm:omni_transfer_tx_s
# vllm:omni_transfer_rx_s
```

**学习要点**:
- KV Cache 的序列化格式
- 异构 TP 下的 KV 路由
- 异步预取和降级机制

**验证标准**:
- [ ] 多阶段流水线正常运行
- [ ] 理解 KV 传输机制
- [ ] 掌握监控指标含义

---

### 阶段 6: 量化和优化复现 (Day 12-13)

**目标**: 验证量化对性能和质量的影响

#### 6.1 FP8 量化

```bash
# 不量化
python examples/offline_inference/text_to_image/text_to_image.py \
    --model stabilityai/stable-diffusion-3.5-medium \
    --prompt "a landscape painting" \
    --output ./output/no_quant.png

# FP8 量化
python examples/offline_inference/text_to_image/text_to_image.py \
    --model stabilityai/stable-diffusion-3.5-medium \
    --prompt "a landscape painting" \
    --quantization fp8 \
    --output ./output/fp8_quant.png

# 对比显存占用和生成质量
```

#### 6.2 GGUF 量化

```bash
# GGUF Q4 量化 (如果模型支持)
vllm serve Qwen/Qwen-Image \
    --omni \
    --diffusion-quantization-config '{"method":"gguf","gguf_model":"QuantStack/Qwen-Image-GGUF/Qwen_Image-Q4_K_M.gguf"}' \
    --port 8101
```

#### 6.3 CPU Offload

```bash
# 层级卸载
vllm serve baidu/ERNIE-Image \
    --omni \
    --enable-layerwise-offload \
    --port 8102

# 模型级卸载
vllm serve Wan-AI/Wan2.2-T2V-14B-Diffusers \
    --omni \
    --enable-cpu-offload \
    --port 8103
```

**学习要点**:
- 量化方法的原理和适用场景
- CPU Offload 的权衡
- 显存和速度的平衡

**验证标准**:
- [ ] 量化后显存占用降低
- [ ] 理解不同量化方法的区别
- [ ] 掌握优化配置方法

---

### 阶段 7: 分布式部署复现 (Day 14-15)

**目标**: 掌握多节点分布式部署

#### 7.1 单节点多阶段分离

```bash
# Stage 0 (主节点)
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen2.5-Omni-7B \
    --omni \
    --stage-id 0 \
    --tensor-parallel-size 2 \
    --omni-master-address 127.0.0.1 \
    --omni-master-port 26000 \
    --port 8104

# Stage 1 (工作节点)
CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen2.5-Omni-7B \
    --omni \
    --stage-id 1 \
    --headless \
    --omni-master-address 127.0.0.1 \
    --omni-master-port 26000
```

#### 7.2 共享内存连接器配置

```yaml
# 自定义部署配置
async_chunk: true

connectors:
  connector_of_shared_memory:
    name: SharedMemoryConnector
    extra:
      initial_codec_chunk_frames: 4
      codec_chunk_frames: 25
      codec_left_context_frames: 25

stages:
  - stage_id: 0
    devices: "0,1"
    tensor_parallel_size: 2
    gpu_memory_utilization: 0.8
    max_num_seqs: 32
  - stage_id: 1
    devices: "2"
    gpu_memory_utilization: 0.6
    max_num_seqs: 32
  - stage_id: 2
    devices: "2"
    gpu_memory_utilization: 0.2
    max_num_seqs: 32
```

**学习要点**:
- OmniConnector 的工作原理
- 共享内存 vs RDMA 的选择
- 多节点协调机制

**验证标准**:
- [ ] 多阶段分离部署成功
- [ ] 理解连接器架构
- [ ] 掌握分布式配置

---

### 阶段 8: 完整流水线复现 (Day 16-18)

**目标**: 运行完整的多模态推理流水线

#### 8.1 Qwen3-Omni 完整流水线 (8 卡)

```bash
# 使用 8 卡部署 Qwen3-Omni
# Thinker: TP=4 (GPU 0-3)
# Talker: GPU 4
# Code2Wav: GPU 4

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --omni \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.85 \
    --port 8105 --host 0.0.0.0

# 测试多模态输入输出
curl http://localhost:8105/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "messages": [
            {"role": "user", "content": "Describe this image and speak the description."}
        ],
        "modalities": ["text", "audio"]
    }'
```

#### 8.2 MiniCPM-o 4.5 完整流水线 (8 卡)

```bash
# 使用专用 8x4090 配置
vllm serve openbmb/MiniCPM-o-4_5 \
    --deploy-config vllm_omni/deploy/minicpmo_4_5_8x4090.yaml \
    --trust-remote-code \
    --port 8106 --host 0.0.0.0
```

**学习要点**:
- 全模态模型的架构设计
- 多阶段流水线的协调
- 异构输出的处理

**验证标准**:
- [ ] 完整流水线正常运行
- [ ] 支持多模态输入输出
- [ ] 理解全模态架构

---

### 阶段 9: 性能基准测试 (Day 19-20)

**目标**: 建立性能基线，验证优化效果

#### 9.1 TTS 基准测试

```bash
# 运行 TTS 基准测试
python benchmarks/tts/benchmark_serving.py \
    --base-url http://localhost:8100 \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --num-prompts 50 \
    --concurrency 1,4,8,16 \
    --output ./benchmarks/results/tts_benchmark.json
```

#### 9.2 图像生成基准测试

```bash
# 运行图像生成基准测试
python benchmarks/diffusion/diffusion_benchmark_serving.py \
    --base-url http://localhost:8094 \
    --model stabilityai/stable-diffusion-3.5-medium \
    --task t2i \
    --dataset vbench \
    --num-prompts 20 \
    --output ./benchmarks/results/image_benchmark.json
```

#### 9.3 监控指标收集

```bash
# 启用详细统计
vllm serve <model> --omni --log-stats --port 8100

# 收集 Prometheus 指标
curl http://localhost:8100/metrics > ./benchmarks/results/metrics.txt

# 关键指标：
# - vllm:omni_e2e_request_latency_s
# - vllm:omni_audio_ttfp_s
# - vllm:omni_audio_rtf
# - vllm:num_requests_running
```

**学习要点**:
- 性能指标的含义和计算方法
- 基准测试的标准化流程
- 性能瓶颈的识别方法

**验证标准**:
- [ ] 建立性能基线数据
- [ ] 理解各项性能指标
- [ ] 掌握性能调优方法

---

## 三、时间规划

| 阶段 | 天数 | 内容 | 产出 |
|------|------|------|------|
| 0 | 1 | 环境搭建 | 可运行的开发环境 |
| 1 | 2 | 单卡 TTS | 3 个 TTS 模型运行成功 |
| 2 | 2 | 单卡图像生成 | 2 个图像模型 + TeaCache 验证 |
| 3 | 2 | 多卡并行 | TP 部署 + MiniCPM-o 运行 |
| 4 | 2 | Async Chunk | TTFP 优化验证 |
| 5 | 2 | KV 传输 | 多阶段流水线 + 监控 |
| 6 | 2 | 量化优化 | FP8/GGUF 量化验证 |
| 7 | 2 | 分布式部署 | 多节点部署成功 |
| 8 | 3 | 完整流水线 | 全模态模型运行 |
| 9 | 2 | 性能测试 | 基准测试报告 |
| **总计** | **20** | | |

---

## 四、风险和应对

| 风险 | 影响 | 应对方案 |
|------|------|----------|
| 4090 OOM | 无法运行大模型 | 使用量化/卸载/降低分辨率 |
| CUDA 兼容性问题 | 编译失败 | 使用预编译 Docker 镜像 |
| 模型下载慢 | 延迟进度 | 提前下载，使用镜像源 |
| 网络问题 | KV 传输失败 | 使用 SharedMemoryConnector |

---

## 五、预期成果

1. **技术掌握**:
   - 理解 vLLM-Omni 的三层架构
   - 掌握多阶段流水线设计
   - 理解 KV Cache 传输机制
   - 掌握并行策略和量化方法

2. **工程能力**:
   - 能够部署和调优多模态模型
   - 能够进行性能基准测试
   - 能够排查常见问题

3. **面试准备**:
   - 准备 5 个以上可讨论的技术点
   - 准备 3 个以上实际案例
   - 准备性能数据和优化经验

---

**计划版本**: v1.0
**最后更新**: 2026-06-27
**硬件环境**: 8x RTX 4090 (24GB)
