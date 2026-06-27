# LLM 算法实习面试准备指南

> 基于 vLLM-Omni 项目深度分析，为 LLM & Agent 应用及后训练方向的面试准备

---

## 目录

1. [项目概述与技术亮点](#1-项目概述与技术亮点)
2. [核心架构理解](#2-核心架构理解)
3. [关键技术深挖点](#3-关键技术深挖点)
4. [系统设计面试题](#4-系统设计面试题)
5. [算法相关深挖点](#5-算法相关深挖点)
6. [面试常见问题与回答要点](#6-面试常见问题与回答要点)
7. [自我介绍建议](#7-自我介绍建议)
8. [扩展知识点](#8-扩展知识点)

---

## 1. 项目概述与技术亮点

### 1.1 项目定位

**vLLM-Omni** 是 vLLM 社区官方维护的**全模态模型推理与服务框架**，将 vLLM 从文本自回归推理扩展到支持：
- **多模态输入/输出**：文本、图像、视频、音频
- **非自回归架构**：Diffusion Transformer (DiT) 等并行生成模型
- **异构输出**：从文本到多模态输出的统一服务

### 1.2 核心技术亮点（面试亮点）

| 技术亮点 | 面试价值 | 关键点 |
|---------|---------|--------|
| **完全解耦架构** | 系统设计能力 | 将复杂多模态推理分解为独立阶段，统一编排 |
| **KV Cache 跨阶段传输** | 缓存管理理解 | 异构 TP 下的 KV 分片、异步预取、序列化优化 |
| **Async Chunk 流水线** | 性能优化思维 | 阶段预热 + 增量传输，TTFP 降低 99.7% |
| **多种并行策略** | 分布式系统理解 | TP/SP/PP/DP/CFG/EP 六维并行 |
| **TeaCache 步缓存** | 算法优化能力 | 基于 L1 距离跳过相似 denoising 步骤 |

### 1.3 性能数据（面试可引用）

**Qwen3-Omni TTS 优化效果**：
- 端到端延迟：336s → 23.78s（**93% 降低**）
- 首包延迟 TTFP：336s → 0.93s（**99.7% 降低**）
- 实时因子 RTF：3.776 → 0.32（**12 倍加速**）

---

## 2. 核心架构理解

### 2.1 三层架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    Entrypoints Layer                         │
│   OmniBase → AsyncOmni / Omni (用户API)                     │
│   OpenAI-compatible API Server                              │
└──────────────────────┬──────────────────────────────────────┘
                       │ janus Queue (sync ↔ async bridge)
┌──────────────────────▼──────────────────────────────────────┐
│                  Engine Layer                                │
│   AsyncOmniEngine (前台薄代理)                               │
│     ↓ 后台线程                                              │
│   Orchestrator (异步事件循环)                                │
│     ├── StagePool[0] → StageClient (AR阶段)                 │
│     ├── StagePool[1] → StageClient (Talker阶段)             │
│     └── StagePool[2] → StageClient (Diffusion阶段)          │
└──────────────────────┬──────────────────────────────────────┘
                       │ ZMQ / IPC
┌──────────────────────▼──────────────────────────────────────┐
│                  Worker Layer                                │
│   GPUARWorker / GPUGenerationWorker                         │
│     ├── ModelRunner (含 OmniConnectorModelRunnerMixin)       │
│     └── Scheduler (OmniARScheduler / OmniGenerationScheduler)│
│   DiffusionEngine                                           │
│     ├── Executor                                            │
│     └── Worker (多GPU分布式)                                 │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 关键设计模式

**面试回答框架**：设计模式 → 应用场景 → 解决的问题

| 设计模式 | 应用位置 | 解决的问题 |
|---------|---------|-----------|
| **Thin Proxy** | AsyncOmniEngine | 前台同步 API 与后台异步执行解耦 |
| **Stage Pool** | StagePool | 逻辑阶段副本管理 + 负载均衡 + 亲和性 |
| **Registry** | OMNI_MODELS, OMNI_PIPELINES | 模型/流水线声明式注册，运行时动态解析 |
| **Strategy** | LoadBalancer, TensorAccumulationStrategy | 可替换的负载均衡和累积策略 |
| **Mixin** | OmniConnectorModelRunnerMixin | 横切关注点代码复用 |

### 2.3 请求执行流程（面试必知）

```
用户请求 → Omni.generate()
    ↓
InputProcessor (tokenization + 多模态处理)
    ↓
StageSubmissionMessage → request_queue
    ↓
Orchestrator._request_handler()
    ↓
StagePool.submit_initial() → Stage-0 StageClient
    ↓
Worker 层 ZMQ 接收 → Scheduler.schedule()
    ↓
EngineCoreOutputs 返回
    ↓
Orchestrator 路由 → _forward_to_next_stage()
    ↓
最终输出 → output_queue → OmniBase
```

---

## 3. 关键技术深挖点

### 3.1 KV Cache 跨阶段传输（高频考点）

**面试问题**：多阶段推理中，KV Cache 如何在不同阶段间高效传输？

**深挖点 1：触发条件控制**
```python
# omni_ar_scheduler.py:136
def _process_kv_transfer_trigger(self, request):
    # 两种触发条件：
    # 1. prefill_finished: confirmed_computed >= num_prompt_tokens
    # 2. special_token: 生成特定标记（如 EOS）时触发
```

**深挖点 2：异构 TP 下的 KV 路由**
- 发送方 TP=2, 接收方 TP=4：发送方预切片，每个目标 rank 只接收对应 head shard
- `KVTPTopology` 类管理源/目标 TP 大小映射
- `ReceiveRole` 枚举：LOCAL（本地直拉）、LEADER（拉取后分发）、FOLLOWER（集合通信接收）

**深挖点 3：异步预取机制**
```python
# kv_transfer_manager.py
class OmniKVTransferManager:
    _prefetch_executor: ThreadPoolExecutor  # 后台线程池
    # 专用 CUDA Stream 进行 H2D 拷贝
    # 内存检查：_has_enough_prefetch_device_mem
    # 内存不足时自动降级为同步接收
```

**深挖点 4：序列化格式选择**
| 格式 | 场景 | 特点 |
|------|------|------|
| 紧凑二进制 | CPU 端连接器 | 4字节头 + JSON头 + 连续tensor |
| GPU Tensor | raw-data 连接器 | 打包为单个 uint8 tensor，避免 CPU-GPU 来回 |
| 字典格式 | 降级方案 | 兼容性最好 |

### 3.2 Async Chunk 流水线（核心创新）

**面试问题**：如何降低多阶段推理的首包延迟？

**核心思想**：阶段预热 + 增量传输

```
时间 ─────────────────────────────────────────────▶

Thinker:  [===prefill===][==chunk1==][==chunk2==][==chunk3==]
                              │           │           │
                         (立即传给Talker)  │           │
                              ▼           ▼           ▼
Talker:                   [==chunk1==][==chunk2==][==chunk3==]
                              │           │           │
                         (立即传给C2W)     │           │
                              ▼           ▼           ▼
Code2Wav:                 [==decode1=][==decode2=][==decode3=]
                              │           │           │
                              ▼           ▼           ▼
输出:                      audio1     audio2     audio3
```

**关键实现**：
1. `_prewarm_async_chunk_stages()`：Stage-0 开始处理时，预先向下游阶段提交占位请求
2. 下游阶段预先分配资源（KV cache block、调度槽位）
3. KV Ready 信号驱动：`_handle_kv_ready_raw_outputs()` 监听 `kv_transfer_params.kv_ready`

**性能收益**：
- TTFP 降低 92%（6.5s → 0.52s）
- E2E 延迟降低 6-17%

### 3.3 多种并行策略（分布式系统考点）

**面试问题**：大规模模型推理有哪些并行策略？各适用于什么场景？

| 并行策略 | 原理 | 适用场景 | vLLM-Omni 实现 |
|---------|------|---------|---------------|
| **张量并行 TP** | 模型权重跨 GPU 分片 | 单层计算量大 | 复用 vLLM 上游 |
| **序列并行 SP** | 序列维度跨 GPU 分片 | 长序列 | Ulysses/Ring Attention |
| **流水线并行 PP** | 模型层跨 GPU 分片 | 深层模型 | AsyncLatents 延迟解析 |
| **数据并行 DP** | 模型复制，批次分割 | 高吞吐需求 | HSDP 2D DeviceMesh |
| **专家并行 EP** | MoE 专家跨 GPU 分片 | MoE 架构 | 与 TP 协同 |
| **CFG 并行** | 正/负预测并行 | 扩散模型 | all_gather 交换结果 |

**深挖点：AsyncLatents 机制**
```python
# diffusion/distributed/pipeline_parallel.py
class AsyncLatents:
    """包装 pending 的 irecv_tensor_dict，延迟 handle.wait() 直到 tensor 被实际消费"""
    # 通过 __getattr__ 和 __torch_function__ 协议实现透明代理
    # 保持 rank 0 在发送接收请求后非阻塞
```

### 3.4 TeaCache 步缓存（算法优化考点）

**面试问题**：扩散模型推理如何加速？

**核心洞察**：连续时间步的调制输入变化缓慢，可复用前一步的计算结果。

**实现原理**：
```python
# diffusion/cache/tea_cache.py
class TeaCache:
    # 1. 计算当前步与前一步的 L1 距离
    # 2. 使用多项式系数估算器预测是否可跳过
    # 3. 跳过时复用前一步的 hidden_states
    # 4. 每种模型有专用校准系数（FLUX、Qwen-Image、Bagel 等）
```

**加速效果**：1.5x-2.0x 加速，质量损失极小

**对比其他缓存策略**：
| 策略 | 原理 | 特点 |
|------|------|------|
| TeaCache | L1 距离 + 多项式估算 | 通用性好，校准简单 |
| MagCache | magnitude 缓存 | 对 magnitude 变化敏感 |
| Cache-DiT | 动态块级缓存 | 细粒度控制 |
| TaylorSeer | 泰勒展开预测 | 理论精度高 |

### 3.5 扩散模型量化（工程优化考点）

**面试问题**：如何在保持精度的前提下减少扩散模型的内存占用和计算量？

**Per-Component 量化**：
```python
# 不同组件使用不同量化方法
ComponentQuantizationConfig(
    transformer=DiffusionMXFP4Config(),  # W4A4
    vae=DiffusionInt8Config(),           # W8A8
    encoder=None                         # 不量化
)
```

**支持的量化方法**：
| 方法 | 精度 | 适用场景 |
|------|------|---------|
| GGUF | 灵活 | 通用扩散模型 |
| Int8 | W8A8 | CUDA + NPU |
| MXFP8 | W8A8 | NPU 专用 |
| MXFP4 | W4A4 | 极致压缩 |
| MXFP4 DualScale | W4A4 + BF16 混合 | 精度与压缩平衡 |

---

## 4. 系统设计面试题

### 4.1 设计一个多模态模型服务系统

**面试问题**：请设计一个支持文本、图像、音频输入，文本、图像、音频输出的模型服务系统。

**回答框架**：

1. **需求分析**
   - 输入模态：文本、图像、音频、视频
   - 输出模态：文本、图像、音频
   - 性能要求：低延迟、高吞吐
   - 扩展性：支持新模态和新模型

2. **架构设计**
   ```
   ┌─────────────────┐
   │   API Gateway   │
   └────────┬────────┘
            │
   ┌────────▼────────┐
   │   Orchestrator  │ ← 请求路由、阶段编排
   └────────┬────────┘
            │
   ┌────────▼────────┐
   │   Stage Pool    │ ← 阶段副本管理
   └────────┬────────┘
            │
   ┌────────▼────────┐
   │  Worker Layer   │ ← GPU 执行
   └─────────────────┘
   ```

3. **关键设计点**
   - **阶段抽象**：将复杂模型分解为独立阶段
   - **异构执行**：AR 阶段用 KV Cache，DiT 阶段用步调度
   - **统一连接器**：阶段间数据传输抽象
   - **流式输出**：Async Chunk 降低首包延迟

4. **扩展讨论**
   - 如何支持新模态？→ 扩展 InputProcessor 和 OutputModality
   - 如何处理长序列？→ 序列并行（Ulysses/Ring Attention）
   - 如何保证低延迟？→ KV 预取 + 流水线重叠

### 4.2 设计一个分布式 KV Cache 管理系统

**面试问题**：在多阶段推理中，如何设计一个高效的 KV Cache 传输和管理系统？

**回答要点**：

1. **触发机制**
   - Prefill 完成触发：当 AR 阶段 prefill 完成后立即提取 KV
   - 特殊 Token 触发：如 EOS 时触发
   - 延迟停止：确保请求在 KV 提取完成前不被停止

2. **序列化策略**
   - 紧凑二进制：4字节头 + JSON头 + 连续tensor（CPU 端）
   - GPU Tensor：打包为单个 uint8 tensor（GPU 端）
   - 选择依据：数据量、传输距离、硬件能力

3. **异构 TP 处理**
   - 发送方 TP < 接收方：预切片，每个目标 rank 只接收对应 shard
   - 发送方 TP > 接收方：合并多个 rank 的分片
   - LEADER-FOLLOWER 模式：LEADER 拉取后分发给 FOLLOWER

4. **异步预取**
   - 后台线程池 + 专用 CUDA Stream
   - 内存检查：预取前检查 GPU 空闲内存
   - 降级策略：内存不足时同步接收

### 4.3 设计一个扩散模型推理加速系统

**面试问题**：如何设计一个高效的扩散模型推理系统？

**回答要点**：

1. **步缓存优化**
   - TeaCache：L1 距离 + 多项式估算
   - 跳过相似步：连续时间步变化缓慢
   - 校准系数：每种模型专用

2. **并行策略**
   - CFG 并行：正/负预测并行，all_gather 交换
   - 序列并行：Ulysses/Ring Attention 处理长序列
   - 流水线并行：AsyncLatents 延迟解析

3. **内存优化**
   - Per-Component 量化：不同组件不同精度
   - CPU 卸载：模型级/层级卸载
   - Regional Compilation：编译单个 transformer block

4. **流式输出**
   - 视频：WebSocket 逐帧传输
   - 音频：逐 token 流式输出 codec
   - 图像：Latent 流式传输

---

## 5. 算法相关深挖点

### 5.1 自回归 vs 非自回归架构（基础概念）

**面试问题**：自回归和非自回归架构有什么区别？各适用于什么场景？

**回答要点**：

| 特性 | 自回归 (AR) | 非自回归 (NAR) |
|------|------------|---------------|
| 生成方式 | 逐 token 生成 | 并行生成所有 token |
| 典型模型 | GPT、LLaMA | DiT、Flow-Matching |
| 优势 | 质量高、连贯性好 | 速度快、可并行化 |
| 劣势 | 速度慢、难以并行 | 质量可能下降、缺乏全局一致性 |
| 适用场景 | 文本生成、TTS codec | 图像/视频生成、语音合成 |

**vLLM-Omni 的处理**：
- AR 阶段：继承 vLLM 的 KV Cache 管理和连续批处理
- NAR 阶段：独立的 DiffusionEngine，支持步级执行

### 5.2 扩散模型原理（算法深挖）

**面试问题**：请解释扩散模型的工作原理。

**核心概念**：

1. **前向过程（加噪）**
   ```
   x_t = √(α_t) * x_{t-1} + √(1-α_t) * ε
   ```
   - 逐步向数据添加高斯噪声
   - α_t 是噪声调度参数

2. **反向过程（去噪）**
   ```
   x_{t-1} = (1/√α_t) * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t, t))
   ```
   - 学习预测噪声 ε_θ
   - 从纯噪声逐步恢复数据

3. **Flow-Matching（现代方法）**
   ```
   dx = v_θ(x, t) * dt
   ```
   - 学习速度场 v_θ
   - 从噪声到数据的连续流

**vLLM-Omni 支持的扩散模型**：
- FLUX、Stable Diffusion 3：文本到图像
- Wan2.2、Cosmos3：文本到视频
- CosyVoice3：语音合成

### 5.3 MoE 架构理解（模型架构考点）

**面试问题**：请解释 MoE（Mixture of Experts）架构的工作原理。

**核心概念**：

1. **门控机制**
   ```
   G(x) = Softmax(TopK(x · W_g, k))
   ```
   - 为每个 token 选择 k 个专家
   - TopK 稀疏激活

2. **专家计算**
   ```
   y = Σ_i G(x)_i * E_i(x)
   ```
   - 每个专家独立处理
   - 加权求和得到输出

3. **负载均衡**
   - 辅助损失：鼓励专家均匀使用
   - 容量因子：限制每个专家处理的 token 数

**vLLM-Omni 中的 MoE 支持**：
- Qwen3-Omni：30B 总参数，3B 激活参数
- 专家并行 EP：专家跨 GPU 分布
- 与 TP 协同：EP + TP 混合并行

### 5.4 注意力机制变体（算法深挖）

**面试问题**：有哪些高效的注意力机制变体？

**回答要点**：

| 机制 | 原理 | 复杂度 | 适用场景 |
|------|------|--------|---------|
| **标准注意力** | Q·K^T / √d | O(n²) | 短序列 |
| **Flash Attention** | 分块 + 重计算 | O(n²) 但常数小 | 通用 |
| **Ring Attention** | 环形传递 KV | O(n²/p) | 长序列 + 多 GPU |
| **Ulysses Attention** | All-to-All 重分布 | O(n²/p) | 中等序列 + 多 GPU |
| **PagedAttention** | 分页 KV Cache | O(n²) 但内存高效 | 高吞吐服务 |

**vLLM-Omni 的实现**：
- AR 阶段：PagedAttention（继承 vLLM）
- Diffusion 阶段：Ring/Ulysses Attention（SP 并行）

### 5.5 量化技术（工程优化考点）

**面试问题**：有哪些模型量化方法？各有什么优缺点？

**回答要点**：

| 方法 | 精度 | 原理 | 优缺点 |
|------|------|------|--------|
| **INT8** | W8A8 | 线性量化 | 简单高效，精度损失小 |
| **FP8** | W8A8 | 浮点量化 | 动态范围大，硬件支持好 |
| **MXFP4** | W4A4 | 微浮点 | 压缩率高，需要校准 |
| **GGUF** | 灵活 | 混合精度 | 通用性好，工具链成熟 |
| **GPTQ** | W4/W8 | 逐层量化 | 精度高，需要校准数据 |
| **AWQ** | W4 | 激活感知 | 保护重要权重 |

**vLLM-Omni 的 Per-Component 量化**：
```python
# 不同组件使用不同量化方法
ComponentQuantizationConfig(
    transformer=DiffusionMXFP4Config(),  # 主干用高压缩
    vae=DiffusionInt8Config(),           # VAE 用中等压缩
    encoder=None                         # 编码器不量化
)
```

---

## 6. 面试常见问题与回答要点

### 6.1 项目相关问题

**Q1：请介绍一下你参与的这个项目。**

**回答要点**（2-3 分钟）：
1. **项目定位**：vLLM-Omni 是 vLLM 社区的全模态模型服务框架
2. **核心问题**：现有系统只支持单一范式（AR 或 DiT），缺乏对 any-to-any 管道的支持
3. **解决方案**：完全解耦的多阶段架构，统一编排 AR、NAR、Diffusion 阶段
4. **关键创新**：Async Chunk 流水线、KV 跨阶段传输、TeaCache 步缓存
5. **性能成果**：TTFP 降低 99.7%，E2E 延迟降低 93%

**Q2：你在这个项目中遇到了什么挑战？如何解决的？**

**回答示例**：
> 最大的挑战是 KV Cache 在异构 TP 下的高效传输。发送方 TP=2，接收方 TP=4 时，需要正确切片和合并 KV 分片。我设计了 KVTPTopology 类管理 TP 映射，实现了预切片和 LEADER-FOLLOWER 分发机制，解决了这个问题。

**Q3：这个项目对你有什么启发？**

**回答示例**：
> 这个项目让我深刻理解了系统设计中"解耦"的重要性。将复杂系统分解为独立阶段，通过统一接口连接，既提高了可扩展性，又便于独立优化。这种思想可以应用到很多场景，比如微服务架构、流水线设计等。

### 6.2 技术深挖问题

**Q4：为什么选择 Async Chunk 而不是 Full Payload？**

**回答要点**：
1. **延迟考虑**：Async Chunk 可以在 chunk 生成后立即传输，无需等待完整输出
2. **流水线重叠**：Stage-0 的 AR 解码与 Stage-1 的资源预分配重叠
3. **内存效率**：增量传输避免大 tensor 暂存
4. **适用场景**：TTS、实时语音等对延迟敏感的场景

**Q5：TeaCache 如何保证生成质量？**

**回答要点**：
1. **校准系数**：每种模型有专用的多项式系数，通过离线校准得到
2. **阈值控制**：L1 距离超过阈值时强制重新计算
3. **渐进退化**：连续跳过步数有限制，避免累积误差
4. **实验验证**：1.5x-2.0x 加速，FID/CLIP 等指标损失极小

**Q6：如何处理多阶段推理中的错误传播？**

**回答要点**：
1. **请求状态机**：每个请求有明确的状态转换（pending → running → finished）
2. **超时控制**：每个阶段有独立的超时设置
3. **错误隔离**：阶段间错误不传播，上游失败自动取消下游
4. **重试机制**：支持请求级重试，避免单点故障

### 6.3 系统设计问题

**Q7：如果要支持一个新的多模态模型，需要做什么？**

**回答步骤**：
1. **模型分析**：确定模型架构（AR/DiT/混合）、输入输出模态
2. **阶段划分**：将模型分解为独立阶段，确定执行类型
3. **注册配置**：在 `_OMNI_MODELS` 和 `OMNI_PIPELINES` 中注册
4. **输入处理**：实现 `stage_input_processors` 中的转换函数
5. **测试验证**：单元测试 + 集成测试 + 性能测试

**Q8：如何评估系统的性能瓶颈？**

**回答框架**：
1. **Profiling 工具**：PyTorch Profiler、NVIDIA Nsight
2. **关键指标**：TTFP、E2E 延迟、RTF、吞吐量
3. **瓶颈分析**：
   - 计算瓶颈：GPU 利用率低 → 优化算子
   - 内存瓶颈：OOM → 量化/卸载
   - 通信瓶颈：传输慢 → 优化连接器
4. **优化策略**：Async Chunk、TeaCache、量化、并行

---

## 7. 自我介绍建议

### 7.1 技术背景介绍模板

```
面试官您好，我是[姓名]，[学校]的[专业]硕士在读。

我的研究方向是[方向]，在[实验室/导师]指导下从事[研究内容]。

在实习期间，我参与了 vLLM-Omni 项目，这是一个面向全模态模型的推理服务框架。
我主要负责[具体工作]，解决了[具体问题]，取得了[具体成果]。

这个经历让我深入理解了[技术点]，也锻炼了我的[能力]。
```

### 7.2 项目经验介绍模板

```
项目名称：vLLM-Omni - 全模态模型推理服务框架

项目背景：
- 现有推理系统只支持单一范式（AR 或 DiT）
- 缺乏对 any-to-any 多模态管道的统一支持
- 多阶段推理的首包延迟过高

我的工作：
1. [具体任务1]：设计并实现了[功能]，解决了[问题]
2. [具体任务2]：优化了[模块]，性能提升了[X]%
3. [具体任务3]：参与了[设计/评审]，贡献了[想法/代码]

技术亮点：
- [亮点1]：Async Chunk 流水线，TTFP 降低 99.7%
- [亮点2]：KV Cache 跨阶段传输，支持异构 TP
- [亮点3]：TeaCache 步缓存，1.5x-2.0x 加速

成果：
- 性能：E2E 延迟降低 93%，RTF 从 3.776 降至 0.32
- 代码：提交了 X 个 PR，合并了 Y 个
- 文档：撰写了 Z 篇技术文档
```

### 7.3 技术能力展示

**可展示的技术栈**：
- **框架理解**：vLLM、PyTorch、Transformers
- **分布式系统**：CUDA、NCCL、Ray、ZMQ
- **优化技术**：量化、缓存、流水线、并行
- **工程能力**：Python、C++、性能调优、系统设计

**可提及的算法知识**：
- 自回归模型：GPT、LLaMA、Qwen
- 扩散模型：DDPM、Flow-Matching、DiT
- 注意力机制：Flash Attention、Ring Attention
- 并行策略：TP、SP、PP、DP、EP

---

## 8. 扩展知识点

### 8.1 相关论文推荐

| 论文 | 主题 | 与项目的关联 |
|------|------|-------------|
| **vLLM** (SOSP '23) | PagedAttention | 项目基座 |
| **Flash Attention** (NeurIPS '22) | 高效注意力 | AR 阶段优化 |
| **Ring Attention** (ICLR '24) | 长序列并行 | SP 并行实现 |
| **DiT** (ICCV '23) | Diffusion Transformer | 扩散模型架构 |
| **Flow-Matching** (ICLR '23) | 连续流生成 | 现代扩散方法 |
| **Qwen3-Omni** | 全模态模型 | 项目核心支持模型 |

### 8.2 面试官可能追问的方向

1. **系统设计追问**
   - 如何处理故障恢复？
   - 如何实现动态扩缩容？
   - 如何保证多租户隔离？

2. **算法深挖追问**
   - KV Cache 的内存占用如何计算？
   - 连续批处理和静态批处理有什么区别？
   - 如何评估量化后的模型质量？

3. **工程实践追问**
   - 如何进行性能调优？
   - 如何设计测试用例？
   - 如何处理代码冲突？

### 8.3 准备建议

1. **深入理解架构**：能够画出三层架构图，解释每个组件的作用
2. **掌握核心创新**：Async Chunk、KV 传输、TeaCache 的原理和实现
3. **准备具体案例**：能够描述一个你解决的技术问题及解决方案
4. **了解性能数据**：记住关键性能指标（TTFP、E2E、RTF）
5. **练习表达**：用 2-3 分钟清晰介绍项目，突出技术亮点

---

## 附录：关键代码位置索引

| 模块 | 文件路径 | 关键类/函数 |
|------|---------|------------|
| 调度器 | `vllm_omni/core/sched/omni_ar_scheduler.py` | `OmniARScheduler` |
| KV 传输 | `vllm_omni/distributed/omni_connectors/kv_transfer_manager.py` | `OmniKVTransferManager` |
| 编排器 | `vllm_omni/engine/orchestrator.py` | `Orchestrator` |
| 阶段池 | `vllm_omni/engine/stage_pool.py` | `StagePool` |
| 连接器 Mixin | `vllm_omni/worker/omni_connector_model_runner_mixin.py` | `OmniConnectorModelRunnerMixin` |
| TeaCache | `vllm_omni/diffusion/cache/tea_cache.py` | `TeaCache` |
| 流水线配置 | `vllm_omni/config/pipeline_registry.py` | `OMNI_PIPELINES` |
| Qwen3-Omni | `vllm_omni/model_executor/models/qwen3_omni/` | 多个文件 |

---

**文档版本**：v1.0  
**最后更新**：2026-06-27  
**适用对象**：LLM 算法实习面试准备
