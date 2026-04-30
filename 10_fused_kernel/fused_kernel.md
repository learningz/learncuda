# 算子融合：未融合 vs 融合在硬件上的数据流差异

配合 `fused_kernel.cu` 阅读。


## 为什么算子融合是 Memory Bound 算子最重要的优化

```
深度学习中绝大多数算子都是 Memory Bound (除了 GEMM):
  ReLU, GELU, Sigmoid, Softmax, LayerNorm, BatchNorm, Dropout...
  它们的算术强度 (AI) 都远低于 GPU 的 Ridge Point。

这意味着: 性能 = 带宽 / 数据搬运量
  → 减少数据搬运量 = 直接等比加速
  → 融合是最有效的手段: 把多次搬运变成一次

一个真实的例子 — Transformer 中的 ReLU + Scale + Bias:
  PyTorch 默认行为: 3 个独立 kernel
    out = F.relu(x)                  ← kernel 1
    out = out * scale                ← kernel 2
    out = out + bias                 ← kernel 3
  
  每个 kernel 都要读写一次主数组 → 按 x/tmp1/tmp2/y 这几块大向量算，总搬运约 6N
  但实际有用的计算只有 3N 次乘加 → 大量时间花在搬运上
  
  融合后: 1 个 kernel
    out[i] = max(x[i], 0) * scale + bias  ← 1 个 kernel
  
  只需读 x、写 y 这两块主数组 → 按主数据流算，总搬运约 2N → 理论上可接近 3× 加速
```


## 未融合 (3 个独立 kernel) 的硬件数据流

```
kernel 1: relu
  HBM → L2 → NoC → SM (读 x) → FP32 ALU (max(x,0)) → SM → NoC → L2 → HBM (写 tmp1)

kernel 2: scale  
  HBM → L2 → NoC → SM (读 tmp1) → FP32 ALU (×scale) → SM → NoC → L2 → HBM (写 tmp2)

kernel 3: bias
  HBM → L2 → NoC → SM (读 tmp2) → FP32 ALU (+bias) → SM → NoC → L2 → HBM (写 y)

每个 kernel 之间:
  GPU 完成 kernel 1 → 所有数据写回 HBM (flush) → CTA Scheduler 分配 kernel 2
  → kernel 2 重新从 HBM 读 tmp1 ...

总数据搬运: 6N (这里按主数组 x/tmp1/tmp2/y 估算，忽略 scale/bias 等标量参数)
每次搬运走: SM ↔ NoC ↔ L2 ↔ MC ↔ HBM 全链路

额外开销:
  - 2 次 kernel launch (每次 ~3-5μs 的 CPU→GPU 命令传递)
  - 2 个中间 buffer 的显存分配 (tmp1, tmp2 各 N×4 bytes)
  - 3 次 CTA 调度 (GPU 硬件分配 Block 到 SM)
```


## 融合 (1 个 kernel) 的硬件数据流

```
fused kernel:
  HBM → L2 → NoC → SM (读 x)
  → 寄存器中: max(x,0) → ×scale → +bias (3 条 ALU 指令, 全在寄存器!)
  → SM → NoC → L2 → HBM (写 y)

中间结果 (relu 输出, scale 输出) 始终在寄存器中:
  寄存器延迟: 0 cycles (流水线内)
  不经过 Shared Memory, 不经过 L1, 不经过 L2, 不经过 HBM!

总数据搬运: 2N (这里按主数组 x/y 估算，忽略 scale/bias 等标量参数)
搬运量减少: 6N → 2N = 3× → 对 Memory Bound 算子, 理论加速 3×

┌────────────────────────────────────────────────────┐
│                                                    │
│  未融合: HBM ←→ SM ←→ HBM ←→ SM ←→ HBM ←→ SM ←→ HBM │
│          │←── 6N 次搬运 ──→│                       │
│                                                    │
│  融合:   HBM → SM → HBM                           │
│          │← 2N ──→│                                │
│                                                    │
│  多出的 4N 次搬运 = 未融合版本浪费在中间数据上的带宽! │
└────────────────────────────────────────────────────┘
```


## float4 向量化在融合上的额外收益

```
标量 LDG.32: 每条指令加载 1 个 float (4B)
  一个 Warp 32 线程 → 32 × LDG.32 = 32 条 LD 指令 → 占 LD/ST pipeline 32 slots

float4 LDG.128: 每条指令加载 4 个 float (16B)  
  一个 Warp 32 线程 → 32 × LDG.128 = 32 条指令, 但每条搬 4× 数据
  → 相同数据量, 指令数减少 4× → LD/ST pipeline 压力降低 4×
  → 空出的 pipeline slots 可以被计算指令使用

融合 + float4 = 最优: 最少的 HBM 访问 + 最少的 LD/ST 指令
```


## 融合在 fused_kernel.cu 中的三个版本对比

```
版本 A — 3 个独立 kernel:
  relu_kernel<<<grid, block>>>(d_x, d_tmp1, N);      // 读 x, 写 tmp1
  scale_kernel<<<grid, block>>>(d_tmp1, d_tmp2, s, N); // 读 tmp1, 写 tmp2
  bias_kernel<<<grid, block>>>(d_tmp2, d_out, b, N);   // 读 tmp2, 写 out
  
  HBM 访问: 读 3N + 写 3N = 6N × 4B  (按主数组估算)
  中间 buffer: 2 × N × 4B 的额外显存
  kernel launch: 3 次

版本 B — 1 个融合 kernel (标量):
  __global__ void fused_rsb(float *x, float *out, float scale, float bias, int n) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < n) {
          float val = x[idx];                    // 从 HBM 读 1 次
          val = fmaxf(val, 0.0f);                // ReLU: 在寄存器中
          val = val * scale;                      // Scale: 在寄存器中
          val = val + bias;                       // Bias: 在寄存器中
          out[idx] = val;                         // 写 HBM 1 次
      }
  }
  
  HBM 访问: 读 N + 写 N = 2N × 4B  (按主数组估算, 减少约 3×)
  中间 buffer: 0
  kernel launch: 1 次

版本 C — 融合 + float4 向量化:
  每线程处理 4 个 float → LDG.128 / STG.128
  HBM 访问: 同版本 B (2N × 4B)
  指令数: 版本 B 的 ~1/4 → LD/ST pipeline 更轻松
  
预期性能 (A100, N=16M):
  A: ~0.12ms (有效带宽 ~1000 GB/s)
  B: ~0.04ms (有效带宽 ~1600 GB/s)  ← 3× 加速
  C: ~0.035ms (有效带宽 ~1830 GB/s)  ← 额外 15% 加速
```


## 哪些算子适合融合？哪些不行？

```
适合融合 (可以直接拼接):
  ✓ Elementwise 链: ReLU → Scale → Bias → Dropout
    每个元素独立 → 直接在一个线程中串联计算
    
  ✓ Elementwise + Reduce: x → GELU → ReduceSum
    先做 GELU，结果留在寄存器，直接参与 reduce
    
  ✓ 相同访问模式的操作: 输入/输出 shape 一致

不容易融合:
  ✗ 改变 shape 的操作: Transpose, Reshape + 后续计算
    数据的物理布局变了 → 融合后的访存模式可能不合并
    
  ✗ 全局 Reduce + 后续操作: ReduceSum → Broadcast → Elementwise
    Reduce 需要跨 Block 同步 → 不能和后续操作放在同一 kernel
    (除非用持久化 kernel + Grid 级同步 → 见 [theory/08_advanced_optimization.md](../theory/08_advanced_optimization.md))
    
  ✗ 不同 Block 配置的操作: 2D Grid 的卷积 + 1D Grid 的 BatchNorm

自动融合工具:
  PyTorch: torch.compile (TorchInductor) 会自动融合 elementwise 链
  TensorRT: 在推理图优化时自动融合
  Triton: 提供 Python 级 DSL，用户控制融合粒度
  
  但手写融合仍然重要:
    - 涉及 Reduce 的复杂融合 (如 Softmax) 自动工具做不好
    - 需要 Shared Memory 的融合 (如 fused attention) 必须手写
    - 训练中的 fwd+bwd 联合融合需要领域知识
```


## 用 ncu 验证融合效果

```
对三个版本分别跑 ncu:
  ncu --set full ./fused_kernel

对比以下指标:

1. DRAM Throughput (HBM 吞吐):
   A: 可能 ~70% (带宽被重复读写浪费)
   B: 可能 ~85% (只读写各一次)
   C: 可能 ~90% (向量化提升)

2. L2 Hit Rate (L2 缓存命中率):
   A 的 kernel 2/3 可能有一定 L2 命中 (tmp 还在 L2 中)
   → 实际加速可能 < 理论的 3× (因为 L2 缓存部分缓解了重复读写)
   → 但当 N 很大 (>> L2 容量) 时，L2 命中率趋近 0 → 加速接近 3×

3. Instruction Count:
   A: 约 3× 的指令量
   C: 约 A 的 1/4 的 LD/ST 指令

4. Launch Overhead:
   用 nsys (不是 ncu) 看:
   nsys profile ./fused_kernel
   → A 有 3 次 kernel launch，每次 ~3-5μs
   → 对小 N 来说，launch 开销占比可能很大
```


## 练习题

| 练习 | 任务 | 核心考点 |
|------|------|---------|
| [ex1_fused_sigmoid_level1.cu](./exercises/ex1_fused_sigmoid_level1.cu) | Sigmoid + Scale 融合 | 两个 kernel 融合成一个（只填 kernel） |

```bash
nvcc -O2 -o ex1_fused_sigmoid_level1 ex1_fused_sigmoid_level1.cu
```
