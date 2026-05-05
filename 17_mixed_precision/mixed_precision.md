# 混合精度实战：FP32 vs FP16 vs BF16

配合 `mixed_precision.cu` 阅读。


## 为什么需要混合精度

深度学习中，FP32（32-bit float）是默认格式。
但现代 GPU 对低精度格式有特殊的硬件加速：

- **FP16/BF16**: 只占 2 bytes（vs FP32 的 4 bytes）→ 内存带宽需求减半
- **FP16 ALU**: Ampere 架构上吞吐是 FP32 的 2×
- **Tensor Core**: 专门为 FP16/BF16 矩阵乘法设计的硬件单元，吞吐比 CUDA Core 高 ~8×

"混合精度" = 关键路径用低精度加速 + 敏感部分保留 FP32 来保持数值稳定。


## 三种精度格式的存储结构

```
FP32 (4 bytes):
  ┌───────┬──────────────────────┬─────────────────────────────────┐
  │ 1 bit │      8 bit 指数      │        23 bit 尾数               │
  │ sign  │       exponent       │         mantissa                 │
  └───────┴──────────────────────┴─────────────────────────────────┘
  范围: ±3.4×10³⁸, 精度: ~7 位有效数字

FP16 (2 bytes):
  ┌───────┬──────────┬───────────────────┐
  │ 1 bit │ 5 bit 指 │   10 bit 尾数     │
  │ sign  │ exponent │    mantissa       │
  └───────┴──────────┴───────────────────┘
  范围: ±65504, 精度: ~3-4 位有效数字
  风险: 梯度 < 2⁻¹⁴ → 下溢为 0! (需要 Loss Scaling)

BF16 (2 bytes):
  ┌───────┬──────────────────────┬───────────────┐
  │ 1 bit │      8 bit 指数      │  7 bit 尾数   │
  │ sign  │       exponent       │   mantissa    │
  └───────┴──────────────────────┴───────────────┘
  范围: 和 FP32 相同 (±3.4×10³⁸), 精度: ~2-3 位有效数字
  优势: 截断 FP32 的高 16 位即可 → 和 FP32 互转非常简单
  不需要 Loss Scaling! (指数范围和 FP32 一样大)

关键区别:
  FP16 尾数 10 bit → 精度更好, 但范围小 → 梯度容易溢出/下溢
  BF16 尾数 7 bit   → 精度稍差, 但范围大 → 和 FP32 一样稳
```

### 用代码感受三种格式的差异

```cuda
// FP16 的最大值
float large_val = 70000.0f;
__half h = __float2half(large_val);
float back = __half2float(h);  // → INF! 因为 70000 > 65504

// BF16 对同样的值没问题
__nv_bfloat16 b = __float2bfloat16(large_val);
float back_bf16 = __bfloat162float(b);  // → 70000.0 (范围相同)

// FP16 需要 Loss Scaling 来保护小梯度
float small_grad = 1e-7f;
__half hs = __float2half(small_grad);
float back_small = __half2float(hs);  // → 0.0! 因为 1e-7 < FP16 最小正数 ~6e-8

// Loss Scaling: 训练时乘以 1024, 反向传播后再除以 1024
float scaled = small_grad * 1024.0f;  // = 1.024e-4 → FP16 能表示
__half hs2 = __float2half(scaled);
float recovered = __half2float(hs2) / 1024.0f;  // → 1e-7 ✓
```

## GEMM 在不同精度下的性能来源

```
FP32 GEMM (naive, CUDA Core):
  每个 FMA = 1 cycle (理论), 实际受限于显存带宽
  4 bytes/元素 → N×N 矩阵 = 4N² bytes 的数据搬运

FP16 GEMM (CUDA Core):
  Ampere 上 FP16 ALU 吞吐是 FP32 的 2× (128 ops/cycle vs 64 ops/cycle)
  但实际加速通常 < 2×, 因为同样的 kernel 还是 Memory Bound
  真正的好处: 2 bytes/元素 → 一半的内存带宽需求 → 1.5-2× 加速

FP16 Tensor Core GEMM (WMMA):
  一条 mma_sync 指令: D = A×B + C, 其中 A [16,16], B [16,16], C [16,16]
  = 16×16×16 = 4096 次 FMA / 指令
  vs CUDA Core 每线程每 cycle 最多 ~2 FMA
  → 每 SM 每 cycle 的吞吐: ~1000× vs CUDA Core 的标量 FMA
  → 实际端到端加速 ~8-16× (受限于 Shared Memory 带宽和寄存器)

为什么 Tensor Core 这么快?
  1. 专门的硬件矩阵乘法单元 (不是通用 ALU)
  2. 一条指令完成了 4096 次乘加
  3. 数据在寄存器之间共享 (Warp 内 32 线程共用同一块数据)
  4. 专用数据通路: ldmatrix 一次从 SMEM 加载 4 个 16-bit 元素到寄存器
```

## Roofline 视角 — 为什么混合精度对某些算子没用

```
算术强度回顾:
  AI = FLOP / Byte

  FP32 GEMM (1024×1024):           AI ≈ 170 → 远在 Ridge Point 右侧 → Compute Bound
    → 换 FP16/BF16 有巨大收益 (计算更快了, 本来就不是 Memory Bound)

  FP32 LayerNorm:                  AI ≈ 5 → 在 Ridge Point 左侧 → Memory Bound
    → 换 FP16 几乎没有性能提升! (瓶颈在内存带宽, 不在计算)
    → 但可以减少显存占用 (一半 → 可以用更大的 batch)

  FP32 GELU (elementwise):         AI ≈ 1.9 → 深度 Memory Bound
    → 换 FP16 基本无加速 (除非配合算子融合)

经验法则:
  GEMM / 大卷积: 精度降低 → 巨大提升 → 用 FP16/BF16 + Tensor Core
  Elementwise / Reduce: 精度降低 → 几乎无提升 → 但省显存
  训练: 混合精度 + Loss Scaling (FP16) 或直接 BF16
  推理: FP16 量化 + Tensor Core → 降低延迟, 减小模型大小
```

## BF16 vs FP16 — 训练中选哪个

```
BF16 的优势:
  ✓ 不需要 Loss Scaling (指数范围 = FP32)
  ✓ 和 FP32 互转是纯截断 (truncate high 16 bits)
  ✓ 训练更稳定, 收敛性和 FP32 几乎一样

BF16 的劣势:
  ✗ 精度比 FP16 低 (7 bit vs 10 bit 尾数)
  ✗ 需要 Ampere (A100) 或更新的 GPU
  ✗ 某些老代码只支持 FP16

FP16 的优势:
  ✓ 精度更高 (10 bit 尾数)
  ✓ 几乎所有的 GPU (Volta+) 都支持
  ✓ 生态更成熟

FP16 的劣势:
  ✗ 需要 Loss Scaling (额外代码 + 调试负担)
  ✗ 范围小 (max=65504), 梯度容易溢出

选择:
  新项目用 Ampere+ → BF16 (简单 + 稳定)
  老 GPU (Volta/Turing) → FP16 + Loss Scaling
  推理 → FP16 (更小的模型 + 精度足够)
```

## 混合精度训练的完整流程

```
1. 前向: FP16/BF16 计算 → 快, 省显存
2. 反向: FP16/BF16 计算梯度 → 快
3. Optimizer: FP32 Master Weights → 保持精度

// 伪代码 (PyTorch 风格):
model_fp16 = model.half()          // 模型用 FP16
optimizer = Adam(model.parameters())  // optimizer 维护 FP32 master weights

for batch in dataloader:
    x = batch.half()               // 输入转 FP16

    output = model_fp16(x)         // 前向: FP16
    loss = criterion(output, target)  // loss 保持 FP32 (更稳定)

    scaler.scale(loss).backward()  // Loss Scaling + 反向

    scaler.step(optimizer)         // optimizer 用 FP32 master weights
    scaler.update()                // 更新 scale factor

// 关键: 模型权重存为 FP32 (master copy), 每个 iteration 转为 FP16 做前向/反向
// 梯度在 FP16 下计算, 但在更新 master weights 前转回 FP32
```

## 练习题

完成 `mixed_precision.cu` 的阅读后，可以在 [`exercises/`](./exercises/) 目录下做以下练习：

| 练习 | 任务 | 核心考点 |
|------|------|---------|
| [ex1_fp16_overflow_level1.cu](./exercises/ex1_fp16_overflow_level1.cu) | FP16 溢出实战 | 理解 FP16 max/min 和 Loss Scaling（只填数值） |
| [ex1_fp16_overflow_level2.cu](./exercises/ex1_fp16_overflow_level2.cu) | 同上（完整实现） | kernel + host + 计时全部自己写 |

编译方式（在 `exercises/` 目录下）：

```bash
nvcc -arch=sm_80 -o ex1_fp16_overflow_level1 ex1_fp16_overflow_level1.cu
./ex1_fp16_overflow_level1
```
