# 第六章：Tensor Core 编程 — 矩阵运算加速器

**难度**: ⭐⭐⭐ 专家
**前置知识**: 第3章的 Shared Memory; 第5章的 GEMM 优化基础
**读完你能做什么**: 理解 Tensor Core 的工作原理; 使用 WMMA API; 读懂 CUTLASS 代码的核心结构
**配套代码**: 无 (建议阅读 CUTLASS 的 examples/)
**新手建议**: 如果你还没写过使用 Shared Memory 的矩阵乘法，先完成第3章和第5章。Tensor Core 是 GEMM 优化的最后一步，不是第一步

> **本章首次出现的术语**:
> - **Tensor Core**: GPU 芯片内的专用硬件单元，专门做小矩阵的乘加运算 (如 16×16)。
>   比用普通 CUDA Core 逐元素计算快 ~16 倍
> - **MMA (Matrix Multiply-Accumulate)**: Tensor Core 执行的操作: D = A × B + C
> - **Fragment**: WMMA API 中的数据容器。一个矩阵被切碎分散到 Warp 的 32 个线程中,
>   每个线程持有矩阵的几个元素
> - **FP16 / BF16 / TF32 / FP8**: 不同的低精度浮点格式 (详见第0章 0.9)
> - **WMMA**: Warp Matrix Multiply-Accumulate, CUDA 提供的 C++ 级 Tensor Core API
> - **CUTLASS**: NVIDIA 开源的高性能 GEMM 模板库，底层使用 Tensor Core
> - **ldmatrix**: 一条硬件指令，让整个 Warp 协作将矩阵从 Shared Memory 加载到寄存器
> - **Swizzle**: 对 Shared Memory 地址做 XOR 重映射，消除 Bank Conflict

## 6.1 Tensor Core 是什么？

Tensor Core 是 NVIDIA 从 Volta (2017) 开始引入的专用矩阵运算硬件。
它不是通用计算单元，而是只做一件事：**矩阵乘累加 (MMA: Matrix Multiply-Accumulate)**

```
D = A × B + C

其中 A, B, C, D 都是小矩阵 (如 16×16)
一条 MMA 指令在几个时钟周期内完成整个矩阵乘！
```

### 为什么 Tensor Core 这么快？

```
CUDA Core 做矩阵乘:
  16×16 × 16×16 的矩阵乘 = 16×16×16×2 = 8192 FLOP
  FP32 Core: 每周期 64 FLOP/SM → 需要 128 个周期

Tensor Core 做矩阵乘:
  一条 HMMA 指令: 完成 16×8×16 (Ampere) 或 16×8×8 (Volta) 的乘累加
  延迟 ~8 个周期, 4 个 Tensor Core 可以并行
  
  A100 FP16 Tensor Core 峰值: 312 TFLOPS
  A100 FP32 CUDA Core 峰值:   19.5 TFLOPS
  → Tensor Core 快 ~16×!

  H100 FP16 Tensor Core: 989 TFLOPS
  → 比 A100 Tensor Core 快 3.2×
```

### 各代 Tensor Core 能力对比

```
             Volta (1st)   Turing (2nd)   Ampere (3rd)    Hopper (4th)
MMA 形状     4×4×4         8×8×4          16×8×16         16×8×16+
             (Warp级)      (Warp级)       (Warp级)        (Warp级)
             
支持精度:
  FP16       ✓             ✓              ✓               ✓
  BF16       ✗             ✗              ✓               ✓
  TF32       ✗             ✗              ✓               ✓
  FP8        ✗             ✗              ✗               ✓
  INT8       ✗             ✓              ✓               ✓
  INT4       ✗             ✓              ✓               ✓
  INT1       ✗             ✓              ✗               ✗
  FP64       ✗             ✗              ✓ (A100)        ✓
  
累加精度:
  FP16→FP16  ✓             ✓              ✓               ✓
  FP16→FP32  ✓             ✓              ✓               ✓
  BF16→FP32  ✗             ✗              ✓               ✓
  TF32→FP32  ✗             ✗              ✓               ✓
  FP8→FP32   ✗             ✗              ✗               ✓
  INT8→INT32 ✗             ✓              ✓               ✓

稀疏支持:
  2:4 结构   ✗             ✗              ✓ (2×吞吐)      ✓ (2×吞吐)
```


## 6.2 数据格式详解

### FP16 (IEEE Half Precision)
```
位布局: 1 sign + 5 exponent + 10 mantissa = 16 bits
范围: ±65504, 最小正数: ~5.96e-8
精度: ~3-4 位有效数字
问题: 范围小, 容易溢出/下溢 → 需要 Loss Scaling
```

### BF16 (Brain Float 16) — Google 发明
```
位布局: 1 sign + 8 exponent + 7 mantissa = 16 bits
范围: 和 FP32 相同! (因为 exponent 位数相同)
精度: ~2-3 位有效数字 (比 FP16 低)
优势: 不需要 Loss Scaling, 可以直接截断 FP32 → BF16
用途: 训练 (范围重要) > 推理 (精度重要)
```

### TF32 (TensorFloat-32) — NVIDIA 发明
```
位布局: 1 sign + 8 exponent + 10 mantissa = 19 bits
(内部格式, 不是存储格式!)

工作方式:
1. 输入 A, B 是 FP32 格式
2. Tensor Core 自动截取高 19 位 (TF32)
3. 用 TF32 做乘法
4. 累加用 FP32

效果: 用户代码不需要改! 只需要启用 TF32:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
性能: 比 FP32 CUDA Core 快 ~8×, 精度损失很小
```

### FP8 (Hopper)
```
两种变体:
E4M3: 1 sign + 4 exp + 3 mantissa → 更高精度, 范围较小
E5M2: 1 sign + 5 exp + 2 mantissa → 更大范围, 精度较低

典型用法:
前向: E4M3 (精度更重要)
反向: E5M2 (梯度范围大)
```


## 6.3 WMMA API — Warp 级矩阵运算

### 基本用法

```cuda
#include <mma.h>
using namespace nvcuda;

// WMMA: 一个 Warp (32 线程) 协作完成一个矩阵乘
// 支持的形状: 16×16×16, 32×8×16, 8×32×16 (Ampere)

__global__ void tensor_core_gemm(const half *A, const half *B, float *C,
                                  int M, int K, int N) {
    // 每个 Warp 处理一个 16×16 的 C tile
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32 * 16;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 32 * 16;
    
    // 声明 fragment (每个线程持有矩阵的一部分)
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    // 初始化累加器为 0
    wmma::fill_fragment(c_frag, 0.0f);
    
    // 沿 K 维度循环
    for (int k = 0; k < K; k += 16) {
        // 从全局内存加载 A 和 B 的 tile 到 fragment
        wmma::load_matrix_sync(a_frag, A + warpM * K + k, K);  // 16×16 tile of A
        wmma::load_matrix_sync(b_frag, B + k * N + warpN, N);  // 16×16 tile of B
        
        // 矩阵乘累加: C += A × B
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // 将结果存回全局内存
    wmma::store_matrix_sync(C + warpM * N + warpN, c_frag, N, wmma::mem_row_major);
}
```

### Fragment 的内部结构

```
Fragment 是 WMMA 的核心概念:
一个 16×16 矩阵被分散存储在一个 Warp 的 32 个线程中。
每个线程持有矩阵的几个元素。

具体分布 (16×16×16 FP16, Ampere):
  matrix_a fragment: 每线程持有 8 个 half 元素
  matrix_b fragment: 每线程持有 8 个 half 元素
  accumulator fragment: 每线程持有 8 个 float 元素

总共: 32 threads × 8 elements = 256 elements = 16×16 ✓

你不需要（也不应该）关心具体哪个线程持有哪个元素。
WMMA API 抽象了这个细节。

但如果需要手动处理 fragment 中的元素:
for (int i = 0; i < c_frag.num_elements; i++) {
    c_frag.x[i] = relu(c_frag.x[i]);  // 对每个元素应用 ReLU
}
```


## 6.4 MMA PTX 指令 — 更底层的控制

### 为什么需要直接用 MMA PTX?

```
WMMA API 的限制:
- 只支持有限的形状 (16×16×16 等)
- 内存布局不够灵活
- 无法精确控制寄存器分配

MMA PTX 指令更灵活:
- 支持更多形状 (m16n8k16, m16n8k8, etc.)
- 可以精确控制数据在寄存器中的布局
- CUTLASS 和 cuBLAS 内部使用 MMA PTX
```

```cuda
// Ampere m16n8k16 FP16 MMA (通过内联 PTX)
__device__ void mma_m16n8k16(
    uint32_t *D,        // 4 个 uint32_t = 8 个 FP16 输出
    uint32_t *A,        // 8 个 uint32_t = 16 个 FP16 (A 矩阵片段)
    uint32_t *B,        // 4 个 uint32_t = 8 个 FP16 (B 矩阵片段)
    uint32_t *C)        // 4 个 uint32_t = 8 个 FP16 (累加器)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0, %1, %2, %3}, "     // D 输出
        "{%4, %5, %6, %7}, "     // A 输入
        "{%8, %9}, "             // B 输入
        "{%10, %11, %12, %13};"  // C 累加器
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3])
    );
}
```

### 数据布局要求

```
Tensor Core 对数据布局有严格要求:

矩阵 A (m×k): 
  行主序 (row_major): A[i][j] = A + i*K + j
  列主序 (col_major): A[i][j] = A + j*M + i

矩阵 B (k×n):
  行主序或列主序

MMA 指令中的标记:
  mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
                              ^^^.^^^
                              A是row, B是col

如果 A 是行主序, B 也是行主序:
  需要隐式转置 B → 可以声明为 col_major 然后存储转置后的数据
  或者用 ldmatrix 指令自动处理布局转换
```


## 6.5 性能优化：如何喂饱 Tensor Core

### Tensor Core 的数据饥饿问题

```
A100 Tensor Core 峰值: 312 TFLOPS (FP16)

一条 m16n8k16 MMA 指令:
  计算量: 16 × 8 × 16 × 2 = 4096 FLOP
  延迟: ~8 cycles
  4 个 TC/SM, 每个 TC 每 8 cycles 一条 → 512 FLOP/cycle/TC × 4 = 2048 FLOP/cycle/SM

需要的数据:
  A 片段: 16×16 × 2 bytes = 512 bytes
  B 片段: 16×8 × 2 bytes = 256 bytes
  总计: 768 bytes / 8 cycles = 96 bytes/cycle

Shared Memory 带宽: 128 bytes/cycle/SM
→ 刚刚够! 但这假设零 bank conflict

如果有 bank conflict 或者 Shared Memory 带宽不够:
→ Tensor Core 空等数据 → 利用率下降

解决方案:
1. 增大 tile 大小 → 更多数据复用 → 降低带宽需求
2. 使用 ldmatrix 指令 (特殊的 Shared → Register 传输, 无 bank conflict)
3. Swizzled Shared Memory 布局 (CUTLASS 方案)
4. 双缓冲 / 多阶段流水线
```

### CUTLASS 的分层 Tiling 策略

```
CUTLASS (CUDA Templates for Linear Algebra Subroutines) 
是 NVIDIA 开源的高性能 GEMM 库。它的核心是 3 层 tiling:

层级 1: Thread Block Tile (CTA Tile)
  每个 Thread Block 负责 C 的一个大块 (如 128×128)
  从 Global Memory → Shared Memory
  
层级 2: Warp Tile
  每个 Warp 负责 CTA Tile 的一个子块 (如 64×64)
  从 Shared Memory → Registers
  
层级 3: MMA 指令 Tile (Thread Tile)
  每条 MMA 指令处理最小块 (如 16×8×16)
  在 Registers 中完成

数据流:
Global Memory → (合并加载) → Shared Memory → (ldmatrix) → Registers → Tensor Core
                    ^                               ^
              cp.async 异步拷贝            ldmatrix 无 bank conflict

流水线:
Stage 0: 加载 tile K=0 到 SMEM buf[0]
Stage 1: 加载 tile K=1 到 SMEM buf[1], 同时计算 buf[0]
Stage 2: 加载 tile K=2 到 SMEM buf[2], 同时计算 buf[1]
...
→ 计算和数据传输完全重叠!
```


## 6.6 混合精度训练的完整流程

```
现代大模型训练的标准配置:

前向传播:
  权重: FP16/BF16 (存储) + FP32 (Master Copy)
  激活: FP16/BF16
  矩阵乘: FP16 × FP16 → FP32 (Tensor Core)
  其他运算: FP16/BF16 (elementwise) 或 FP32 (reduce/norm)

反向传播:
  梯度: FP16/BF16
  矩阵乘: FP16 × FP16 → FP32

权重更新:
  FP32! (必须用 FP32 精度, 否则小梯度会被截断为 0)
  FP32_weight += learning_rate * FP32_gradient
  FP16_weight = FP16(FP32_weight)  // 截断

Loss Scaling (仅 FP16 需要, BF16 通常不需要):
  问题: FP16 最小正数 ~6e-8, 很多梯度比这还小 → 变成 0
  解决: 
    1. 把 loss 乘以一个大数 (如 1024)
    2. 反向传播得到放大的梯度
    3. 更新权重前把梯度除以 1024
    4. 动态调整 scale: 如果发现 INF/NaN, 减小 scale; 否则逐渐增大

PyTorch 实现:
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():  # 自动选择 FP16/FP32
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```


## 6.7 ldmatrix — 高效的 Shared → Register 传输

### 为什么需要 ldmatrix

```
Tensor Core 的数据存在寄存器中 (Fragment)。
传统方式: 每个线程用 LDS (Shared Memory Load) 加载自己需要的元素。

问题:
  一个 16×16 FP16 矩阵 = 256 个 half = 512 字节
  32 个线程各自加载 8 个 half → 32 × LDS.64 指令
  容易产生 Bank Conflict (因为 Fragment 的分布不连续)

ldmatrix:
  一条指令让整个 Warp 协作加载一个矩阵 Fragment
  硬件自动处理 Shared Memory 地址计算和 Bank Conflict 避免
  
  SASS 指令: LDMATRIX.SYNC.ALIGNED.M8N8.X4 R4, [R0];
  一条指令加载 4 个 8×8 矩阵 = 32×16 bytes → 全 Warp 参与
```

### ldmatrix 的具体工作方式

```
ldmatrix.sync.aligned.m8n8.x4:
  每个线程提供一个 Shared Memory 地址 (指向矩阵的一行)
  硬件从 32 个地址各加载 16 bytes (128 bits)
  → 总计 512 bytes = 一个 16×16 FP16 矩阵
  
  每个线程获得 4 个 32-bit 寄存器 (= 8 个 FP16 值)
  → 直接就是 MMA 指令需要的 Fragment 布局!

线程 Lane 与矩阵行的对应:
  Lane 0  → 加载 row 0 的 16 bytes (8 个 half)
  Lane 1  → 加载 row 1
  ...
  Lane 7  → 加载 row 7
  Lane 8  → 加载 row 8 (重复: 因为 m8n8.x4 需要 4 个 8×8 子矩阵)
  ...
  Lane 15 → 加载 row 15
  Lane 16-31 → 加载另一半矩阵 (用于 B 矩阵的转置布局)

关键优势:
  1. 一条指令完成, 减少指令发射数
  2. 硬件自动避免 Bank Conflict
  3. 输出直接是 MMA Fragment 布局, 不需要额外的 shuffle/transpose
```

### 使用 ldmatrix (内联 PTX)

```cuda
__device__ void load_matrix_sync_m16n16(
    uint32_t frag[4],           // 输出: 4 个 uint32 = 8 个 FP16
    const void *smem_ptr,       // Shared Memory 基址
    int stride_bytes) {         // 行步长 (字节)
    
    int lane = threadIdx.x % 32;
    // 每个线程计算自己负责的行地址
    uint32_t addr = __cvta_generic_to_shared(
        (const char*)smem_ptr + (lane % 16) * stride_bytes);
    
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3])
        : "r"(addr)
    );
}
```


## 6.8 Shared Memory Swizzle — 消除 Bank Conflict 的终极方案

### 问题: Tensor Core 数据布局导致 Bank Conflict

```
矩阵 A 在 Shared Memory 中按行存储:
  Row 0: addr 0,  2,  4,  6,  8, 10, 12, 14  (8 个 FP16 = 16 bytes)
  Row 1: addr 16, 18, 20, 22, 24, 26, 28, 30
  ...

ldmatrix 每个线程加载一行 (16 bytes = 4 个 bank):
  Lane 0 → Row 0: Bank 0,1,2,3
  Lane 1 → Row 1: Bank 4,5,6,7
  ...
  Lane 8 → Row 8: Bank 0,1,2,3  ← 和 Lane 0 相同的 Bank! 冲突!

如果行步长 = 16 bytes (8 个 half):
  Row 0 起始 Bank = 0
  Row 8 起始 Bank = (8 × 16 / 4) % 32 = 0  → 冲突!
```

### Padding 方案 (简单但浪费内存)

```cuda
// 加 padding 使相邻行不在同一 Bank
__shared__ half smemA[16][16 + 8];  // 每行多 8 个 half = 16 bytes
// Row 0 起始 Bank: 0
// Row 1 起始 Bank: (24 × 2 / 4) % 32 = 12
// Row 8 起始 Bank: (8 × 24 × 2 / 4) % 32 = (96) % 32 = 0  → 还是冲突!

// 需要更仔细地选择 padding 大小...
// 通常 padding = 8 bytes (4 half) 就够了:
__shared__ half smemA[16][16 + 4];
```

### Swizzle 方案 (零额外内存, CUTLASS 使用)

```
Swizzle 是一种地址重映射:
  物理地址 = 逻辑地址 XOR (某个基于行号的模式)

例: XOR Swizzle with B=3, M=2, S=3
  对于逻辑地址的 byte offset: byte_offset
  行号: row
  
  swizzled_offset = byte_offset XOR ((row & mask) << shift)
  
  效果: 每行的数据被 "旋转" 到不同的 Bank 组
  → 相隔 8 行的数据不再在同一 Bank → 无冲突

具体例子 (简化):
  逻辑布局 (Bank 分配):
    Row 0: Bank 0,1,2,3,4,5,6,7
    Row 1: Bank 0,1,2,3,4,5,6,7  ← 冲突!
    
  XOR Swizzle (row XOR offset):
    Row 0: Bank 0,1,2,3,4,5,6,7  (不变)
    Row 1: Bank 1,0,3,2,5,4,7,6  (XOR 1)
    Row 2: Bank 2,3,0,1,6,7,4,5  (XOR 2)
    Row 3: Bank 3,2,1,0,7,6,5,4  (XOR 3)
    → 4 行内完全无 Bank Conflict!

在 CUTLASS 中的实现:
  cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<...>
  自动应用 Swizzle 模式, 用户不需要手动计算
```


## 6.9 CUTLASS GEMM 的完整流水线

### Ampere 3 阶段异步流水线

```
CUTLASS 3.x 的 Ampere GEMM 使用 3 阶段 (或更多) 的软件流水线:

内存层级:
  Global Memory → [cp.async] → Shared Memory → [ldmatrix] → Registers → [mma] → Registers

3 阶段:
  Stage 0: cp.async 加载 tile K=0 到 SMEM[0]
  Stage 1: cp.async 加载 tile K=1 到 SMEM[1]; ldmatrix SMEM[0] → Regs; 
  Stage 2: cp.async 加载 tile K=2 到 SMEM[2]; ldmatrix SMEM[1] → Regs; mma(Regs_0);

  主循环体:
  ┌──────────────────────────────────────────────────────────┐
  │ 1. cp.async_commit()                                     │
  │ 2. cp.async_wait<Stages-2>()   // 等待最老的一批完成       │
  │ 3. __syncthreads()                                       │
  │                                                          │
  │ 4. for each sub-tile in current SMEM stage:              │
  │      a. ldmatrix: SMEM → Registers (下一个 sub-tile)     │
  │      b. mma.sync: 计算当前 sub-tile                      │
  │                                                          │
  │ 5. cp.async: 开始加载下一个 K-tile 到 SMEM[next_stage]    │
  └──────────────────────────────────────────────────────────┘

时间线 (理想情况):
  CP.ASYNC: ████████████████████████████████  (持续加载)
  LDMATRIX:    ████████████████████████████    (持续传输)
  MMA:            ████████████████████████      (持续计算)
  
  三条流水线完全重叠 → 接近硬件峰值!
```

### Hopper 的 Warp Specialization (TMA + WGMMA)

```
Hopper 引入了全新的编程模型:

传统 (Ampere): 所有 Warp 都做加载+计算
  每个 Warp: cp.async → ldmatrix → mma → cp.async → ...
  问题: 每个 Warp 在加载和计算之间切换, 效率损失

Hopper Warp Specialization: 不同 Warp 专门负责不同任务
  Producer Warps: 专门做 TMA 加载 (Global → Shared)
  Consumer Warps: 专门做 WGMMA 计算 (Shared → Register → TC)
  
  通过 Barrier + Async Pipeline 通信:
    Producer 完成加载 → 通知 Consumer (通过 barrier)
    Consumer 完成消费 → 通知 Producer 可以覆盖 SMEM (通过 barrier)

好处:
  1. Producer 不需要寄存器做计算 → 更多寄存器留给 Consumer
  2. TMA 是硬件单元, 几乎不占 Warp 调度器资源
  3. 计算和加载真正解耦

WGMMA (Warp Group MMA):
  4 个 Warp 组成一个 Warp Group, 协作执行一条 MMA 指令
  支持更大的 MMA 形状 (如 64×256×16)
  直接从 Shared Memory 读取 B 矩阵 (不需要 ldmatrix!)
```


## 6.10 MMA 指令中线程与矩阵元素的精确映射

### m16n8k16 FP16 MMA 的线程-数据对应关系

```
HMMA.16816.F16 指令:  D = A × B + C
  A: 16×16 (FP16)
  B: 16×8  (FP16)
  C: 16×8  (FP32 累加器)
  D: 16×8  (FP32 输出)

32 个线程 (一个 Warp) 如何分配 A 矩阵的 16×16 元素:

  Thread   →  A 矩阵的哪些元素 (每线程 8 个 half = 4 个 uint32)
  T0         A[0,0:1]  A[0,2:3]  A[8,0:1]   A[8,2:3]
  T1         A[1,0:1]  A[1,2:3]  A[9,0:1]   A[9,2:3]
  T2         A[2,0:1]  A[2,2:3]  A[10,0:1]  A[10,2:3]
  T3         A[3,0:1]  A[3,2:3]  A[11,0:1]  A[11,2:3]
  T4         A[4,0:1]  A[4,2:3]  A[12,0:1]  A[12,2:3]
  T5         A[5,0:1]  A[5,2:3]  A[13,0:1]  A[13,2:3]
  T6         A[6,0:1]  A[6,2:3]  A[14,0:1]  A[14,2:3]
  T7         A[7,0:1]  A[7,2:3]  A[15,0:1]  A[15,2:3]
  T8-T15     A[row, 4:7] 和 A[row+8, 4:7]  (k 维度 4-7)
  T16-T23    A[row, 8:11] 和 A[row+8, 8:11] (k 维度 8-11)
  T24-T31    A[row, 12:15] 和 A[row+8, 12:15] (k 维度 12-15)

  规律: 
  - lane % 8 决定行 (0-7 对应 row 0-7 和 row 8-15)
  - lane / 8 决定 K 维度的哪个 4 列块 (0,1,2,3 对应 k=0-3, 4-7, 8-11, 12-15)
  - 每线程持有 2 个行的 2 列 = 4 个 half = 2 个 uint32 → ×2(两个行块) = 4 uint32

这个布局直接决定了:
  1. ldmatrix 如何从 Shared Memory 加载到寄存器
  2. 从 Global Memory 到 SMEM 的数据排列 (Swizzle 就是为了匹配这个布局)
  3. 算子 epilogue 时如何将 Fragment 写回全局内存
```

### CUTLASS 的概念模型 (Concepts)

```
CUTLASS 用几个核心抽象来组织 GEMM:

1. Tile (切片):
   ├── CtaTile:    一个 CTA 负责的 C 子矩阵大小, 如 128×128
   ├── WarpTile:   一个 Warp 负责的子矩阵大小, 如 64×64
   └── MmaTile:    一条 MMA 指令处理的大小, 如 16×8×16

2. Layout (布局):
   描述逻辑坐标 (row, col) → 物理偏移 (offset) 的映射。
   ├── RowMajor: offset = row × stride + col
   ├── ColumnMajor: offset = col × stride + row
   └── 各种 Swizzled Layout: 对 SMEM 做 XOR 重映射消除 bank conflict

3. Copy (数据搬运):
   ├── G2S: Global → Shared (用 cp.async 或 TMA)
   ├── S2R: Shared → Register (用 ldmatrix)
   └── R2G: Register → Global (epilogue 写回)

4. MMA (矩阵乘):
   ├── SM70_16x8x8_F16F16F16F16   (Volta)
   ├── SM80_16x8x16_F16F16F16F16  (Ampere)
   └── SM90_64x256x16_F16F16F16F16 (Hopper WGMMA)

5. Epilogue (后处理):
   GEMM 结果在寄存器中, 写回前可以做:
   ├── Bias 加法
   ├── Activation (ReLU, GELU)
   ├── Scale + Quantize
   └── 和另一个矩阵的 elementwise 运算
   → 这就是 "Epilogue Fusion", 避免单独的 elementwise kernel

CUTLASS 3.x (Hopper) 的新抽象:
  CollectiveMainloop: 主循环 (加载 + 计算)
  CollectiveEpilogue: 后处理 + 写回
  TiledMMA: 描述 MMA 在 Warp/线程级别的切分
  TiledCopy: 描述数据搬运的切分
```


## 6.11 Hopper TMA — 硬件自动搬运多维张量

> TMA (Tensor Memory Accelerator) 是 Hopper 架构最重要的新硬件。
> 它将"计算地址+搬运数据"从软件移到硬件 → 释放 Warp 和寄存器资源。

### TMA 解决的问题

```
传统方式 (Ampere 及更早) 从全局内存加载一个 2D tile 到 Shared Memory:

  每个线程:
    1. 计算自己负责的源地址: addr = base + row*stride + col  (若干条指令)
    2. 发起加载: cp.async(&smem[offset], &global[addr], 16)
    3. 计算下一个地址... 发起下一次加载...
  
  问题:
    - 32 个线程 × 多次加载 = 大量地址计算指令 (浪费 INT ALU)
    - 需要寄存器存放中间地址 (浪费寄存器)
    - 源数据可能是多维张量 (如 [B, H, S, D]) → 地址计算更复杂

TMA 方式 (Hopper):
  只有 1 个线程发起 1 条 TMA 指令:
    "从全局内存地址 (base + tensor坐标), 加载一个 tile[32×64] 到 smem[offset]"
  
  TMA 硬件自动:
    1. 计算多维张量的所有地址 (按 tensor layout 自动遍历)
    2. 发起所有内存请求 (合并 + 最优顺序)
    3. 处理边界 (out-of-bounds 自动填 0)
    4. 数据直接写入 Shared Memory
    5. 完成后通过 mbarrier 通知 Consumer Warps
  
  → 省掉了所有地址计算指令
  → 省掉了存放地址的寄存器
  → 只用 1 个线程的 1 条指令, 其他线程可以同时做计算
```

### TMA Descriptor — 描述张量布局

```
使用 TMA 前, CPU 端需要创建一个 "TMA Descriptor":

CUtensorMap tensorMap;
cuTensorMapEncodeTiled(
    &tensorMap,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT16,  // 数据类型
    4,                                  // 张量维度数
    globalAddress,                      // 全局内存基址
    globalDim,                          // 全局张量大小 {D, S, H, B}
    globalStrides,                      // 每维的步长 (字节)
    boxDim,                             // 每次 TMA 搬运的 tile 大小 {32, 64, 1, 1}
    elementStrides,                     // 元素内步长 (通常 {1, 1, 1, 1})
    CU_TENSOR_MAP_INTERLEAVE_NONE,     // 不交织
    CU_TENSOR_MAP_SWIZZLE_128B,        // Shared Memory 的 Swizzle 模式
    CU_TENSOR_MAP_L2_PROMOTION_L2_256B,// L2 预取提示
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE  // 越界处理
);

// 将 descriptor 传给 kernel (通过 Constant Memory)
// Kernel 内:
//   cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
//       [smem_ptr], [tensorMap, {x_coord, y_coord}], [mbar];
// 
// 一条指令搬一个 32×64 的 FP16 tile!
```

### TMA + Warp Specialization 的完整流程

```
Hopper GEMM 的典型架构:

Block 内的 Warp 分工:
  Producer Warps (1-2 个 Warp):
    只做 TMA 加载: 发 TMA 指令 → 通过 mbarrier 通知 Consumer
    几乎不消耗寄存器和计算资源
    
  Consumer Warps (剩余 Warp):
    只做 WGMMA 计算: 等 mbarrier → 从 SMEM 计算 → 通知 Producer 可以覆写
    拥有大量寄存器 (因为 Producer 不需要)

流水线时间线:
  Producer:  [TMA tile0] [TMA tile1] [TMA tile2] [TMA tile3] ...
  Consumer:             [WGMMA t0 ] [WGMMA t1 ] [WGMMA t2 ] ...
  
  完美重叠: 当 Consumer 算 tile0 时, Producer 同时搬 tile1

mbarrier (异步屏障):
  不是 __syncthreads()! mbarrier 是硬件异步屏障:
  - arrive: "我完成了" (非阻塞, 立即返回)
  - wait: "等别人完成" (阻塞, 但 GPU 会切换到其他 Warp)
  - TMA 完成时自动 arrive → Producer Warp 甚至不需要显式等

  mbarrier 还支持 "transaction counting":
  初始化时告诉 barrier "总共要搬多少字节"
  TMA 每搬完一批自动更新计数 → 计数归零时 Consumer 自动被唤醒
```


## 6.12 Thread Block Cluster — 跨 Block 共享片上存储

```
传统: 每个 Block 的 Shared Memory 互相不可见。
      不同 Block 需要相同数据? 各自从 HBM 加载一份。

Cluster: 2-16 个物理相邻的 Block 组成一个 Cluster:
  - 保证被调度到相邻的 SM
  - 可以直接读写彼此的 Shared Memory (称为 Distributed Shared Memory, DSMEM)
  - 有 Cluster 级同步: cluster.sync()

Cluster GEMM 的应用:
  传统: 每个 Block 独立加载 K/V tile → 如果相邻 Block 需要相同的 K/V → 重复加载!
  Cluster: 只有一个 Block 加载 K/V → 通过 DSMEM 广播给 Cluster 内其他 Block
          → 全局内存访问减少 Cluster_size 倍!

DSMEM 的带宽和延迟:
  同一 SM 的 Shared Memory: ~5 cycles, ~19 TB/s
  相邻 SM 的 DSMEM: ~10-20 cycles (跨 SM, 但仍在片上)
  vs L2 Cache: ~200 cycles
  → DSMEM 比 L2 快 10-20×, 容量可达 Cluster_size × 228KB

声明 Cluster:
  __global__ void __cluster_dims__(4, 1, 1) my_kernel() {
      // 4 个 Block 组成一个 Cluster
      namespace cg = cooperative_groups;
      auto cluster = cg::this_cluster();
      
      unsigned int other_block = (cluster.block_rank() + 1) % cluster.num_blocks();
      float *other_smem = cluster.map_shared_rank(my_shared_data, other_block);
      
      // 直接读另一个 Block 的 Shared Memory!
      float val = other_smem[threadIdx.x];
  }

FlashAttention-3 的 Cluster 用法:
  Cluster 内的多个 Block 各负责不同的 Q 行
  但共享同一份 K/V tile (通过 DSMEM + TMA Multicast)
  → K/V 只加载一次, 多个 Block 共用 → 带宽节省 Cluster_size×
```


## 6.13 本章总结

```
Tensor Core 是现代 GPU 性能的核心来源:
  FP16 TC 性能 = FP32 CUDA Core 的 ~16× (A100: 312 vs 19.5 TFLOPS)

关键知识链:
  数据格式 (FP16/BF16/TF32/FP8/INT8) → 决定精度和吞吐
  MMA 指令 (m16n8k16 等) → 硬件原语
  Fragment 布局 (线程-元素映射) → ldmatrix 和 Swizzle 的基础
  CUTLASS 抽象 (Tile/Layout/Copy/MMA/Epilogue) → 构建高性能 GEMM
  流水线 (cp.async → SMEM → ldmatrix → Reg → MMA) → 多阶段重叠

Hopper 的跳跃:
  TMA: 硬件自动多维地址计算 + 数据搬运
  WGMMA: 4 个 Warp 協作, 直接从 SMEM 读 B 矩阵
  Warp Specialization: Producer/Consumer 解耦
```


## 6.12 Q&A

### Q: 我用了 FP16 为什么没有自动用 Tensor Core?

```
Tensor Core 只在矩阵乘法中被使用, 不是所有 FP16 运算都用 TC!

使用 Tensor Core 的条件:
  1. 必须是矩阵乘法 (torch.matmul, nn.Linear, nn.Conv2d)
  2. 矩阵维度必须是 8 的倍数 (TC 的 tile 大小要求)
  3. 必须调用支持 TC 的库 (cuBLAS, cuDNN) 或显式用 WMMA/MMA PTX
  4. 张量必须 contiguous 且 alignment 正确

Elementwise 的 FP16 运算 (如 GELU) 用的是 FP16 CUDA Core, 不是 TC。
FP16 CUDA Core 同样比 FP32 快 (Ampere 上 2×), 但跟 TC 无关。

检查是否用了 TC:
  ncu 看 "Tensor Core Utilization" 指标 > 0
  或看 SASS 中是否有 HMMA/IMMA 指令
```

### Q: TF32 是不是“偷偷降精度”? 会影响模型精度吗?

```
TF32 确实降了精度 — 从 23 bit mantissa 到 10 bit (= FP16 级别)。
但它保留了 FP32 的 8 bit exponent (= FP32 的范围)。

实际影响:
  大多数训练任务: 无可觉影响 (已经用 FP16 训练了, TF32 精度相同)
  极少数敏感任务: 可能需要关闭 TF32
    torch.backends.cuda.matmul.allow_tf32 = False

优势: 用户代码完全不用改! 输入输出还是 FP32 格式,
Tensor Core 内部自动截取为 TF32 做乘法, 累加用 FP32。
性能: 比 FP32 CUDA Core 快 ~8×, 精度损失微乎其微。
```

### 概念辨析: WMMA vs MMA PTX vs CUTLASS

```
WMMA (编程接口):
  CUDA C++ API, wmma::fragment, wmma::mma_sync
  易用, 但形状有限 (16×16×16 等)
  内存布局不够灵活, 性能距极致有差距

MMA PTX (硬件原语):
  通过内联 PTX 汇编调用, mma.sync.aligned.m16n8k16...
  完全控制 Fragment 布局和寄存器分配
  cuBLAS 和 CUTLASS 内部使用

CUTLASS (库/框架):
  C++ 模板库, 提供从 Tile 到 MMA 的完整抽象
  包含数据搬运 (cp.async, ldmatrix) + Swizzle + 流水线
  可定制度极高, 但学习曲线陡峭

关系: CUTLASS 内部调用 MMA PTX, WMMA 是 MMA PTX 的简化封装。
```


## 6.13 练习题

配套代码在 [`theory/exercises/`](./exercises/) 目录下: [`ch06_ex1_wmma.cu`](./exercises/ch06_ex1_wmma.cu) / [`ch06_ex2_tf32.cu`](./exercises/ch06_ex2_tf32.cu)

### 练习 1: WMMA 基础 [难度: ⭐⭐⭐]

```
写一个使用 WMMA API 的 16×16 矩阵乘法:

1. 初始化两个 16×16 的 half 矩阵 A, B (在 CPU 上, 用简单值如 A[i][j] = i+j)
2. 拷到 GPU
3. 用 wmma::load_matrix_sync + wmma::mma_sync + wmma::store_matrix_sync 计算
4. 拷回 CPU, 和 CPU 参考结果对比

需要:
  #include <mma.h>
  编译时加 -arch=sm_70 或更高 (需要 Volta+)
  用 half 类型: #include <cuda_fp16.h>

提示: 每个线程不是处理一个元素, 而是一个 Warp (32线程) 协作处理整个 16×16 矩阵。
所以你的 kernel 只需要 1 个 Warp = blockDim=32, gridDim=1。
(配合理论: 本章 6.3 "WMMA API")
```

### 练习 2: TF32 的影响 [难度: ⭐⭐]

```
如果你有 PyTorch 环境:

import torch
A = torch.randn(1024, 1024, device='cuda')
B = torch.randn(1024, 1024, device='cuda')

# 关闭 TF32
torch.backends.cuda.matmul.allow_tf32 = False
C1 = A @ B

# 开启 TF32 (Ampere+ 默认开启)
torch.backends.cuda.matmul.allow_tf32 = True
C2 = A @ B

print("最大差异:", (C1 - C2).abs().max().item())
print("相对误差:", ((C1 - C2) / C1.abs().clamp(min=1e-7)).abs().max().item())

观察: 误差有多大? 对于深度学习训练来说能接受吗?
用 %%timeit 对比速度差异。
(配合理论: 本章 6.12 Q&A "TF32 是不是偷偷降精度")
```
