# WMMA GEMM：用 Tensor Core 做矩阵乘法

配合 `wmma_gemm.cu` 阅读。

**难度**: ⭐⭐⭐ 专家
**前置知识**: Shared Memory Tiled 矩阵乘法（[`02_matrix_mul/`](../02_matrix_mul/)）；混合精度（[`17_mixed_precision/`](../17_mixed_precision/)）
**读完你能做什么**: 能自己写出一个完整的 WMMA GEMM kernel，理解 Warp 到 tile 的映射和 K 维度循环

> **编译要求**: nvcc 需要 `-arch=sm_70` 或更高（Volta+，Tensor Core 从这里开始）。


## 1. Tensor Core 是什么，和 CUDA Core 有什么区别

```
CUDA Core (通用):
  一条 FFMA 指令 = a × b + c = 2 FLOP
  延迟: ~4 cycles
  1 个 SM 有 64 个 FP32 Core → 64 × 2 FLOP / 4 cycles ≈ 32 FLOP/cycle/SM

Tensor Core (专用矩阵乘法电路):
  一条 HMMA 指令 = D[16×16] = A[16×16] × B[16×8] + C[16×8] = 4096 FLOP
  延迟: ~8 cycles
  1 个 SM (A100) 有 4 个 Tensor Core → 4 × 4096 / 8 = 2048 FLOP/cycle/SM

  2048 / 32 ≈ 64× → Tensor Core 的 SM 级吞吐是 CUDA Core FP32 的 ~64 倍!

为什么快这么多?
  → Tensor Core 不是通用电路, 它只会做固定尺寸的矩阵乘法
  → 内部是一个专用乘加器阵列, 一拍能处理整个 16×16×16 矩阵块
```

### 使用的代价

```
1. 输入精度: 必须是 FP16/BF16/TF32/INT8/FP8 — FP32 不能直接用
2. 矩阵大小: 不是 16 的倍数时需要 padding
3. 编程接口: 用 WMMA API (C++) 或 MMA PTX (汇编)
4. 数据布局: Tensor Core 需要矩阵分布在 32 线程的寄存器中 → Fragment 抽象
```


## 2. 完整 WMMA GEMM Kernel

```cuda
#include <mma.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void wmma_gemm(const half *A, const half *B, float *C,
                           int m, int n, int k) {
    // ① 计算本 Warp 负责 C 的哪个 16×16 tile
    int warpM = blockIdx.y * blockDim.y + threadIdx.y;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    if (warpM * WMMA_M >= m || warpN * WMMA_N >= n) return;

    // ② 声明 Fragment — 分散在 32 线程寄存器中的矩阵块
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);  // C = 0

    // ③ 沿 K 维度循环 — 这是"外层 tiling"
    for (int ki = 0; ki < k; ki += WMMA_K) {
        // ④ 加载 A 的一个 16×16 tile (从 HBM)
        wmma::load_matrix_sync(a_frag,
            A + warpM * WMMA_M * k + ki,  // 基址: A[warpM*16][ki]
            k);                            // leading dimension

        // ⑤ 加载 B 的一个 16×16 tile (从 HBM, 列主序!)
        wmma::load_matrix_sync(b_frag,
            B + ki + warpN * WMMA_N * k,  // 基址: B[ki][warpN*16] (列主序)
            k);

        // ⑥ Tensor Core 执行: C += A × B  (一条指令, 4096 FLOP!)
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // ⑦ 把结果写回 HBM
    wmma::store_matrix_sync(
        C + warpM * WMMA_M * n + warpN * WMMA_N,  // C[warpM*16][warpN*16]
        c_frag, n, wmma::mem_row_major);
}
```


## 3. 逐段拆解

### ① Warp 到 C tile 的映射 — 最关键的一步

```cuda
int warpM = blockIdx.y * blockDim.y + threadIdx.y;
int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
```

```
Host 端 Launch 配置:
  dim3 block(128, 1);   // blockDim.x=128, blockDim.y=1
  dim3 grid(N/16/(128/32), M/16);

关键约束: 每个 Warp (32 线程) 独立处理一个 16×16 的 C tile。
         WMMA 的所有 API (load/mma/store) 都是 Warp 级操作 —
         一个 Warp 的 32 线程必须一起调用它们。

所以 Block 内有 blockDim.x / 32 = 4 个 Warp，每个 Warp 处理不同的 C tile。

warpM 的计算:
  blockIdx.y 从 0 到 M/16-1
  threadIdx.y = 0 (blockDim.y=1)
  → warpM = blockIdx.y → 直接对应"第几行 tile"

warpN 的计算:
  blockIdx.x 从 0 到 N/16/(128/32)-1
  threadIdx.x 从 0 到 127 → 4 个 Warp
  threadIdx.x / 32 = 0,1,2,3 → 4 个不同的"列 tile"
  → warpN = blockIdx.x * 4 + (threadIdx.x / 32)

具体例子 (M=N=512):
  gridDim.y = 512/16 = 32        → blockIdx.y: 0..31
  gridDim.x = 32 / 4 = 8         → blockIdx.x: 0..7

  Block(0, 0) 内的 4 个 Warp:
    Thread  0..31:  warpM=0, warpN=0×4+0=0  → 处理 C 的 (0,0) tile
    Thread 32..63:  warpM=0, warpN=0×4+1=1  → 处理 C 的 (0,1) tile
    Thread 64..95:  warpM=0, warpN=0×4+2=2  → 处理 C 的 (0,2) tile
    Thread 96..127: warpM=0, warpN=0×4+3=3  → 处理 C 的 (0,3) tile

  Block(1, 5) 内的 4 个 Warp:
    Thread  0..31:  warpM=5, warpN=1×4+0=4  → 处理 C 的 (5,4) tile
    ...

  32×8×4 = 1024 个 Warp 各处理一个 16×16 tile
  → 1024 × 16 × 16 = 262144 元素
  → 但 C 是 512×512 = 262144 → 刚好覆盖! ✓
```

**为什么 blockDim=(128, 1) 而不是 (16, 16)？**

```
WMMA 的 API 是 Warp 级操作 (warp-synchronous)。
同一 Warp 的 32 个线程必须一起调用 load_matrix_sync / mma_sync / store_matrix_sync。

如果用 blockDim(16, 16)、threadIdx.y 来区分 Warp:
  → 同一 Warp 内 threadIdx.y 只有 2 个值 (y=0 的 16 线程 + y=1 的 16 线程)
  → 计算 warpM/warpN 变复杂且容易出错

用 blockDim(128, 1):
  → threadIdx.y 固定为 0 → warpM 直接用 blockIdx.y
  → 所有线程的 x 维度连续 → 自然分成 4 个 Warp (0..31, 32..63, 64..95, 96..127)
  → 简化映射逻辑
```

### ② Fragment 声明

```cuda
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
```

```
模板参数的语义:
  fragment<角色, M, N, K, 数据类型, 内存布局>

  matrix_a:     A 矩阵 (形状 M×K = 16×16)
  matrix_b:     B 矩阵 (形状 K×N = 16×16)
  accumulator:  C/D 矩阵 (形状 M×N = 16×16, FP32 累加)

  row_major vs col_major:
    A 是 row_major  → 内存中正常行主序 (和 C/CUDA 默认一致)
    B 是 col_major  → 内存中列主序!

    为什么 B 用列主序?
      → 这是 Tensor Core 硬件的内部要求
      → A 的 K 维度连续 (row_major), B 的 K 维度也连续 (col_major)
      → 这样 load_matrix_sync 内部的 ldmatrix 指令能高效加载

Fragment 不是普通数组!
  → 16×16 = 256 个元素, 分散在 32 线程中
  → 每个线程持有 256/32 = 8 个元素, 存在该线程的寄存器中
  → 你不能用 a_frag[i][j] 访问单个元素
  → 只能通过 load / mma / store 操作整个 Fragment
```

### ③⑤ K 维度的外层 tiling 循环

```cuda
for (int ki = 0; ki < k; ki += WMMA_K) {
    wmma::load_matrix_sync(a_frag, A + warpM * 16 * k + ki, k);
    wmma::load_matrix_sync(b_frag, B + ki + warpN * 16 * k, k);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
}
```

```
这和 02_matrix_mul 的 Shared Memory Tiling 完全一样的思想!

只是每一轮:
  → 不需要手动搬数据到 Shared Memory (load_matrix_sync 直接从 HBM 加载)
  → 不需要手动写乘加循环 (mma_sync 一条指令做 16×16×16)
  → 不需要 __syncthreads() (load_matrix_sync 和 mma_sync 内部有隐式同步)

以 M=N=K=512 为例: K 维度需要 512/16 = 32 轮循环
  每轮: load A tile + load B tile + mma → 累加到 c_frag
  32 轮后: c_frag = 完整的 C tile!

A 地址计算:
  warpM * 16 * k + ki
  → A[warpM*16][ki] = A 的第 warpM*16 行、第 ki 列
  → 加载 16×16 的 A tile

B 地址计算 (列主序!):
  ki + warpN * 16 * k
  → B = [K, N] 的列主序矩阵
  → B[ki][warpN*16] = B 的第 ki 行、第 warpN*16 列 (在列主序布局下)
  → 加载 16×16 的 B tile

如果你的 B 是行主序存储的 (通常情况!):
  → 不能直接给 wmma::col_major
  → 需要: 要么在调用前转置 B, 要么加载时调整 leading dimension
  → 更简单的做法: B 在 host 端就以"列主序"解释, 即 B[col][row] 但存储顺序反过来
```

### B 列主序的解释 — 最容易犯错的地方

```
B 是 [K, N] 矩阵。如果按 row_major 存储:
  B[0][0], B[0][1], ..., B[0][N-1], B[1][0], ...

Tensor Core 需要 B 在内存中 K 维度连续 (即 B[ki][col] 和 B[ki+1][col] 地址相邻):
  → col_major 布局: B[0][col], B[1][col], B[2][col], ...

host 端如何满足:
  在 host 端, B 的逻辑形状是 [K, N], 但用列主序存储:
    B 的第 col 列的元素连续存储 → B[0][col], B[1][col], ..., B[K-1][col]
  
  即内存中: B 的第 0 列, 第 1 列, ..., 第 N-1 列

  CPU 验证时的访问: B[ki + col * k] → 这就是列主序访问模式!

检查: 如果你的数据本来是 row_major, 加载到 B_frag (col_major) 会怎样?
  → load_matrix_sync 会读错误的地址 → 结果完全错误
  → 解决办法: 用转置后的 B, 或在 host 端就按列主序存储
```

### ⑥ `wmma::mma_sync` — Tensor Core 执行

```cuda
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
//             ↑D     ↑A      ↑B      ↑C
//             C += A × B
```

```
这条 C++ 调用编译为一条 SASS 指令:
  HMMA.16816.F32 Rdest, Ra, Rb, Rc

HMMA  = Half-precision Matrix Multiply-Accumulate
.16816 = M16 × N8 × K16 (但 WMMA 自动拆成 2 步, 覆盖 N=16)
.F32  = 累加器精度 FP32

计算量: 16 × 16 × 16 = 4096 次乘加 = 8192 FLOP
延迟: ~8 cycles
→ 8192 / 8 ≈ 1024 FLOP/cycle (单指令!)

对比: CUDA Core FFMA = 2 FLOP, 4 cycles → 0.5 FLOP/cycle
→ 单条指令快 ~2000×, SM 级快 ~64× (考虑到 SM 有 64 CUDA Core vs 4 Tensor Core)
```

### ⑦ 写回结果

```cuda
wmma::store_matrix_sync(
    C + warpM * 16 * n + warpN * 16,  // 目标地址
    c_frag, n, wmma::mem_row_major);  // leading dim + 输出布局
```

```
目标地址: C[warpM*16][warpN*16], row-major 布局
mem_row_major: 输出用行主序存储 (和 C/CUDA 默认一致)
```


## 4. Host 端 Grid/Block 配置推导

```cuda
dim3 block(128, 1);
dim3 grid_wmma(N / WMMA_N / (128/32), M / WMMA_M);
```

```
推导过程:

1. 每个 Warp 处理一个 16×16 的 C tile
2. Block 内有 blockDim.x / 32 = 128/32 = 4 个 Warp
3. 4 个 Warp 在 x 方向并排, 各处理相邻的 4 个 C tile 列
4. 所以一个 Block 覆盖: 1 行 tile × 4 列 tile

gridDim.x = (N/16) / 4 = N/64
  → N 方向需要 N/16 个 16-wide tile
  → 每个 Block 处理 4 个 → gridDim.x = (N/16) / 4

gridDim.y = (M/16) / 1 = M/16
  → M 方向需要 M/16 个 16-tall tile
  → 每个 Block 处理 1 个 → gridDim.y = M/16

对于 M=N=512:
  gridDim.y = 512/16 = 32
  gridDim.x = 512/64 = 8
  → 32 × 8 = 256 个 Block
  → 256 × 4 = 1024 个 Warp
  → 1024 × 256 = 262144 线程
  → 每个线程写 8 个输出元素 (Fragment 的 8 个 FP32)
  → 覆盖 512 × 512 ✓
```


## 5. 和其他 GEMM 实现的对比

`wmma_gemm.cu` 中包含了 4 种实现，跑一遍就能看到差距：

```
1. FP32 naive (CUDA Core, 无 tiling):
   → 从 HBM 重复读, 无 Shared Memory → Memory Bound
   → ~1% peak FP32

2. FP32 tiled (Shared Memory, CUDA Core):
   → 有 SMEM 复用 → 好很多, 但仍是 CUDA Core
   → ~5% peak FP32 (但 peak 只有 19.5 TFLOPS)

3. FP16 naive (CUDA Core, 无 tiling):
   → FP16 只省了一半带宽 → 仍然是 Memory Bound
   → ~2% peak FP16

4. FP16 WMMA (Tensor Core):
   → Tensor Core 专用加速 → Compute Bound!
   → 一个小 kernel 就能接近 peak FP16 Tensor Core (312 TFLOPS)
   → 比方案 1 快 ~10-30×

实际性能受限于:
  → 矩阵大小: 512×512 仍然偏小, launch overhead 明显
  → load_matrix_sync 直接从 HBM 加载 (没有 Shared Memory 预取!)
  → 真正的生产级实现 (cuBLAS/CUTLASS) 有 Shared Memory staging + double buffering
```

### WMMA 初学者版本 vs 生产级实现的差异

```
本 kernel (教学版):
  load_matrix_sync 直接从 HBM 加载 A 和 B
  → 没有 Shared Memory 预取
  → 每个 K tile 都从 HBM 读一次 → 仍然有数据重用问题!

生产级 (cuBLAS/CUTLASS):
  1. 外层: Shared Memory tiling (把 A 和 B 的更大块先搬进 SMEM)
  2. 内层: WMMA mma_sync 从 SMEM 用 ldmatrix 指令加载
  3. 软件流水线 (double/triple buffering): 用异步拷贝预取下一轮的数据
  → 这才能接近 Tensor Core 的理论峰值!

但本教学版已经展示了 WMMA 的核心概念:
  → Fragment 抽象
  → mma_sync 一条指令完成 16×16×16 矩阵乘
  → Warp 到 tile 的映射
  → K 维度循环

理解了这些, 再去学习 CUTLASS 就能看懂它在做什么。
```


## 6. Fragment 在线程中的分布 (选读)

你通常不需要知道这个, 但理解它对高级优化有帮助：

```
wmma::fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;

16×16 FP16 矩阵 = 256 个 half = 512 bytes
分给 32 线程 → 每线程 8 个 half = 4 个 32-bit 寄存器

Ampere m16n8k16 的线程分布 (简化):
  Thread 0-7:   持有 A 的第 0-7 行  (每个线程连续 2 个元素)
  Thread 8-15:  持有 A 的第 8-15 行
  Thread 16-23: A 的 k=4..7 列部分
  Thread 24-31: A 的 k=12..15 列部分

你完全不需要手动管理这个!
  load_matrix_sync → 自动把内存布局转换为 Fragment 布局
  store_matrix_sync → 自动把 Fragment 布局转换为内存布局

为什么需要 Fragment? 为什么不像 Shared Memory 一样用 [16][16] 数组?
  → 如果用 Shared Memory, 32 个线程通过 SMEM 交换数据 → 需要 __syncthreads
  → Fragment 把数据直接存在寄存器中 → mma_sync 内部通过硬件数据通路交换
  → 零延迟的 Warp 内数据共享!
```


## 7. 常见错误

- **B 的 layout 搞错** → 症状: 结果完全不对。B 需要 col_major! 如果你的 B 是 row_major，要么先转置，要么在 host 端按列主序存储数据
- **grid/block 算错** → 症状: 只处理了部分 tile。验证: 总 Warp 数 = gridDim.x × gridDim.y × (blockDim.x / 32) = (N/16) × (M/16) → 刚好覆盖所有 C tile
- **矩阵大小不是 16 的倍数** → 症状: 超出边界的 warp 访问越界。需要在 kernel 开头加边界检查，并在 host 端 padding 输入矩阵
- **忘了 `#include <cuda_fp16.h>`** → `half` 类型未定义
- **忘了 `-arch=sm_70` 编译选项** → Volta+ 才有 Tensor Core
- **把 `c_frag` 初始化为 0 但忘了在 K 循环外做** → 症状: 每个 K tile 的结果被覆盖而不是累加。`fill_fragment` 在 K 循环之前做一次
- **load_matrix_sync 的参数 k (leading dimension)** → A 是 row_major，leading dim = k (行宽)；B 是 col_major，leading dim = k (列高)。不要写成 n!


## 练习题

| 练习 | 任务 | 核心考点 |
|------|------|---------|
| [ex1_wmma_add_level1.cu](./exercises/ex1_wmma_add_level1.cu) | WMMA 矩阵加法 | fragment load/store 基本用法（只填 kernel） |
| [ex1_wmma_add_level2.cu](./exercises/ex1_wmma_add_level2.cu) | 同上（完整实现） | kernel + host 全部自己写 |

```bash
nvcc -O2 -arch=sm_75 -o ex1_wmma_add_level1 ex1_wmma_add_level1.cu
./ex1_wmma_add_level1
```
