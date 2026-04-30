# Register Tiling GEMM：数据在寄存器中的精确流动

配合 `gemm_register.cu` 阅读。


## 为什么需要 Register Tiling

```
GEMM 的优化是一层层叠加的:

第 1 层: 全局内存 → Shared Memory (Tiling)
  02_matrix_mul 做了这一步
  效果: 全局内存访问量减少 ~TILE_SIZE 倍
  瓶颈转移: Shared Memory 读取成为新的瓶颈

第 2 层: Shared Memory → 寄存器 (Register Tiling) ← 本课!
  本程序做的事
  效果: Shared Memory 读取量减少 ~TM (或 TN) 倍
  瓶颈转移: 计算成为瓶颈 → 接近 Compute Bound → 好事!

第 3 层: 用 Tensor Core 加速计算
  15_wmma_gemm 展示了这一步
  效果: 计算吞吐提升 ~8×

直觉:
  没有 Register Tiling: 每做 1 次乘加，就要从 SMEM 读 2 次
  有 Register Tiling:   每做 16 次乘加，只需从 SMEM 读 8 次
  → SMEM 读取的"性价比"提高了 4 倍
```


## 朴素 vs Register Tiled 的硬件数据流对比

```
朴素版 (每线程 1 个 C 元素, Thread Tile = 1×1):
  K-loop 每一步:
    从 SMEM 读 A 的 1 个元素: LDS R_a, [smem_A + ty*BK + k]   → 5 cycles
    从 SMEM 读 B 的 1 个元素: LDS R_b, [smem_B + k*BN + tx]   → 5 cycles
    做 1 次 FMA:              FFMA R_c, R_a, R_b, R_c          → 4 cycles
    
    SMEM 读取: 2 次/FMA
    数据复用率: 1×1 / (1+1) = 0.5

Register Tiled 版 (Thread Tile = 4×4):
  K-loop 每一步:
    从 SMEM 读 A 的 4 个元素: 4× LDS → a_frag[0..3]           → 5 cycles (可流水线)
    从 SMEM 读 B 的 4 个元素: 4× LDS → b_frag[0..3]           → 5 cycles
    做 16 次 FMA (外积):
      for i=0..3:
        for j=0..3:
          FFMA c_reg[i][j], a_frag[i], b_frag[j], c_reg[i][j]
    
    SMEM 读取: 8 次 → 产生 16 次 FMA
    数据复用率: 4×4 / (4+4) = 2.0 (比朴素版好 4×!)

外积 (Outer Product) 在硬件上:
  ┌── 寄存器 ──────────────────────────────┐
  │ a_frag: [a0, a1, a2, a3]               │
  │ b_frag: [b0, b1, b2, b3]               │
  │                                         │
  │ c_reg[0][0] += a0 * b0  ← FFMA 指令 1  │
  │ c_reg[0][1] += a0 * b1  ← FFMA 指令 2  │
  │ c_reg[0][2] += a0 * b2  ← FFMA 指令 3  │
  │ c_reg[0][3] += a0 * b3  ← FFMA 指令 4  │
  │ c_reg[1][0] += a1 * b0  ← FFMA 指令 5  │
  │ ...                                     │
  │ c_reg[3][3] += a3 * b3  ← FFMA 指令 16 │
  │                                         │
  │ 全部在寄存器中! 零 SMEM 访问, 零延迟!   │
  └─────────────────────────────────────────┘
  
  16 条 FFMA 背靠背发射, 流水线填满 → 接近计算峰值!
  (前提: a_frag 和 b_frag 已经在寄存器中 → LDS 的延迟被流水线隐藏)
```


## Block Tile 和 Thread Tile 的空间分解

```
整个 GEMM:  C[M × N] = A[M × K] × B[K × N]

第 1 级: Grid → Block 的分解 (Block Tile)
  每个 Block 负责 C 的一个 BM×BN 的子块
  gridDim = (N/BN, M/BM) = (1024/64, 1024/64) = (16, 16) = 256 个 Block

  ┌────────────── C [1024 × 1024] ──────────────┐
  │ Block(0,0)   Block(1,0)   ...   Block(15,0)  │
  │ [64×64]      [64×64]            [64×64]       │
  │                                               │
  │ Block(0,1)   Block(1,1)   ...   Block(15,1)  │
  │ ...                                           │
  │ Block(0,15)  ...                Block(15,15)  │
  └───────────────────────────────────────────────┘

第 2 级: Block → Thread 的分解 (Thread Tile)
  每个 Block 有 (BM/TM) × (BN/TN) = 16 × 16 = 256 个线程
  每个线程负责 C 子块中的 TM×TN = 4×4 = 16 个元素

  Block(0,0) 的 64×64 子块:
  ┌─────────────────────────────────────────────────┐
  │ T(0,0)    T(1,0)    ...    T(15,0)              │
  │ [4×4]     [4×4]            [4×4]                │
  │                                                  │
  │ T(0,1)    T(1,1)    ...    T(15,1)              │
  │ ...                                              │
  │ T(0,15)   ...              T(15,15)             │
  └─────────────────────────────────────────────────┘
  
  每个 T(tx, ty) 在寄存器中维护 c_reg[4][4] = 16 个累加器

第 3 级: K 方向的循环 (BK = 8)
  每次迭代:
    1. 整个 Block 协作加载 A[BM×BK] + B[BK×BN] 到 SMEM
    2. 每个线程从 SMEM 加载 a_frag[4] 和 b_frag[4]
    3. 做 4×4 外积累加到 c_reg
  循环 K/BK = 1024/8 = 128 次
```


## 寄存器使用量的影响

```
每线程的寄存器占用:
  c_reg[4][4] = 16 个 float = 16 个寄存器
  a_frag[4]   = 4 个寄存器
  b_frag[4]   = 4 个寄存器
  地址计算等  ≈ 10 个寄存器
  总计: ~34 个寄存器/线程

  256 线程/Block → 256 × 34 = 8704 个寄存器/Block
  SM 有 65536 个寄存器 → 可以同时驻留 65536/8704 ≈ 7 个 Block
  7 Block × 256 线程 = 1792 线程 = 56 Warp → Occupancy ≈ 87% ✓

  如果 TM=TN=8 (更大的 Thread Tile):
  c_reg = 64, 总 ~80 寄存器/线程
  → 65536 / (256×80) ≈ 3 Block → 24 Warp → 37.5% Occupancy
  → 可能性能更好 (更多复用) 或更差 (Occupancy 太低) → 需要实测!

权衡:
  ┌──────────────────────────────────────────────────────────┐
  │  Thread Tile   │ 复用率  │ 寄存器  │ Occupancy │ 效果   │
  │  1×1          │ 0.5    │ ~10    │ 100%     │ 慢     │
  │  4×4          │ 2.0    │ ~34    │ 87%      │ 甜点   │
  │  8×8          │ 4.0    │ ~80    │ 37%      │ 需实测 │
  │  16×16        │ 8.0    │ ~280   │ <10%     │ 太慢   │
  └──────────────────────────────────────────────────────────┘
  
  核心矛盾: 更大的 Tile = 更好的数据复用 + 更低的 Occupancy
  → 通常 4×4 ~ 8×8 是甜点区间
  → 精确最优点需要在具体 GPU 上实测
```


## 从 Register Tiling 到 CUTLASS/cuBLAS

```
本程序的 Register Tiling 是 GEMM 优化的核心思想。
实际的高性能 GEMM 库 (CUTLASS, cuBLAS) 在此基础上还有很多优化:

1. 双缓冲 (Double Buffering)
   问题: __syncthreads() 后才能开始计算，加载和计算不能重叠
   方案: 用 2 份 SMEM，加载下一块的同时计算当前块
   → 隐藏 SMEM 加载延迟

2. 向量化加载 (Vectorized Load)
   从全局内存到 SMEM 用 float4/LDG.128 加载
   → 减少指令数，提高带宽利用率

3. Swizzle 布局
   调整 SMEM 中的数据排列，消除 Bank Conflict
   → SMEM 读取不再有串行化

4. Tensor Core ([theory/06_tensor_core.md](../theory/06_tensor_core.md))
   用 WMMA/MMA 替代 FFMA
   → 16×16×16 的 FMA 用一条指令完成

5. Warp Tile
   在 Thread Tile 之上还有 Warp Tile 的概念
   一个 Warp 的 32 线程协作处理一个更大的子矩阵
   → 进一步减少 SMEM 访问

这些优化叠加后，A100 上的 GEMM 可以达到 >90% 的理论计算峰值。
本程序是理解这些优化的起点。
```


## 练习题

| 练习 | 任务 | 核心考点 |
|------|------|---------|
| [ex1_reg_tile_2x2_level1.cu](./exercises/ex1_reg_tile_2x2_level1.cu) | 2×2 Register Tiled GEMM | 简化版 register blocking（只填 kernel） |

```bash
nvcc -O2 -o ex1_reg_tile_2x2_level1 ex1_reg_tile_2x2_level1.cu
```
