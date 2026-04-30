# 矩阵乘法：Shared Memory 在硬件上的真实行为

本文档配合 `matmul.cu` 阅读。重点解释 Shared Memory 在硬件层面是什么、
数据在 SM 内部怎么流动、为什么 Tiled 版比朴素版快。

> **前置阅读**: [`theory/software_hardware_mapping.md`](../theory/software_hardware_mapping.md) 讲了 Grid→SM→Warp 的通用映射。
> 本文档聚焦于**矩阵乘法用 2D Block 时**的具体映射。


## 2D Block 的线程到 Warp 的映射

```
matmul.cu 用 dim3 block(16, 16) → 256 线程, 排列成 16×16 的 2D 网格。

但 GPU 硬件不认识 "2D 线程" — 只认识 Warp (32 个连续线程)。
256 线程 = 8 Warp。那 2D 坐标怎么映射到 Warp?

映射规则: 线性 ID = threadIdx.x + threadIdx.y × blockDim.x

  threadIdx.y=0:  (0,0) (1,0) (2,0) ... (15,0) → 线性ID 0-15
  threadIdx.y=1:  (0,1) (1,1) (2,1) ... (15,1) → 线性ID 16-31
  threadIdx.y=2:  (0,2) (1,2) ...                → 线性ID 32-47
  ...

  Warp 0: 线性ID  0-31  = (y=0 全部16个) + (y=1 全部16个)
  Warp 1: 线性ID 32-63  = (y=2 全部16个) + (y=3 全部16个)
  Warp 2: 线性ID 64-95  = (y=4 全部16个) + (y=5 全部16个)
  ...
  Warp 7: 线性ID 224-255 = (y=14 全部16个) + (y=15 全部16个)

这意味着:
  同一 Warp 的 32 线程, threadIdx.x 从 0-15 循环两次 (y 不同)。
  → 读 B[k][col] 时, col = blockIdx.x×16 + threadIdx.x
  → 同一 Warp 的 threadIdx.x = 0,1,...15,0,1,...15
  → B 的列地址连续 → 合并访问!

  如果 blockDim 改成 (16,16) 但映射反过来 (x 对应行):
  → 同一 Warp 的 threadIdx.x 连续 → row 连续
  → B[k][row] 的地址跨度很大 → 不合并 → 性能暴跌!
```


## Tiled 版: __syncthreads 时 SM 内的 Warp 状态快照


## 编译阶段的差异

朴素版和 Tiled 版编译后的 SASS 有关键区别：

```
朴素版 kernel 的核心循环 (SASS 简化):
  LOOP:
    LDG.E R2, [R_addr_A] ;    // 从 HBM 加载 A[row][k] → ~500 cycles 延迟!
    LDG.E R3, [R_addr_B] ;    // 从 HBM 加载 B[k][col] → ~500 cycles 延迟!
    FFMA R4, R2, R3, R4 ;     // R4 += R2 * R3 (4 cycles, 但要等 LDG 回来)
    ... 更新地址 ...
    BRA LOOP ;

  → 每次循环 2 次 HBM 访问 + 1 次 FMA
  → 大量时间在等 HBM 数据 (Memory Bound)

Tiled 版 kernel 的核心结构 (SASS 简化):
  TILE_LOOP:
    // 阶段1: 合作加载 A/B tile 到 Shared Memory
    LDG.E R2, [R_addr_A] ;    // 从 HBM 加载 → 慢, 但整个 tile 只加载 1 次
    STS [R_smem_A], R2 ;      // 存入 Shared Memory → 快
    LDG.E R3, [R_addr_B] ;
    STS [R_smem_B], R3 ;
    BAR.SYNC 0 ;              // __syncthreads(): 等所有线程加载完

    // 阶段2: 从 Shared Memory 读取计算
    INNER_LOOP:
      LDS R5, [R_smem_A_k] ;  // 从 Shared Memory 加载 → 5 cycles (快 100×!)
      LDS R6, [R_smem_B_k] ;
      FFMA R4, R5, R6, R4 ;   // 累加
      BRA INNER_LOOP ;

    BAR.SYNC 0 ;              // 等所有人算完再加载下一个 tile
    BRA TILE_LOOP ;

  → 每个 tile: HBM 访问 1 次 (加载), Shared Memory 访问 TILE_SIZE 次 (计算)
  → HBM 访问减少 TILE_SIZE 倍!
```


## Shared Memory 在硬件上是什么

```
Shared Memory 不是"另一种显存"——它是 SM 芯片内部的 SRAM:

┌── SM (Streaming Multiprocessor) ──────────────────────────┐
│                                                            │
│  ┌── Processing Block 0 ──┐  ┌── Processing Block 1 ──┐  │
│  │ Warp Scheduler         │  │ Warp Scheduler          │  │
│  │ 16× FP32 Core          │  │ 16× FP32 Core           │  │
│  │ Register File (64KB)   │  │ Register File (64KB)    │  │
│  └─────────┬──────────────┘  └──────────┬──────────────┘  │
│            │                            │                  │
│            └────────────┬───────────────┘                  │
│                         │                                  │
│  ┌──────────────────────▼────────────────────────────┐    │
│  │         L1 Cache / Shared Memory (192KB)           │    │
│  │         ← 同一块物理 SRAM! 可配置分割比例 →         │    │
│  │                                                    │    │
│  │  物理结构: 32 个 Bank, 每 Bank 宽 4 字节            │    │
│  │  ┌─────┬─────┬─────┬─────┬─────┬───── ─┐         │    │
│  │  │Bank0│Bank1│Bank2│Bank3│ ... │Bank31 │         │    │
│  │  └─────┴─────┴─────┴─────┴─────┴───────┘         │    │
│  │  32 个 Bank 可以同时被 32 个线程并行访问!           │    │
│  │  → 1 个 Warp 的 32 线程如果访问不同 Bank → 1 cycle │    │
│  │  → 多线程访问同一 Bank → 串行 → Bank Conflict!     │    │
│  └────────────────────────────────────────────────────┘    │
│                                                            │
└────────────────────────────────────────────────────────────┘

关键区别:
  全局显存 (HBM): 在 GPU 芯片外部, 通过 Memory Controller + NoC 访问 → ~500 cycles
  Shared Memory:  在 SM 芯片内部, 直接通过 Bank 访问 → ~5 cycles
  
  距离差异 → 延迟差 100×。这就是 Tiled 版快的物理原因。
```


## __syncthreads() 在硬件上做了什么

```
代码中的 __syncthreads() 编译为 SASS 指令: BAR.SYNC 0

硬件行为:
  SM 内有一个 Barrier Unit (屏障硬件):
  
  ┌── Barrier Unit ──────────────────────┐
  │ Barrier #0:                          │
  │   participant_count = 256 (blockDim) │
  │   arrived_count = 0                  │
  └──────────────────────────────────────┘
  
  当 Warp 0 执行到 BAR.SYNC 0:
    arrived_count += 32 (这 32 个线程到达)
    Warp 0 进入 Stall (Barrier Stall)
    
  当 Warp 1 执行到 BAR.SYNC 0:
    arrived_count += 32 → 现在 = 64
    Warp 1 进入 Stall
    
  ... 直到 Warp 7 到达 ...
    arrived_count += 32 → 现在 = 256 = participant_count!
    → 所有 Warp 被释放, 继续执行!

  耗时: 取决于最慢的 Warp 到达的时间。
  如果某些 Warp 还在等 HBM 数据 → 其他 Warp 在 Barrier 处等它们 → 浪费时间。
  → 这就是 ncu 中 "Stall Barrier" 的来源。
```


## 数据流时间线 (Tiled 版, 一次 Tile 循环)

```
时间 ────────────────────────────────────────────────────────────►

Warp 0: [LDG A tile][LDG B tile]--等HBM--→[STS smem][BAR][LDS×16][FFMA×16][BAR]
Warp 1: [LDG A tile][LDG B tile]--等HBM--→[STS smem][BAR][LDS×16][FFMA×16][BAR]
...                                         ↑               ↑
Warp 7: [LDG A tile][LDG B tile]--等HBM--→[STS smem][BAR][LDS×16][FFMA×16][BAR]

        │←── ~500 cyc (等HBM) ──→│         │←── ~80 cyc ──→│
        HBM 延迟被多 Warp 隐藏              Shared Mem 极快

注意:
  LDG (全局加载) 延迟 ~500 cycles, 但 8 个 Warp 交替执行 → 延迟被隐藏
  LDS (Shared 加载) 延迟 ~5 cycles, 几乎无需隐藏
  FFMA (浮点乘加) 延迟 4 cycles, 流水线执行
  BAR 是所有 Warp 的等待点 → 最慢的 Warp 决定整体速度
```


## Tiled 版完整的一次 Tile 循环中 SM 内部的资源占用变化

```
初始状态 (K-loop 开始前):
  Register File: 8 Warp 各占 1792 寄存器 = 14336 / 65536 (21.9%)
  Shared Memory: As[16][16] + Bs[16][16] = 2 × 1024B = 2KB
  Warp 状态: 8 个全部 Eligible

阶14: 协作加载 A/B tile 到 Shared Memory
  每个线程执行:
    LDG.E R_tmp, [addr_A]     → 从 HBM 加载 (全局显存)
    STS [smem_A], R_tmp       → 写入 Shared Memory
    LDG.E R_tmp, [addr_B]
    STS [smem_B], R_tmp
  
  硬件变化:
    LDG 发射后 → MSHR 分配 entry → 请求发往 L2/HBM
    Warp 可能在 LDG 后短暂 Stall (Long Scoreboard)
    但 STS 只需要等 LDG 回来 → STS 和 LDG 有依赖
    Scheduler 在 Warp 0 Stall 时切到 Warp 1, 2, ... → 延迟隐藏
  
  Shared Memory 状态:
    加载完成前: As/Bs 内容不完整 (部分线程还没写完)
    加载完成后: As = A 的一个 16×16 块, Bs = B 的一个 16×16 块

阶24: BAR.SYNC 0 (__syncthreads #1)
  硬件变化:
    Barrier Unit 计数器递增:
      Warp 0 到达 → arrived += 32 → arrived = 32
      Warp 1 到达 → arrived = 64
      ...
      Warp 7 到达 → arrived = 256 = participant_count → 释放!
    
    在等待期间:
      先到达的 Warp 在 "Stall Barrier" 状态
      Scheduler 无法选择它们 (即使 Scoreboard 清了)
      如果 SM 上有其他 Block 的 Warp → Scheduler 可以调度它们
      如果没有 → PB 空闲 → 浪费!
  
  此时 Shared Memory 的状态:
    As[16][16] 和 Bs[16][16] 完整可用 ← 因为 BAR 保证了所有线程都写完

阶34: 从 Shared Memory 读取并计算 (TILE_SIZE 次循环)
  每次循环:
    LDS R_a, [smem_A + ty*16 + k]   → 从 Shared Memory 读 (~5 cyc)
    LDS R_b, [smem_B + k*16 + tx]   → 从 Shared Memory 读 (~5 cyc)
    FFMA R_c, R_a, R_b, R_c         → 累加 (~4 cyc)
  
  硬件变化:
    LDS 不经过 L2/HBM → 不占用 NoC/MC 带宽!
    LDS 延迟只有 5 cyc → Scheduler 几乎不需要切换 Warp
    16 次循环: 16 × (5+5+4) = 224 cycles (vs HBM 的 500 cyc/次!)
    
    Bank Conflict 检查:
      LDS R_a: ty*16+k → 不同线程的 ty 不同 → 不同 Bank → 无冲突 ✓
      LDS R_b: k*16+tx → 不同线程的 tx 不同 → 不同 Bank → 无冲突 ✓
      (如果 TILE_SIZE=32 且读列方向 → 可能 32-way 冲突! 见 [05_bank_conflict/](../05_bank_conflict/))

阶44: BAR.SYNC 0 (__syncthreads #2)
  硬件变化:
    同阶24。确保所有线程都算完后, 再覆盖 Shared Memory 加载下一个 tile。
    如果不同步: Warp 0 可能开始加载下一个 tile 的 STS,
    而 Warp 7 还在读上一个 tile 的 LDS → 数据被覆盖 → 结果错误!

完整一次 Tile 循环的资源占用变化:
  │←─ 加载阶段 ─→│ BAR │─ 计算阶段 ─→│ BAR │
  HBM 带宽:  ████████      0      0      0           0
  SMEM 写:   ████████      0      0      0           0
  SMEM 读:   0      0      0      ██████████████  0
  FP32 ALU:  0      0      0      ██████████████  0
  Barrier:   0      ██     0      0             ██
  
  加载阶段: HBM 带宽是瓶颈 (Warp 在等 LDG)
  计算阶段: FP32 ALU 和 SMEM 读是主角 (LDS+FFMA 流水线)
  BAR: 所有 Warp 等待最慢的那个 (Amdahl 定律!)
```


## 练习题

完成 `matmul.cu` 的阅读后，可以在 [`exercises/`](./exercises/) 目录下做以下练习：

| 练习 | 公式 | 核心考点 |
|------|------|---------|
| [ex1_transpose_level1.cu](./exercises/ex1_transpose_level1.cu) | `B[j][i] = A[i][j]` | 2D 索引 + row-major 读写（只填 kernel） |
| [ex1_transpose_level2.cu](./exercises/ex1_transpose_level2.cu) | 同上 | kernel + dim3 配置 + host 端全部自己写 |
| [ex2_matadd_level1.cu](./exercises/ex2_matadd_level1.cu) | `C[i][j] = A[i][j] + B[i][j]` | 2D 版 vector_add（只填 kernel） |
| [ex2_matadd_level2.cu](./exercises/ex2_matadd_level2.cu) | 同上 | kernel + dim3 配置 + host 端全部自己写 |
| [ex3_gemv_tiled_level1.cu](./exercises/ex3_gemv_tiled_level1.cu) | `y[i] = Σ A[i][k]*x[k]` | Shared Memory 归约 + `__syncthreads()`（只填 kernel） |
| [ex3_gemv_tiled_level2.cu](./exercises/ex3_gemv_tiled_level2.cu) | 同上 | kernel + host 端全部自己写 |

编译方式（在 `exercises/` 目录下）：

```bash
nvcc -O2 -o ex1_transpose_level1 ex1_transpose_level1.cu
./ex1_transpose_level1
```
