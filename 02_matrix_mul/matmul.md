# 矩阵乘法：从零写出 Shared Memory Tiling

本文档配合 `matmul.cu` 阅读。目标：**读完能自己写出 Tiled 矩阵乘法**。

> **前置阅读**: 先完成 [`tutorial.md`](../tutorial.md) Part 1，理解最基本的 kernel 写法（`<<<grid, block>>>`、`threadIdx`、`blockIdx`、`blockDim`）。
> 本文档聚焦于矩阵乘法中 Shared Memory Tiling 的完整推导过程。


## 1. 矩阵乘法在算什么

```
C = A × B

其中 A: M×K, B: K×N, C: M×N

C[i][j] = Σ(k=0..K-1) A[i][k] × B[k][j]

具体例子: M=2, K=3, N=2

      A (2×3)          B (3×2)           C (2×2)
  ┌         ┐     ┌         ┐     ┌                   ┐
  │ a00 a01 a02│   │ b00 b01 │     │ a00*b00+a01*b10+a02*b20  ... │
  │ a10 a11 a12│ × │ b10 b11 │  =  │ a10*b00+a11*b10+a12*b20  ... │
  └         ┘     │ b20 b21 │     └                   ┘
                  └         ┘

C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0]
        = a00*b00 + a01*b10 + a02*b20
```

关键观察：**计算 C 的每个元素，需要 A 的一整行 和 B 的一整列。
而且所有同行/同列的 C 元素会重复用同一份 A/B 数据。**


## 2. 朴素 GPU 版 — 把 for 循环直接映射到线程

### 2.1 每个线程算一个 C 元素

最直接的思路：让每个线程负责 C 的一个 `[row][col]`，线程自己跑一个 for 循环沿 K 累加。

```cuda
__global__ void matmul_naive(const float *A, const float *B, float *C,
                             int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

### 2.2 逐行理解

```
第 1 行: int row = blockIdx.y * blockDim.y + threadIdx.y;
         ↑ 我负责 C 的第几行

第 2 行: int col = blockIdx.x * blockDim.x + threadIdx.x;
         ↑ 我负责 C 的第几列

  为什么 x 对应列、y 对应行?
  → 因为同一 Warp 的 32 个线程 threadIdx.x 连续
  → 让它们访问 B[k][col] 时 col 连续 → 地址连续 → 合并访问
  → 如果反过来 (x 对应行), B 的读取会变成 stride=N → 慢 30×
```

### 2.3 线程映射图

```
假设 M=N=4, K=3, blockDim=(2,2), gridDim=(2,2)

C 矩阵 (4×4):
        col=0  col=1  col=2  col=3
row=0: [ 0,0   0,1   1,0   1,1 ]    ← 每个格子标注 (blockIdx.x, blockIdx.y)
row=1: [ 0,0   0,1   1,0   1,1 ]       gridDim.x=2, gridDim.y=2
row=2: [ 2,0   2,1   3,0   3,1 ]       总共 8 个 Block
row=3: [ 2,0   2,1   3,0   3,1 ]

以 Block (0,0) 为例, blockDim=(2,2):
  Thread (0,0): row=0, col=0 → 算 C[0][0]
  Thread (1,0): row=0, col=1 → 算 C[0][1]  ← threadIdx.x=1, threadIdx.y=0
  Thread (0,1): row=1, col=0 → 算 C[1][0]
  Thread (1,1): row=1, col=1 → 算 C[1][1]
```

### 2.4 朴素版的问题 — 数据被反复从 HBM 读取

```
以 Block (0,0) 的 4 个线程为例, K=3:

  Thread (0,0) 读: A[0][0],A[0][1],A[0][2] + B[0][0],B[1][0],B[2][0]
  Thread (1,0) 读: A[0][0],A[0][1],A[0][2] + B[0][1],B[1][1],B[2][1]
  Thread (0,1) 读: A[1][0],A[1][1],A[1][2] + B[0][0],B[1][0],B[2][0]
  Thread (1,1) 读: A[1][0],A[1][1],A[1][2] + B[0][1],B[1][1],B[2][1]

  A[0][0] 被 Thread(0,0) 和 Thread(1,0) 各读了一次 → 从 HBM 读了 2 次!
  B[0][0] 被 Thread(0,0) 和 Thread(0,1) 各读了一次 → 从 HBM 读了 2 次!

  推广到 TILE_SIZE=16 (Block 有 256 个线程):
    同一行的 16 个线程都需要 A[row][k] → 同一个值被读了 16 次!
    同一列的 16 个线程都需要 B[k][col] → 同一个值被读了 16 次!

  每次读都是 HBM (~500 cycles)。这是巨大的浪费。
```

**(k=0)** 中，4 个线程访问 A 和 B 的情况 | **(k=1)** 和 **(k=2)** 与此相同——每轮都在重复读
:---:|:---:
![fig01](diagrams/matmul_fig01.svg) | 每轮都在重复读


## 3. Tiled 版 — 让 Block 内的线程共享数据

### 3.1 核心思想

```
问题: 同一 Block 内 16×16=256 个线程都需要读同一块 A 和 B 的数据。
     但每个线程各自从 HBM 读 → 同一数据被读了 256 次。

解决: 让 Block 内的线程协作——先把需要的数据集体搬进 Shared Memory，
     然后大家都从 Shared Memory 读。
     
     Shared Memory 在 SM 芯片内部, 延迟 ~5 cycles (vs HBM ~500 cycles)
     → 同一份数据被 256 个线程读 256 次, 但只需要从 HBM 搬 1 次!

具体做法:
  把 K 维度切成若干个 TILE_SIZE 大小的"块"(tile)。
  每一步:
    1. 256 个线程各搬 A 的 1 个元素 + B 的 1 个元素 → Shared Memory
    2. __syncthreads() → 确保大家都搬完了
    3. 从 Shared Memory 读 TILE_SIZE 次, 做 TILE_SIZE 次乘加
    4. __syncthreads() → 确保大家都算完了, 才能覆盖 Shared Memory
```

### 3.2 完整代码 (先看全貌, 再逐段拆解)

```cuda
#define TILE_SIZE 16

__global__ void matmul_tiled(const float *A, const float *B, float *C,
                             int M, int K, int N) {
    // ① Shared Memory 声明
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // ② 我负责 C 的哪一行哪一列
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // ③ 沿 K 方向滑动窗口
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {

        // ④ 每个线程搬 A 的一个元素到 As, B 的一个元素到 Bs
        int aCol = t * TILE_SIZE + threadIdx.x;   // 线程在 A tile 中负责的列
        int bRow = t * TILE_SIZE + threadIdx.y;   // 线程在 B tile 中负责的行

        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K)
            ? A[row * K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N)
            ? B[bRow * N + col] : 0.0f;

        // ⑤ 同步: 等所有人都写完 Shared Memory
        __syncthreads();

        // ⑥ 从 Shared Memory (快!) 读数据, 做 TILE_SIZE 次乘加
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        // ⑦ 同步: 等所有人都算完, 才能覆盖 Shared Memory
        __syncthreads();
    }

    // ⑧ 写回结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### 3.3 逐段拆解

#### ① Shared Memory 声明

```cuda
__shared__ float As[TILE_SIZE][TILE_SIZE];  // 存 A 的一个 16×16 小块
__shared__ float Bs[TILE_SIZE][TILE_SIZE];  // 存 B 的一个 16×16 小块
```

`__shared__` 关键字把这 2KB 的数据放在 SM 的片上 SRAM 中。
Block 内的 256 个线程都可以读写它。延迟 ~5 cycles（快 100 倍）。

#### ② 线程 → C 矩阵的映射

```cuda
int row = blockIdx.y * TILE_SIZE + threadIdx.y;
int col = blockIdx.x * TILE_SIZE + threadIdx.x;
```

```
示例: M=N=64, TILE_SIZE=16 → gridDim=(4,4), blockDim=(16,16)

C 矩阵 (64×64), 被分成 4×4 = 16 个 16×16 的块:

  Block(0,0) 负责 C[0:16][0:16],    Block(1,0) 负责 C[0:16][16:32],  ...
  Block(0,1) 负责 C[16:32][0:16],   Block(1,1) 负责 C[16:32][16:32], ...
  ...

以 Block(0,0) 为例, 其内部的 256 个线程:
  Thread(0,0): row=0,  col=0   → 算 C[0][0]
  Thread(0,1): row=0,  col=1   → 算 C[0][1]
  ...
  Thread(0,15): row=0, col=15  → 算 C[0][15]
  Thread(1,0):  row=1,  col=0  → 算 C[1][0]
  ...
  Thread(15,15): row=15, col=15 → 算 C[15][15]
```

#### ③ K 方向的分块循环

```cuda
for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
```

`(K + TILE_SIZE - 1) / TILE_SIZE` = 向上取整的除法, 即 `ceil(K / TILE_SIZE)`。

```
例如 K=50, TILE_SIZE=16:
  (50 + 16 - 1) / 16 = 65 / 16 = 4 → 需要 4 轮

  t=0: 处理 A 的 col 0..15,  B 的 row 0..15
  t=1: 处理 A 的 col 16..31, B 的 row 16..31
  t=2: 处理 A 的 col 32..47, B 的 row 32..47
  t=3: 处理 A 的 col 48..49, B 的 row 48..49 (最后 2 个, 超出矩阵的部分填 0)
```

#### ④ 协作加载 — 每个线程搬一个元素

这是整个算法最关键的一步。Block 内 256 个线程, 各搬 A 的一个元素和 B 的一个元素:

```cuda
int aCol = t * TILE_SIZE + threadIdx.x;   // 当前 tile 在 A 中的列号
int bRow = t * TILE_SIZE + threadIdx.y;   // 当前 tile 在 B 中的行号

As[threadIdx.y][threadIdx.x] = (row < M && aCol < K)
    ? A[row * K + aCol] : 0.0f;
Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N)
    ? B[bRow * N + col] : 0.0f;
```

**为什么每个线程只搬一个元素?**

```
As 是 16×16 = 256 个元素。Block 有 256 个线程。
每个线程负责 As 的一个位置 → 256 个线程填满 256 个位置 → 刚好!

哪个线程负责 As 的哪个位置?
  线程 (threadIdx.y, threadIdx.x) → 负责 As[threadIdx.y][threadIdx.x]

Thread(0,0)  → As[0][0]  + Bs[0][0]
Thread(0,1)  → As[0][1]  + Bs[0][1]
...
Thread(0,15) → As[0][15] + Bs[0][15]
Thread(1,0)  → As[1][0]  + Bs[1][0]
...
Thread(15,15) → As[15][15] + Bs[15][15]
```

**As 中存的是什么?**

```
以 t=0 (第一个 tile) 为例, Block(0,0) 负责 C[0:16][0:16]:

  As[threadIdx.y][threadIdx.x] = A[row][aCol]
                               = A[threadIdx.y][t*16 + threadIdx.x]
                               = A[threadIdx.y][threadIdx.x]

  所以 As = A 的前 16 行 × 前 16 列:
         col=0 1 2 ... 15
  row=0: [a00 a01 a02 ... a0_15]
  row=1: [a10 a11 a12 ... a1_15]
  ...
  row=15:[a150 ...          a15_15]

  Bs[threadIdx.y][threadIdx.x] = B[bRow][col]
                               = B[t*16 + threadIdx.y][threadIdx.x]
                               = B[threadIdx.y][threadIdx.x]

  所以 Bs = B 的前 16 行 × 前 16 列:
         col=0 1 2 ... 15
  row=0: [b00 b01 b02 ... b0_15]
  ...
  row=15:[b150 ...          b15_15]
```

**边界检查是什么意思?**

```
当矩阵大小不是 TILE_SIZE 的倍数时, 最后一个 tile 会超出矩阵边界。

例如 K=50, t=3 时 (最后一个 tile):
  aCol = 3*16 + threadIdx.x = 48 + threadIdx.x

  threadIdx.x=0..1:  aCol=48..49 < K=50 → 正常读
  threadIdx.x=2..15: aCol=50..63 ≥ K=50 → 填 0

  为什么填 0? → 0 乘以任何数 = 0 → 不影响最终的 sum
```

#### ⑤ 第一个 __syncthreads() — 等大家都搬完

```cuda
__syncthreads();
```

```
256 个线程搬运速度不一样 (有些 Warp 可能被其他 Block 的 Warp 抢占)。
__syncthreads() 确保: 所有 256 个线程都执行到这里之后, 才继续往下。

如果不等: Thread 0 已经搬完开始从 As 读数据,
         但 Thread 255 还没写完 As[15][15]
         → Thread 0 读到的是旧数据 → 结果错!

硬件上: 编译为 BAR.SYNC 指令, SM 的 Barrier Unit 计数:
  每到达一个 Warp (32线程) → arrived_count += 32
  所有 8 个 Warp 都到达 (arrived=256) → 全部释放继续执行
```

#### ⑥ 从 Shared Memory 计算

```cuda
for (int i = 0; i < TILE_SIZE; i++) {
    sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
}
```

```
这就是一个普通的 K 维累加——但数据从 Shared Memory 读而不是 HBM!

Thread(2, 3) 为例 (row=2, col=3, 即 C[2][3]):
  i=0:  sum += As[2][0] * Bs[0][3]
        = A[2][0] * B[0][3]
  i=1:  sum += As[2][1] * Bs[1][3]
        = A[2][1] * B[1][3]
  ...
  i=15: sum += As[2][15] * Bs[15][3]
        = A[2][15] * B[15][3]

  总共 16 次乘加, 全部从 Shared Memory 读数据 (~5 cyc/次)
  vs 朴素版从 HBM 读 (~500 cyc/次)

下一个 tile (t=1): 处理 A 的 col 16..31, B 的 row 16..31
  再次累加到 sum → 最终覆盖整个 K 维度

注意: As[threadIdx.y][i] → 同一 Warp 的线程读 As 的 [threadIdx.y][i]
      不同的 threadIdx.y → 不同的行 → 不同的 Bank → 无 Bank Conflict ✓
      Bs[i][threadIdx.x] → 同一 Warp 读 Bs 的 [i][连续列]
      → 不同的 threadIdx.x → 不同的列 → 不同的 Bank → 无 Bank Conflict ✓
```

#### ⑦ 第二个 __syncthreads() — 等大家都算完

```cuda
__syncthreads();
```

```
为什么还需要第二个?

如果不等: Thread 0 已经算完, 进入下一轮 t=1 开始往 As 写新数据
          Thread 255 还在做 t=0 的计算, 读 As 的旧数据
          → Thread 0 覆盖了 Thread 255 还在读的数据 → 结果错!

两个 __syncthreads() 各司其职:
  第 1 个: 保证"写完 Shared Memory 之后才能读" (写→读 barrier)
  第 2 个: 保证"读完 Shared Memory 之后才能写下一轮"  (读→写 barrier)
```

#### ⑧ 写回结果

```cuda
if (row < M && col < N) {
    C[row * N + col] = sum;
}
```

和朴素版一样。只有负责合法 row/col 的线程才写。


## 4. 用一组具体数字走一遍完整流程

这是理解算法最好的方式。拿一个小例子手动走一遍。

```
条件: M=K=N=8, TILE_SIZE=4

C (8×8) 被分成 4 个 4×4 的 tile (gridDim=(2,2)):
  Block(0,0): C[0:4][0:4]     Block(1,0): C[0:4][4:8]
  Block(0,1): C[4:8][0:4]     Block(1,1): C[4:8][4:8]

以 Block(0,0) 为例, 16 个线程 (4×4):
  Thread(0,0)→C[0][0], Thread(0,1)→C[0][1], ... Thread(3,3)→C[3][3]

K=8, TILE_SIZE=4 → 需要 2 轮 (t=0, t=1)

轮次 t=0 (处理 A[:,0:4] 和 B[0:4,:]):
  ┌────────── 步骤 1: 加载 ──────────┐
  │ 16 个线程各从 HBM 加载 1 个元素:   │
  │                                    │
  │ Thread(0,0): As[0][0]=A[0][0]     │
  │ Thread(0,1): As[0][1]=A[0][1]     │
  │ Thread(0,2): As[0][2]=A[0][2]     │
  │ Thread(0,3): As[0][3]=A[0][3]     │
  │ Thread(1,0): As[1][0]=A[1][0]     │
  │ ...                                │
  │ 同理加载 Bs 的 16 个元素           │
  │                                    │
  │ As 变成了 A[0:4][0:4] 的副本       │
  │ Bs 变成了 B[0:4][0:4] 的副本       │
  └────────────────────────────────────┘

  __syncthreads()  ← 等 16 个线程都搬完

  ┌────── 步骤 2: 计算 (从 Shared Memory) ────┐
  │ 每个线程做 TILE_SIZE=4 次乘加:               │
  │                                              │
  │ Thread(0,0): sum += As[0][0]*Bs[0][0]       │
  │                   + As[0][1]*Bs[1][0]       │
  │                   + As[0][2]*Bs[2][0]       │
  │                   + As[0][3]*Bs[3][0]       │
  │             = A[0][0]*B[0][0]               │
  │             + A[0][1]*B[1][0]               │
  │             + A[0][2]*B[2][0]               │
  │             + A[0][3]*B[3][0]               │
  │             ← 这算完了 K 维度前 4 个的累加    │
  └──────────────────────────────────────────────┘

  __syncthreads()  ← 等 16 个线程都算完当前 tile

轮次 t=1 (处理 A[:,4:8] 和 B[4:8,:]):
  同样步骤, 加载 A[0:4][4:8] 到 As, B[4:8][0:4] 到 Bs

  Thread(0,0): sum += As[0][0]*Bs[0][0]  + ...
             = ... + A[0][4]*B[4][0]
                   + A[0][5]*B[5][0]
                   + A[0][6]*B[6][0]
                   + A[0][7]*B[7][0]

  现在 sum = A[0][0]*B[0][0] + ... + A[0][7]*B[7][0]
         = 完整的 C[0][0]! ✓

最后写回: C[0][0] = sum
```

### 4.1 同一个 tile 内的数据复用

```
以 t=0 的加载为例, 看数据如何被复用:

As[0][0] = A[0][0] → 被谁使用?
  Thread(0,0): 读 As[0][0] 一次 (乘以 Bs[0][0])
  Thread(0,1): 读 As[0][0] 一次 (乘以 Bs[0][1])   ← 复用!
  Thread(0,2): 读 As[0][0] 一次 (乘以 Bs[0][2])   ← 复用!
  Thread(0,3): 读 As[0][0] 一次 (乘以 Bs[0][3])   ← 复用!

  As[0][0] 被同一行的 4 个线程各读了一次 → 4 次复用!
  但只从 HBM 加载了 1 次 → HBM 读取量减少 4×!

推广到 TILE_SIZE=16: 每个元素被同一行的 16 个线程复用 → HBM 读取量减少 16×
```

这就是 Tiling 加速的物理本质: **Shared Memory 让同一个 Block 内的线程共享数据, 避免了每个线程各自从 HBM 重复读。**


## 5. Shared Memory 在硬件上是什么

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


## 6. __syncthreads() 在硬件上做了什么

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


## 7. Tiled 版一次 Tile 循环的数据流

```
时间 ────────────────────────────────────────────────────────────►

Warp 0: [LDG As][LDG Bs]--等HBM--→[STS smem][BAR][LDS×16][FFMA×16][BAR]
Warp 1: [LDG As][LDG Bs]--等HBM--→[STS smem][BAR][LDS×16][FFMA×16][BAR]
...                                     ↑               ↑
Warp 7: [LDG As][LDG Bs]--等HBM--→[STS smem][BAR][LDS×16][FFMA×16][BAR]

        │←── ~500 cyc (等HBM) ──→│         │←── ~100 cyc ──→│
        HBM 延迟被多 Warp 隐藏              从 Shared Memory 读, 极快

注意:
  LDG (全局加载) 延迟 ~500 cycles, 但 8 个 Warp 交替执行 → 延迟被隐藏
  LDS (Shared 加载) 延迟 ~5 cycles
  BAR 是所有 Warp 的等待点 → 最慢的 Warp 决定整体速度
```


## 2D Block 的线程到 Warp 的映射

```
blockDim(16, 16) → 256 线程, 排列成 16×16 的 2D 网格。

硬件只认识 Warp (32 个连续线程), 不认识 "2D 线程"。
256 线程 = 8 Warp。映射规则: 线性ID = threadIdx.y * blockDim.x + threadIdx.x

  threadIdx.y=0:  (0,0) (1,0) (2,0) ... (15,0) → 线性ID 0-15
  threadIdx.y=1:  (0,1) (1,1) (2,1) ... (15,1) → 线性ID 16-31
  ...

  Warp 0: 线性ID  0-31  = (y=0 全部 16 个) + (y=1 全部 16 个)
  Warp 1: 线性ID 32-63  = (y=2 全部 16 个) + (y=3 全部 16 个)
  ...
  Warp 7: 线性ID 224-255 = (y=14 全部 16 个) + (y=15 全部 16 个)

这意味着同一 Warp 的 32 个线程, threadIdx.x 从 0-15 循环两次 (y 不同)。

读 B[k][col] 时, col = blockIdx.x*TILE_SIZE + threadIdx.x
→ 同一 Warp 的 threadIdx.x 连续 → B 的列地址连续 → 合并访问! ✓

读 A[row][k] 时, row = blockIdx.y*TILE_SIZE + threadIdx.y
→ 同一 Warp 的 threadIdx.y 只有两个值 (y=0 和 y=1)
→ A 的地址间隔 = 1×K 字节 (两个行, 不完全连续)
→ 2 个 cache line → 2 次事务 → 效率还可以接受 (对于 TILE_SIZE=16)

如果 x 对应行、y 对应列:
→ 读 B 时 col 的 threadIdx 不连续 → stride=N → 完全不合并 → 性能暴跌!
→ 这就是为什么 matmul 中 x 对应列、y 对应行
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


## 常见错误

- **忘了 `__syncthreads()`** → 症状: 结果不稳定/每次跑不同。读 Shared Memory 时可能读到其他线程还没写完的数据
- **只写一个 `__syncthreads()`** → 症状: 结果部分正确部分错。需要两个! 循环内必须在加载后和计算后各同步一次
- **`__syncthreads()` 放循环外面** → 症状: 每个 tile 间没同步。搬运和计算之间没有屏障保护
- **As/Bs 索引写错** → `As[threadIdx.x][threadIdx.y]` 而不是 `As[threadIdx.y][threadIdx.x]` → 合并访问变跨步访问
- **TILE_SIZE 不是 16 但 Shared Memory 声明是 `[16][16]`** → 越界写 SMEM → compute-sanitizer 会报错
- **threadIdx 和矩阵索引搞反** → `threadIdx.y` 对应矩阵的行, `threadIdx.x` 对应矩阵的列
- **忘了边界检查** → 矩阵不是 TILE_SIZE 的倍数时, 最后一个 tile 超出范围 → 越界访问
- **TILE_SIZE × sizeof(float) > Shared Memory 上限** → 编译错误。Ampere 默认 SMEM 上限 100KB, TILE_SIZE ≤ 128 是安全的 (128×128×4B×2 = 128KB, 需要配置 SMEM 上限)
- **不加边界检查直接用全局数组索引** → `A[row * K + aCol]` 中 aCol 可能 >= K → GPU 上不会崩溃但读到垃圾值
