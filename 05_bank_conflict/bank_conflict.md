# Bank Conflict：32 个 Bank 在硬件上的真实行为

本文档配合 `bank_conflict.cu` 阅读。

**难度**: ⭐⭐ 进阶
**前置知识**: Shared Memory 基本概念（[tutorial Part 2](../tutorial.md#part-2-让它更快--shared-memory-和矩阵乘法)）
**读完你能做什么**: 理解 Shared Memory 的 32 个 Bank 硬件结构，能诊断和消除 Bank Conflict


## 什么是 Bank Conflict

在 Part 2 你学会了用 Shared Memory 加速数据访问。
但 Shared Memory 有一个隐藏的性能陷阱：Bank Conflict。

### Shared Memory 不是"一整块内存"

你可能以为 Shared Memory 就像一个大数组，读哪里都一样快。
实际上，它在物理上被切成了 **32 个独立的小存储体（Bank）**，
每个 Bank 有自己独立的读写端口。

为什么切成 32 个？因为一个 Warp 正好 32 个线程。
理想情况下，32 个线程各访问一个不同的 Bank → 全部同时完成 → 1 cycle。

```
地址如何分配给 Bank:

  地址按 4 字节 (一个 float) 轮流分配:
    地址 0-3    → Bank 0
    地址 4-7    → Bank 1
    地址 8-11   → Bank 2
    ...
    地址 124-127 → Bank 31
    地址 128-131 → Bank 0    ← 又回到 Bank 0! (32 个一轮回)
    地址 132-135 → Bank 1
    ...

  公式: Bank编号 = (字节地址 / 4) % 32

  对 float 数组 smem[] 来说:
    smem[0] → Bank 0
    smem[1] → Bank 1
    smem[31] → Bank 31
    smem[32] → Bank 0    ← 回到 Bank 0
    smem[33] → Bank 1
```

### 冲突发生的时刻

当一个 Warp 的多个线程要访问**同一个 Bank 里的不同地址**时，
这些访问必须排队串行执行：

```
无冲突: 32 个线程各访问一个不同的 Bank
  Thread 0 → Bank 0, Thread 1 → Bank 1, ..., Thread 31 → Bank 31
  → 32 个端口同时读 → 1 cycle 搞定!

2-way 冲突: 每个 Bank 被 2 个线程访问
  Thread 0 → Bank 0, Thread 16 → Bank 0 (都访问 Bank 0!)
  Thread 1 → Bank 2, Thread 17 → Bank 2 (都访问 Bank 2!)
  → 每个 Bank 排队 2 次 → 2 cycles → 慢了 2 倍

32-way 冲突: 所有线程都打在同一个 Bank
  Thread 0,1,2,...,31 全部 → Bank 0
  → 排队 32 次 → 32 cycles → 慢了 32 倍!
```

### 什么样的代码会导致冲突

来看一个具体的例子——矩阵转置：

```
你有一个 32×32 的矩阵存在 Shared Memory:
  __shared__ float tile[32][32];

写入时: tile[threadIdx.y][threadIdx.x] = input[row][col];
  同一 Warp 的 32 个线程 threadIdx.x = 0,1,...,31
  → 访问 tile[y][0], tile[y][1], ..., tile[y][31]
  → Bank 0, Bank 1, ..., Bank 31  → 无冲突! ✓

读出时 (转置): output[col][row] = tile[threadIdx.x][threadIdx.y];
  同一 Warp 的 32 个线程 threadIdx.x = 0,1,...,31
  → 访问 tile[0][y], tile[1][y], ..., tile[31][y]
  
  让我们逐线程算 Bank 编号:
    规则: Bank = (字节地址 / 4) % 32
    对于 tile[row][col], 字节地址 = (row × 32 + col) × 4
    
    Thread 0  (threadIdx.x=0):  读 tile[0][y]  → 地址 (0×32 + y)×4  → Bank (0×32 + y) % 32 = y
    Thread 1  (threadIdx.x=1):  读 tile[1][y]  → 地址 (1×32 + y)×4  → Bank (1×32 + y) % 32 = y
    Thread 2  (threadIdx.x=2):  读 tile[2][y]  → 地址 (2×32 + y)×4  → Bank (2×32 + y) % 32 = y
    ...
    Thread 31 (threadIdx.x=31): 读 tile[31][y] → 地址 (31×32 + y)×4 → Bank (31×32 + y) % 32 = y
    
  全部 % 32 = y → 全部落在 Bank y!
  关键: stride=32, Bank 数=32, 32×row 对 32 取模 = 0 → 多行的同一列 offset 对 Bank 的贡献为 0
  → 这就是为什么 stride=32 是 32-way conflict!
```

**问题的根源**：列方向的步长 = 32，而 Bank 数也是 32，
所以 stride=32 意味着每一步走完一整圈回到同一个 Bank。


### Padding 为什么能消除冲突

解法极其简单——声明时多加 1 列：

```
原来: __shared__ float tile[32][32];    // 每行 32 个 float
现在: __shared__ float tile[32][33];    // 每行 33 个 float (多了 1 个!)
```

看看这一列如何打破冲突：

```
原来 (32 列, stride=32):
  tile[0][y] → 位置 y        → Bank y%32
  tile[1][y] → 位置 32+y     → Bank (32+y)%32 = y  ← 同一 Bank!
  tile[2][y] → 位置 64+y     → Bank (64+y)%32 = y  ← 同一 Bank!
  → 列方向全撞在同一 Bank!

加 padding (33 列, stride=33):
  tile[0][y] → 位置 y        → Bank y%32
  tile[1][y] → 位置 33+y     → Bank (33+y)%32 = (y+1)%32  ← 不同 Bank!
  tile[2][y] → 位置 66+y     → Bank (66+y)%32 = (y+2)%32  ← 又不同!
  → 每行偏移 1 个 Bank → 32 行 × 每行偏 1 = 32 个不同 Bank → 无冲突!

代价: 每行浪费 4 字节 (1 个 float) → 32 行 × 4B = 128B
     vs 32× 性能提升 → 完全值得!
```

这就是 padding 的原理：**通过让每行多一个元素，
让列方向的步长从 32 (Bank 数的整倍数) 变成 33 (互质)，
从而让相邻行的同一列落在不同的 Bank 上。**

下面用三种 stride 模式在硬件上的真实行为来验证。


## Shared Memory 的物理结构

```
Shared Memory 是一块 SRAM, 被切成 32 个独立的 Bank:

┌─────┬─────┬─────┬─────┬─────┬───────────┬─────┐
│Bank0│Bank1│Bank2│Bank3│Bank4│    ...     │Bk31│
│addr │addr │addr │addr │addr │           │addr │
│ 0   │ 4   │ 8   │ 12  │ 16  │           │ 124 │
│128  │132  │136  │140  │144  │           │ 252 │
│256  │260  │264  │268  │272  │           │ 380 │
│...  │...  │...  │...  │...  │           │ ... │
└─────┴─────┴─────┴─────┴─────┴───────────┴─────┘

地址到 Bank 的映射:
  Bank 编号 = (字节地址 / 4) % 32

  地址 0    → Bank 0   (0/4 % 32 = 0)
  地址 4    → Bank 1   (4/4 % 32 = 1)
  地址 124  → Bank 31  (124/4 % 32 = 31)
  地址 128  → Bank 0   (128/4 % 32 = 0)  ← 又回到 Bank 0!

每个 Bank 有独立的读/写端口:
  → 32 个不同 Bank 可以同时被访问 → 1 cycle 完成 32 次读
  → 同一 Bank 被多次访问 → 必须串行 → N 次访问需要 N cycles
```


## 三种 stride 在硬件上的行为

### stride=1 (无冲突)

```
smem[threadIdx.x * 1]  →  每线程访问连续地址

Thread 0  → smem[0]   → addr 0   → Bank 0
Thread 1  → smem[1]   → addr 4   → Bank 1
Thread 2  → smem[2]   → addr 8   → Bank 2
...
Thread 31 → smem[31]  → addr 124 → Bank 31

32 个线程 → 32 个不同 Bank → 全并行 → 1 cycle!

硬件行为:
  ┌────────────────────────────────────────┐
  │ Cycle 0: 32 个 Bank 同时读出数据      │
  │          → 32 个值同时到达 32 个线程   │
  │          总时间: 1 × 5 cycles = 5 cyc │
  └────────────────────────────────────────┘
```

### stride=2 (2-way 冲突)

```
smem[threadIdx.x * 2]  →  每线程访问偶数地址

Thread 0  → smem[0]   → addr 0   → Bank 0
Thread 1  → smem[2]   → addr 8   → Bank 2
Thread 2  → smem[4]   → addr 16  → Bank 4
...
Thread 15 → smem[30]  → addr 120 → Bank 30
Thread 16 → smem[32]  → addr 128 → Bank 0   ← 和 Thread 0 同一个 Bank!
Thread 17 → smem[34]  → addr 136 → Bank 2   ← 和 Thread 1 同一个 Bank!
...

每个 Bank 被 2 个线程访问 → 2-way conflict → 分两轮:

硬件行为:
  ┌────────────────────────────────────────┐
  │ Cycle 0: Thread 0,1,2,...15 访问      │
  │          (16 个不同 Bank, 并行)        │
  │ Cycle 1: Thread 16,17,...31 访问       │
  │          (同样的 16 个 Bank, 串行!)    │
  │          总时间: 2 × 5 cycles = 10 cyc│
  └────────────────────────────────────────┘

  → 延迟是无冲突的 2 倍!
```

### stride=32 (32-way 冲突, 最坏)

```
smem[threadIdx.x * 32]  →  每线程间隔 128 字节

Thread 0  → smem[0]    → addr 0    → Bank 0
Thread 1  → smem[32]   → addr 128  → Bank 0  ← 同一个 Bank!
Thread 2  → smem[64]   → addr 256  → Bank 0  ← 同一个 Bank!
...
Thread 31 → smem[992]  → addr 3968 → Bank 0  ← 全都是 Bank 0!

32 个线程全部打在 Bank 0 → 完全串行!

硬件行为:
  ┌────────────────────────────────────────┐
  │ Cycle 0:  Thread 0 访问 Bank 0        │
  │ Cycle 1:  Thread 1 访问 Bank 0        │
  │ Cycle 2:  Thread 2 访问 Bank 0        │
  │ ...                                    │
  │ Cycle 31: Thread 31 访问 Bank 0       │
  │          总时间: 32 × 5 = 160 cycles! │
  └────────────────────────────────────────┘

  → 延迟是无冲突的 32 倍! 完全丧失了并行性。
```


## Broadcast 例外 — 同一地址多次访问不冲突

```
前面说"多个线程访问同一 Bank 不同地址 = 冲突"。
但有一个重要的例外: 如果多个线程访问同一 Bank 的同一个地址,
Shared Memory 硬件会自动广播!

广播 (Broadcast):
  Thread 0, Thread 1, ..., Thread 7 全部读 smem[0]
  → 地址 0 → Bank 0
  → 所有 8 个线程要的是同一个地址 (完全相同!)
  → 硬件检测到 → 通过广播机制, 1 cycle 内把数据发给所有线程
  → 不算冲突!

  广播 vs 冲突:
    线程要同一 Bank 同一地址 → 广播 → 1 cycle ✓
    线程要同一 Bank 不同地址 → 冲突 → N cycles ✗
    
  硬件怎么区分?
    Smem 的 Bank 内部有一个地址比较器:
      如果 N 个请求的地址完全匹配 → 合并为 1 次读 + 广播
      如果地址不同 → 必须串行, 1 次读 1 个

  这类似 CPU 缓存的"多读不冲突"(Multiple Readers OK):
    只读 + 地址相同 = 无冲突
    读写混合 或 地址不同 = 冲突
```

## 一般化的冲突度公式

```
上面只看了 stride=1,2,32 三种特殊情况。这里给出通用公式。

给定一个 Warp 的 32 个访问地址: addr[0..31]

定义:
  Bank[i] = (addr[i] / 4) % 32   // 线程 i 的地址归属哪个 Bank

冲突度 k = max_{b in [0,31]} (访问 Bank b 的不同地址数)

  如果 k = 1: 无冲突 (每 Bank 最多 1 个地址) → 1 cycle
  如果 k = 2: 2-way 冲突 → 2 cycles
  如果 k = N: N-way 冲突 → N cycles (最坏 ~32×)

对于 stride=s 的均匀访问 addr[i] = base + i * s * 4:

  Bank[i] = (base/4 + i*s) % 32

  Bank[i + 1] = (base/4 + (i+1)*s) % 32
             = (Bank[i] + s) % 32
  → 每步在 Bank 空间中位移 s

  每个 Bank 被访问的次数 = 32 / gcd(32, s)  (如果 s 能被分发均匀)
  
  具体:
    s=1:  gcd(32,1)=1  → 每 Bank 被 1 次 → 无冲突 ✓
    s=2:  gcd(32,2)=2  → 每 Bank 被 2 次 → 2-way 冲突
    s=4:  gcd(32,4)=4  → 每 Bank 被 4 次 → 4-way 冲突
    s=8:  gcd(32,8)=8  → 每 Bank 被 8 次 → 8-way 冲突
    s=16: gcd(32,16)=16→ 每 Bank 被 16 次 → 16-way 冲突
    s=32: gcd(32,32)=32→ 每 Bank 被 32 次 → 32-way 冲突 ← 全部撞在一个 Bank!
    s=33: gcd(32,33)=1 → 每 Bank 被 1 次 → 无冲突 (padding 的原理!)
    s=3:  gcd(32,3)=1  → 每 Bank 被 1 次 → 无冲突 (奇数 stride 通常安全)

关键洞察: stride 和 32 互质时无冲突。
  → 32 的因数 (1,2,4,8,16,32) 作为 stride 会导致不同程度的冲突
  → 非因数 stride (3,5,7,...33,35...) 无冲突
```

## 在 ncu 中看到 Bank Conflict

```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st \
./bank_conflict

输出示例:
  kernel bank_conflict_test<1>:  conflicts = 0       ← stride=1
  kernel bank_conflict_test<2>:  conflicts = 131072  ← stride=2
  kernel bank_conflict_test<32>: conflicts = 4063232 ← stride=32
  
conflicts 不是"冲突次数", 而是"额外的串行化 cycle 总数"。
数字越大 → 性能越差。
```


## 练习题

完成 `bank_conflict.cu` 的阅读后，可以在 [`exercises/`](./exercises/) 目录下做以下练习：

| 练习 | 任务 | 核心考点 |
|------|------|---------|
| [ex1_padding_level1.cu](./exercises/ex1_padding_level1.cu) | 给 stride=32 的 SMEM 加 padding | 只需改 2 处声明/索引 |
| [ex1_padding_level2.cu](./exercises/ex1_padding_level2.cu) | 同上 | 从零写 padded kernel |
| [ex2_transpose_smem_level1.cu](./exercises/ex2_transpose_smem_level1.cu) | SMEM 矩阵转置, 对比有/无 padding | 填两个 kernel（无 padding + 有 padding） |
| [ex2_transpose_smem_level2.cu](./exercises/ex2_transpose_smem_level2.cu) | 同上 | kernel + host + 计时全部自己写 |

编译方式（在 `exercises/` 目录下）：

```bash
nvcc -O2 --extended-lambda -o ex1_padding_level1 ex1_padding_level1.cu
./ex1_padding_level1
```
