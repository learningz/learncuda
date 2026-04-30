# Bank Conflict：32 个 Bank 在硬件上的真实行为

本文档配合 `bank_conflict.cu` 阅读。


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
