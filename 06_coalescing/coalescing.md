# 合并访问：Cache Line 和内存事务在硬件上的行为

配合 `coalescing.cu` 阅读。

**难度**: ⭐⭐ 进阶
**前置知识**: GPU 内存层级基础（[tutorial Part 2](../tutorial.md#part-2-让它更快--shared-memory-和矩阵乘法)）
**读完你能做什么**: 理解 Cache Line 和内存事务的硬件行为，能写出合并访问的 kernel


## 什么是合并访问 (Coalesced Access)

### 显存不是一次给你一个数

你可能以为 `data[i]` 就是从显存读一个 float (4 字节)。
实际上，GPU 显存有**两层最小传输单位**：

- **Sector (32 字节)**: HBM 到 L2 Cache 的最小传输单位。
  即使你只要 4 字节，DRAM 芯片也会返回整个 32 字节 sector 给 L2。
- **Cache Line (128 字节)**: L2 Cache 到 L1 Cache/SM 的最小传输单位。
  1 个 cache line = 4 个连续的 sector（128 = 4 × 32）。

也就是说，哪怕你只要 1 个 float (4 字节)，
硬件实际上传输了一整个 cache line (128 字节) 到 L1 — 其余 124 字节白传了。

分层来看：

```
HBM (DRAM 芯片)
  │
  │ sector = 32B (最小 burst, DRAM 物理限制)
  ▼
L2 Cache (片上, 所有 SM 共享)
  │
  │ cache line = 128B = 4 sectors
  ▼
L1 Cache / SMEM (每个 SM 独立)
  │
  │ 实际请求的 4B 写入寄存器
  │ 剩余 124B 留在 L1 (可能被后续访问命中)
  ▼
Register
```

### 关键在于：一个 Warp 的 32 个线程一起发请求

GPU 不是一个线程一个线程地发内存请求，而是一个 Warp (32 线程) 一起发。
硬件会把这 32 个地址"打包"，看看它们一共触及了多少个 Cache Line：

```
情况 A — 合并访问:
  Thread 0 要 data[0]   → 地址 0
  Thread 1 要 data[1]   → 地址 4
  Thread 2 要 data[2]   → 地址 8
  ...
  Thread 31 要 data[31] → 地址 124

  32 个 float × 4B = 128B → 正好 1 个 Cache Line!
  → 硬件只发 1 次内存请求 → 传输 128B, 全部有用 → 效率 100%

情况 B — stride=32 访问:
  Thread 0 要 data[0]      → 地址 0      → Cache Line 0
  Thread 1 要 data[32]     → 地址 128    → Cache Line 1
  Thread 2 要 data[64]     → 地址 256    → Cache Line 2
  ...
  Thread 31 要 data[31×32] → 地址 3968   → Cache Line 31

  32 个 float 分散在 32 个不同的 Cache Line!
  → 硬件发 32 次内存请求!
  → 传输 32 × 128B = 4096B, 只用了 32 × 4B = 128B
  → 效率 128/4096 = 3%! 浪费 97% 的带宽!

情况 C — 随机访问:
  Thread 0 要 data[random_0]
  Thread 1 要 data[random_1]
  ...
  32 个随机地址，大概率落在 20-32 个不同的 Cache Line
  → 效率更低，通常 < 5%
```

### 为什么差距这么大

用快递来理解：

```
合并访问 = 你们整栋楼 32 户人家合买的东西恰好装在 1 个包裹里
  → 快递员来一趟就送完了

stride 访问 = 32 户人家的东西分装在 32 个包裹里，每个包裹只有 1 件
  → 快递员要跑 32 趟!
  → 而且每个包裹箱子一样大 (128B Cache Line)，浪费了 97% 的箱子空间!
```

**这不是 10% 的性能差异，而是 10-30 倍的差距。**

### 怎么写出合并的代码

规则很简单：**让同一 Warp 的相邻线程访问相邻的内存地址。**

```cuda
// ✓ 合并: 相邻线程访问相邻地址
output[idx] = input[idx];
// Thread 0 读 input[0], Thread 1 读 input[1], ... → 地址连续 → 合并!

// ✗ 不合并: 相邻线程访问间隔很大的地址
output[idx] = input[idx * stride];
// Thread 0 读 input[0], Thread 1 读 input[stride], ... → 间隔 stride → 不合并!

// 常见陷阱 — 行优先 vs 列优先:
// 矩阵 A[M][N], 如果让 threadIdx.x 对应行 (M 维度)
//   Thread 0 读 A[0][col], Thread 1 读 A[1][col]
//   → 地址间隔 N×4 字节 → 不合并!
// 正确做法: threadIdx.x 对应列 (N 维度)
//   Thread 0 读 A[row][0], Thread 1 读 A[row][1]
//   → 地址连续 → 合并!
```

下面用三种具体的访问模式在硬件上看到差异。


## 三种访问模式在硬件上的真实差异

```
一个 Warp (32 线程) 发起全局内存加载时, LD/ST Unit 做的事:

1. 收集 32 个地址
2. 映射到 128-byte 对齐的 Cache Line
3. 统计触及了多少个不同的 Cache Line
4. 每个 Cache Line 生成一个内存事务 (Memory Transaction)

连续访问 (stride=1):
  Thread 0 → addr+0, Thread 1 → addr+4, ..., Thread 31 → addr+124
  全部在 1 个 128B Cache Line 内 → 1 次事务
  
  ┌── Cache Line (128 bytes) ────────────────────────────────┐
  │ T0 T1 T2 T3 T4 T5 T6 T7 T8 ... T31                     │
  └──────────────────────────────────────────────────────────┘
  传输 128B, 使用 128B → 效率 100%

stride=32 访问:
  Thread 0 → addr+0, Thread 1 → addr+128, Thread 2 → addr+256, ...
  每线程在不同 Cache Line → 32 次事务!
  
  ┌── CL 0 ──┐ ┌── CL 1 ──┐ ┌── CL 2 ──┐     ┌── CL 31 ─┐
  │ T0 ..... │ │ T1 ..... │ │ T2 ..... │ ... │ T31 .... │
  └──────────┘ └──────────┘ └──────────┘     └──────────┘
  传输 32×128B = 4096B, 使用 32×4B = 128B → 效率 3.125%!

随机访问:
  每线程地址完全不可预测 → 最坏 32 个不同 CL → 32 次事务
  且 L2 Cache 命中率低 (地址散布整个显存) → 每次都走 HBM
```


## 有效带宽的物理含义

```
程序输出的"有效带宽"怎么算:

  有效带宽 = 实际传输的有用数据量 / 耗时

  连续访问 (stride=1):
    32 线程 × 4B = 128B 有用数据
    触发了 1 次 128B cache line 传输 = 128B 的总传输
    效率 = 128/128 = 100%
    实测: N=4M, 耗时 0.02ms → 有效带宽 = 4M × 4B / 0.02ms = 800 GB/s
  
  stride=32 访问:
    32 线程 × 4B = 128B 有用数据
    触发了 32 次 128B cache line 传输 = 4096B 的总传输
    效率 = 128/4096 = 3.125%
    实测: N=4M, 耗时 0.6ms → 有效带宽 = 16MB / 0.6ms = 26.7 GB/s

  差距 = 800/26.7 ≈ 30×, 全部来自浪费的 cache line 传输。
  GPU 的 HBM 物理带宽没变, 但 97% 的传输被浪费在不需要的数据上。
  → 非合并访问让你的 2TB/s 带宽退化到 ~27GB/s 可用带宽。
```


## LD/ST Unit 的合并逻辑 — 硬件内部到底怎么工作的

```
每个 SM 有若干 LD/ST Unit (Load/Store Unit)，负责处理内存请求。

当一个 Warp 执行 LDG (Global Load) 指令时:

Step 1: Address Generation (地址生成)
  32 个线程各自计算自己的地址:
    addr[i] = base_ptr + idx[i] * sizeof(float)
  
  例如连续访问:
    addr[0] = 0x1000, addr[1] = 0x1004, ..., addr[31] = 0x107C

Step 2: Coalescing (合并)
  LD/ST Unit 的合并逻辑将 32 个地址按 128B 对齐的 Cache Line 分组:
  
  连续访问:
    0x1000 ~ 0x107C → 全在 CL [0x1000, 0x107F] 内
    → 合并为 1 个请求 (1 × 128B sector group = 4 × 32B sectors)
  
  stride=2 访问:
    0x1000, 0x1008, 0x1010, ..., 0x10F8
    → 跨 2 个 CL: [0x1000, 0x107F] 和 [0x1080, 0x10FF]
    → 合并为 2 个请求 (2 × 4 = 8 sectors)
  
  stride=32 访问:
    0x1000, 0x1080, 0x1100, ..., 每线程跨不同 CL
    → 32 个独立请求 (32 × 4 = 128 sectors)

Step 3: MSHR 队列
  合并后的请求进入 MSHR (Miss Status Holding Register) 队列。
  每个请求先查 L1 Cache:
    命中 → ~28 cycles 返回 (L1 延迟)
    未命中 → 转发到 L2 → ~200 cycles
    L2 也未命中 → 走 HBM → ~400-500 cycles
  
  MSHR 的容量有限 (通常 ~48-64 个 pending request per SM)。
  如果非合并访问产生大量请求，MSHR 会被填满 → Stall MIO Throttle。

Step 4: 数据返回
  数据从 Cache/HBM 返回后，分发给各线程的目标寄存器。
  只有该线程实际请求的 4B 被写入寄存器，Cache Line 的其余部分留在 L1 Cache 中。
```


## 常见的非合并访问模式及修复方法

```
模式 1: AoS (Array of Structures) — 最常见的陷阱

  struct Particle { float x, y, z, w; };  // 16 bytes/粒子
  Particle *particles = ...;
  
  // 只读 x 坐标:
  float x = particles[tid].x;
  
  问题: 每个 Particle 16B, 相邻线程的 .x 相隔 16B (而不是 4B)
    Thread 0 → addr+0   (Particle[0].x)
    Thread 1 → addr+16  (Particle[1].x)  ← 间隔 16B!
    → 32 线程跨越 512B = 4 条 CL → 效率 25%
    → 而且 y, z, w 被传输了但没使用 → 75% 浪费

  内存布局 (AoS — Array of Structures):
  
  地址 →   +0   +4   +8  +12  +16  +20  +24  +28  +32  +36  +40  +44
          ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
          │ x0 │ y0 │ z0 │ w0 │ x1 │ y1 │ z1 │ w1 │ x2 │ y2 │ z2 │ w2 │...
          └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
           ← 线程 0 要的 →           ← 线程 1 要的 →   ← 线程 2 要的 →
           
  只读 x 时: 真正需要的 float 分布在 addr+0, +16, +32, +48,... 
  → 每 16 字节才有一个有用的 4 字节 → Cache Line 中 75% 的数据浪费!
  → 而且 32 线程 × 16B 间隔 = 512B = 4 个 cache line → 4 次事务 → 效率 25%

  修复: SoA (Structure of Arrays)

  内存布局 (SoA — Structure of Arrays):

  地址 →   +0   +4   +8  +12  +16  +20  +24  +28
          ┌────┬────┬────┬────┬────┬────┬────┬────┐
          │ x0 │ x1 │ x2 │ x3 │ x4 │ x5 │ x6 │ x7 │...  (所有 x 连续存储)
          └────┴────┴────┴────┴────┴────┴────┴────┘
           ← 线程 0-7 要的, 全部连续! →

          ┌────┬────┬────┬────┬────┬────┬────┬────┐
          │ y0 │ y1 │ y2 │ y3 │ y4 │ y5 │ y6 │ y7 │...  (所有 y 连续存储)
          └────┴────┴────┴────┴────┴────┴────┴────┘

          ┌────┬────┬────┬────┬────┬────┬────┬────┐
          │ z0 │ z1 │ z2 │ z3 │ z4 │ z5 │ z6 │ z7 │...  (所有 z 连续存储)
          └────┴────┴────┴────┴────┴────┴────┴────┘

  float *x_array = ..., *y_array = ..., *z_array = ..., *w_array = ...;
  float x = x_array[tid];  // 相邻线程连续 → 完美合并!
  // 32 线程需要 128B → 刚好 1 个 cache line → 1 次事务 → 效率 100%

模式 2: 矩阵的列访问

  float *matrix;  // row-major, M × N
  float val = matrix[tid * N + col];  // 读第 col 列
  
  问题: 相邻线程的 tid 差 1 → 地址差 N*4B → 完全不合并!
  修复: 转置矩阵, 或用 Shared Memory 做中间缓存

模式 3: 间接索引 (Gather)

  float val = data[index[tid]];  // index 是随机排列
  
  这是最难优化的模式。可能的缓解策略:
    - 对 index 排序, 让相近的索引被相邻线程处理
    - 先把需要的数据收集到 Shared Memory
    - 使用 L2 persistence (CUDA 11.0+)

模式 4: Reduction 的 stride 错误

  // 错误: stride 从大到小 → 前半部分 Warp 活跃, 后半不活跃
  //        但地址不连续!
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s) sdata[tid] += sdata[tid + s];
      __syncthreads();
  }
  // 这里的 Shared Memory 访问模式没问题 (连续 Bank)
  // 但如果对全局内存这样做就会有合并问题
```


## 写合并 (Write Coalescing) — 写入也有同样的规则

```
前面说了读取需要合并, 写入同理!

STG (Store Global) 指令:
  STG.E.32 [Raddr], Rval  → 写 1 个 float (4B)
  STG.E.128 [Raddr], Rval  → 写 4 个 float (16B), 向量化写入

Warp 的 32 线程一起执行 STG 时:
  LD/ST Unit 同样收集 32 个地址 → 合并到同一 Cache Line → 1 次写事务

写入的关键硬件路径:

  情况 A — 合并写入:
    Thread 0 → addr+0, Thread 1 → addr+4, ..., Thread 31 → addr+124
    → 全在同一 128B Cache Line → 1 次写事务 → 效率 100%

  情况 B — strided 写入:
    每线程写不同 Cache Line → 32 次写事务 → 效率 3%

写入和读取的关键区别:
  1. L2 Write-Combine: 短时间内同一 Cache Line 的多次写入
     在 L2 中合并成一次写回 HBM (通常是 Evict 时)
     → 即使看起来是分散的小写, L2 也能部分缓冲

  2. Sector Write Mask: 非合并写入时, 每次写事务只写一个 sector
     L2 中的 Cache Line 是部分脏的 (Partial Write)
     → 后续 Evict 时需要 Read-Modify-Write (读整行 → 改 → 写整行)
     → 增加了 HBM 带宽浪费!

  3. Write-Through vs Write-Back:
     GPU L1 对全局内存一般是 Write-Through (直接写到 L2)
     L2 是 Write-Back (标记脏, 延迟写回 HBM)
     → 合并写入让 L2 更高效: 完整 Cache Line 脏 → 写回时一次性传输

写入合并的常见问题:
  问题 1: 分散写入
    for (int i = tid; i < n; i += stride) output[i] = val;
    → stride 大时, 每次写到不同 Cache Line → 不合并

  问题 2: Atomic scatter
    atomicAdd(&output[bin[tid]], val);
    → 写入地址由 index 决定 → 随机 → 完全不合并

  问题 3: Write Mask 不完整
    只有部分线程活跃 (Warp Divergence) → 写入的数据 < 128B
    但 Cache Line 仍然被标记为脏 → 同样需要 Read-Modify-Write

ncu 查看写合并:
  ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum \
  ./your_program
  → 写入的 sector 数 / 实际有用的字节数 → 写合并效率
```


## float4 向量化加载的原理

```
float4 v = reinterpret_cast<const float4*>(ptr)[tid];

编译后的 SASS:
  LDG.E.128 R4, [R0]    // 一条指令加载 128 bits = 4 个 float

vs 四次标量加载:
  LDG.E.32 R4, [R0]     // 加载 1 个 float
  LDG.E.32 R5, [R0+4]
  LDG.E.32 R6, [R0+8]
  LDG.E.32 R7, [R0+12]

两者传输的数据量完全相同 (都是 16 bytes/线程, 512 bytes/Warp)。
但向量化版本的指令数是 1/4 → LD/ST pipeline 发射的指令少了 4 倍。

为什么指令数少能提速?
  LD/ST Unit 每周期只能发射有限条指令 (通常 1 条/cycle)。
  标量版: 4 条 LDG.32 需要 4 个 cycle 来发射
  向量版: 1 条 LDG.128 只需 1 个 cycle 来发射
  → 节省了 3 个 cycle 的指令发射时间
  → 对于 Memory Bound kernel, 这个差距 = 5-15% 的带宽提升

使用条件:
  1. 地址必须 16B 对齐 (float4 = 16 bytes)
  2. N 必须是 4 的倍数 (否则需要处理尾部元素)
  3. 指针 reinterpret_cast 时要确保对齐
```


## 练习题

完成 `coalescing.cu` 的阅读后，可以在 [`exercises/`](./exercises/) 目录下做以下练习：

| 练习 | 任务 | 核心考点 |
|------|------|---------|
| [ex1_aos_soa_level1.cu](./exercises/ex1_aos_soa_level1.cu) | AoS vs SoA 读取对比 | 最经典的非合并陷阱（只填 kernel） |
| [ex1_aos_soa_level2.cu](./exercises/ex1_aos_soa_level2.cu) | 同上 | kernel + host + 计时全部自己写 |
| [ex2_write_pattern_level1.cu](./exercises/ex2_write_pattern_level1.cu) | 合并写 vs 非合并写 | 写入也需要合并（只填 kernel） |
| [ex2_write_pattern_level2.cu](./exercises/ex2_write_pattern_level2.cu) | 同上 | kernel + host + 计时全部自己写 |

编译方式（在 `exercises/` 目录下）：

```bash
nvcc -O2 --extended-lambda -o ex1_aos_soa_level1 ex1_aos_soa_level1.cu
./ex1_aos_soa_level1
```
