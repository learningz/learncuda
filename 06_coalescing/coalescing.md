# 合并访问：Cache Line 和内存事务在硬件上的行为

配合 `coalescing.cu` 阅读。


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

  严格来说, 有效带宽 = 程序真正完成的有用字节数 / 时间。

  在这个示例里, kernel 的主要工作是读取 N 个 float，写回只有一个很小的结果槽位。
  所以更准确的口径应该把它看成"以读取为主的带宽测试"。

  为了便于和常见的 streaming kernel 直觉对齐，代码里用了一个简化估算:
    有效带宽 ≈ N × 4 bytes / 时间

  例: N=4M, 连续访问耗时 0.02ms:
  按读取流量估算, 有效带宽 ≈ 4M × 4B / 0.02ms = 800 GB/s

  如果把程序改成读 N 写 N 的标准 streaming kernel，
  那么同样的 0.02ms 会对应约 1600 GB/s。

  stride=32 耗时 0.6ms:
  按读取流量估算, 有效带宽 ≈ 16MB / 0.6ms = 26.7 GB/s → 极差!

  差距仍然主要来自合并/非合并访问的差异。
  GPU 的 HBM 带宽没变, 但大量带宽被浪费在传输不需要的数据上。
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

  修复: SoA (Structure of Arrays)
  float *x_array = ..., *y_array = ..., *z_array = ..., *w_array = ...;
  float x = x_array[tid];  // 相邻线程连续 → 完美合并!

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
