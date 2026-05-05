# 并行归约 7 版演进：从朴素到极致的优化之路

配合 `reduce.cu` 阅读。本文档先解释什么是归约，再逐版解释 V0-V6 在硬件上的行为差异。

> **前置阅读**: [`theory/software_hardware_mapping.md`](../theory/software_hardware_mapping.md) 讲了 Warp Scheduler 的通用调度机制。
> 本文档聚焦于**归约操作**中每一步优化的硬件原理。


## 什么是归约 (Reduce)

```
归约 = 把一组数据"汇总"成一个值。

日常例子:
  全班 50 人的考试成绩 → 求平均分          (sum → 然后除以 N)
  一个月 30 天的气温   → 求最高气温        (max)
  一篇文章的每个词     → 统计某个词出现几次  (count)

数学上:
  sum([3, 1, 4, 1, 5]) = 14
  max([3, 1, 4, 1, 5]) = 5
  这些都是归约: 多个值 → 1 个值
```

CPU 上很简单——一个 for 循环就行：
```c
float sum = 0;
for (int i = 0; i < n; i++) sum += data[i];
```
但这是串行的，1 亿个数就要 1 亿次加法，一个接一个。


## 为什么归约可以并行

关键观察：**加法满足结合律**。`(a+b)+c = a+(b+c)`。
这意味着我们可以把加法任意分组，先算一部分，再把部分结果合起来。

```
串行:     3+1+4+1+5+9+2+6   一次加一个，7 步
          ↓
并行:     把 8 个数两两配对，同时加

第 1 轮: [3+1]  [4+1]  [5+9]  [2+6]     4 次加法同时做!
          = 4     = 5    = 14    = 8

第 2 轮: [4+5]       [14+8]              2 次加法同时做!
          = 9          = 22

第 3 轮: [9+22]                           1 次加法
          = 31

3 轮就算完了 (log₂8 = 3)。如果有足够多的"工人"(线程)，
每轮的耗时和加一次法一样 → 总时间从 O(N) 变成 O(log N)!
```

这个"两两配对、逐轮减半"的过程就叫**树形归约**——
画出来像一棵倒过来的树。


## 树形归约怎么映射到 GPU 线程

现在关键问题来了：**哪个线程负责加哪两个数？**

GPU 有 threadIdx（线程编号），我们用 threadIdx 决定每个线程的工作。

### V0 的做法（朴素交错）

最直觉的想法——让 Thread 0 加位置 0 和 1，Thread 2 加位置 2 和 3...

```
初始:   sdata[0] sdata[1] sdata[2] sdata[3] sdata[4] sdata[5] sdata[6] sdata[7]
          3        1        4        1        5        9        2        6

stride=1: Thread 0 执行 sdata[0] += sdata[1]  → sdata[0]=4
          Thread 2 执行 sdata[2] += sdata[3]  → sdata[2]=5
          Thread 4 执行 sdata[4] += sdata[5]  → sdata[4]=14
          Thread 6 执行 sdata[6] += sdata[7]  → sdata[6]=8
          判断条件: if (tid % 2 == 0) → 只有偶数号线程工作

stride=2: Thread 0 执行 sdata[0] += sdata[2]  → sdata[0]=9
          Thread 4 执行 sdata[4] += sdata[6]  → sdata[4]=22
          判断条件: if (tid % 4 == 0) → 只有 0,4 号线程工作

stride=4: Thread 0 执行 sdata[0] += sdata[4]  → sdata[0]=31
          判断条件: if (tid % 8 == 0) → 只有 0 号线程工作
```

代码直接对应：
```cuda
for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if (tid % (2 * stride) == 0) {            // 每隔 2×stride 个线程选一个
        sdata[tid] += sdata[tid + stride];    // 加上右边 stride 处的值
    }
    __syncthreads();  // 等所有人写完再进入下一轮!
}
```

**但是这个做法有一个严重的性能问题**——Warp Divergence。

### V0 为什么慢：Warp Divergence 详解

GPU 以 Warp (32 线程) 为单位执行指令。同一 Warp 里的线程必须执行同一条指令。

看 stride=1 时发生了什么：
```
Warp 0 的 32 个线程:
  Thread 0:  tid%2==0 ✓ → 走 if 分支，做加法
  Thread 1:  tid%2==1 ✗ → 走 else 分支，空等
  Thread 2:  tid%2==0 ✓ → 走 if 分支，做加法
  Thread 3:  tid%2==1 ✗ → 走 else 分支，空等
  ...

同一个 Warp 里，一半走 if、一半走 else
→ GPU 必须先执行 if 路径 (偶数线程工作，奇数空闲)
→ 再执行 else 路径 (奇数线程"什么都不做"，偶数空闲)
→ 两条路径串行! 即使 else 什么都不做也浪费了一轮!
→ 这就是 Warp Divergence，每轮实际效率只有 50%。
```

### V1 怎么解决：让连续线程工作

V1 换了一个思路——从另一个方向遍历：

```
stride=128: Thread 0-127 工作, Thread 128-255 不工作
  → Warp 0-3 (Thread 0-127) 全部执行 if → 无分歧!
  → Warp 4-7 (Thread 128-255) 全部跳过 if → 也无分歧!

stride=64:  Thread 0-63 工作
  → Warp 0-1 全部执行, Warp 2-7 全部跳过 → 仍然无分歧!

stride=32:  Thread 0-31 工作
  → 只有 Warp 0 工作 → 还是无分歧!

stride=16:  Thread 0-15 工作
  → 这是 Warp 0 里的一半线程 → 出现分歧了
  → 但此时只剩最后几轮了, 影响小得多
```

代码对比：
```cuda
// V0: stride 从小到大, 每轮都有分歧
for (int stride = 1; stride < blockDim.x; stride *= 2)
    if (tid % (2 * stride) == 0)  sdata[tid] += sdata[tid + stride];

// V1: stride 从大到小, 前几轮完全无分歧
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    if (tid < stride)  sdata[tid] += sdata[tid + stride];
```

两者在数学上完全等价（都是树形归约），但 **线程分配策略不同，
导致 Warp Divergence 行为完全不同**。这就是 GPU 优化的精髓：
同一个算法，线程映射方式决定了性能。


## 从 Block 级到全局：为什么需要多阶段

上面的树形归约在一个 Block 内完成。但一个 Block 最多 1024 个线程。
如果有 100 万个数，怎么办？

```
策略: 两阶段归约

阶段 1: 分成很多 Block，每个 Block 内部做树形归约
  Block 0: data[0..255]     → 1 个部分和 partial[0]
  Block 1: data[256..511]   → 1 个部分和 partial[1]
  Block 2: data[512..767]   → 1 个部分和 partial[2]
  ...
  Block 4095: data[1048320..1048575] → 1 个部分和 partial[4095]

  现在有 4096 个部分和。

阶段 2: 再启动一个 kernel，对 4096 个部分和做归约 → 1 个最终结果

但 4096 个 Block 开销很大 (调度 + SMEM 分配)。
→ 这就引出了 V2 的 Grid-Stride Loop 优化。
```


## V2 Grid-Stride Loop：每个线程先自己累加

V2 的核心思想：**别急着用树形归约，先让每个线程自己串行加一堆数**。

```
V1 的做法: N=4M 个数, blockSize=256 → 需要 16384 个 Block
  每个 Block 只处理 256 个数 → Block 调度开销很大

V2 的做法: 固定只用 256 个 Block (gridSize=256)
  总共 256×256 = 65536 个线程
  每个线程处理 N/65536 ≈ 64 个数
  怎么分配? 用 Grid-Stride Loop:

  int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 我的起始位置
  int stride = blockDim.x * gridDim.x;              // 总线程数
  float local_sum = 0;
  for (int i = idx; i < n; i += stride)             // 每隔 stride 取一个
      local_sum += input[i];
```

举个具体的例子（假设只有 4 个线程，N=16）：
```
Thread 0: 加 data[0], data[4], data[8],  data[12]  → local_sum_0
Thread 1: 加 data[1], data[5], data[9],  data[13]  → local_sum_1
Thread 2: 加 data[2], data[6], data[10], data[14]  → local_sum_2
Thread 3: 加 data[3], data[7], data[11], data[15]  → local_sum_3

注意: 相邻线程访问的地址是连续的! (0,1,2,3 → 4,5,6,7 → ...)
→ 合并访问! 带宽利用率 100%!

然后只需要对 4 个 local_sum 做一次树形归约 → 最终结果。
```

**为什么快得多？**
1. 每线程的 64 次加法完全在寄存器中，零同步开销
2. Block 数从 16384 降到 256，调度开销降 64 倍
3. 内存访问天然合并（相邻线程地址连续）


## V3 Warp Shuffle：不经过 Shared Memory 的归约

V2 做完 Grid-Stride Loop 后，Block 内还是用 Shared Memory 做树形归约。
这意味着：写 SMEM → `__syncthreads()` → 读 SMEM → 写 SMEM → 同步 → ...
8 轮归约 = 8 次同步屏障。

V3 发现：**同一 Warp 的 32 个线程可以直接读彼此的寄存器值，
不需要经过 Shared Memory，也不需要 `__syncthreads()`！**

这就是 Warp Shuffle 指令：
```cuda
float neighbor = __shfl_down_sync(0xffffffff, my_val, offset);
// 我直接拿到了"编号比我大 offset"的线程的 my_val
// 不经过任何内存! 直接寄存器到寄存器!
```

### __shfl_down_sync 的 mask 参数 — 为什么不能省略

```
__shfl_down_sync(mask, val, delta):

  mask (4-byte bitmap): 指定哪些线程参与这次 shuffle
    0xffffffff = 所有 32 个 lane 都参与
    0x0000ffff = 只有 lane 0-15 参与 (lane 16-31 被排除)

  关键规则: mask 必须在所有参与线程中完全相同!
    如果 Thread 0 传 0xffffffff, Thread 1 传 0xfffffffe
    → 未定义行为! → 可能导致死锁或错误数据

  为什么旧版 __shfl_down (无 _sync 后缀) 被废弃?
    旧版: __shfl_down(val, delta)  ← 没有 mask 参数!
    问题: 隐式假设所有 32 个 lane 都参与
    但 Volta+ 引入 Independent Thread Scheduling (ITS):
      线程可能在不同的位置执行 → 不能假设所有 lane 同步
      → 需要显式指定 mask 来保证正确性
    
    示例 — 为什么 mask 必须和活跃线程一致:
      // ✗ 危险: 如果只有前 16 个 lane 活跃
      if (tid < 16) {
          // 只有 lane 0-15 到达这里
          float v = __shfl_down_sync(0xffffffff, val, 8);
          // mask=0xffffffff 暗示所有 32 lane 都参与!
          // 但 lane 16-31 没有执行到这里 → 它们不会提供数据
          // → 未定义行为!
      }
      
      // ✓ 安全: mask 匹配实际活跃的 lane
      unsigned mask = __activemask();  // 获取当前活跃线程的 mask
      if (tid < 16) {
          float v = __shfl_down_sync(mask, val, 8);
          // mask 只包含实际执行到这里的线程
      }

  编译前 vs 运行时:
    如果 mask 在编译期是常量 (如 0xffffffff):
      → 编译器可以消除对非活跃 lane 的等待 → 更快
    如果 mask 是变量 (如 __activemask()):
      → 运行时等待对应 lane → 稍慢但安全

  规则总结:
    1. 所有参与线程必须传相同的 mask
    2. mask 中标记为参与的线程必须都执行了这条指令
    3. 优先用编译期常量 mask (性能最好)
    4. 不确定时用 __activemask() (最安全)
```

### __shfl_down vs __shfl_xor vs __shfl — 选哪个

```
__shfl_down_sync: 从 lane_id + delta 拿值
  用途: 树形归约 → 每轮 delta 减半

__shfl_xor_sync:  从 lane_id ^ delta 拿值
  用途: 蝶形归约 → 每轮 lane 和相邻 lane 交换
  优势: 所有 lane 都得到结果 (而 shfl_down 只有低 lane 有)

__shfl_sync:      所有 lane 广播同一个 lane 的值
  用途: 归约完成后把结果广播给所有 lane
  示例: max_val = __shfl_sync(0xffffffff, max_val, 0);
        → lane 0 的 max_val 广播给所有 32 个 lane

归约中常用组合:
  __shfl_down 做归约 (delta=16,8,4,2,1) → 结果在 lane 0
  __shfl_sync 广播 lane 0 的结果 → 所有 lane 都有最终值
```

用 Shuffle 做 Warp 内归约（32 → 1）：
```
初始: 32 个线程各持有一个 local_sum

  __shfl_down 16: Thread 0 拿到 Thread 16 的值，加到自己的值上
                  Thread 1 拿到 Thread 17 的值，加到自己的值上
                  ... Thread 15 拿到 Thread 31 的值
                  → 32 个值变成 16 个部分和 (在 Thread 0-15 中)

  __shfl_down 8:  Thread 0 += Thread 8 的值, Thread 1 += Thread 9 的值...
                  → 16 个变成 8 个 (在 Thread 0-7 中)

  __shfl_down 4:  → 8 个变成 4 个
  __shfl_down 2:  → 4 个变成 2 个
  __shfl_down 1:  → 2 个变成 1 个 (在 Thread 0 中)

  5 步搞定! 而且:
    - 零 Shared Memory 读写
    - 零 __syncthreads()
    - 每步延迟 ~1 cycle (vs SMEM 的 ~5 cycles)
```

但一个 Block 有多个 Warp (256 线程 = 8 个 Warp)，怎么把 8 个 Warp 的结果合起来？

```
步骤 1: 每个 Warp 内用 Shuffle 归约 → 8 个 Warp 各得 1 个值
步骤 2: 8 个值写入 Shared Memory (只写 8 个 float, 不是 256 个!)
        __syncthreads()   ← 整个 Block 只需这 1 次同步!
步骤 3: 第一个 Warp 读出 8 个值, 再做一次 Shuffle 归约 → 1 个值

原来: 8 次 __syncthreads, 256×4B SMEM
现在: 1 次 __syncthreads, 8×4B SMEM
```


## 七个版本的全景

```
┌────────┬──────────────────────────────┬────────────────────────────┐
│ 版本   │ 核心改进                      │ 消除的瓶颈                  │
├────────┼──────────────────────────────┼────────────────────────────┤
│ V0     │ 朴素交错归约                  │ (基准)                     │
│ V1     │ 连续线程工作                  │ Warp Divergence            │
│ V2     │ Grid-Stride Loop             │ Block 调度开销 + 寄存器复用 │
│ V3     │ Warp Shuffle                 │ __syncthreads 开销 (8→1次) │
│ V4     │ 4 路循环展开                  │ 内存延迟 (ILP)             │
│ V5     │ float4 向量化                 │ 指令发射瓶颈               │
│ V6     │ atomicAdd                    │ CPU 回传汇总的开销          │
└────────┴──────────────────────────────┴────────────────────────────┘
```


## V0 vs V1: Warp Divergence 的消除

```
V0 的问题 — 交错归约:
  stride=1 时: if (tid % 2 == 0)
    Warp 0 的 Thread 0 工作, Thread 1 不工作, Thread 2 工作, Thread 3 不工作...
    → 同一 Warp 内奇偶线程走不同分支 → Warp Divergence!
    → 两条路径串行 → 性能减半

  stride=2 时: if (tid % 4 == 0)
    → 更严重: 每 4 个线程只有 1 个工作 → 75% 线程空闲

V1 的改进 — 让连续线程工作:
  stride=128 时: if (tid < 128)
    Warp 0-3 (Thread 0-127) 全部工作 → 无分歧!
    Warp 4-7 (Thread 128-255) 全部不工作 → 也无分歧!
    → Divergence 只在 stride < 32 的最后几轮才出现

  硬件差异:
    V0: 每轮都有 Divergence → 8 轮 × 50% 效率 = ~4× 等效轮数
    V1: 前 5 轮无 Divergence → 只有最后 3 轮效率下降
    → V1 快 ~1.3-1.5× (视数据量)
```


## V1 vs V2: Grid-Stride Loop 的威力

```
V1 的问题:
  N=4M, blockSize=256 → gridSize = 16384 个 Block
  每个 Block 只处理 256 个元素 → 做 8 轮归约就结束了
  但 Block 的调度、SMEM 分配、__syncthreads 等固定开销不变!
  → 大量时间花在"管理"而不是"计算"上

V2 的改进:
  固定 gridSize=256 → 每线程处理 N / (256×256) ≈ 64 个元素
  for (int i = idx; i < n; i += stride) local_sum += input[i];
  
  每线程的 64 次累加完全在寄存器中 → 零同步开销!
  之后 Block 级归约从 256 个局部和开始 (而不是 256 个原始元素)
  
  硬件差异:
    V1: 16384 个 Block × 每 Block 开销 ≈ 大量调度时间
    V2: 256 个 Block × 每 Block 开销 ≈ 极少调度时间
    且 V2 的加载模式天然是合并的: 连续线程访问连续地址
    → V2 通常快 2-5×
```


## V2 vs V3: Shared Memory 归约 → Warp Shuffle

```
V2 的 Block 级归约仍然用 Shared Memory 树形:
  8 轮 → 8 次 __syncthreads → 每次 BAR.SYNC ~25 cycles = ~200 cycles 屏障开销

V3 改为两阶段:
  阶段 1: Warp 内 Shuffle (5 次 SHFL+FADD, ~30 cycles, 零屏障!)
  阶段 2: Warp 间 SMEM 通信 (1 次 BAR.SYNC, ~25 cycles)
  总屏障: ~25 cycles (vs V2 的 ~200 cycles)

SHFL.DOWN 在硬件上的路径:
  寄存器 → Warp 内 Crossbar → 目标线程寄存器
  不经过 Shared Memory, 不经过 L1 Cache, 不经过 NoC!
  延迟: ~1-2 cycles (vs SMEM 的 ~5 cycles)

┌────────────────────┬───── V2 (SMEM 归约) ──┬───── V3 (Warp Shuffle) ──┐
│ 指令类型           │ LDS + STS + BAR.SYNC  │ SHFL + FADD             │
│ 数据通路           │ 寄存器 → SMEM → 寄存器│ 寄存器 → 寄存器 (直连)  │
│ 延迟 (每步)        │ ~40 cycles            │ ~6 cycles               │
│ __syncthreads 次数 │ 8 次                  │ 1 次 (仅 Warp 间)       │
│ BAR.SYNC 总开销    │ ~200 cycles           │ ~25 cycles              │
│ SMEM 使用          │ 256 × 4B = 1KB        │ 8 × 4B = 32B           │
└────────────────────┴───────────────────────┴─────────────────────────┘
```


## V3 vs V4: ILP (指令级并行) 隐藏内存延迟

```
V3 的 Grid-Stride Loop:
  for (int i = idx; i < n; i += stride) local_sum += input[i];
  
  编译后:
    LDG R1, [addr]         // 发射加载, 延迟 ~300 cycles
    --- 等待 300 cycles --- // Warp Stall: Long Scoreboard
    FADD R0, R0, R1        // 加载完成后才能执行
    IADD addr, addr, stride
    循环...

  每个元素: ~300 cycles (几乎全在等内存)

V4 的 4 路展开:
  for (; i + 3*stride < n; i += 4*stride) {
      local_sum += input[i] + input[i+stride]
                 + input[i+2*stride] + input[i+3*stride];
  }
  
  编译后:
    LDG R1, [addr]          // 发射加载 0
    LDG R2, [addr+stride]   // 立刻发射加载 1 (不依赖 R1!)
    LDG R3, [addr+stride*2] // 立刻发射加载 2
    LDG R4, [addr+stride*3] // 立刻发射加载 3
    // 4 个加载同时在飞! 只等最慢那个 = ~300 cycles (而不是 4×300)
    FADD R0, R0, R1
    FADD R0, R0, R2
    FADD R0, R0, R3
    FADD R0, R0, R4

  每 4 个元素: ~300 + 16 ≈ 316 cycles → 平均 ~79 cycles/元素
  加速: 300/79 ≈ 3.8× (理论值, 实际取决于 Occupancy)

为什么编译器不自动展开?
  -O2 下编译器可能做一定程度的展开, 但:
  1. 编译器不知道 n 有多大, 过度展开可能浪费寄存器
  2. 手动展开让你精确控制 ILP 度 + 寄存器用量的权衡
  3. 实际中 2-8 路展开是甜点, 需要实测
```


## V4 vs V5: 向量化加载

```
V4 的 4 条标量加载:
  LDG.E.32 R1, [addr]        // 每条传 4 bytes
  LDG.E.32 R2, [addr+4]
  LDG.E.32 R3, [addr+8]
  LDG.E.32 R4, [addr+12]
  → 4 条指令, 占用 LD/ST pipeline 4 个 slot

V5 的 1 条向量化加载:
  LDG.E.128 R4, [addr]       // 一条传 16 bytes (4 个 float)
  → 1 条指令, 占用 LD/ST pipeline 1 个 slot

数据量相同, 但指令数减少 4×!
LD/ST pipeline 的发射带宽有限 → 指令少 = 发射更快 = 带宽利用率更高

对于 Memory Bound 的归约:
  V4 的 LD/ST pipeline 可能成为瓶颈 (指令太多, 发不过来)
  V5 消除了这个瓶颈 → 通常额外加速 5-15%
```


## V5 vs V6: atomicAdd 消除 CPU 回传

```
V0-V5 的流程:
  GPU kernel: N 个元素 → 256 个部分和 (每 Block 一个)
  cudaMemcpy: 256 个 float 从 GPU → CPU
  CPU for 循环: 256 个 float → 1 个总和
  
  问题: 需要一次 D2H 传输 + CPU 计算 + 可能还要传回 GPU
  对于大规模 pipeline, 这个来回增加延迟

V6 的改进:
  每个 Block 归约完后: atomicAdd(&output[0], block_sum)
  256 个 Block 只做 256 次 atomicAdd → 结果直接在 GPU 显存中!
  
  atomicAdd 的硬件行为:
    在 L2 Cache 中执行 Read-Modify-Write (原子操作)
    多个 Block 同时 atomicAdd → L2 会串行化 → 有一定开销
    但只有 256 次 (不是 N 次!) → 开销 < 1μs
    vs D2H + CPU 汇总 ≈ 5-10μs
  
  注意:
    float atomicAdd 的精度: 加法顺序不确定 → 结果有微小差异
    → 对 ML 场景完全可接受 (误差 << 1e-5)
    → 如果需要确定性结果, 仍需用确定顺序的归约
```


## 练习题

完成 `reduce.cu` 的阅读后，可以在 [`exercises/`](./exercises/) 目录下做以下练习：

| 练习 | 任务 | 核心考点 |
|------|------|---------|
| [ex1_reduce_max_level1.cu](./exercises/ex1_reduce_max_level1.cu) | 求数组最大值 | 把 += 换成 fmaxf（只填 kernel） |
| [ex1_reduce_max_level2.cu](./exercises/ex1_reduce_max_level2.cu) | 同上 | kernel + host 端全部自己写 |
| [ex2_dot_level1.cu](./exercises/ex2_dot_level1.cu) | 两个数组的点积 | elementwise 乘 + 归约（只填 kernel） |
| [ex2_dot_level2.cu](./exercises/ex2_dot_level2.cu) | 同上 | kernel + host 端全部自己写 |
| [ex3_count_level1.cu](./exercises/ex3_count_level1.cu) | 计数 > 阈值的元素 | 条件判断 + 归约 + atomicAdd（只填 kernel） |
| [ex3_count_level2.cu](./exercises/ex3_count_level2.cu) | 同上 | kernel + host 端全部自己写 |

编译方式（在 `exercises/` 目录下）：

```bash
nvcc -O2 -o ex1_reduce_max_level1 ex1_reduce_max_level1.cu
./ex1_reduce_max_level1
```


## 性能基准参考

以下数据在 N=2^22 (4M) 元素上测得，仅供参考对比。绝对值因 GPU 型号而异，**相对提升倍数**是关注重点：

| 版本 | A100 (sm_80) | RTX 3090 (sm_86) | 相对 V0 提升 |
|------|:---:|:---:|:---:|
| V0 朴素交错 | ~400 GB/s | ~200 GB/s | 1.0× (baseline) |
| V1 连续线程 | ~600 GB/s | ~300 GB/s | ~1.5× |
| V2 Grid-Stride | ~800 GB/s | ~400 GB/s | ~2.0× |
| V3 Warp Shuffle | ~1200 GB/s | ~600 GB/s | ~3.0× |
| V4 ILP 展开 | ~1400 GB/s | ~700 GB/s | ~3.5× |
| V5 float4 向量化 | ~1550 GB/s | ~780 GB/s | ~3.9× |
| V6 atomicAdd | ~1500 GB/s | ~750 GB/s | ~3.8× |

> **怎么读**: 如果你的 V0 是 200 GB/s, V3 是 550 GB/s, 但参考值是 400→1200 (3×),
> 说明 V0→V3 相对提升是 2.75× — 接近参考, 说明优化路径正确。
> 如果 V3 只比 V0 快 1.2×, 检查 Shuffle 是否真的用了 (不用 SMEM)。

**A100 峰值带宽**: ~2 TB/s。V5 达到 ~1550 GB/s = 77% 峰值 — 很好!
Reduce 是 Memory Bound (AI ≈ 0.08), 77% 峰值带宽已经是优秀水平。
