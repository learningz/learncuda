# 并行归约 7 版演进：从朴素到极致的优化之路

配合 `reduce.cu` 阅读。本文档解释 V0-V6 七个版本在硬件上的行为差异。

> **前置阅读**: [`theory/software_hardware_mapping.md`](../theory/software_hardware_mapping.md) 讲了 Warp Scheduler 的通用调度机制。
> 本文档聚焦于**归约操作**中每一步优化的硬件原理。


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
