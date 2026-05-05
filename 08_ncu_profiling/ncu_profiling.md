# ncu Profiling：从输出到诊断的硬件级解读

配合 `ncu_demo.cu` 阅读。

**难度**: ⭐⭐ 进阶
**前置知识**: Roofline 模型概念（[tutorial Part 4](../tutorial.md#part-4-内存是瓶颈--合并访问和性能分析)）
**读完你能做什么**: 用 ncu 分析 kernel 瓶颈，解读 SOL、Warp Stall Reasons 等关键指标


## 什么是 ncu, 为什么需要它

### 计时只告诉你"慢了", 不告诉你"为什么慢"

你已经学会用 `cudaEvent` 给 kernel 计时了。但如果 kernel 慢，
计时只告诉你"花了 2ms"，你不知道下一步该优化什么：

```
可能的原因 (完全不同的优化方向):
  A. 内存带宽被打满了 → 应该减少内存访问 (融合, Tiling)
  B. 计算单元被打满了 → 应该用 Tensor Core 或优化算法
  C. 内存访问不合并   → 应该重新设计访问模式
  D. Bank Conflict    → 应该加 padding
  E. Warp 分歧       → 应该重新设计分支
  F. Occupancy 太低   → 应该减少寄存器/SMEM 使用

→ 你需要一个工具来回答"为什么慢" → 这就是 ncu
```

### ncu 怎么工作

```
ncu (Nsight Compute) 通过读取 GPU 的硬件性能计数器来诊断 kernel。

使用:
  ncu --set full ./your_program
  → 对每个 kernel 输出一份完整的性能报告

  ncu -o report ./your_program
  → 保存到文件, 用 Nsight Compute GUI 打开 (更直观, 有图表)

ncu 对 kernel 做了什么:
  1. 截获 kernel launch
  2. 重放这个 kernel 5-10 次, 每次采集不同的硬件计数器
     (因为计数器数量有限, 同一时刻不能全部采集)
  3. 汇总所有指标, 生成报告

→ 所以 ncu 会让程序慢 10-100 倍! 只在分析时用, 不要在生产中用。
```

### 看报告时最重要的几个指标

```
1. GPU Speed Of Light (SOL):
   Memory%:  你用了内存带宽的百分之几
   Compute%: 你用了计算能力的百分之几
   
   Memory% >> Compute% → Memory Bound → 优化内存访问
   Compute% >> Memory% → Compute Bound → 优化计算 / 用 Tensor Core
   两者都低 → 可能是 Occupancy 低或 Warp 停顿太多

2. Memory Workload:
   Global Load Efficiency:  合并效率 (100% = 完美合并, 3% = 随机访问)
   Shared Bank Conflicts:   Bank 冲突次数 (0 = 无冲突)
   
3. Occupancy:
   Achieved Occupancy: 实际活跃的 Warp 占理论最大值的百分比
   < 25% → 可能寄存器或 SMEM 用太多, 限制了并发 Warp 数

4. Warp State:
   Stall Reasons: 告诉你 Warp 为什么在等待
     Long Scoreboard → 等内存加载 → 可以通过 ILP/Occupancy 缓解
     Wait → 等 __syncthreads → 可以减少同步或用 Shuffle
     Barrier → 等其他 Warp → Block 内负载不均衡
```

本示例包含 3 个故意有"问题"的 kernel，帮你练习用这些指标定位瓶颈。


## ncu 采集到的数据从哪来

```
当你运行 ncu --set full ./ncu_demo 时:

ncu (Nsight Compute) 做了什么:
  1. 接管 GPU 的 kernel launch (通过注入 CUDA Driver 钩子)
  2. 对每个 kernel, 重放多次并采集不同类别的硬件计数器:
     - 第 1 遍: 采集 SM 吞吐相关计数器
     - 第 2 遍: 采集内存吞吐相关计数器
     - 第 3 遍: 采集 Warp 调度相关计数器
     - ...可能 5-10 遍

  为什么要重放多次?
    GPU 的硬件性能计数器 (Performance Counter) 数量有限。
    同一时刻不能同时采集所有指标。
    → 重放同一 kernel 多次, 每次采集不同指标。
    → 这就是 ncu 比手动计时慢很多的原因!

硬件计数器在哪里?
  每个 SM 内部有专用的 Performance Monitor 电路:
  ┌── SM ──────────────────────────────────────┐
  │ Warp Scheduler → 计数: 发射了多少条指令     │
  │ LD/ST Unit → 计数: 多少次全局加载/存储      │
  │ L1 Cache → 计数: 命中/未命中次数            │
  │ Barrier → 计数: 线程在 barrier 等了多久     │
  │ ...                                        │
  │ Performance Monitor: 将计数器值汇总上报     │
  └────────────────────────────────────────────┘
```


## 关键指标的硬件含义

```
SOL (Speed Of Light) — 你的 kernel 用了多少硬件能力:

  Compute (SM) Throughput = 实际计算指令数 / 理论最大指令数
    50% → SM 一半时间在算, 一半时间在等
    
  Memory (DRAM) Throughput = 实际 HBM 传输量 / 理论最大传输量
    85% → 几乎吃满带宽 (对 Memory Bound kernel 很好!)

  如果 Memory >> Compute → Memory Bound (大多数情况)
  如果 Compute >> Memory → Compute Bound (GEMM 等)
  如果两者都低 → 可能是 Launch Overhead, 同步开销, 或分歧

Sectors per Request — 合并效率:
  理想: 4 sectors/request (32 线程 × 4B = 128B = 4 个 32B sector)
  实际: 如果 > 4 → 有非合并访问 → 带宽浪费
  
  硬件计算方式:
    LD/ST Unit 的 Coalescing Logic 输出的总 sector 数
    ÷ 请求数 (合并后的 Memory Transaction 数)
    
Warp Stall Reasons — Warp 在等什么:
  Long Scoreboard:  等 LDG 回来 (HBM 延迟) → Memory Bound 的信号
  MIO Throttle:     MSHR 满了 (内存请求太多) → 需要减少同时在飞的请求
  Barrier:          等 __syncthreads() → 需要减少同步次数
  Short Scoreboard: 等计算完成 (依赖链) → 需要更多 ILP
  Not Selected:     Warp 就绪但没被选 → 好事! 说明并行度充足
```


## 三个 kernel 的 ncu 诊断实战

```
对 ncu_demo.cu 的三个 kernel 逐个分析:

kernel_A_strided (stride-2 访问):
  ─────────────────────────────────────────────
  症状: 有效带宽只有 B 版本的 ~50%

  ncu 诊断:
    SOL Memory: ~40%              ← 没吃满带宽
    Sectors/Request: ~8           ← 比理想值 4 多了一倍!
    Global Load Efficiency: ~50%  ← 一半传输的数据是"废的"

  根因分析:
    每个线程读 input[idx * 2], 相邻线程地址间隔 8B (2 个 float)
    同一 Warp 的 32 个线程:
      Thread 0 → addr + 0
      Thread 1 → addr + 8    ← 间隔 8B, 不是 4B
      Thread 2 → addr + 16
      ...
      Thread 31 → addr + 248

    32 线程跨越 256B = 8 个 32B sector
    但如果是连续访问只需 4 个 sector
    → 多传输了 128B 的无用数据
    → 带宽效率 = 128B有用 / 256B传输 = 50%

  优化方案: 重新组织数据布局，让相邻线程访问连续地址


kernel_B_coalesced (合并但标量):
  ─────────────────────────────────────────────
  症状: 带宽不错但还有提升空间

  ncu 诊断:
    SOL Memory: ~75%              ← 接近峰值
    Sectors/Request: ~4           ← 完美合并!
    Global Load Efficiency: ~100% ← 没有浪费

  指令分析:
    每个元素需要: 1 条 LDG.32 + 1 条 FMA + 1 条 STG.32 = 3 条指令
    16M 元素 = 48M 条指令
    LD/ST pipeline 是瓶颈 (指令太多)

  为什么没到 85%+?
    每元素 1 条 LDG.32 → LD/ST 指令队列可能成为瓶颈
    指令发射率限制了请求发起速度
    → 即使地址完美合并，指令太多也限制带宽利用率


kernel_C_vectorized (合并 + float4):
  ─────────────────────────────────────────────
  症状: 最高带宽

  ncu 诊断:
    SOL Memory: ~85%              ← 接近硬件极限
    Sectors/Request: ~4           ← 完美合并
    Instruction Count: 约 B 的 1/4 ← 指令大幅减少

  指令分析:
    每 4 个元素: 1 条 LDG.128 + 4 条 FMA + 1 条 STG.128 = 6 条指令
    vs B 版本:   4 条 LDG.32  + 4 条 FMA + 4 条 STG.32  = 12 条指令
    指令数减半! LD/ST pipeline 压力大幅降低

  为什么 LDG.128 更好?
    硬件角度: LDG.128 和 LDG.32 使用相同的 LD/ST Unit
    但 LDG.128 一条指令搬 16B vs LDG.32 搬 4B
    → 同样的指令发射带宽，传输 4x 的数据
    → LD/ST pipeline 不再是瓶颈，带宽利用率提升
```


## --set full vs --set basic — 什么时候用哪个

```
ncu 的 section/rule 体系:

  --set basic (默认, ~5 遍重放):
    覆盖: SOL (Compute + Memory), Occupancy, 基本内存指标
    耗时: 较短, 适合日常开发
    适用: 快速定位瓶颈类型 (Memory vs Compute Bound)

  --set full (推荐深入分析, ~10 遍重放):
    覆盖: basic + 详细内存分析 + Warp State + Scheduler + 指令统计
    耗时: 较长, 所有指标都采集
    适用: 深入理解为什么慢

  --set roofline:
    覆盖: Roofline 图所需指标
    耗时: 中等
    适用: 获得 Roofline 图

  --set source (需要 -lineinfo):
    覆盖: 基本指标 + 源码/指令关联
    耗时: 较长
    适用: 定位到具体行/SASS 指令

  自定义 section (只采集你需要的):
    ncu --section SpeedOfLight --section MemoryWorkloadAnalysis ./prog
    → 比 --set full 快, 只看相关指标

  经验法则:
    第一次诊断: --set basic → 看 SOL 决定是 Memory/Compute Bound
    深入分析: --set full → 看具体是哪个子系统的瓶颈
    定位代码: --set source → 找到具体是哪行代码
```

## Profiling 的最佳实践 — 避开常见陷阱

```
1. 预热 GPU
   第一次 kernel 启动有初始化开销 (CUDA Context, JIT)
   → 先跑一个 dummy kernel 再开始 profile
   → 或 --launch-skip 1 跳过第一个 kernel

2. 只 profile 一个 kernel
   多个 kernel 一起 profile 导致报告混淆
   → --launch-count 1 --launch-skip N 只采集目标 kernel

3. N 要足够大
   太小 (N<1000): launch overhead 主导, 看不到真实性能
   推荐: N >= 100万 元素, 让 kernel 的测量有意义

4. 区分 warmup 和 steady state
   第一个 kernel 和后续 kernel 性能可能不同 (Cache 冷 vs 热)
   → 连续跑 3 次, 看稳定后的数据

5. compute-sanitizer vs ncu 不要同时用
   两者都会插入检查代码 → 同时用会互相干扰
   → 先 sanitizer 查正确性, 再 ncu 查性能

6. 竞态不影响 ncu 指标 (但会影响计时)
   ncu 重放 kernel 多次 → 看到的是稳态平均
   但 cudaEvent 计时包含所有波动 → 两者可能不一致

7. 用 nsys 看端到端, ncu 看 kernel 内部
   nsys (Nsight Systems): 看整个程序的 timeline, kernel 间 overlap
   ncu (Nsight Compute): 看单个 kernel 的微架构行为
   → 互补关系, 不是替代!
```

## SOL 的三种结果 — 第三个维度: Stall

```
之前提到: SOL Memory% vs SOL Compute%, 两种情况:
  • Memory Bound (Memory% >> Compute%)
  • Compute Bound (Compute% >> Memory%)

但还有第三种: 两者都低 (各 < 30%)

  原因 A: 延迟瓶颈 (Latency Bound)
    现象: Warp Stall Reasons 以 Long Scoreboard / Wait 为主
    含义: 在等数据或同步, 但内存带宽也没占满
    解决: 增加 Occupancy (更多 Warp 隐藏延迟), 或用 ILP

  原因 B: 发射瓶颈 (Issue Bound)
    现象: Warp Stall 以 Short Scoreboard / Not Selected 为主
    含义: 计算依赖链太长, 或者 Warp 太多导致调度拥塞
    解决: 减少寄存器压力 (提高 Occupancy), 或打破依赖链 (ILP)

  原因 C: Launch 开销
    现象: 小 N 时 SOL 都很低
    含义: kernel 太短, 大部分时间花在 launch/调度上
    解决: 增大 grid size, 或融合到更大的 kernel

  SOL Memory% 高但 SOL Compute% 低 → Memory Bound (正常, 大多数 kernel 都是)
  SOL Compute% 高但 SOL Memory% 低 → Compute Bound (GEMM 等)
  SOL Memory% 低且 SOL Compute% 低 → 看 Warp Stall 找原因

Stall Reasons 到优化行动的映射:
  Long Scoreboard 主导 → 减少访存 (融合/Tiling/向量化)
  Barrier 主导 → 减少 __syncthreads (Shuffle/调整 Block 大小)
  MIO Throttle 主导 → 减少同时在飞的内存请求 (降低 Occupancy)
  Short Scoreboard 主导 → ILP 打散依赖链
  Not Selected 主导 → Occupancy 足够, 不需要优化 (好事!)
```

## ncu 的已知限制

```
1. 重放假设: ncu 假设每次重放时 kernel 行为完全相同
   如果 kernel 有 randomness (如 Dropout) → 指标不准
   → 对这类 kernel, 设置固定 seed

2. 对非常短 kernel (<1μs) 的测量精度有限
   硬件计数器有最小分辨率 → 小 kernel 的相对误差大

3. 并发 kernel (不同 Stream) 会互相污染计数器
   ncu 通常只 profile 一个 kernel → 需要串行化运行

4. 虚拟化/容器的 GPU 透传可能影响某些计数器
   不是所有计数器在虚拟化环境中都可用

5. 某些指标是派生量, 不是直接测量的
   如 Occupancy = 活跃 Warp / 理论最大 Warp × 活跃周期占比
   派生涉及多个计数器 → 可能有累积误差

6. --set full 显著增加 kernel 运行时间 (10-50×)
   如果 kernel 中有时效性相关逻辑 (如 polling), 结果可能不准
```


## ncu 输出的完整阅读顺序

```
拿到 ncu 报告后，按这个顺序看:

Step 1: GPU Speed Of Light (SOL) — 30 秒定位瓶颈类型
  看 Compute% 和 Memory% 的柱状图
  → Memory 高、Compute 低 = Memory Bound (最常见)
  → Compute 高、Memory 低 = Compute Bound (GEMM)
  → 都低 = Latency Bound (需要更多并行度或更少同步)

Step 2: Memory Workload Analysis — 内存效率
  看 Global Load/Store Efficiency
  看 Sectors per Request
  → 如果 Efficiency < 80%: 有非合并访问，优先修复
  → 如果 Efficiency > 95%: 内存访问模式没问题

Step 3: Compute Workload Analysis — 计算效率
  看 Achieved Occupancy (实际占用率)
  → < 25%: 考虑减少寄存器/SMEM 用量，或增加 Block 数
  → > 50%: Occupancy 不是瓶颈

Step 4: Warp State Statistics — 瓶颈细化
  看 Stall Reasons 的饼图
  → Long Scoreboard 主导: 等内存，想办法减少内存访问
  → Barrier 主导: 同步太多，考虑减少 __syncthreads()
  → Short Scoreboard 主导: 计算依赖链长，用 ILP 打断

Step 5: Scheduler Statistics — 指令级分析
  看每周期发射的指令数 (Instructions per Clock)
  → 接近 2.0 = 双发射利用得好
  → < 0.5 = 严重的指令饥饿

Step 6: Source Counters (需要 -lineinfo 编译) — 定位到代码行
  看哪行代码贡献了最多的 Stall 周期
  → 精确定位热点行，针对性优化
```


## 练习题

完成 `ncu_demo.cu` 的阅读后，可以在 [`exercises/`](./exercises/) 目录下做以下练习：

| 练习 | 任务 | 核心考点 |
|------|------|---------|
| [ex1_vectorize_level1.cu](./exercises/ex1_vectorize_level1.cu) | 把标量 kernel 改成 float4 | float4 读写 + reinterpret_cast（只填 kernel） |
| [ex2_fix_kernel_level1.cu](./exercises/ex2_fix_kernel_level1.cu) | 修复有性能 bug 的 kernel | 分析 stride-2 → 改成合并访问（只填 kernel） |

编译方式（在 `exercises/` 目录下）：

```bash
nvcc -O2 -o ex1_vectorize_level1 ex1_vectorize_level1.cu
./ex1_vectorize_level1
```


## Profile-Driven Development 完整案例

下面演示一个真实的 "profile → 诊断 → 优化 → 验证" 循环。
以 Softmax V1 (3-pass) → V2 (2-pass online) 为例。

### Round 0: 跑 V1, 建立基线

```
$ ./07_softmax/softmax
V1 (3-pass):    0.850 ms, 有效带宽 980 GB/s
```

### Round 1: ncu 分析瓶颈

```
$ ncu --set basic ./07_softmax/softmax

V1 kernel 的关键指标:
  Memory Throughput:  982 GB/s   (49% of peak 2000)
  SOL DRAM:           49%
  Compute (SM) Throughput: 78 GFLOPS  (0.4% of peak!)
  SOL SM:             0.4%

  Stall Reasons:
    Long Scoreboard:  82%     ← Warp 在等内存!
    Short Scoreboard: 3%
    Not Selected:     10%

诊断: SOL DRAM 49% + SOL SM 0.4% → Memory Bound。
      Long Scoreboard 82% → Warp 大部分时间在等数据。
      3-pass 意味着读 3 次输入数据 → 3× 不必要的 HBM 访问。
```

### Round 2: 提出优化方案

```
目标: 减少 HBM 读取次数
方案: V2 (2-pass Online) — max 和 sum 合并为一遍扫描

预期: HBM 读取从 3N 降到 2N → 带宽利用率 ~1.5x
      加上修正因子的计算 → 计算量略有增加 (但仍是 Memory Bound, 无所谓)
```

### Round 3: 实现 V2

在 `softmax.cu` 中加入 V2 kernel (Online Softmax),
重新编译运行:

```
$ ./07_softmax/softmax
V1 (3-pass):    0.850 ms,  980 GB/s
V2 (2-pass):    0.610 ms, 1360 GB/s    ← 1.39× faster
```

### Round 4: ncu 验证优化效果

```
$ ncu --set basic ./07_softmax/softmax

V2 kernel 的关键指标:
  Memory Throughput:  1360 GB/s  (68% of peak) ← 从 49% 提升到 68%!
  SOL DRAM:           68%
  SOL SM:             0.6%                      ← 仍然是 Memory Bound (正常)

  Stall Reasons:
    Long Scoreboard:  68%     ← 从 82% 降到 68%
    Short Scoreboard: 5%      ← 修正因子的 exp/log 计算增加了一点计算等待
    Not Selected:     15%

对比 Round 1:
  ✓ Memory Throughput: 982 → 1360 GB/s (+38%)
  ✓ 耗时: 0.850 → 0.610 ms (-28%)
  ✓ Long Scoreboard: 82% → 68% (Warp 等内存减少了, 因为读的次数少了)
  ✓ 优化方向正确!
```

### Round 5: 进一步优化?

```
SOL DRAM = 68%, 还有提升空间。下一步可以:
  1. float4 向量化 → 预计 68% → 80% (LDG.128 vs LDG.32)
  2. 和后续算子融合 → 省掉写回+重新读取的开销
  3. 检查 Early Kill 优化吗? (后面一半 block 做无用功?)

如果这是模型的热点 (如 LLM 的最后输出层), 继续优化。
如果它只占 2% 的总体时间, 停止优化 → 转向更热的部分。
```

### 核心教训

```
成功的 Profile-Driven 优化循环:

1. 测基线 (cudaEvent 计时)
2. Profile (ncu, 关注 SOL DRAM vs SOL SM)
3. 诊断瓶颈 (Memory Bound? Compute Bound? Latency Bound?)
4. 提出具体假设 ("减少 pass 数 → 带宽利用率提升 1.5×")
5. 实现 (只改一个变量!)
6. 验证 (重新 profile, 对比关键指标)
7. 决定: 继续优化还是转向下一个瓶颈

不要跳过第 2 步直接改代码!
你不知道瓶颈是什么, 优化方向可能完全是错的。
```
