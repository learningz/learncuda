# ncu Profiling：从输出到诊断的硬件级解读

配合 `ncu_demo.cu` 阅读。


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


## 常用 ncu 命令速查

```
基础采集 (推荐入门用):
  ncu --set full ./your_program
  → 采集所有指标，输出到终端

保存报告文件 (推荐离线分析):
  ncu --set full -o my_report ./your_program
  → 生成 my_report.ncu-rep
  → 用 ncu-ui my_report.ncu-rep 打开 GUI 查看

只看某些指标 (快速诊断):
  ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
    ./your_program

只分析某个 kernel:
  ncu --kernel-name "kernel_B_coalesced" --set full ./your_program

采集 Roofline 数据:
  ncu --set roofline -o roofline_report ./your_program
  → 在 ncu-ui 中可以看到 Roofline 图

显示 SASS 源码级分析:
  nvcc -O2 -lineinfo -o prog prog.cu       ← 编译时加 -lineinfo
  ncu --set source -o source_report ./prog  ← 采集源码关联
  → ncu-ui 中可以看到每行 CUDA 代码对应的 SASS 指令和性能数据

跳过前 N 个 kernel (避免初始化 kernel 干扰):
  ncu --launch-skip 2 --launch-count 1 --set full ./your_program
  → 跳过前 2 个 kernel, 只采集第 3 个
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
