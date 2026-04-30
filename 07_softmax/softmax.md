# Softmax 三版本：指令流和硬件行为对比

本文档配合 `softmax.cu` 阅读。重点对比三个版本在硬件上的差异。


## 三版本的核心指令差异

```
V1 (3-pass):
  Pass 1 (max):   LDG → Shared Memory reduce (8× BAR.SYNC) → broadcast max
  Pass 2 (sum):   LDG → exp → Shared Memory reduce (8× BAR.SYNC) → broadcast sum
  Pass 3 (norm):  LDG → exp → div → STG
  
  全局显存访问: 3N 次读 + N 次写 = 4N
  BAR.SYNC: 16 次 (每 pass 8 次)
  
V2 (2-pass Online):
  Pass 1 (max+sum): LDG → online max/sum → Shared Memory reduce (8× BAR.SYNC)
  Pass 2 (norm):    LDG → exp → div → STG
  
  全局显存访问: 2N 次读 + N 次写 = 3N  (比 V1 少 1N!)
  BAR.SYNC: 8 次 (只有 1 pass 的 reduce)
  
V3 (Warp-level, N≤32):
  1 pass: LDG → SHFL reduce max → exp → SHFL reduce sum → div → STG
  
  全局显存访问: N 次读 + N 次写 = 2N (最少!)
  BAR.SYNC: 0 次! (全用 SHFL, 无 Shared Memory)
```


## V1 每条指令的硬件路径

```
以 Pass 1 (求 max) 的第一步为例:

Thread 0 执行: local_max = fmaxf(local_max, x[i]);

SASS:
  LDG.E R2, [R_addr] ;     // 从 HBM 加载 x[i]
  
  硬件路径:
    Thread 0 的地址 → LD/ST Unit
    32 线程的地址合并 (如果连续 → 1 事务)
    → L1 Cache 查询 (可能 miss)
    → NoC → L2 → MC → HBM → 数据返回
    延迟: ~500 cycles
    (其他 Warp 在这段时间执行 → 延迟隐藏)
  
  FMNMX R0, R0, R2, !PT ;  // max(local_max, x[i])
  
  硬件路径:
    R0 和 R2 从 Register File 读出
    → FP32 ALU 比较 (FMNMX = Float Min/Max)
    → 结果写回 R0
    延迟: ~4 cycles

Pass 1 的 reduce 阶段:
  STS [R_smem], R0 ;        // local_max → Shared Memory
  BAR.SYNC 0 ;              // 等所有线程写完
  LDS R1, [R_smem + stride*4] ; // 读邻居的 max
  FMNMX R0, R0, R1, !PT ;  // 取更大的
  STS [R_smem], R0 ;        // 写回
  BAR.SYNC 0 ;              // 等所有线程算完
  // 重复 8 轮...

这段循环的硬件开销:
  8 × (LDS 5cyc + FMNMX 4cyc + STS 5cyc + BAR ~25cyc) ≈ 312 cycles
  主要花在 BAR.SYNC 上!
```


## V2 Online Softmax 的硬件特殊性

```
Online 版本的关键指令:

  // 修正旧的 sum: sum = sum * exp(old_max - new_max) + exp(x - new_max)
  FSUB R3, R_old_max, R_new_max ;    // old_max - new_max
  MUFU.EX2 R3, R3 ;                  // exp2(R3) → 需要先乘 log2(e)
  FMUL R_sum, R_sum, R3 ;            // sum *= correction
  FSUB R4, R_x, R_new_max ;          // x - new_max
  MUFU.EX2 R4, R4 ;                  // exp(x - new_max)
  FADD R_sum, R_sum, R4 ;            // sum += exp(x - new_max)

MUFU.EX2 (exp2) 在硬件上:
  使用 SFU (Special Function Unit) 执行
  SFU 的吞吐只有 FP32 Core 的 1/4!
  延迟: ~28 cycles (vs FADD 的 4 cycles)
  
  → Online 版本每元素多了 2 次 MUFU 调用 (exp 修正 + exp 新值)
  → 但节省了一整遍全局显存读 (N 次 LDG)
  → 对于 Memory Bound 的 Softmax, 节省 LDG 的收益 >> 多几条 MUFU 的代价
```


## V3 Warp Shuffle 版本的极致效率

```
V3 只在 N ≤ 32 时有效: 一个 Warp 的 32 线程各持有 1 个元素。

SASS (完整的 warp reduce max):
  SHFL.BFLY R1, R0, 0x10, 0x1f ;   // R1 = lane 对面 (XOR 16) 的 R0
  FMNMX R0, R0, R1, !PT ;           // max
  SHFL.BFLY R1, R0, 0x8, 0x1f ;
  FMNMX R0, R0, R1, !PT ;
  SHFL.BFLY R1, R0, 0x4, 0x1f ;
  FMNMX R0, R0, R1, !PT ;
  SHFL.BFLY R1, R0, 0x2, 0x1f ;
  FMNMX R0, R0, R1, !PT ;
  SHFL.BFLY R1, R0, 0x1, 0x1f ;
  FMNMX R0, R0, R1, !PT ;
  SHFL.IDX R_max, R0, RZ, 0x1f ;   // 广播 lane 0 的 max 给所有线程

  // SHFL.BFLY 使用蝶形模式: 比 SHFL.DOWN 好, 因为所有 lane 都得到结果

整个 V3 kernel 的指令计数:
  SHFL: 10 条 (max 5 + sum 5)
  FMNMX: 5 条
  FADD: 5 条
  MUFU.EX2: 1 条 (exp)
  MUFU.RCP: 1 条 (1/sum)
  FMUL: 2 条
  FSUB: 1 条
  LDG: 1 条
  STG: 1 条
  SHFL.IDX: 2 条 (广播)
  
  总: ~30 条指令, 0 条 BAR.SYNC, 0 次 Shared Memory 访问
  → 极致高效, 但只适用于 N ≤ 32
```


## 三版本硬件开销对比 (N=4096, 每行)

```
┌──────────────┬──── V1 (3-pass) ──┬── V2 (online) ──┬── V3 (warp) ──┐
│ HBM 读取      │ 3×4096=12288 次  │ 2×4096=8192 次  │ N/A (N≤32)   │
│ HBM 写入      │ 4096 次          │ 4096 次         │              │
│ BAR.SYNC      │ 16 次            │ 8 次            │ 0 次         │
│ MUFU (exp)    │ 2×4096 条        │ 3×4096 条       │ ~32 条       │
│ SHFL          │ 0 条             │ 0 条            │ ~10 条       │
│ SMEM 读写     │ ~4096 条         │ ~2048 条        │ 0 条         │
│ 瓶颈          │ HBM 带宽         │ HBM 带宽        │ 指令数       │
└──────────────┴──────────────────┴─────────────────┴──────────────┘

关键洞察:
  V1→V2: HBM 读取减少 33% → 对 Memory Bound 算子, 理论加速 ~1.3×
  V2→V3: 完全消除 SMEM 和 BAR → 但只适用于短行
  所有版本: 瓶颈都是 HBM 带宽 (AI < 1), 不是计算
```


## 练习题

完成 `softmax.cu` 的阅读后，可以在 [`exercises/`](./exercises/) 目录下做以下练习：

| 练习 | 任务 | 核心考点 |
|------|------|---------|
| [ex1_logsumexp_level1.cu](./exercises/ex1_logsumexp_level1.cu) | LogSumExp（Softmax 前半段） | 2-pass SMEM 归约: 求 max + 求 sum（只填 kernel） |
| [ex2_l2norm_level1.cu](./exercises/ex2_l2norm_level1.cu) | L2 Normalize | 换公式但结构同 Softmax: 求 norm² + 归一化（只填 kernel） |
| [ex3_softmax_v1_level2.cu](./exercises/ex3_softmax_v1_level2.cu) | 从零写 3-pass Softmax | 完整的 3-pass 归约 + kernel + host 全部自己写 |

编译方式（在 `exercises/` 目录下）：

```bash
nvcc -O2 -o ex1_logsumexp_level1 ex1_logsumexp_level1.cu
./ex1_logsumexp_level1
```
