# 简化版 FlashAttention：Online Softmax 在硬件上的数据流

配合 `flash_attention.cu` 阅读。

**难度**: ⭐⭐⭐ 专家
**前置知识**: Online Softmax 算法（[`07_softmax/`](../07_softmax/)）；Shared Memory Tiling（[`02_matrix_mul/`](../02_matrix_mul/)）
**读完你能做什么**: 理解 FlashAttention 如何通过 Tiling + Online Softmax + Recomputation 消除 N×N 中间矩阵的显存瓶颈


## 什么是 Attention (注意力机制)

### 从直觉到公式

Attention 是 Transformer (GPT, BERT, LLaMA 等) 的核心操作。

直觉：你在读一个句子，读到"它"这个词时，
你的大脑会自动回去找"它"指代的是什么——可能是前面的"猫"。
Attention 就是让模型做同样的事：对于每个位置，
回头看所有其他位置，算出"应该关注哪里"，然后把信息加权取出来。

```
数学公式: O = softmax(Q × K^T / √d) × V

  Q (Query):  [N, d]  → "我在找什么" (每个位置发出的查询)
  K (Key):    [N, d]  → "我有什么"  (每个位置的标签)
  V (Value):  [N, d]  → "我的内容"  (每个位置的实际信息)

  Q × K^T = [N, N] → "每个位置和其他位置的匹配度"
    → 这是一个 N×N 矩阵！每一行 = 某个位置对所有位置的关注程度
  
  softmax(每一行) → 归一化为概率 (总和=1)
    → 现在每一行是"注意力权重"
  
  × V → 按权重取出内容
    → 权重大的位置贡献多, 权重小的贡献少
```

### 标准 Attention 的问题：N×N 矩阵太大

```
N = 序列长度 (一句话的 token 数)

N=512:   N×N = 26 万  → 1 MB   → 没问题
N=4096:  N×N = 1600 万 → 64 MB  → 开始吃力
N=32768: N×N = 10 亿   → 4 GB   → 放不下!
N=131072 (GPT-4 级别): N×N = 64 GB → 远超显存!

而且这个 N×N 矩阵不只是存着占空间, 还要:
  1. 写入显存 (Q×K^T 的结果)
  2. 从显存读出来做 Softmax
  3. Softmax 结果写回显存
  4. 再从显存读出来乘以 V
  → 至少 3 次读写 → 在 Memory Bound 的瓶颈下, 带宽耗尽!
```


### FlashAttention 的核心思想：分块 + Online Softmax

FlashAttention 的关键洞察：**N×N 矩阵不需要完整存在！**

```
标准做法: 算完整个 N×N, 存下来, 再做 Softmax, 再乘 V
  → 必须存 N×N → 内存爆炸

FlashAttention: 把 Q, K, V 分成小块, 一块一块算
  每次只算 N×N 矩阵的一小部分 (比如 64×64)
  这一小块在寄存器/Shared Memory 中短暂存在, 用完就扔
  → N×N 矩阵从未完整存在于显存中!
```

但这引出一个问题：**Softmax 需要看完一整行才能算**（因为要先求 max 和 sum），
如果一行被切成了很多块，每次只看到一小块，怎么算 Softmax？

答案就是 **Online Softmax**——和 07_softmax 中 V2 用的完全一样的修正因子技巧！

```
处理第 1 块 K_1:
  计算 S_1 = Q × K_1^T / √d        (小矩阵, 在寄存器中)
  m_1 = max(S_1)                     (这块的局部 max)
  l_1 = sum(exp(S_1 - m_1))         (这块的局部 sum)
  o_1 = exp(S_1 - m_1) / l_1 × V_1  (用这块的局部 Softmax 加权)

处理第 2 块 K_2:
  计算 S_2 = Q × K_2^T / √d
  m_2 = max(S_2)
  
  new_max = max(m_1, m_2)           (全局 max 更新了!)
  
  修正之前的结果:
    l_1 要修正: l_1 = l_1 × exp(m_1 - new_max)     ← 旧基准→新基准
    o_1 也要修正: o_1 = o_1 × exp(m_1 - new_max)   ← 同样的因子!
  
  加上新块的贡献:
    l_2 = sum(exp(S_2 - new_max))
    o_2 = exp(S_2 - new_max) × V_2
  
  合并:
    l = l_1 + l_2
    o = o_1 + o_2
  
  最后归一化: O = o / l

关键:
  - 修正因子 exp(m_1 - new_max) 和 Softmax V2 里的一模一样!
  - 不同的是, FlashAttention 不仅修正 sum (l), 还修正输出向量 (o)
  - 这就是为什么 tutorial.md 说 "V2 的修正因子是 FlashAttention 的秘密武器"
```

**效果**：
```
标准 Attention:
  额外显存: N×N × 4B (S 矩阵) + N×N × 4B (P 矩阵) = 2 × N² × 4B
  HBM 读写: ~5 × N² × 4B
  
FlashAttention:
  额外显存: N × 4B (每行的 m 和 l) ≈ 几乎为 0
  HBM 读写: ~4 × N × d × 4B (只需读写 Q/K/V/O)
  
  N=4096, d=64:
    标准: HBM 访问 ~320 MB, 额外内存 ~128 MB
    Flash: HBM 访问 ~4 MB,  额外内存 ~32 KB
    → HBM 访问减少 80 倍! 额外内存减少 4000 倍!
    → 又快又省, 这在优化中是非常罕见的双赢!
```


## 标准 Attention vs FlashAttention 的硬件数据流对比

```
标准 Attention (O = softmax(QK^T/√d) × V):

  Step 1: S = Q × K^T         GPU 计算 GEMM, 结果写入 HBM (N×N 矩阵!)
             ↓ 写 HBM
  Step 2: P = softmax(S)       GPU 从 HBM 读 S, 计算 softmax, 结果写回 HBM
             ↓ 写 HBM
  Step 3: O = P × V            GPU 从 HBM 读 P 和 V, 计算 GEMM, 写 HBM
  
  S 和 P 各 N×N: N=4096 时 = 64MB → 必须存在 HBM → 3 次 HBM 全量读写!

FlashAttention:

  外循环: for j = 0 to N/Bc:    遍历 K/V 的块
    加载 K_j (Bc×d) 到寄存器      从 HBM 读 1 次
    加载 V_j (Bc×d) 到寄存器      从 HBM 读 1 次
    
    内循环: for i = 0 to N/Br:   遍历 Q 的块
      加载 Q_i (Br×d) 到寄存器    从 HBM 读 1 次
      
      S_ij = Q_i × K_j^T / √d   在寄存器中! (小矩阵 Br×Bc, 不写 HBM)
      
      Online Softmax 更新:        在寄存器中! (只维护 m, l, o)
        m_new = max(m_old, max(S_ij))
        o = o × exp(m_old - m_new) + exp(S_ij - m_new) × V_j   ← 关键!
        l = l × exp(m_old - m_new) + sum(exp(S_ij - m_new))
      
      写回 O_i = o / l            到 HBM, 1 次写
  
  S 矩阵 (N×N) 从未完整存在! 只有 Br×Bc 的小块在寄存器中短暂存在。
```


## Br 和 Bc 的选择 — 为什么通常是 64 或 128?

FlashAttention 最重要的两个超参数: Br (Q 的分块大小) 和 Bc (K/V 的分块大小).
它们的选择直接决定了 register 用量、SMEM 用量和性能。

```
约束条件 (以 FP16 + A100 SMEM=164KB 为例):

1. Register 约束 (每线程最多 255 个 32-bit 寄存器):
   内循环中需要的寄存器:
     Q_i 块: Br × d 个 half (存在寄存器中作为 fragment)
     K_j 块: Bc × d 个 half
     V_j 块: Bc × d 个 half
     S_ij 累加器: Br × Bc 个 float (累加器必须 FP32, 即使是 FP16 输入)
     O_i 累加器: Br × d 个 float
     l/m 标量: Br 个 float
   
   每个线程只持有这些数据的一部分 (Warp 内 32 线程分摊).
   以 Br=Bc=64, d=64, WMMA 为例:
     每线程持有的寄存器片段:
       Q_i:   (64×64) / 32 = 128 half = 64 float 寄存器
       S_ij:  (64×64) / 32 = 128 float 寄存器
       O_i:   (64×64) / 32 = 128 float 寄存器
       K_j:   同上 ~64 float
       总计: ~400 寄存器/线程 ← 接近 255 的上限!
   
   → 如果 Br=Bc=128, 寄存器需求 ~4× → 一定溢出 → 性能崩溃.

2. SMEM 约束:
   每个 Block 需要的 SMEM:
     Q_i  tile: Br × d × 2B (FP16) = 64×64×2 = 8 KB
     K_j  tile: Bc × d × 2B = 8 KB
     V_j  tile: Bc × d × 2B = 8 KB
     其他 (l, m, D): ~1 KB
     总计: ~25 KB
   
   → 远小于 A100 的 164KB → SMEM 不是瓶颈!
   → 真正决定 Br/Bc 大小的是寄存器.

3. Occupancy 约束:
   400 寄存器/线程 → 每 Warp 需要 400×32 = 12800 寄存器
   向上取整到分配粒度 (256): ceil(12800/256)×256 = 12800
   每 SM 65536 寄存器 → 最多 5 Warp/SM → 5/64 = 7.8% Occupancy!
   
   → 这是 FlashAttention 的典型 Occupancy (很低!)
   → 但没关系, 因为它是 Compute Bound — 低 Occupancy 被高 ILP 弥补.

推导过程:
  max_rf_per_thread = 255
  rf_per_thread ≈ (Br×d + Bc×d + Br×Bc + Br×d) / 32 × 2  (FP16 in, FP32 accum)
  
  对于 d=64: rf ≈ (64×64 + 64×64 + 64×64 + 64×64) / 32 × 2 = 512 × 2 / 32 = ...
  → 简化: rf ≈ Br × d / 32 × 4 + Bc × d / 32 × 2 + Br × Bc / 32
  
  试算:
    Br=Bc=64, d=64:  rf ≈ 128 + 64 + 128 = ~320 → 接近极限, 可行
    Br=Bc=128, d=64: rf ≈ 256 + 128 + 512 = ~900 → 远超 255 → 必然 spill!
    Br=128, Bc=64, d=128: rf ≈ 512 + 128 + 256 = ~900 → 也不可行
  
  → 这就是为什么 FlashAttention 的分块大小不会很大.
  → 实践中 Br=Bc=64 或 Br=128, Bc=64 是最常见的配置.
  → FlashAttention-2 通过交换内外循环, 改善了寄存器分配, 但分块大小仍然受此约束.
```

## FlashAttention 的反向传播 — 概述

> 完整的 FlashAttention 反向传播推导见 [`theory/07_classic_operators.md`](../theory/07_classic_operators.md) §7.5.

```
反向传播的核心挑战:

  前向: O = softmax(QK^T/√d) × V
    只保存了 O (输出), lse (log-sum-exp), m (max) → O(Nd) 显存
    没有保存 P = softmax(QK^T/√d) → O(N²) 的矩阵!
  
  反向: 给定 dO (loss 对 O 的梯度), 求 dQ, dK, dV.
    需要 P 才能计算 dV = P^T × dO 和 dQ = dS × K
    
  解决: 在前向只存 lse_i, 反向时重新计算 S 和 P
    P_ij = exp(S_ij - lse_i)  ← 用保存的 lse 重建 P, 不需要存 N×N 矩阵!
    
  反向的计算量: ~5 次 GEMM (前向只需 2 次)
    dP = dO × V^T
    dS = P ⊙ (dP - rowsum(dO ⊙ O))  ← Softmax 反向
    dQ = dS × K / √d
    dK = dS^T × Q / √d
    dV = P^T × dO
  
  但因为仍然不需要存 P, 显存节省 ~N× vs 标准 Attention.
  → 训练时可以跑更长的序列!
```

## "修正因子" 的硬件操作

```
Online Softmax 中, 当 max 更新时:
  o = o × exp(m_old - m_new) + new_contribution

这在 SASS 级别是:

  FSUB R_diff, R_m_old, R_m_new ;      // m_old - m_new (负值)
  FMUL R_diff, R_diff, R_LOG2E ;       // × log2(e) (为 EX2 准备)
  MUFU.EX2 R_correction, R_diff ;      // exp(m_old - m_new)  ← SFU, ~28 cycles
  
  // 修正 d 维度的 o 向量 (d=64 → 64 次乘法)
  FMUL R_o[0], R_o[0], R_correction ;  // o[0] *= correction
  FMUL R_o[1], R_o[1], R_correction ;  // o[1] *= correction
  ... (64 次 FMUL, 可以流水线执行)
  
  // 加上新贡献
  MUFU.EX2 R_p, R_s_minus_m ;          // exp(s - m_new) for 新块
  FFMA R_o[0], R_p, R_v[0], R_o[0] ;   // o[0] += p × V[0]
  ... (64 次 FFMA)

  总共 ~64+64 = 128 条 FP 指令 per Q-K 对, 外加 2 条 MUFU。
  这些全在寄存器中执行, 不经过 Shared Memory 或 HBM!
```


## 内存节省的硬件层面解释

```
N = 4096, d = 64, float32:

标准 Attention:
  S = N×N × 4B = 64 MB   → 存在 HBM
  P = N×N × 4B = 64 MB   → 存在 HBM
  总额外 HBM: 128 MB
  
  HBM 读写次数: 读 S + 写 S + 读 S + 写 P + 读 P = 5 × 64MB = 320 MB

FlashAttention:
  m = N × 4B = 16 KB     → 存在 HBM (每行 1 个 max)
  l = N × 4B = 16 KB     → 存在 HBM (每行 1 个 sum)
  总额外 HBM: 32 KB
  
  HBM 读写: 读 Q + 读 K + 读 V + 写 O ≈ 4 × N × d × 4B = 4 MB
  
  HBM 减少: 320 MB → 4 MB ≈ 80×!
  额外内存: 128 MB → 32 KB ≈ 4000×!
  
  这就是为什么 FlashAttention:
    1. 速度快 (HBM 带宽是瓶颈, 访问量减少 80×)
    2. 省内存 (可以处理更长的序列)
    3. 两者同时实现 (很少有优化能同时改善速度和内存!)
```


## 常见错误

- **修正因子只乘了 l (sum), 忘了乘 o (output)** → 症状: 当 max 更新时, 旧的 o 没有修正, 新块的 o 和旧块的 o 不在同一数值基准上 → 结果局部正确但整体错误。`o *= correction` 和 `l *= correction` 必须同时做!
- **修正因子方向写反** → 症状: `exp(new_max - old_max)` 而不是 `exp(old_max - new_max)` → 如果 new_max > old_max, 修正因子 > 1, 旧结果被指数放大 → output 爆炸成 Inf/NaN
- **Q×K^T 忘了除以 √d** → 症状: 点积值太大, softmax 输出变成 one-hot (只有一个位置为 1, 其余为 0) → 梯度消失, 模型不收敛。`scale = 1.0f / sqrtf(d)` 不是可选的优化, 是必须的数值稳定步骤
- **分块大小超过 Shared Memory 上限** → 症状: 编译错误或运行时 cudaErrorInvalidConfiguration。每块需要 Br×d + Bc×d + Br×Bc 的 SMEM。典型配置: Br=Bc=64, d=64 → 每块约 20KB, 安全
- **内外循环搞反** → 症状: 正确性没问题但性能差。外层应该遍历 K/V (写 O 的循环), 内层遍历 Q (读 Q 的循环) → 这样 O 的中间结果在寄存器中, 不用频繁写 HBM
- **最后忘了除以 l 做归一化** → 症状: o 的数值量级完全错误。FlashAttention 全程维护的是未归一化的 o (加权和) 和 l (权重和), 最后一轮循环结束后必须 `O = o / l`


## 练习题

| 练习 | 任务 | 核心考点 |
|------|------|---------|
| [ex1_block_mm_level1.cu](./exercises/ex1_block_mm_level1.cu) | 分块矩阵乘 (FlashAttn 子问题) | 从零写 tiled matmul（只填 kernel） |

```bash
nvcc -O2 -o ex1_block_mm_level1 ex1_block_mm_level1.cu
```
