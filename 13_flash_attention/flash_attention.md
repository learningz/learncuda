# 简化版 FlashAttention：Online Softmax 在硬件上的数据流

配合 `flash_attention.cu` 阅读。


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


## 练习题

| 练习 | 任务 | 核心考点 |
|------|------|---------|
| [ex1_block_mm_level1.cu](./exercises/ex1_block_mm_level1.cu) | 分块矩阵乘 (FlashAttn 子问题) | 从零写 tiled matmul（只填 kernel） |

```bash
nvcc -O2 -o ex1_block_mm_level1 ex1_block_mm_level1.cu
```
