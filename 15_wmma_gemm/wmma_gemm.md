# WMMA GEMM：Tensor Core 指令在硬件上的行为

配合 `wmma_gemm.cu` 阅读。


## WMMA 调用到 Tensor Core 指令的映射

```
代码:
  wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

编译后的 SASS:
  HMMA.16816.F32 R4, R0, R8, R4 ;    // Tensor Core MMA 指令!
  
  HMMA = Half-precision Matrix Multiply-Accumulate
  .16816 = M16 × N8 × K16 (一条指令处理这么大的矩阵块)
  .F32 = 累加器精度 FP32

  一条 HMMA 指令做了什么:
    D[16×8] = A[16×16] × B[16×8] + C[16×8]
    计算量: 16 × 8 × 16 × 2 = 4096 FLOP
    延迟: ~8 cycles
    → 4096 FLOP / 8 cycles = 512 FLOP/cycle (一条指令!)
    
    vs FP32 CUDA Core:
    FFMA 一条指令: 2 FLOP (1 mul + 1 add), 4 cycles
    → 0.5 FLOP/cycle (一条指令)
    
    Tensor Core 单指令吞吐是 CUDA Core 的 ~1000×!
```


## Fragment 在线程中的分布

```
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;

a_frag 不是一个普通数组! 它是"分散在 32 个线程寄存器中的矩阵":

  一个 16×16 FP16 矩阵 = 256 个 half = 512 bytes
  分给 32 个线程: 每线程 256/32 = 8 个 half = 4 个 uint32 寄存器

  哪个线程持有哪些元素? (Ampere m16n8k16):
    Thread 0:  A[0,0:1] A[0,2:3] A[8,0:1] A[8,2:3]
    Thread 1:  A[1,0:1] A[1,2:3] A[9,0:1] A[9,2:3]
    ...
    Thread 7:  A[7,0:1] A[7,2:3] A[15,0:1] A[15,2:3]
    Thread 8-15:  A 的 k=4..7 列部分
    Thread 16-23: A 的 k=8..11 列部分
    Thread 24-31: A 的 k=12..15 列部分

  你不需要手动管理这个映射!
  wmma::load_matrix_sync 自动将连续内存布局转换为 Fragment 布局。
  wmma::store_matrix_sync 自动将 Fragment 布局转换回连续内存。

  但理解这个映射对高级优化很重要:
    → ldmatrix 指令直接按 Fragment 布局从 SMEM 加载 (零额外 shuffle!)
    → Swizzle 模式就是为了让 SMEM 布局匹配 Fragment 布局, 消除 Bank Conflict
```


## FP16 输入 + FP32 累加的硬件原因

```
代码中:
  A, B 是 half (FP16, 16-bit)
  C, D 是 float (FP32, 32-bit)

为什么这样设计?

  FP16 的精度只有 ~3 位有效数字。
  16×16 矩阵乘中, 每个输出元素 = 16 次乘加:
    如果 FP16 累加: 每次加法都损失精度 → 16 次后误差可能很大
    如果 FP32 累加: 乘法用 FP16 (快), 但加法用 FP32 (准) → 精度接近 FP32

  Tensor Core 硬件内部:
    乘法器: FP16 × FP16 → FP32 的乘法结果 (不损失精度)
    加法器: FP32 + FP32 → FP32 的累加 (不损失精度)
    → 只有输入量化为 FP16 时有精度损失, 累加过程无损!

  这就是为什么 AI 训练的 "混合精度" 能工作:
    权重和激活用 FP16 → 省带宽和存储
    累加用 FP32 → 保持精度
    → Tensor Core 在硬件上直接支持这个模式
```


## 练习题

| 练习 | 任务 | 核心考点 |
|------|------|---------|
| [ex1_wmma_add_level1.cu](./exercises/ex1_wmma_add_level1.cu) | WMMA 矩阵加法 | 最简单的 fragment load/store（只填 kernel） |

```bash
nvcc -O2 -arch=sm_75 -o ex1_wmma_add_level1 ex1_wmma_add_level1.cu
```
