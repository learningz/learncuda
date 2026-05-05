# 练习题参考答案

> **重要**: 先自己做完再看答案! 答案是用来验证的, 不是用来抄的。
> 如果你卡住了, 先看每题的"提示", 再看配合的理论章节, 最后才看答案。


## Ch2 练习题答案

### 练习 1: blockSize 对性能的影响

```
blockSize=32:  gridSize=32768, 性能一般 (每 Block 只有 1 个 Warp, SM 利用率低)
blockSize=128: gridSize=8192,  通常接近最优
blockSize=256: gridSize=4096,  通常最优或接近
blockSize=512: gridSize=2048,  可能因寄存器压力导致 Occupancy 降低
blockSize=1024: gridSize=1024, 可能只有 1-2 个 Block/SM, 灵活性差

blockSize=100: 可以运行! 但最后一个 Warp 只有 100%32=4 个活跃线程,
  其余 28 个浪费。总浪费比例: 28/128 = 22% 的线程空转。
  → 所以 blockSize 应该总是 32 的倍数。
```

### 练习 2: 2D Grid 索引

```
Block(0,0) Thread(0,0): row = 0*16+0 = 0, col = 0*16+0 = 0 ✓

为什么 threadIdx.x 对应列?
  Warp 的 32 个线程的 threadIdx.x 是连续的 (0,1,2,...31)。
  让 threadIdx.x 对应列 → 同一 Warp 的线程访问 B 矩阵的连续列 →
  内存地址连续 → 合并访问 → 1 次事务而不是 32 次!
  
  如果反过来 (threadIdx.x 对应行):
  同一 Warp 的线程访问 B 的不同行 → 地址相隔 N×4 字节 → 完全不合并!
```


## Ch3 练习题答案

### 练习 1: 合并访问探索

```
stride=4 的有效带宽 ≈ stride=1 的 25%:
  32 线程 × 4B, stride=4 → 地址跨越 32×4×4=512B → 4 条 Cache Line
  vs stride=1 的 128B = 1 条 Cache Line
  → 传输量多 4× → 有效带宽 ÷ 4

float4 的有效带宽通常比 float 高 5-15%:
  LDG.128 vs LDG.32 → 指令数减少 4× → LD/ST pipeline 压力降低
  但传输的数据总量相同 → 带宽提升来自指令效率, 不是数据效率。

ncu Global Load Efficiency:
  stride=1: ~100% (所有传输字节都是有用的)
  stride=4: ~25%  (每条 CL 只用了 1/4)
  随机:     ~3-12% (大多数 CL 只有 1 个有用元素)
```

### 练习 2: Bank Conflict 消除

```
stride=3: GCD(3, 32) = 1 → 无 Bank Conflict! ✓
  (奇数 stride 通常没问题, 因为 32 是偶数, 奇数和偶数互素)

32×32 矩阵转置:
  __shared__ float tile[32][32];
  写: tile[threadIdx.y][threadIdx.x] → 按行写, 无冲突 (tx 连续 → 不同 Bank)
  读: tile[threadIdx.x][threadIdx.y] → 按列读, 32-way 冲突!
      (tx 连续但 y 相同 → 地址间隔 32×4=128B → Bank 0 冲突)
  
  加 padding: __shared__ float tile[32][33];
  现在每行 33 个 float = 132B
  列读: 地址间隔 33×4B → Bank 间隔 = 33%32=1 → 连续 Bank → 无冲突! ✓
```

### 练习 3: Roofline 分析

```
向量加法:
  AI = 1 FLOP / 12 Byte = 0.083 FLOP/Byte
  远低于 A100 Ridge Point (9.6 FLOP/Byte) → 强 Memory Bound
  
  理论最短时间 = 12MB / 2039 GB/s = 5.9 μs
  如果实测 ~7 μs → 效率 = 5.9/7 = 84% → 不错!
```


## Ch4 练习题答案

### 练习 1: 分支分歧

```
threadIdx.x % 4: 4 条路径, 但编译器很可能谓词化短分支。
  如果每条路径只有 1-2 条指令 → 被谓词化 → 几乎无分歧代价。
  如果每条路径 10+ 条指令 → 真正的分支跳转 → 代价明显。

加长路径 (10 次乘法) 后, 分歧代价更明显:
  无分歧: ~T cycles (1 条路径)
  50% 分歧: ~2T cycles (2 条路径串行)
  → 加速比接近 2×, 而短路径时可能只有 1.05×
```

### 练习 2: Warp Shuffle 归约

```
__shfl_xor_sync 的蝶形归约:
  val += __shfl_xor_sync(0xffffffff, val, 16);  // lane i ↔ lane i^16
  val += __shfl_xor_sync(0xffffffff, val, 8);   // lane i ↔ lane i^8
  val += __shfl_xor_sync(0xffffffff, val, 4);
  val += __shfl_xor_sync(0xffffffff, val, 2);
  val += __shfl_xor_sync(0xffffffff, val, 1);
  
  和 __shfl_down 的区别: xor 让所有 32 个线程都得到最终结果!
  (down 只有 lane 0 有完整的和)
  
warp_reduce_max: 把 += 换成 fmaxf:
  val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 16));
  // ... 同理 ...
```


## Ch5 练习题答案

### 练习 3: 判断瓶颈类型

```
(a) 向量加法: AI = 1/12 = 0.083 → Memory Bound (远低于 9.6)
(b) GELU:     AI = 15/8 = 1.875 → Memory Bound (仍低于 9.6)
(c) GEMM 1024: AI = 2×1024³ / (3×1024²×4) = 2048/12 = 170.7 → Compute Bound!

结论: 几乎所有 elementwise/reduce 算子都是 Memory Bound。
      只有 GEMM (和大卷积) 在矩阵足够大时是 Compute Bound。
```


## Ch7 练习题答案

### 练习 1: Softmax 数值稳定性

```
输入 x[i] = 100.0 + rand:
  不减 max: exp(100) = 2.7e43 → exp(200) = INF → NaN!
  减 max:   exp(0) ~ exp(-10) → 正常

float vs double:
  V1 和 V2 在 float 下精度应该完全一致 (数学上等价)。
  但如果用 double 参考, 两者的误差也应该一致 (都是 float 精度的限制)。
```

### 练习 3: FlashAttention 核心思考题

```
关键: 在 Online Softmax 中, 你维护了 running_max (m) 和 running_sum (l)。
当 m 更新时, l 乘以修正因子 exp(m_old - m_new)。

现在加一个 running_output (o):
  o = Σ P[i,:] × V[i,:]  的部分和

当处理新的 K/V block 时:
  m_new = max(m_old, max(S_new_block))
  correction = exp(m_old - m_new)
  
  l = l * correction + sum(exp(S_new - m_new))        ← sum 修正
  o = o * correction + exp(S_new - m_new) × V_block   ← output 也修正!

最终: O = o / l  (归一化)

这就是 FlashAttention! 完整代码见 [`13_flash_attention/flash_attention.cu`](./13_flash_attention/flash_attention.cu)
```


## Ch6 练习题答案

### 练习 1: WMMA 基础

```
完整 kernel 代码框架:

#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

__global__ void wmma_16x16(half *A, half *B, float *C) {
    // 声明 fragment — 这些不是普通数组，是分散在 32 个线程寄存器中的矩阵块
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>              c_frag;

    // 清零累加器 — 每个线程只清自己持有的那部分
    wmma::fill_fragment(c_frag, 0.0f);

    // 从全局内存加载到 fragment
    // 第二个参数是 leading dimension（行优先时 = 列数 = 16）
    // 32 个线程协作完成加载，每个线程自动搬运自己负责的元素
    wmma::load_matrix_sync(a_frag, A, 16);
    wmma::load_matrix_sync(b_frag, B, 16);

    // Tensor Core 矩阵乘加: C = A × B + C
    // 编译后变成 HMMA.16816 SASS 指令，一条指令完成 16×16×16 的乘加！
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // 将结果写回全局内存
    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}

// launch: wmma_16x16<<<1, 32>>>(d_A, d_B, d_C);
//         只需要 1 个 Warp (32 线程)!
//         不需要 1024 线程 — Tensor Core 的并行度在指令级别

CPU 参考验证:
  for (i in 0..15)
    for (j in 0..15)
      ref[i][j] = sum(A[i][k] * B[k][j] for k in 0..15)

  注意 half 精度有限 (~3 位有效数字):
  如果 A[i][j] = (float)(i+j)，最大值 = 30.0
  乘积最大 = 30 × 30 = 900，累加 16 次最大 ~14400
  half 能表示到 65504，这个范围没问题

  但如果矩阵值更大（如 i*j），累加可能溢出 half 范围
  → 这就是为什么累加器用 float！

常见错误:
  1. blockDim 不是 32 → WMMA 要求恰好一个完整 Warp 协作
  2. leading dimension 写错 → fragment 内数据错位
  3. 忘了 fill_fragment → 累加器初始值是垃圾
```

### 练习 2: TF32 的影响

```
典型输出 (A100):

  最大差异: ~0.001 (千分之一级别)
  相对误差: ~0.0001 (万分之一级别)

为什么有差异?
  FP32:  1 + 8 + 23 = 32 bit (23 bit 尾数, ~7 位有效数字)
  TF32:  1 + 8 + 10 = 19 bit (10 bit 尾数, ~3 位有效数字)

  TF32 对每个输入元素截断了 13 bit 尾数精度
  但累加仍然用 FP32 → 最终误差主要来自输入截断

  具体误差分析:
    单次乘法的相对误差 ≈ 2^(-10) ≈ 0.001
    累加 1024 次 (K=1024) → 误差可能累积
    但由于正负抵消，实际误差通常 << 理论最大值
    实测相对误差 ~0.01% → 对 DL 训练完全可以接受

速度差异:
  TF32 开启: ~快 2-4× (取决于矩阵大小和 GPU)
  原因: TF32 使用 Tensor Core，FP32 使用 CUDA Core
  Tensor Core 吞吐是 CUDA Core 的 ~8× (Ampere)

对深度学习的影响:
  训练收敛性: 几乎无影响 (SGD 本身就有随机性，0.01% 误差远小于梯度噪声)
  推理精度: 极少数场景可能需要关闭 (如科学计算)
  实际建议: 默认开启，只在遇到精度问题时才关闭

  这就是 NVIDIA 在 Ampere+ 上默认开启 TF32 的理由：
  用几乎不可感知的精度代价，换来数倍的计算加速。
```


## Ch8 练习题答案

### 练习 1: ILP 实验

```
版本 A (无 ILP) 的指令依赖链:

  LDG R0, [input+idx]    // 加载 a，延迟 ~300 cycles (L2 命中) 或 ~500 cycles (HBM)
  --- 等待 300+ cycles ---
  FMUL R1, R0, 2.0       // b = a * 2.0，依赖 R0，等加载完才能发射
  --- 等待 4 cycles ---
  FADD R2, R1, 1.0       // c = b + 1.0，依赖 R1
  --- 等待 4 cycles ---
  STG [output+idx], R2   // 存储 c

  每个元素的指令延迟: ~300 + 4 + 4 = ~308 cycles
  Warp 调度器可以通过切换到其他 Warp 来隐藏延迟
  但如果 Warp 数量不够（低 Occupancy），就隐藏不完 → 性能差

版本 B (4路 ILP) 的指令调度:

  LDG R0, [input+idx]           // 发射加载 0
  LDG R1, [input+idx+stride]    // 立刻发射加载 1 (不依赖 R0!)
  LDG R2, [input+idx+stride*2]  // 立刻发射加载 2
  LDG R3, [input+idx+stride*3]  // 立刻发射加载 3
  // 4 个加载同时在路上！只等最慢那个 = ~300 cycles (而不是 4×300)

  FMUL+FADD R0, ...   // 处理第 0 个
  FMUL+FADD R1, ...   // 处理第 1 个 (第 0 个的 FMA 在流水线中，无需等待)
  FMUL+FADD R2, ...
  FMUL+FADD R3, ...
  STG ×4

  4 个元素的总延迟 ≈ 300 + 4×(4+4) ≈ 332 cycles
  平均每元素: ~83 cycles (vs 版本 A 的 ~308 cycles)

预期结果:
  版本 B 的有效带宽 ≈ 版本 A 的 1.5~3× (取决于 Occupancy)
  高 Occupancy 时差距较小（足够的 Warp 已经能隐藏延迟）
  低 Occupancy 时差距巨大（ILP 是唯一的延迟隐藏手段）

  实测技巧:
    故意降低 blockSize (如 32) 减少 Occupancy → ILP 差距更明显
    blockSize=256 + 足够多 Block → 差距可能只有 10-20%
    → ILP 和 Occupancy 是互补的延迟隐藏手段！

代价:
  版本 B 用了 4× 的寄存器 → 可能降低 Occupancy
  → 寄存器和 ILP 之间存在权衡，需要实测找最优展开因子
  → 通常 2~8 路展开是甜点，太多反而变慢
```

### 练习 2: 算子融合对比

```
版本 A (3 个独立 kernel) 的数据流:

  relu_kernel:   读 x (N×4B) → 写 tmp1 (N×4B)     = 2N×4 字节的 HBM 访问
  scale_kernel:  读 tmp1     → 写 tmp2               = 2N×4 字节
  bias_kernel:   读 tmp2     → 写 out                = 2N×4 字节
  ─────────────────────────────────────────────
  总 HBM 访问: 6N×4 = 24N 字节
  额外开销: 2 次 kernel launch (~5μs each) + 2 个中间 buffer 的显存

版本 B (融合 kernel) 的数据流:

  fused_kernel:  读 x → 计算 relu+scale+bias → 写 out  = 2N×4 字节
  ───────────────────────────────────────────────
  总 HBM 访问: 2N×4 = 8N 字节
  节省: 3× 的 HBM 访问 + 0 次额外 launch + 0 个中间 buffer

版本 C (融合 + float4):

  __global__ void fused_vec4(float4 *x, float4 *out, float scale, float bias, int n4) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < n4) {
          float4 v = x[idx];          // 一条 LDG.128 加载 4 个 float
          v.x = fmaxf(v.x, 0.0f) * scale + bias;  // ReLU + Scale + Bias
          v.y = fmaxf(v.y, 0.0f) * scale + bias;
          v.z = fmaxf(v.z, 0.0f) * scale + bias;
          v.w = fmaxf(v.w, 0.0f) * scale + bias;
          out[idx] = v;               // 一条 STG.128 存储 4 个 float
      }
  }

  和版本 B 相比: 数据传输量相同 (8N 字节)
  但指令数减少 4× (LDG.128 vs 4×LDG.32) → LD/ST 指令 pipeline 压力更低
  → 通常额外提速 5-15%

预期的性能对比 (N = 16M, A100):
  版本 A: ~0.12ms  (有效带宽 ~1000 GB/s, 因为大量无用读写)
  版本 B: ~0.04ms  (有效带宽 ~1600 GB/s, 接近峰值)
  版本 C: ~0.035ms (有效带宽 ~1830 GB/s, 逼近硬件上限)

  加速比: A→B ≈ 3×, B→C ≈ 1.15×, A→C ≈ 3.4×

核心教训:
  Memory Bound 算子的优化空间主要在"减少 HBM 访问次数"
  融合 = 把中间结果留在寄存器（0 延迟）而不是写回 HBM（500 周期延迟）
  向量化 = 减少指令数，让硬件更高效地利用已有的带宽
  两者正交，可以叠加
```


## 补充答案: Ch4 练习 3

### 练习 3: 手写 __syncthreads 死锁

```
为什么 Divergent __syncthreads 会死锁?

__syncthreads() 是 Block 级别的屏障: Block 内所有线程必须都到达这个点才能继续。

代码问题:
  if (threadIdx.x % 2 == 0) {
      do_work();
      __syncthreads();   // 只有偶数号到达!
  } else {
      do_other_work();
      __syncthreads();   // 只有奇数号到达!
  }

死锁原因:
  线程 0 (偶数): 进 if → do_work() → __syncthreads() → 等待其余 255 个线程
  线程 1 (奇数):  进 else → do_other_work() → __syncthreads() → 等待其余 255 个线程
  → 线程 0 永不到达 else 的 __syncthreads(), 线程 1 永不到达 if 的 __syncthreads()
  → 死锁! GPU 不会超时, 程序永久挂起

正确做法:
  if (threadIdx.x % 2 == 0) { do_work(); }
  else { do_other_work(); }
  __syncthreads();  // 所有线程都会到达 ✓
```


## 补充答案: Ch5 练习 1、2

### 练习 1: GELU 的 Roofline 分析

```
GELU: y = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
  读: 4 bytes, 写: 4 bytes
  计算: tanh(多项式) + 几次乘加 ≈ 15 FLOP
  AI = 15 / 8 ≈ 1.9 FLOP/Byte

A100 Ridge Point ≈ 10 FLOP/Byte
  AI(GELU) = 1.9 << 10 → 深处 Memory Bound

理论最大 = 1.9 × 2000 GB/s = 3800 GFLOPS
  vs FP32 peak 19500 GFLOPS → 利用率上限 ≈ 19%

优化: 算子融合 + float4 向量化 + FP16/BF16
不要优化计算! 计算已经远超内存能力
```

### 练习 2: GELU + Dropout 算子融合

```
未融合: gelu (读x写tmp) + dropout (读tmp写out)
  HBM: 4N×4 = 16N bytes

融合: gelu+dropout (读x读mask写out)
  HBM: 3N×4 = 12N bytes
  加速比 ≈ 16/12 ≈ 1.3×

融合 kernel:
  __global__ void gelu_dropout_fused(const float *x, const uint8_t *mask,
      float *out, float scale, int N) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i >= N) return;
      if (mask[i]) {
          float v = x[i];
          float cdf = 0.5f*(1+tanhf(0.79788456f*(v+0.044715f*v*v*v)));
          out[i] = v * cdf * scale;
      } else out[i] = 0;
  }
```
```


## 补充答案: Ch7 练习 2

### 练习 2: Warp-level Softmax 变种 (N=64)

```
挑战: N=64 > 32, 每线程处理 2 元素

__device__ void warp_softmax_64(float *x, int lane) {
    float v0 = x[lane], v1 = x[lane + 32];

    // Step 1: 蝴蝶交换找 max (shfl_xor: 所有 lane 都能参与)
    float m = fmaxf(v0, v1);
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 16));
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 8));
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 4));
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));

    // Step 2: exp sum
    float s = expf(v0 - m) + expf(v1 - m);
    s += __shfl_xor_sync(0xffffffff, s, 16);
    s += __shfl_xor_sync(0xffffffff, s, 8);
    s += __shfl_xor_sync(0xffffffff, s, 4);
    s += __shfl_xor_sync(0xffffffff, s, 2);
    s += __shfl_xor_sync(0xffffffff, s, 1);

    // Step 3: 归一化
    x[lane] = expf(v0 - m) / s;
    x[lane + 32] = expf(v1 - m) / s;
}

扩展: N=128 → 每线程 4 元素 (寄存器限制)
纯 Shuffle 方案 = 0 次 __syncthreads → 比 Block+Shared Memory 方案更快
```

## 补充答案: Ch4 练习 3

### 练习 3: 手写 __syncthreads 死锁

__syncthreads() 是 Block 级别的屏障: Block 内所有线程必须都到达这个点才能继续。

代码问题:
  if (threadIdx.x % 2 == 0) {
      do_work();
      __syncthreads();   // 只有偶数号到达!
  } else {
      do_other_work();
      __syncthreads();   // 只有奇数号到达!
  }

死锁原因:
  线程 0 (偶数): 进 if -> do_work() -> __syncthreads() -> 等待其余 255 个线程
  线程 1 (奇数):  进 else -> do_other_work() -> __syncthreads() -> 等待其余 255 个线程
  -> 线程 0 永不到达 else 的 __syncthreads(), 线程 1 永不到达 if 的 __syncthreads()
  -> 死锁! 程序永久挂起

正确做法:
  if (threadIdx.x % 2 == 0) { do_work(); }
  else { do_other_work(); }
  __syncthreads();  // 所有线程都会到达


## 补充答案: Ch5 练习 1、2

### 练习 1: GELU 的 Roofline 分析

GELU: y = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
  读: 4 bytes, 写: 4 bytes, 计算: ~15 FLOP
  AI = 15 / 8 = 1.9 FLOP/Byte

A100 Ridge Point = 10 FLOP/Byte
  AI(GELU) = 1.9 << 10 -> Memory Bound

理论最大性能 = 1.9 x 2000 GB/s = 3800 GFLOPS
  vs FP32 peak 19500 GFLOPS -> 利用率上限 19%

优化: 融合 + float4 + FP16/BF16 (不要优化计算!)

### 练习 2: GELU + Dropout 算子融合

未融合: gelu(读x写tmp) + dropout(读tmp写out) -> HBM: 16N bytes
融合: gelu+dropout(读x读mask写out) -> HBM: 12N bytes
加速比: 16/12 = 1.3x

融合 kernel:
  __global__ void gelu_dropout_fused(const float *x, const uint8_t *mask,
      float *out, float scale, int N) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i >= N) return;
      if (mask[i]) {
          float v = x[i];
          float cdf = 0.5f*(1+tanhf(0.79788456f*(v+0.044715f*v*v*v)));
          out[i] = v * cdf * scale;
      } else out[i] = 0;
  }


## 补充答案: Ch7 练习 2

### 练习 2: Warp-level Softmax 变种 (N=64)

挑战: N=64 > 32, 每线程处理 2 元素

__device__ void warp_softmax_64(float *x, int lane) {
    float v0 = x[lane], v1 = x[lane + 32];
    float m = fmaxf(v0, v1);
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 16));
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 8));
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 4));
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
    float s = expf(v0 - m) + expf(v1 - m);
    s += __shfl_xor_sync(0xffffffff, s, 16);
    s += __shfl_xor_sync(0xffffffff, s, 8);
    s += __shfl_xor_sync(0xffffffff, s, 4);
    s += __shfl_xor_sync(0xffffffff, s, 2);
    s += __shfl_xor_sync(0xffffffff, s, 1);
    x[lane] = expf(v0 - m) / s;
    x[lane + 32] = expf(v1 - m) / s;
}

扩展: N=128 -> 每线程 4 元素. 纯 Shuffle = 零同步 -> 比 Block+SMEM 更快
