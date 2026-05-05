# Softmax 三版本：从代码到硬件

本文档配合 `softmax.cu` 阅读。目标：**读完能自己写出 3-pass、2-pass Online、Warp-level 三种 Softmax kernel**。

> **前置阅读**: 理解 Warp Shuffle 归约（[`03_reduce/`](../03_reduce/)）和 Shared Memory 用法（[`02_matrix_mul/`](../02_matrix_mul/)）。


## 1. Softmax 在算什么

```
公式:  softmax(x_i) = exp(x_i) / Σ_j exp(x_j)

例子:  输入  [2.0, 1.0, 0.1]
       exp   [7.39, 2.72, 1.11]
       sum   = 11.22
       输出  [0.659, 0.242, 0.099]  → 总和 = 1.0 ✓
```

把任意实数变成概率分布——输出全在 0~1 之间且总和为 1。大的值对应大的概率。

### 为什么不能直接按公式算

```
如果输入是 [100, 200, 300]:
  exp(100) ≈ 2.7 × 10^43
  exp(200) ≈ 7.2 × 10^86
  exp(300) ≈ 1.9 × 10^130    ← 远超 float 最大值 3.4 × 10^38!
  → +Inf / +Inf = NaN → 全完了!
```

**解决：先减最大值**。数学上等价（分子分母同时乘 exp(-max)）：

```
softmax(x_i) = exp(x_i - max) / Σ_j exp(x_j - max)

最大指数变成 exp(0) = 1，永远不会溢出。
```

### GPU 上实现 Softmax 需要什么

```
要算 softmax(x), 需要:
  1. 求 max(x)           ← 归约 (所有线程通信)
  2. 求 sum(exp(x-max))  ← 归约
  3. y[i] = exp(x[i]-max) / sum  ← elementwise

所以 Softmax = 2 次归约 + 1 次 elementwise。
归约 = 第 3 章的核心技术!
```

### Grid/Block 怎么配

```
输入 shape [rows, cols]

和 LayerNorm 一样: 每一行完全独立 → 每个 Block 处理一行。

  gridDim = rows    (有多少行就有多少个 Block)
  blockDim = 256    (一行内 256 个线程协作归约)

为什么 blockDim=256?
  → Softmax 用 Shared Memory 做归约
  → 需要 smem[blockDim] 存中间结果
  → 256 × 4B = 1KB → 很小, 完全没问题
  → 如果 blockDim=1024 → 4KB, 也可以但 Warp 更多 = 更多轮 BAR.SYNC
```


## 2. V1: 3-pass — 最直接的实现

### 完整代码

```cuda
__global__ void softmax_3pass(const float *input, float *output,
                               int rows, int cols) {
    extern __shared__ float smem[];  // 动态分配: 大小 = 2 * blockDim * sizeof(float)
    int row = blockIdx.x;            // 每个 Block 处理一行
    int tid = threadIdx.x;
    const float *x = input + row * cols;
    float *y = output + row * cols;

    // ---- Pass 1: 求每行的 max ----
    float local_max = -INFINITY;
    for (int i = tid; i < cols; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }

    // Shared Memory 归约求全局 max
    smem[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float max_val = smem[0];   // 所有线程现在都有全局 max
    __syncthreads();           // 等所有人读完 max_val

    // ---- Pass 2: 求 sum(exp(x - max)) ----
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        local_sum += expf(x[i] - max_val);
    }

    smem[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float sum_val = smem[0];
    __syncthreads();

    // ---- Pass 3: 归一化 ----
    for (int i = tid; i < cols; i += blockDim.x) {
        y[i] = expf(x[i] - max_val) / sum_val;
    }
}
```

### 逐段拆解

#### ① Shared Memory 声明 — `extern __shared__`

```cuda
extern __shared__ float smem[];
```

```
没写大小? → 大小在 host 端 launch kernel 时指定:
  size_t smem = 2 * blockSize * sizeof(float);
  softmax_3pass<<<rows, blockSize, smem>>>(d_in, d_out, rows, cols);
  //                               ^^^^ 第三个参数 = 动态 shared memory 大小

为什么是 2×?
  → Pass 1 用 smem[0..blockDim-1] 存 max 的部分值
  → Pass 2 用同样的空间存 sum 的部分值
  → V2 (online) 需要同时存 max 和 sum → 所以预留了 2×
```

#### ② Pass 1: 局部 max + 树形归约

```cuda
// Step A: 每个线程先算自己负责的那几个元素的最大值
float local_max = -INFINITY;
for (int i = tid; i < cols; i += blockDim.x) {
    local_max = fmaxf(local_max, x[i]);
}

// Step B: 所有线程把 local_max 写入 Shared Memory, 然后树形归约
smem[tid] = local_max;
__syncthreads();
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
    __syncthreads();
}
float max_val = smem[0];
```

```
Step B 的树形归约 (blockDim=8 为例):

初始: smem = [m0, m1, m2, m3, m4, m5, m6, m7]

s=4:  线程 0 读 smem[0]和[4] → max(m0,m4) → 写入 smem[0]
      线程 1 读 smem[1]和[5] → max(m1,m5) → 写入 smem[1]
      线程 2 读 smem[2]和[6] → max(m2,m6) → 写入 smem[2]
      线程 3 读 smem[3]和[7] → max(m3,m7) → 写入 smem[3]
      __syncthreads()

s=2:  线程 0: max(smem[0], smem[2]) → smem[0]
      线程 1: max(smem[1], smem[3]) → smem[1]
      __syncthreads()

s=1:  线程 0: max(smem[0], smem[1]) → smem[0]
      __syncthreads()

最终 smem[0] = max(m0..m7) = 整行的最大值! ✓

为什么每轮都要 __syncthreads()?
  → 线程 0 在 s=4 时读了 smem[2], 但线程 2 可能还没写完 smem[2]
  → 不等就错! 每轮归约后必须 barrier
```

#### ③ Pass 2: sum(exp(x-max)) — 同样的归约模式

和 Pass 1 完全相同的结构，只是 `fmaxf` 换成 `+`，`expf` 换 `-INFINITY` 初始化。

```
为什么累加 expf(x[i] - max_val) 而不是直接 expf(x[i])?
  → max_val 已经减去 → exp 不会溢出
  → 而且 max_val 是全局 max → 即使局部元素很大, max_val ≥ 它们
```

#### ④ Pass 3: 归一化

第三次遍历数据，每个线程对自己负责的元素做 `exp(x-max)/sum` 然后写回。

**V1 总计: 读 3N 次 + 写 N 次 = 4N 次显存访问。**


## 3. V2: 2-pass Online — 把前两次遍历合并

### 核心观察

Pass 1 和 Pass 2 都是遍历同一组数据做归约。能不能合并？

**不能直接合并** — 因为 Pass 2 需要 Pass 1 的 max_val 才能算 `exp(x-max)`。

但 Online 算法有一个技巧：**边扫描边修正**。

### 推导

```
已经扫描了 [3, 1, 7]:
  max = 7
  sum = exp(3-7) + exp(1-7) + exp(7-7) = 0.018 + 0.0025 + 1.0 = 1.0205

来了第 4 个数 10:
  new_max = 10  (max 更新了!)
  
  问题: 之前的 sum=1.0205 是基于 old_max=7 的, 需要修正成基于 new_max=10 的

  修正: new_sum = old_sum × exp(old_max - new_max) + exp(10 - new_max)
                = 1.0205 × exp(7-10) + exp(0)
                = 1.0205 × 0.0498 + 1.0
                = 1.0508

  验证: exp(3-10)+exp(1-10)+exp(7-10)+exp(10-10) = 0.0009+0.00012+0.0498+1.0 = 1.0508 ✓

为什么一个乘法就修正了所有旧项?
  exp(x_i - new_max) = exp(x_i - old_max) × exp(old_max - new_max)
  因子 exp(old_max - new_max) 对所有旧项都一样 → 整个 sum 乘一次即可!
```

### 完整代码

```cuda
__global__ void softmax_online(const float *input, float *output,
                                int rows, int cols) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float *x = input + row * cols;
    float *y = output + row * cols;

    // ---- Pass 1: Online 求 max 和 sum ----
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = x[i];
        float old_max = local_max;
        local_max = fmaxf(local_max, val);
        local_sum = local_sum * expf(old_max - local_max)  // 修正旧 sum
                  + expf(val - local_max);                  // 加上新项
    }

    // ---- Block 级 Online Reduce ----
    // smem 前 blockDim 存 max, 后 blockDim 存 sum
    smem[tid] = local_max;
    smem[tid + blockDim.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float m1 = smem[tid],     d1 = smem[tid + blockDim.x];
            float m2 = smem[tid + s], d2 = smem[tid + s + blockDim.x];
            float new_max = fmaxf(m1, m2);
            smem[tid] = new_max;
            // 合并两个 (max, sum) 对: 用同样的 online 修正公式
            smem[tid + blockDim.x] =
                d1 * expf(m1 - new_max) + d2 * expf(m2 - new_max);
        }
        __syncthreads();
    }
    float max_val = smem[0];
    float sum_val = smem[blockDim.x];
    __syncthreads();

    // ---- Pass 2: 归一化 ----
    for (int i = tid; i < cols; i += blockDim.x) {
        y[i] = expf(x[i] - max_val) / sum_val;
    }
}
```

### 逐段拆解

#### ① Online 累加 — 关键的三行代码

```cuda
float old_max = local_max;
local_max = fmaxf(local_max, val);
local_sum = local_sum * expf(old_max - local_max) + expf(val - local_max);
```

```
这句是整个算法的灵魂。分两种情况:

情况 A: val ≤ local_max → local_max 不变 → old_max == local_max
  → expf(old_max - local_max) = expf(0) = 1.0
  → local_sum = local_sum * 1.0 + expf(val - local_max)
  → 没修正, 正常累加 ✓

情况 B: val > local_max → local_max 更新 → new_max > old_max
  → expf(old_max - new_max) < 1.0  (修正因子, 缩小旧 sum)
  → local_sum = local_sum * 修正因子 + expf(val - new_max)
  → 旧项被修正到新基准, 新项直接按新基准加 ✓

注意: expf(old_max - local_max) 中 old_max - local_max ≤ 0
  → 修正因子 ≤ 1 → 不会溢出! (这是减 max 技巧的另一个好处)
```

#### ② Online Reduce — 合并两个 (max, sum) 对

```
归约树中, 每个节点不再是一个标量, 而是一对 (max, sum)。

s=某轮时, 线程 tid 要把 (m1, d1) 和 (m2, d2) 合并成一个 (new_max, new_sum):

             (new_max, new_sum)
                   │
          ┌────────┴────────┐
    (m1, d1)           (m2, d2)

合并公式 (和在线累加完全一样的逻辑):
  new_max = max(m1, m2)
  new_sum = d1 * exp(m1 - new_max) + d2 * exp(m2 - new_max)

  假设 m1 ≥ m2 → new_max = m1
    d1 * exp(m1 - m1) = d1 * 1.0 = d1  (不需要修正!)
    d2 * exp(m2 - m1) = d2 × 小于1的因子  (d2 从 m2 基准修正到 m1 基准)

  结果存在 smem[tid] (new_max) 和 smem[tid + blockDim.x] (new_sum)
```

#### ③ `smem[tid + blockDim.x]` — 用后半段存 sum

```
smem 布局:
  [0 .. blockDim-1]:          存 max 的归约中间值
  [blockDim .. 2*blockDim-1]: 存 sum 的归约中间值

这就是为什么 host 端分配了 2 * blockSize * sizeof(float):
  softmax_online<<<rows, blockSize, 2 * blockSize * sizeof(float)>>>
```

**V2 总计: 读 2N 次 + 写 N 次 = 3N 次显存访问** (比 V1 少读一遍!)


## 4. V3: Warp-level — 当一行 ≤ 32 个元素

当 cols ≤ 32 时, 一行只需要一个 Warp (32 线程) 就能处理。不需要 Shared Memory, 不需要 `__syncthreads()`!

### 完整代码

```cuda
// 辅助函数: Warp 级 max 归约
__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return __shfl_sync(0xffffffff, val, 0);  // lane 0 广播给所有人
}

// 辅助函数: Warp 级 sum 归约
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return __shfl_sync(0xffffffff, val, 0);
}

__global__ void softmax_warp(const float *input, float *output,
                              int rows, int cols) {
    // 一个 Warp (32 线程) 处理一行 → 需要自己算出负责哪一行
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= rows) return;

    const float *x = input + warp_id * cols;
    float *y = output + warp_id * cols;

    // 每个 lane 持有一行中的一个元素 (如果 cols < 32, 多余的 lane 填 -INF/0)
    float val = (lane < cols) ? x[lane] : -INFINITY;

    float m = warp_reduce_max(val);                     // Warp 级 max 归约
    float e = (lane < cols) ? expf(val - m) : 0.0f;
    float s = warp_reduce_sum(e);                       // Warp 级 sum 归约
    if (lane < cols) y[lane] = e / s;
}
```

### 逐段拆解

#### ① Warp 到行的映射 — 不再是一个 Block 一行

```cuda
int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
int lane = threadIdx.x % 32;
```

```
这里 grid 和 block 的配置变了:

  warps_per_block = 8;  // blockDim=256 → 8 个 Warp
  grid = ceil(rows / warps_per_block);

  例如 rows=32768, warps_per_block=8:
    grid = 4096 个 Block
    每个 Block 有 8 个 Warp → 8 个 Warp × 4096 Blocks = 32768 Warp
    → 每个 Warp 处理一行 → 刚好!

  warp_id 的计算:
    Block 0, Thread 0..31:    warp_id = (0*256 + 0)/32   = 0
    Block 0, Thread 32..63:   warp_id = (0*256 + 32)/32  = 1
    ...
    Block 0, Thread 224..255: warp_id = (0*256 + 224)/32 = 7
    Block 1, Thread 0..31:    warp_id = (1*256 + 0)/32   = 8
    ...

  同一个 Block 内的 8 个 Warp 处理 8 个不同的行!
  这是 Warp-level kernel 的关键: 一个 Block 可以有多个 Warp, 各干各的。
```

#### ② `warp_reduce_max` — 寄存器归约

```cuda
__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return __shfl_sync(0xffffffff, val, 0);
}
```

```
和 03_reduce 中的 warp_reduce_sum 完全一样的结构, 换成了 fmaxf。

但返回值不同!
  return __shfl_sync(0xffffffff, val, 0);
  → 从 lane 0 广播给所有 32 个 lane
  → 所有 lane 都得到全局 max (不只是 lane 0)

为什么 sum 版本也广播?
  → 归一化时每个 lane 都需要 sum 做除法
  → 广播省了一次 Shared Memory 写入
```

#### ③ 为什么没有 `__syncthreads()`

```
一个 Block 有 8 个 Warp, 但每个 Warp 处理不同的行 → 互不干扰!
同一 Warp 内所有 32 个线程同步执行 (SIMT) → 不需要显式 barrier。
→ 零同步开销!
```

**V3 总计: 读 N 次 + 写 N 次 = 2N 次显存访问, 零 `__syncthreads()`。**


## 5. 三版本指令和硬件路径概要

```
V1 (3-pass):
  LDG × 3N + STG × N = 4N 显存操作
  2 × 树形 SMEM 归约 (每轮都 __syncthreads)
  → 瓶颈: HBM 带宽 (算术强度 < 1)

V2 (2-pass Online):
  LDG × 2N + STG × N = 3N 显存操作  (比 V1 少 25%)
  1 × Online SMEM 归约 (同时归约 max 和 sum)
  多了一些 MUFU.EX2 (exp) 指令 → 但多几条 ALU 的代价 << 省一遍 HBM 读
  → 瓶颈: 仍然是 HBM 带宽, 但更少的数据搬运 → ~1.3× 加速

V3 (Warp-level, N ≤ 32):
  LDG × N + STG × N = 2N
  全用 SHFL → 延迟 ~1 cycle (vs SMEM ~5 cycles)
  零 BAR.SYNC → 零 Stall Barrier
  → 瓶颈: 变成指令发射 (不再是 Memory Bound, 因为数据量太小)
```

### 什么时候用哪个

```
cols > 32, 对正确性要求很高 → V1 (代码最简单, 不容易出错)
cols > 32, 追求性能        → V2 (少一次遍历)
cols ≤ 32                  → V3 (Warp-level, 极致效率)

实际场景:
  Transformer 的 QKV projection  (cols=64~128)   → V1 或 V2
  Attention weights 的每行 softmax (cols=seq_len) → 短序列用 V3, 长序列用 V2
  Classification logits (cols=1000~10000)         → V1 或 V2
```


## 6. Softmax 的反向传播 (Backward Pass)

### 数学推导

```
前向: y_i = exp(x_i - max) / sum,  sum = Σ exp(x_j - max)

给定上游梯度 dy (∂L/∂y), 求 dx (∂L/∂x):

Softmax 的 Jacobian:
  ∂y_i/∂x_j = y_i × (δ_ij - y_j)
  其中 δ_ij = 1 if i==j else 0

链式法则:
  dx_i = Σ_j dy_j × ∂y_j/∂x_i
       = Σ_j dy_j × y_j × (δ_ji - y_i)
       = dy_i × y_i - Σ_j dy_j × y_j × y_i
       = y_i × dy_i - y_i × dot(y, dy)
       = y_i × (dy_i - dot(y, dy))

直觉:
  dy_i 本身                              ← 直接: 改变 x_i 直接影响 y_i
  - y_i × dot(y, dy)                     ← 间接: 改变 x_i 也通过分母影响所有 y_j

反向 kernel 结构:
  Pass 1: 求 dot(y, dy) = Σ y_j × dy_j → Warp Shuffle 归约 → 1 个标量
  Pass 2: dx_i = y_i × (dy_i - dot_val) → elementwise → 写回

实现要点:
  → 需要前向的 y → 前向保存 y (save_for_backward)
  → 或者重算 y (recompute): 需要 x 和 max/sum → 重算 exp 和 div
```


## 7. 常见错误

- **忘了减 max** → 症状: 输入中有 >88 的值时输出全是 NaN。`exp(89)` = 溢出。减 max 是必须的, 不是优化。
- **Online 修正因子写反** → 症状: `sum * expf(new_max - old_max)` 而不是 `sum * expf(old_max - new_max)`。如果 new_max > old_max, sum 被指数放大 → 结果爆炸。
- **Online Reduce 中忘了同时修正两个 sum** → `d1 * exp(m1-new) + d2 * exp(m2-new)`. 两个 d 都需要修正到 new_max 基准, 不是只修正大的那个!
- **归约树每轮后忘了 `__syncthreads()`** → 症状: 结果不稳定/随机。线程 A 在读 smem 的同时线程 B 在写。
- **Warp-level 中每 Warp 处理一行, 但 grid/block 没配对** → 症状: 只处理了部分行。grid = ceil(rows / warps_per_block)。


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
