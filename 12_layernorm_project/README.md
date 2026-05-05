# LayerNorm 实战：从 CUDA Kernel 到 PyTorch 接入

配合 `layernorm_cuda.cu` 和 `layernorm_starter.cu` 阅读。

> **前置阅读**: 理解 Warp Shuffle 归约（[`03_reduce/`](../03_reduce/)）、Shared Memory（[`02_matrix_mul/`](../02_matrix_mul/)）和 PyTorch C++ 扩展（[`04_pytorch_extension/`](../04_pytorch_extension/)）。


## 1. LayerNorm 在算什么

```
输入:  x    shape [rows, cols]   例如 rows=1024, cols=768
参数:  gamma shape [cols]       可学习的缩放参数
      beta  shape [cols]       可学习的偏移参数

对每一行独立做归一化:

  mean = (1/cols) × Σ x[i]
  var  = (1/cols) × Σ (x[i] - mean)²

  y[i] = gamma[i] × (x[i] - mean) / sqrt(var + eps) + beta[i]

其中 eps = 1e-5，防止除零。
```

关键观察：**每一行完全独立——行与行之间不需要通信。** 这让 GPU 并行非常自然：每个 Block 处理一行，所有行并行处理。


## 2. 每个 Block 处理一行 — Grid/Block 是怎么定的

```cuda
int block_size = 256;
layernorm_forward_kernel<<<rows, block_size>>>(...);
//                         ↑ grid   ↑ blockDim
```

```
假设 rows=1024, cols=768, blockDim=256:

  gridDim = 1024 个 Block   (每个 Block 独立处理一行)
  blockDim = 256 个线程      (每行内 256 个线程协作)

  Block 0   → 处理第 0 行
  Block 1   → 处理第 1 行
  ...
  Block 1023 → 处理第 1023 行

每个 Block 内部:
  256 个线程协作计算这一行的 mean 和 variance（需要归约）
  然后 256 个线程各自归一化自己负责的元素

为什么 blockDim=256?
  → 32 的倍数（Warp 大小）、不超过 1024（硬件限制）
  → 256 = 8 个 Warp → 归约只需 2 级（Warp 内 + Warp 间），代码清晰
  → 如果 cols=768，每个线程处理 ceil(768/256) = 3 个元素（Grid-Stride Loop）
```

关键问题：**为什么 mean 和 variance 需要 256 个线程协作？**
因为 mean = Σx/cols，每个线程只看到了自己负责的那几个 x[i]，需要把所有线程的部分和汇总起来。这就是"归约"（Reduce）——第三章的核心技术。


## 3. Forward Kernel — 完整代码逐段拆解

先看全貌，再看每一段做什么：

```cuda
__global__ void layernorm_forward_kernel(
    const float *x, float *y, const float *gamma, const float *beta,
    float *mean_out, float *rstd_out,
    int rows, int cols, float eps)
{
    int row = blockIdx.x;                         // ① 我负责哪一行
    int tid = threadIdx.x;
    const float *xr = x + row * cols;             // 指向本行起始
    float *yr = y + row * cols;

    // ---- Pass 1: 计算 mean 和 variance ----
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {  // ② Grid-Stride Loop
        float val = xr[i];
        local_sum += val;
        local_sq_sum += val * val;
    }

    // ③ Warp 级归约
    local_sum = warp_reduce_sum(local_sum);
    local_sq_sum = warp_reduce_sum(local_sq_sum);

    // ④ Warp 间归约 (Shared Memory)
    __shared__ float s_sum[32], s_sq[32];
    int warp_id = tid / 32, lane = tid % 32;
    if (lane == 0) {
        s_sum[warp_id] = local_sum;
        s_sq[warp_id] = local_sq_sum;
    }
    __syncthreads();

    int num_warps = blockDim.x / 32;
    if (warp_id == 0) {
        local_sum = (lane < num_warps) ? s_sum[lane] : 0.0f;
        local_sq_sum = (lane < num_warps) ? s_sq[lane] : 0.0f;
        local_sum = warp_reduce_sum(local_sum);
        local_sq_sum = warp_reduce_sum(local_sq_sum);
    }

    // ⑤ 计算 mean 和 rstd，广播给所有线程
    __shared__ float s_mean, s_rstd;
    if (tid == 0) {
        float mean = local_sum / cols;
        float var = local_sq_sum / cols - mean * mean;
        float rstd = rsqrtf(var + eps);
        s_mean = mean;
        s_rstd = rstd;
        mean_out[row] = mean;   // 保存给反向传播
        rstd_out[row] = rstd;
    }
    __syncthreads();

    float mean = s_mean;
    float rstd = s_rstd;

    // ---- Pass 2: 逐元素归一化 ----
    for (int i = tid; i < cols; i += blockDim.x) {  // ⑥ 归一化
        yr[i] = gamma[i] * (xr[i] - mean) * rstd + beta[i];
    }
}
```

### ① 线程 → 行的映射

```cuda
int row = blockIdx.x;
const float *xr = x + row * cols;
```

```
这就是前面说的：blockIdx.x 直接对应行号。

Block 0 的 256 个线程 → 全部处理第 0 行
Block 1 的 256 个线程 → 全部处理第 1 行
...

同一 Block 的 256 个线程通过 threadIdx.x 分工：
  线程 0   负责 col 0, 256, 512, ...   (Grid-Stride)
  线程 1   负责 col 1, 257, 513, ...
  ...
  线程 255 负责 col 255, 511, 767
```

### ② Grid-Stride Loop — 每个线程处理多个元素

```cuda
for (int i = tid; i < cols; i += blockDim.x) {
    float val = xr[i];
    local_sum += val;
    local_sq_sum += val * val;
}
```

```
cols=768, blockDim=256 → 每线程处理 3 个元素:

  线程 0:   xr[0],   xr[256],   xr[512]
  线程 1:   xr[1],   xr[257],   xr[513]
  ...
  线程 255: xr[255], xr[511],   xr[767]

每个线程累加出两个局部统计量:
  local_sum    = Σ x[i]      (这 3 个元素的和)
  local_sq_sum = Σ x[i]²     (这 3 个元素的平方和)

为什么同时累加 sum 和 sum_of_squares?
  → 只需一次遍历!
  → mean = sum / N
  → var  = E[x²] - E[x]² = (sq_sum/N) - (sum/N)²
  → 不需要先算 mean 再回头算 var（那样就是两次遍历了）

注意: 这不是 Welford 算法（Welford 是逐元素更新，不需要存 sum 和 sq_sum）。
对于 FP32，用 sum + sum_of_squares 公式完全够用。
Welford 的优势在 FP16（避免精度损失），见 7.2 节的理论文档。
```

### ③ Warp 级归约 — `warp_reduce_sum`

```cuda
local_sum = warp_reduce_sum(local_sum);
```

`warp_reduce_sum` 的定义：

```cuda
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
```

```
以 1 个 Warp (32 线程) 为例, 假设各线程的 local_sum 为 [v0..v31]:

  offset=16: 线程 0..15 收到 线程 16..31 的值
    lane 0:  val = v0  + v16
    lane 1:  val = v1  + v17
    ...
    lane 15: val = v15 + v31

  offset=8:  线程 0..7 收到 线程 8..15 的值
    lane 0:  val = (v0+v16) + (v8+v24)
    ...

  offset=4 → offset=2 → offset=1

  最后: lane 0 的 val = v0+v1+...+v31 (所有 32 个线程的和)
       其他 lane 的 val = 部分和（不完整）

  为什么用 __shfl_down_sync 而不是 Shared Memory?
    → shuffle 是寄存器直接交换，延迟 ~1 cycle
    → Shared Memory 需要写 → __syncthreads → 读，延迟 ~5 cycles
    → shuffle 快 5× 且不需要 barrier!

  执行完之后:
    每个 Warp 的 lane 0 持有该 Warp 32 个线程的部分和
    其他 lane 的值是中间结果（后续不再使用）
```

### ④ Warp 间归约 — 用 Shared Memory 汇总 8 个 Warp

```
256 线程 = 8 个 Warp。上一步得到了 8 个部分和（每个 Warp 的 lane 0 持有）。
还需要把这 8 个部分和加起来得到整行的总和。
```

```cuda
__shared__ float s_sum[32], s_sq[32];
int warp_id = tid / 32;    // 0..7  (我在第几个 Warp)
int lane = tid % 32;       // 0..31 (我在 Warp 内的第几个 lane)

// 每个 Warp 的 lane 0 把部分和写入 Shared Memory
if (lane == 0) {
    s_sum[warp_id] = local_sum;
    s_sq[warp_id] = local_sq_sum;
}
__syncthreads();  // ← 等 8 个 Warp 都写完!

// 第一个 Warp (warp_id==0) 的 32 个线程读出这些部分和，再做一次 shuffle 归约
int num_warps = blockDim.x / 32;
if (warp_id == 0) {
    local_sum = (lane < num_warps) ? s_sum[lane] : 0.0f;
    local_sq_sum = (lane < num_warps) ? s_sq[lane] : 0.0f;
    local_sum = warp_reduce_sum(local_sum);
    local_sq_sum = warp_reduce_sum(local_sq_sum);
}
```

```
时间线 — 8 个 Warp 各自完成自己的 shuffle 后:

  Warp 0: lane0 有 [第 0..31  号线程的和] → 写入 s_sum[0]
  Warp 1: lane0 有 [第 32..63 号线程的和] → 写入 s_sum[1]
  Warp 2: lane0 有 [第 64..95 号线程的和] → 写入 s_sum[2]
  ...
  Warp 7: lane0 有 [第 224..255 号线程的和] → 写入 s_sum[7]

  __syncthreads()  ← 确保 8 个都写完了

  然后 Warp 0 的 32 个线程读 s_sum[0..7]:
    lane 0 → s_sum[0], lane 1 → s_sum[1], ..., lane 7 → s_sum[7]
    lane 8..31 → 0.0f (只有 8 个 Warp)

  再做一次 warp_reduce_sum:
    lane 0 的 local_sum = s_sum[0]+s_sum[1]+...+s_sum[7]
                        = 整行所有 768 个元素的总和! ✓

为什么只让 Warp 0 做?
  → 只需要一个 Warp 汇总就够了
  → 其他 7 个 Warp 可以闲置（它们后续从 Shared Memory 读结果就行）
```

### ⑤ 计算 mean 和 rstd，广播

```cuda
__shared__ float s_mean, s_rstd;
if (tid == 0) {
    float mean = local_sum / cols;
    float var = local_sq_sum / cols - mean * mean;
    float rstd = rsqrtf(var + eps);    // rsqrt = 1/sqrt, 硬件单条指令
    s_mean = mean;
    s_rstd = rstd;
    mean_out[row] = mean;    // 保存！反向传播要用
    rstd_out[row] = rstd;
}
__syncthreads();

float mean = s_mean;    // 所有线程从 Shared Memory 读到同一个值
float rstd = s_rstd;
```

```
只有 tid==0（Warp 0 的 lane 0）持有完整的总和。
它计算出 mean 和 rstd，写入 Shared Memory。
__syncthreads() 后，所有 256 个线程都能读到。

var 的计算：
  var = E[x²] - E[x]² = sq_sum/N - (sum/N)²

  为什么不用 Σ(x-mean)²/N？
    → 那需要先知道 mean 才能算 → 需要两遍遍历
    → E[x²] - E[x]² 可以一遍遍历同时累加 sum 和 sq_sum

  数值稳定性？
    → FP32 下 E[x²]-E[x]² 对 LayerNorm 的典型数据范围完全够用
    → 极端情况（mean 很大、var 很小）才需要 Welford
```

### ⑥ 逐元素归一化

```cuda
for (int i = tid; i < cols; i += blockDim.x) {
    yr[i] = gamma[i] * (xr[i] - mean) * rstd + beta[i];
}
```

```
又是 Grid-Stride Loop——每线程处理 3 个元素。

以 cols=768, blockDim=256 为例:
  线程 0:   y[0]   = gamma[0]   * (x[0]   - mean) * rstd + beta[0]
             y[256] = gamma[256] * (x[256] - mean) * rstd + beta[256]
             y[512] = gamma[512] * (x[512] - mean) * rstd + beta[512]
  线程 1:   类似处理 col 1, 257, 513
  ...
```

**整个 kernel 遍历了数据 2 次**（Pass 1 算统计量 + Pass 2 归一化），这是 LayerNorm 的理论下限——必须先知道 mean 和 var 才能归一化。


## 4. 用一组小例子走一遍完整流程

```
rows=2, cols=8, blockDim=4 (故意用小的方便看)

输入 x:
  row 0: [1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]
  row 1: [0.0, 2.0, 4.0, 6.0, 1.0, 3.0, 5.0, 7.0]

gamma = [1, 1, 1, 1, 1, 1, 1, 1], beta = [0, 0, 0, 0, 0, 0, 0, 0], eps = 0

以 Block 0 (处理 row 0) 为例，4 个线程：

=== Pass 1: Grid-Stride Loop (cols=8, blockDim=4 → 每线程 2 个) ===

  线程 0: 读 x[0]=1.0, x[4]=2.0  → local_sum=3.0,   local_sq_sum=1+4=5.0
  线程 1: 读 x[1]=3.0, x[5]=4.0  → local_sum=7.0,   local_sq_sum=9+16=25.0
  线程 2: 读 x[2]=5.0, x[6]=6.0  → local_sum=11.0,  local_sq_sum=25+36=61.0
  线程 3: 读 x[3]=7.0, x[7]=8.0  → local_sum=15.0,  local_sq_sum=49+64=113.0

=== Warp 级归约 (4 线程, 1 个 Warp) ===

  offset=2: 线程 0 从线程 2 拿: sum=3.0+11.0=14.0,   sq=5.0+61.0=66.0
            线程 1 从线程 3 拿: sum=7.0+15.0=22.0,   sq=25.0+113.0=138.0
  offset=1: 线程 0 从线程 1 拿: sum=14.0+22.0=36.0,  sq=66.0+138.0=204.0

  lane 0: total_sum=36.0, total_sq=204.0  ← 校验: 1+3+5+7+2+4+6+8=36 ✓

=== 计算 mean 和 rstd ===

  mean = 36.0 / 8 = 4.5
  var  = 204.0 / 8 - 4.5² = 25.5 - 20.25 = 5.25
  rstd = 1 / sqrt(5.25) = 1 / 2.291 = 0.4364

=== Pass 2: 归一化 ===

  线程 0: y[0] = (1.0 - 4.5) * 0.4364 = -1.527
          y[4] = (2.0 - 4.5) * 0.4364 = -1.091
  线程 1: y[1] = (3.0 - 4.5) * 0.4364 = -0.655
          y[5] = (4.0 - 4.5) * 0.4364 = -0.218
  线程 2: y[2] = (5.0 - 4.5) * 0.4364 =  0.218
          y[6] = (6.0 - 4.5) * 0.4364 =  0.655
  线程 3: y[3] = (7.0 - 4.5) * 0.4364 =  1.091
          y[7] = (8.0 - 4.5) * 0.4364 =  1.527

  校验: mean(y_row0) ≈ 0.0, std(y_row0) ≈ 1.0 ✓ (归一化成功)
```


## 5. 反向传播 Kernel

### 5.1 数学推导

```
前向 (对每一行):
  xhat[i] = (x[i] - mean) * rstd
  y[i] = gamma[i] * xhat[i] + beta[i]

给定上游梯度 dy (∂L/∂y)，求：
  dx    (∂L/∂x, 传给前一层)
  dgamma (∂L/∂gamma, 更新参数)
  dbeta  (∂L/∂beta,  更新参数)

dgamma 和 dbeta 很简单：
  dgamma[i] = Σ_rows dy[row][i] * xhat[row][i]     ← 跨所有行累加!
  dbeta[i]  = Σ_rows dy[row][i]

dx 的推导 (链式法则, 略去中间步骤):
  dx[i] = rstd * gamma[i] * (dy[i] - mean_dy - xhat[i] * mean_dy_xhat)

  其中:
    mean_dy      = (1/cols) * Σ_j dy[j] * gamma[j]
    mean_dy_xhat = (1/cols) * Σ_j dy[j] * gamma[j] * xhat[j]

直觉:
  dy[i] 本身                                    ← 直接梯度
  - mean_dy                                     ← 因为 mean 变了会影响所有 y
  - xhat[i] * mean_dy_xhat                      ← 因为 var (即 rstd) 变了会影响所有 y

  三项加起来才是完整的 dx[i]。
```

### 5.2 Kernel 代码逐段拆解

```cuda
__global__ void layernorm_backward_kernel(
    const float *dy, const float *x, const float *gamma,
    const float *mean_saved, const float *rstd_saved,
    float *dx, float *dgamma, float *dbeta,
    int rows, int cols)
{
    int row = blockIdx.x;          // 和前向一样, 一 Block 一行
    int tid = threadIdx.x;

    float mean = mean_saved[row];  // 从 ctx.save_for_backward 恢复
    float rstd = rstd_saved[row];

    const float *dy_row = dy + row * cols;
    const float *x_row = x + row * cols;
    float *dx_row = dx + row * cols;
```

和前向完全相同的 Grid/Block 映射：一 Block 一行。

#### 第一步 — 计算两个中间归约量

```cuda
    float local_sum_dy = 0.0f;
    float local_sum_dy_xhat = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float xhat = (x_row[i] - mean) * rstd;
        float dy_gamma = dy_row[i] * gamma[i];
        local_sum_dy += dy_gamma;
        local_sum_dy_xhat += dy_gamma * xhat;
    }
```

```
和前向 Pass 1 一样: Grid-Stride Loop 累加两个标量。
需要一次遍历, 同时累加 Σ dy*gamma 和 Σ dy*gamma*xhat。
```

#### 第二步 — Block 级归约 (和前向完全相同)

```cuda
    local_sum_dy = warp_reduce_sum(local_sum_dy);
    local_sum_dy_xhat = warp_reduce_sum(local_sum_dy_xhat);

    __shared__ float s1[32], s2[32];
    int warp_id = tid / 32, lane = tid % 32;
    if (lane == 0) { s1[warp_id] = local_sum_dy; s2[warp_id] = local_sum_dy_xhat; }
    __syncthreads();

    int num_warps = blockDim.x / 32;
    if (warp_id == 0) {
        local_sum_dy = (lane < num_warps) ? s1[lane] : 0.0f;
        local_sum_dy_xhat = (lane < num_warps) ? s2[lane] : 0.0f;
        local_sum_dy = warp_reduce_sum(local_sum_dy);
        local_sum_dy_xhat = warp_reduce_sum(local_sum_dy_xhat);
    }

    __shared__ float ss_dy, ss_dy_xhat;
    if (tid == 0) { ss_dy = local_sum_dy; ss_dy_xhat = local_sum_dy_xhat; }
    __syncthreads();
```

```
和前向的归约模式一摸一样，只是这次归约的是两个不同的量。
归约完成后，ss_dy 和 ss_dy_xhat 里是该行的 Σ dy*gamma 和 Σ dy*gamma*xhat。
```

#### 第三步 — 计算 dx, dgamma, dbeta

```cuda
    float inv_cols = 1.0f / cols;

    for (int i = tid; i < cols; i += blockDim.x) {
        float xhat = (x_row[i] - mean) * rstd;

        dx_row[i] = rstd * gamma[i] *
            (dy_row[i] - inv_cols * ss_dy - inv_cols * xhat * ss_dy_xhat);

        atomicAdd(&dgamma[i], dy_row[i] * xhat);
        atomicAdd(&dbeta[i], dy_row[i]);
    }
}
```

```
dx 公式: rstd * gamma[i] * (dy[i] - mean_dy - xhat[i] * mean_dy_xhat)

  其中 mean_dy = ss_dy / cols = (Σ dy*gamma) / N
       mean_dy_xhat = ss_dy_xhat / cols = (Σ dy*gamma*xhat) / N

dgamma/dbeta 为什么用 atomicAdd?
  → gamma 和 beta 是 [cols] 形状，但输入是 [rows, cols]
  → 每行都贡献该行对 dgamma[i] 的梯度: dy[row][i] * xhat[row][i]
  → 多个 Block 同时写同一个 dgamma[i] → 数据竞争!
  → atomicAdd 保证累加正确

  性能: atomicAdd 有开销，但对 LayerNorm 这种 Memory Bound 的 kernel 影响小于 5%。
  如果极致优化可以用两阶段归约（先 Block 内归约再 atomicAdd 一次），
  但代码复杂度显著增加，对教学不友好。
```


## 6. PyTorch 接入 — 三段式桥接

`layernorm_cuda.cu` 包含三段代码：

```
┌─ CUDA Kernel ─────────────────┐
│ layernorm_forward_kernel      │  ← 在 GPU 上跑的 kernel
│ layernorm_backward_kernel     │
└──────────┬────────────────────┘
           │ 被 C++ 函数调用
┌──────────▼────────────────────┐
│ layernorm_forward()           │  ← C++ 封装: 校验输入、分配输出、调 kernel
│ layernorm_backward()          │
└──────────┬────────────────────┘
           │ 被 pybind11 暴露
┌──────────▼────────────────────┐
│ PYBIND11_MODULE               │  ← 注册为 Python 可调用的函数
│   m.def("forward", ...)       │
│   m.def("backward", ...)      │
└──────────┬────────────────────┘
           │ Python import
┌──────────▼────────────────────┐
│ import custom_layernorm       │  ← Python 中直接调用
│ custom_layernorm.forward(...) │
└───────────────────────────────┘
```

C++ 封装做三件事：

```cpp
std::vector<torch::Tensor> layernorm_forward(
    torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, float eps)
{
    // 1. 输入校验
    TORCH_CHECK(x.is_cuda(), "x 必须在 GPU 上");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "目前只支持 float32");
    TORCH_CHECK(x.is_contiguous(), "x 必须是 contiguous");

    int rows = x.size(0), cols = x.size(1);

    // 2. 分配输出 tensor — PyTorch 内部调用 cudaMalloc
    auto y = torch::empty_like(x);
    auto mean = torch::empty({rows}, x.options());
    auto rstd = torch::empty({rows}, x.options());

    // 3. Launch kernel
    int block_size = 256;
    layernorm_forward_kernel<<<rows, block_size>>>(
        x.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        y.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
        rows, cols, eps);

    return {y, mean, rstd};  // mean 和 rstd 会被 ctx.save_for_backward 保存
}
```

Python 侧（`test_layernorm.py`）用 `torch.autograd.Function` 包装：

```python
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, eps):
        y, mean, rstd = custom_layernorm.forward(x, gamma, beta, eps)
        ctx.save_for_backward(x, gamma, mean, rstd)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, gamma, mean, rstd = ctx.saved_tensors
        dx, dgamma, dbeta = custom_layernorm.backward(
            grad_output.contiguous(), x, gamma, mean, rstd)
        return dx, dgamma, dbeta, None  # eps 不需要梯度
```

## 7. 性能分析 — LayerNorm 是 Memory Bound

```
LayerNorm 的算术强度:

  前向 FLOP ≈ 5 × cols (每元素: 1减 + 1乘 + 1加 = 3, 加上 mean/var 计算的 2)
  Byte ≈ 2 × cols × 4B (读 x 和 gamma, 写 y)

  忽略 gamma/beta（每行重复用，从 L2 命中）:
    AI ≈ 5 / 8 ≈ 0.6 FLOP/Byte → 极端 Memory Bound!

ncu 分析要点:
  → SOL Memory% 应该接近 100%（瓶颈在带宽，不在计算）
  → Warp Stall Reasons 以 Long Scoreboard 为主（等内存）
  → 优化方向: float4 向量化加载 → 减少指令数 → 让 LD/ST pipeline 更高效
```


## 练习题

在项目目录下：

```bash
# 编译 PyTorch 扩展
pip install -e .

# 运行完整测试 (正确性 + 梯度 + 性能)
python test_layernorm.py
```

如果你还没准备好写完整的 PyTorch 接入，可以先从纯 CUDA starter 开始：

```bash
nvcc -O2 -o layernorm_starter layernorm_starter.cu
./layernorm_starter
# 填空完成 kernel，通过正确性检查
```

也可以做 exercises 目录下的独立练习：

| 练习 | 任务 | 核心考点 |
|------|------|---------|
| [ex1_layernorm_level1.cu](./exercises/ex1_layernorm_level1.cu) | 单行统计量归约 | Warp Shuffle + SMEM 两阶段归约求 mean/rstd（只填 kernel） |

## 常见错误

- **归约后只用 lane 0 的值，但没有 `if (lane == 0)` 保护** → 症状: mean/variance 完全错误。其他 lane 的 shuffle 结果是中间值，不能直接写入 Shared Memory
- **忘了 `__syncthreads()` 在 Warp 间归约前** → 症状: 第一个 Warp 可能在别的 Warp 写完 Shared Memory 前就读了 → 读到旧数据或 0
- **`rsqrtf(var + eps)` 忘了加 eps** → 症状: var=0 时（全行相同）输出全是 NaN
- **`E[x²] - E[x]²` 在大数下精度不够** → 症状: 输入值在 10^6 量级时 var 可能算成负数（因为 sq_sum/N 和 mean² 两个大数相减抵消）。修复: 用 Welford 算法或先减去 mean 估计值
- **dgamma/dbeta 调用前没清零** → 症状: 每次 backward 结果不同，梯度不断累加。需要在 Python 端 `torch.zeros_like(gamma)` 或 kernel 开头 memset
- **`blockDim.x` 超过 `cols`** → 症状: 部分线程空闲是 OK 的（Grid-Stride Loop 的 `i < cols` 条件保护），但如果 blockDim 远大于 cols（如 blockDim=256 但 cols=32），只有 32 个线程干活，其余 224 个浪费。考虑减小 blockDim 或让多个 Block 处理同一行
