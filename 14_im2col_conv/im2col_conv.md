# im2col Convolution：把卷积变成矩阵乘法

配合 `im2col_conv.cu` 阅读。

**难度**: ⭐⭐ 进阶
**前置知识**: 矩阵乘法（[`02_matrix_mul/`](../02_matrix_mul/)）；合并访问（[`06_coalescing/`](../06_coalescing/)）
**读完你能做什么**: 能自己写出 im2col kernel，理解 index 反算的推导过程


## 1. 卷积在算什么，以及直接写 kernel 为什么难

### 卷积的基本操作

```
输入: 5×5 单通道图像      卷积核: 3×3
┌───┬───┬───┬───┬───┐     ┌───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │ 5 │     │ 1 │ 0 │-1 │
├───┼───┼───┼───┼───┤     ├───┼───┼───┤
│ 6 │ 7 │ 8 │ 9 │10 │     │ 1 │ 0 │-1 │
├───┼───┼───┼───┼───┤     ├───┼───┼───┤
│11 │12 │13 │14 │15 │     │ 1 │ 0 │-1 │
├───┼───┼───┼───┼───┤     └───┴───┴───┘
│16 │17 │18 │19 │20 │
├───┼───┼───┼───┼───┤
│21 │22 │23 │24 │25 │
└───┴───┴───┴───┴───┘

滑窗在位置 (0,0): 取 3×3 子区域, 和核做逐元素乘加
  1×1+2×0+3×(-1)+6×1+7×0+8×(-1)+11×1+12×0+13×(-1) = -6

滑窗在位置 (0,1): 取右边的 3×3 子区域 → -6

3×3 滑窗 → 输出 3×3 的结果。
```

### 直接写卷积 kernel 的问题

```
直接实现的伪代码:
  for oh, ow:                     // 输出位置
    for ci:                       // 输入通道
      for kh, kw:                 // 核内位置
        output[oh][ow] += input[ci][oh+kh][ow+kw] * weight[ci][kh][kw]

GPU 上直接写的困难:
  → input 访问模式不规则: (oh+kh, ow+kw) 对不同输出位置是错开的
  → 不同输出位置的 3×3 窗口有重叠, 但很难做 Shared Memory tiling
  → 无法直接使用 Tensor Core (需要规则的矩阵乘法输入)
  → 手写优化极其困难
```

### im2col 的核心思路

**把每个滑动窗口"展平"成一行，所有窗口叠成矩阵，和权重做标准矩阵乘法！**

```
输入图像 (5×5):                    卷积核 (3×3):
┌──┬──┬──┬──┬──┐                   ┌────┬────┬────┐
│a │b │c │d │e │                   │ w0 │ w1 │ w2 │
├──┼──┼──┼──┼──┤                   ├────┼────┼────┤
│f │g │h │i │j │                   │ w3 │ w4 │ w5 │
├──┼──┼──┼──┼──┤                   ├────┼────┼────┤
│k │l │m │n │o │                   │ w6 │ w7 │ w8 │
├──┼──┼──┼──┼──┤                   └────┴────┴────┘
│p │q │r │s │t │
├──┼──┼──┼──┼──┤
│u │v │w │x │y │
└──┴──┴──┴──┴──┘

窗口 (0,0): [a,b,c, f,g,h, k,l,m] → 展平成 1 行
窗口 (0,1): [b,c,d, g,h,i, l,m,n] → 展平成 1 行
...
9 个窗口 → col 矩阵 [9行 × 9列]:

  窗口位置 ↓    窗口内容 (展平的 3×3) →
       (0,0):  [a b c f g h k l m]
       (0,1):  [b c d g h i l m n]
       (0,2):  [c d e h i j m n o]
       (1,0):  [f g h k l m p q r]
       (1,1):  [g h i l m n q r s]
       (1,2):  [h i j m n o r s t]
       (2,0):  [k l m p q r u v w]
       (2,1):  [l m n q r s v w x]
       (2,2):  [m n o r s t w x y]

权重展开: [w0 w1 w2 w3 w4 w5 w6 w7 w8]  (1行 × 9列)

然后: output = col × weight^T = [9,9] × [9,1] = [9,1]
→ 这 9 个值就是手算的 9 个输出位置的卷积结果! ✓

多通道 (实际场景):
  C_in=64, C_out=128, Kh=Kw=3, H=W=32
  → col:  [H_out×W_out, C_in×Kh×Kw] = [1024, 576]
  → weight: [C_out, C_in×Kh×Kw]     = [128, 576]
  → output: weight × col^T          = [128, 1024]
  → 这就是标准 GEMM!
```


## 2. im2col Kernel — 完整代码逐段拆解

这是整个算法的核心。输入是一张图像 `[C, H, W]`，输出是展开后的列矩阵 `[H_out×W_out, C×Kh×Kw]`。

```cuda
__global__ void im2col_kernel(
    const float *data_im,
    float *data_col,
    int C, int H, int W,
    int Kh, int Kw,
    int pad,
    int H_out, int W_out)
{
    // ① 每个线程负责 col 矩阵的 1 个元素
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H_out * W_out * C * Kh * Kw;
    if (idx >= total) return;

    // ② 从一维 idx 反算出 5 维坐标 — 这是最难的一步
    int kw_idx = idx % Kw;
    int tmp    = idx / Kw;
    int kh_idx = tmp % Kh;
    tmp        = tmp / Kh;
    int c      = tmp % C;
    tmp        = tmp / C;
    int ow     = tmp % W_out;
    int oh     = tmp / W_out;

    // ③ 从输出位置反算输入位置
    int ih = oh - pad + kh_idx;
    int iw = ow - pad + kw_idx;

    // ④ 确定在 col 矩阵中的位置 (行, 列)
    int col_row = oh * W_out + ow;
    int col_col = c * Kh * Kw + kh_idx * Kw + kw_idx;

    // ⑤ 读 input 或填 0 (padding)
    float val = 0.0f;
    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
        val = data_im[c * H * W + ih * W + iw];
    }
    data_col[col_row * (C * Kh * Kw) + col_col] = val;
}
```

### ① Grid/Block — 线程怎么映射

```cuda
int total = H_out * W_out * C * Kh * Kw;     // col 矩阵的总元素数
im2col_kernel<<<(total+255)/256, 256>>>(...); // 1 线程 → 1 个元素
```

```
例如 C=64, H=W=32, Kh=Kw=3, pad=1:
  H_out = 32-3+2+1 = 32, W_out = 32
  total = 32 × 32 × 64 × 3 × 3 = 589824

  grid = ceil(589824 / 256) = 2304 Blocks
  每个 Block 256 线程 → 589824 线程 → 每个线程写 col 矩阵的 1 个元素

这是最简单的映射: 一个线程负责输出矩阵的一个位置。
```

### ② 一维 idx 反算五维坐标 — 最关键的推导

col 矩阵有 5 个索引维度:
```
data_col[col_row][col_col]
  col_row = oh * W_out + ow              ← 表示"哪个输出位置"
  col_col = c * Kh * Kw + kh_idx * Kw + kw_idx  ← 表示"核内的哪个元素"

总元素数 = H_out × W_out × C × Kh × Kw
```

每个线程拿到一个一维 `idx`，需要反算出 `(oh, ow, c, kh_idx, kw_idx)`。**反算的顺序决定了遍历的顺序**——这里选的是 kw_idx 在最低位、oh 在最高位：

```
idx 的空间排布 (row-major, 最右是最内层):

  idx 变化最快的是 kw_idx (最内层)
  然后是             kh_idx
  然后是             c
  然后是             ow
  变化最慢的是 oh      (最外层)

即:
  idx = oh * (W_out * C * Kh * Kw)
      + ow * (C * Kh * Kw)
      + c  * (Kh * Kw)
      + kh_idx * Kw
      + kw_idx

反算:
  kw_idx = idx % Kw            ← 最内层: 周期 = Kw
  tmp = idx / Kw               ← 去掉 Kw 那维
  kh_idx = tmp % Kh            ← 下一层: 周期 = Kh
  tmp = tmp / Kh
  c = tmp % C                  ← 周期 = C
  tmp = tmp / C
  ow = tmp % W_out             ← 周期 = W_out
  oh = tmp / W_out             ← 剩余的就是最外层
```

```
具体例子: idx = 1234, Kw=3, Kh=3, C=64, W_out=32

  kw_idx = 1234 % 3  = 1           → 核内第 1 列
  tmp    = 1234 / 3  = 411
  kh_idx = 411 % 3   = 0           → 核内第 0 行
  tmp    = 411 / 3   = 137
  c      = 137 % 64  = 9           → 第 9 个通道
  tmp    = 137 / 64  = 2
  ow     = 2 % 32    = 2           → 输出列 = 2
  oh     = 2 / 32    = 0           → 输出行 = 0

  所以 idx=1234 对应: oh=0, ow=2, c=9, kh=0, kw=1
  即输出位置 (0,2) 的 col 矩阵行中, 通道 9、核位置 (0,1) 的元素
```

**为什么这个排布顺序很重要？**

```
kw_idx 在最内层 → 相邻 idx 对应 kw_idx 连续
  → 相邻线程写相邻的 col_col (kw_idx 变化)
  → col 矩阵同一行内连续 → 写入合并! ✓

如果反过来 (oh 在最内层):
  → 相邻线程写同一行的不同 oh → 跨越整个 col 行 → 写入不合并!
```

### ③ 从输出位置反算输入位置

```cuda
int ih = oh - pad + kh_idx;   // stride=1
int iw = ow - pad + kw_idx;
```

```
以输出位置 (0, 0)、核位置 (1, 1)、pad=1 为例:
  oh=0, ow=0, kh_idx=1, kw_idx=1
  ih = 0 - 1 + 1 = 0
  iw = 0 - 1 + 1 = 0
  → 输入位置 (0, 0)

以输出位置 (0, 0)、核位置 (0, 0)、pad=1 为例:
  ih = 0 - 1 + 0 = -1
  iw = 0 - 1 + 0 = -1
  → 越界! padding 区域 → 填 0

padding 的作用:
  没有 padding: 3×3核在 5×5图像上 → 输出只有 3×3
  有 padding=1: 在图像四周各补一圈 0 → 输出 = 5×5 (和输入一样大!)
```

### ④⑤ 读 input 并写入 col 矩阵

```cuda
int col_row = oh * W_out + ow;                        // col 矩阵的行
int col_col = c * Kh * Kw + kh_idx * Kw + kw_idx;     // col 矩阵的列

float val = 0.0f;
if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
    val = data_im[c * H * W + ih * W + iw];            // 读输入图像
}
data_col[col_row * (C * Kh * Kw) + col_col] = val;    // 写 col 矩阵
```

```
data_im 的 layout: [C, H, W], row-major
  data_im[c * H * W + ih * W + iw]

data_col 的 layout: [H_out*W_out, C*Kh*Kw], row-major
  data_col[col_row * (C*Kh*Kw) + col_col]

  col_row 和 col_col 的范围:
    col_row: 0 .. H_out*W_out-1  (1024 行)
    col_col: 0 .. C*Kh*Kw-1      (576 列)
```


## 3. GEMM 阶段 — weight × col^T

展开完成后，问题变成了标准的矩阵乘法：

```cuda
__global__ void gemm_simple(
    const float *A, float *B, float *C_out,
    int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C_out[row * N + col] = sum;
}
```

```
host 端调用:
  int M = C_out;                    // 128 (权重矩阵的行数)
  int K = C_in * Kh * Kw;          // 576 (内积维度)
  int N = H_out * W_out;           // 1024 (col^T 的列数)

  dim3 block(16, 16);
  dim3 grid((N+15)/16, (M+15)/16);
  gemm_simple<<<grid, block>>>(d_wt, d_col, d_out, M, K, N);
  //                                ↑A    ↑B^T   ↑C

注意: B = d_col, 即 col 矩阵 (不是 col^T)。
  kernel 里读取 B[k * N + col] 时，k 遍历 K，col 遍历 N
  → B 的 layout 是 [K, N] = [576, 1024]  (列优先视角)
  → 和 col 矩阵的 [N, K] = [1024, 576]  (行优先视角)
  → 因为 row-major 的 [N,K] 在列优先视角下就是 [K,N] 的转置
  → 所以不需要显式转置! 直接改访问模式就行
```

**这个 GEMM 是普通朴素版**（无 Shared Memory Tiling），仅用于教学。实际生产中应替换为 cuBLAS：

```c
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
            C_out, H_out*W_out, C_in*Kh*Kw,
            &alpha, d_wt, C_out, d_col, H_out*W_out,
            &beta, d_out, C_out);
```


## 4. im2col 的代价和收益

```
代价:
  1. 额外内存: col 矩阵 = H_out×W_out×C×Kh×Kw 元素
     例: 32×32×64×9 = 589824 float = 2.25 MB (输入的 9×!)
     3×3 核 → 每个输入元素被提取到 ~9 个不同的窗口中 → ~9× 膨胀

  2. 额外带宽: 写 col 矩阵到 HBM, 然后 GEMM 再读回来

收益:
  1. 卷积变成了标准 GEMM → 可以直接用 cuBLAS/CUTLASS
  2. GEMM 经过极致优化 (Shared Memory Tiling + Register Blocking + Tensor Core)
  3. 算术强度高 (Compute Bound) → Tensor Core 能跑满
  4. 总性能通常远优于手写直接卷积

算术强度分析 (上面参数的 GEMM):
  FLOP = 2 × C_out × (C_in×Kh×Kw) × H_out×W_out ≈ 151M FLOP
  Byte = 4 × (weight + col + output) ≈ 3.2 MB
  AI ≈ 47 FLOP/Byte >> A100 Ridge Point (~9.7) → Compute Bound! ✓
```


## 5. 练习题

| 练习 | 任务 | 核心考点 |
|------|------|---------|
| [ex1_im2col_level1.cu](./exercises/ex1_im2col_level1.cu) | im2col 展开 kernel | 一维 idx → 五维坐标的反算（只填 kernel） |
| [ex1_im2col_level2.cu](./exercises/ex1_im2col_level2.cu) | 同上（完整实现） | kernel + host + 验证全部自己写 |

```bash
nvcc -O2 -o ex1_im2col_level1 ex1_im2col_level1.cu
./ex1_im2col_level1
```

## 常见错误

- **idx 反算顺序写错** → 症状: 输出结果完全对不上。五个 % / 的顺序必须和内层的排布一致（最内层最先取模）。如果 innermost 是 kw_idx，那第一步必须是 `idx % Kw` → `kw_idx`
- **忘了考虑 padding** → 症状: `ih < 0` 时访问 `data_im[c*H*W + (-1)*W + iw]` → GPU 不会崩溃但读到垃圾值。必须检查 `ih >= 0 && ih < H && iw >= 0 && iw < W`
- **col 矩阵索引写错** → `col_col = c*Kh*Kw + kh_idx*Kw + kw_idx`，容易把 Kh 和 Kw 搞反
- **stride != 1 时 ih/iw 计算公式不同** → `ih = oh * stride - pad + kh_idx`。本例 stride=1 省略了
- **gemm_simple 的 B 矩阵理解错误** → kernel 里 `B[k * N + col]` 读的是 col^T（即 col 的 k 列 row col 行）。如果 col 是 `[N, K]` 的 row-major 矩阵，`B[k*N+col]` 的含义是"第 k 列第 col 个元素"——这在 row-major 下就是 data_col[col * K + k]
