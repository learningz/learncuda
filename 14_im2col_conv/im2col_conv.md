# im2col Convolution：把卷积展开成矩阵乘法的硬件视角

配合 `im2col_conv.cu` 阅读。


## im2col 在内存中做了什么

```
输入图像: [C_in=64, H=32, W=32], 卷积核: [C_out=128, C_in=64, 3, 3], padding=1

im2col 展开:
  对输出的每个位置 (oh, ow), 提取输入中的 3×3×64 = 576 个元素:
  
  输出位置 (0,0): 提取 input[:, -1:-1+3, -1:-1+3] (有 padding, 边缘为 0)
  输出位置 (0,1): 提取 input[:, -1:-1+3,  0:0+3]
  ...
  输出位置 (31,31): 提取 input[:, 30:33, 30:33]
  
  每个补丁展平成一行 (576 个元素), 共 32×32 = 1024 行:
  
  列矩阵 col: [1024 行, 576 列]
    ┌─ col 矩阵 ────────────────────────────────────────┐
    │ row 0   = input 在位置 (0,0) 的 3×3×64 展平       │
    │ row 1   = input 在位置 (0,1) 的 3×3×64 展平       │
    │ ...                                               │
    │ row 1023 = input 在位置 (31,31) 的 3×3×64 展平    │
    └───────────────────────────────────────────────────┘
  
  然后: output = weight × col^T
    weight: [128, 576]    (C_out × C_in×Kh×Kw)
    col^T:  [576, 1024]   (C_in×Kh×Kw × H_out×W_out)
    output: [128, 1024]   = [C_out, H_out×W_out]
    → 标准 GEMM!
```


## im2col kernel 在硬件上的执行

```
im2col_kernel 的每个线程:
  1. 从全局索引 idx 反算出 (oh, ow, c, kh, kw)
     → 5 次整数除法和取模 (GPU 上整数除法很慢, ~80 cycles!)
     → 编译器会尝试用乘法+移位替代
  
  2. 计算输入位置 ih = oh - pad + kh, iw = ow - pad + kw
     → 2 次整数加法
  
  3. 边界检查: if (ih >= 0 && ih < H && iw >= 0 && iw < W)
     → padding 区域填 0 (不需要 HBM 访问!)
  
  4. 从 HBM 读 input[c][ih][iw]
     → 这是散射读取! 相邻线程的 (oh, ow, c, kh, kw) 不同
     → 地址不连续 → 合并度取决于具体的索引映射
  
  5. 写到 col 矩阵的正确位置
     → col[oh*W_out+ow][c*Kh*Kw+kh*Kw+kw] = 值
     → 输出地址是连续的 (线程按 kw, kh, c, ow, oh 排列)
     → 写入可能有较好的合并度

HBM 数据流:
  ┌── 输入 ──────┐     ┌── im2col kernel ──┐     ┌── col 矩阵 ──┐
  │ [64,32,32]   │ →→→ │ 读: 散射          │ →→→ │ [1024, 576]  │
  │ = 256 KB     │     │ 写: 连续          │     │ = 2.25 MB    │
  └──────────────┘     └───────────────────┘     └──────────────┘
  
  im2col 的代价:
    额外内存: col 矩阵 2.25MB (输入的 ~9× — 因为 3×3 核, 每个元素被提取 ~9 次)
    额外带宽: 写 2.25MB 到 HBM, 然后 GEMM 再读 2.25MB
    
  好处: GEMM 的实现可以直接用 cuBLAS (经过极致优化, 接近峰值)
```


## GEMM 阶段在硬件上的执行

```
weight × col^T: [128, 576] × [576, 1024]

使用 cuBLAS 时:
  cuBLAS 自动选择最优的 GEMM kernel (Tiled + Register Blocking + Tensor Core)
  
  矩阵足够大 (M=128, K=576, N=1024) → 可以较好地利用 Tensor Core
  算术强度: 2×128×576×1024 / (4×(128×576 + 576×1024 + 128×1024))
          = 150994944 / 3276800 ≈ 46 FLOP/Byte
  → 远超 Ridge Point (9.6) → Compute Bound → GEMM 效率高!

使用本示例的朴素 GEMM 时:
  没有 Shared Memory tiling → 大量 HBM 重复读取 → 性能差很多
  → 这就是为什么生产中要用 cuBLAS 而不是手写朴素 GEMM
```


## 为什么不直接做卷积而要 im2col?

```
直接卷积:
  5 重循环 (batch, cout, oh, ow, cin×kh×kw)
  每个输出元素独立计算 → 可以并行
  但: 输入的访问模式不规则 (每个输出位置读一个 3×3 补丁)
       → 难以做好 Shared Memory tiling
       → 难以利用 Tensor Core
       → 手写很难达到好的性能

im2col + GEMM:
  im2col: 一次性展开 (不太高效, 但只做一次)
  GEMM: 规则的矩阵乘 → 极度优化的实现现成可用
  → 总性能通常优于手写的直接卷积

  cuDNN 对不同场景选择不同策略:
    大 kernel (5×5, 7×7): im2col + GEMM
    3×3: Winograd (减少乘法) 或 im2col
    1×1: 直接就是 GEMM (不需要 im2col!)
    小 batch: 直接卷积可能更好 (im2col 的额外内存不值得)
```


## 练习题

| 练习 | 任务 | 核心考点 |
|------|------|---------|
| [ex1_im2col_level1.cu](./exercises/ex1_im2col_level1.cu) | im2col 展开 kernel | 理解卷积窗口展平成矩阵行（只填 kernel） |

```bash
nvcc -O2 -o ex1_im2col_level1 ex1_im2col_level1.cu
```
