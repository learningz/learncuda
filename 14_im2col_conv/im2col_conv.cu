#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ============================================================
// 14: im2col Convolution — 把卷积变成矩阵乘法
//
// 配合理论: theory/05_operator_development.md 5.5 节 (Convolution 三种策略)
//
// 卷积 (Convolution): 用一个小窗口 (如 3×3) 在图像上滑动, 每个位置做点积。
//
// 为什么用 im2col?
//   直接实现卷积需要 5 重循环 (batch, cout, h, w, cin×kh×kw) → 难优化。
//   im2col 的思路: 把每个滑窗位置的补丁展开成一行 → 变成矩阵乘法 → 直接用 cuBLAS!
//
// im2col 展开过程:
//   输入图像: [C_in, H, W]
//   卷积核:   [C_out, C_in, Kh, Kw]
//
//   对每个输出位置 (oh, ow):
//     从输入中提取 [C_in, Kh, Kw] 的补丁 → 展平成 1 行 (长度 = C_in × Kh × Kw)
//     所有输出位置的行拼起来 → 矩阵 col: [H_out × W_out, C_in × Kh × Kw]
//
//   然后: output = weight × col^T
//     weight 已经是 [C_out, C_in×Kh×Kw] 的形状 → 标准矩阵乘!
//
// 新概念:
//   Padding: 在图像边缘补 0, 使输出大小不缩小
//   Stride:  滑窗每次移动的步长 (本例 stride=1)
//
// 本程序实现:
//   im2col kernel: 将输入图像展开成列矩阵
//   简单的 GEMM:  weight × col^T → 输出
//   对比 CPU 直接卷积验证正确性
// ============================================================

#define CUDA_CHECK(call) do {                                              \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                cudaGetErrorString(err)); exit(1); }                       \
} while(0)

// ---- im2col kernel ----
// 将输入图像展开成列矩阵
// 输入: data_im [C, H, W]
// 输出: data_col [H_out * W_out, C * Kh * Kw]
__global__ void im2col_kernel(
    const float *data_im,    // 输入图像
    float *data_col,         // 输出列矩阵
    int C, int H, int W,    // 输入的 通道数, 高度, 宽度
    int Kh, int Kw,          // 卷积核大小
    int pad,                 // padding
    int H_out, int W_out) {  // 输出大小

    // 每个线程处理输出列矩阵的一个元素
    // 全局索引 idx → 映射到 (output_pos, kernel_element)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H_out * W_out * C * Kh * Kw;
    if (idx >= total) return;

    // 从 idx 反算出:
    //   oh, ow: 输出位置 (对应滑窗的中心位置)
    //   c, kh, kw: 在卷积核中的位置
    int kw_idx = idx % Kw;
    int tmp = idx / Kw;
    int kh_idx = tmp % Kh;
    tmp = tmp / Kh;
    int c = tmp % C;
    tmp = tmp / C;
    int ow = tmp % W_out;
    int oh = tmp / W_out;

    // 计算对应的输入位置 (考虑 padding)
    int ih = oh - pad + kh_idx;  // stride=1
    int iw = ow - pad + kw_idx;

    // 输出列矩阵的位置:
    //   行 = oh * W_out + ow (输出位置)
    //   列 = c * Kh * Kw + kh_idx * Kw + kw_idx (展平的核元素)
    int col_row = oh * W_out + ow;
    int col_col = c * Kh * Kw + kh_idx * Kw + kw_idx;

    float val = 0.0f;
    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
        val = data_im[c * H * W + ih * W + iw];
    }
    // padding 区域的值为 0 (val 的初始值)

    data_col[col_row * (C * Kh * Kw) + col_col] = val;
}

// ---- 简单的 GEMM (weight × col^T) ----
// 为了教学清晰, 不用 cuBLAS, 手写一个朴素 GEMM
// 实际生产中一定要用 cuBLAS / CUTLASS!
__global__ void gemm_simple(
    const float *A,   // [M, K] = weight [C_out, C_in*Kh*Kw]
    const float *B,   // [K, N] = col^T  [C_in*Kh*Kw, H_out*W_out]
    float *C_out,     // [M, N] = output [C_out, H_out*W_out]
    int M, int K, int N) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C_out[row * N + col] = sum;
}

// ---- CPU 直接卷积 (参考实现) ----
void conv2d_cpu(
    const float *input,   // [C_in, H, W]
    const float *weight,  // [C_out, C_in, Kh, Kw]
    float *output,        // [C_out, H_out, W_out]
    int C_in, int H, int W,
    int C_out, int Kh, int Kw, int pad) {

    int H_out = H + 2 * pad - Kh + 1;
    int W_out = W + 2 * pad - Kw + 1;

    for (int co = 0; co < C_out; co++) {
        for (int oh = 0; oh < H_out; oh++) {
            for (int ow = 0; ow < W_out; ow++) {
                float sum = 0;
                for (int ci = 0; ci < C_in; ci++) {
                    for (int kh = 0; kh < Kh; kh++) {
                        for (int kw = 0; kw < Kw; kw++) {
                            int ih = oh - pad + kh;
                            int iw = ow - pad + kw;
                            float v = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                      ? input[ci*H*W + ih*W + iw] : 0.0f;
                            sum += v * weight[co*C_in*Kh*Kw + ci*Kh*Kw + kh*Kw + kw];
                        }
                    }
                }
                output[co*H_out*W_out + oh*W_out + ow] = sum;
            }
        }
    }
}

int main() {
    // 参数: 64 通道输入, 128 通道输出, 32×32 图像, 3×3 卷积核, padding=1
    int C_in = 64, C_out = 128, H = 32, W = 32, Kh = 3, Kw = 3, pad = 1;
    int H_out = H + 2*pad - Kh + 1;  // = 32
    int W_out = W + 2*pad - Kw + 1;  // = 32

    printf("im2col Convolution\n");
    printf("  输入: [%d, %d, %d], 卷积核: [%d, %d, %d, %d], padding=%d\n",
           C_in, H, W, C_out, C_in, Kh, Kw, pad);
    printf("  输出: [%d, %d, %d]\n\n", C_out, H_out, W_out);

    size_t in_bytes = C_in * H * W * sizeof(float);
    size_t wt_bytes = C_out * C_in * Kh * Kw * sizeof(float);
    size_t out_bytes = C_out * H_out * W_out * sizeof(float);
    // im2col 展开后的列矩阵大小
    size_t col_bytes = H_out * W_out * C_in * Kh * Kw * sizeof(float);

    printf("  im2col 列矩阵: [%d, %d] = %.1f MB\n",
           H_out*W_out, C_in*Kh*Kw, col_bytes/1e6);
    printf("  (这就是 im2col 的代价: 额外的内存占用)\n\n");

    // 分配和初始化
    float *h_in = (float*)malloc(in_bytes);
    float *h_wt = (float*)malloc(wt_bytes);
    float *h_out_cpu = (float*)malloc(out_bytes);
    float *h_out_gpu = (float*)malloc(out_bytes);
    srand(42);
    for (int i = 0; i < C_in*H*W; i++) h_in[i] = (rand()%200-100)/100.0f;
    for (int i = 0; i < C_out*C_in*Kh*Kw; i++) h_wt[i] = (rand()%200-100)/100.0f;

    // CPU 参考
    conv2d_cpu(h_in, h_wt, h_out_cpu, C_in, H, W, C_out, Kh, Kw, pad);

    // GPU
    float *d_in, *d_wt, *d_col, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, in_bytes));
    CUDA_CHECK(cudaMalloc(&d_wt, wt_bytes));
    CUDA_CHECK(cudaMalloc(&d_col, col_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wt, h_wt, wt_bytes, cudaMemcpyHostToDevice));

    // Step 1: im2col — 将输入展开成列矩阵
    int total_col = H_out * W_out * C_in * Kh * Kw;
    im2col_kernel<<<(total_col+255)/256, 256>>>(
        d_in, d_col, C_in, H, W, Kh, Kw, pad, H_out, W_out);

    // Step 2: GEMM — weight × col^T = output
    // weight: [C_out, C_in*Kh*Kw], col^T: [C_in*Kh*Kw, H_out*W_out]
    // → output: [C_out, H_out*W_out]
    int M = C_out, K = C_in*Kh*Kw, NN = H_out*W_out;
    dim3 block(16, 16);
    dim3 grid((NN+15)/16, (M+15)/16);
    gemm_simple<<<grid, block>>>(d_wt, d_col, d_out, M, K, NN);

    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, out_bytes, cudaMemcpyDeviceToHost));

    // 验证
    float max_err = 0;
    for (int i = 0; i < C_out*H_out*W_out; i++)
        max_err = fmaxf(max_err, fabsf(h_out_cpu[i] - h_out_gpu[i]));
    printf("CPU vs GPU (im2col) 最大误差: %.2e %s\n\n", max_err, max_err < 1e-3 ? "✓" : "✗");

    printf("im2col 的本质:\n");
    printf("  1. 把复杂的卷积 → 变成简单的矩阵乘\n");
    printf("  2. 矩阵乘有 cuBLAS/CUTLASS 这些极度优化的实现 → 直接复用!\n");
    printf("  3. 代价: 列矩阵占额外内存 (%.1f MB), 但换来了计算效率\n", col_bytes/1e6);
    printf("  4. cuDNN 在很多情况下内部就是用 im2col + GEMM\n");
    printf("\n其他策略 (不需要额外内存):\n");
    printf("  - 直接卷积 + Shared Memory tiling (见 Ch5.5)\n");
    printf("  - Winograd 变换: 3×3 卷积减少乘法次数 (见 Ch5.5)\n");

    cudaFree(d_in); cudaFree(d_wt); cudaFree(d_col); cudaFree(d_out);
    free(h_in); free(h_wt); free(h_out_cpu); free(h_out_gpu);
    return 0;
}
