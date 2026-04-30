// ============================================================
// 练习 1: im2col 展开 kernel — 把卷积窗口展平成矩阵行
// 难度: Level 1 (只需填写 kernel 函数体)
//
// 编译: nvcc -O2 -o ex1_im2col_level1 ex1_im2col_level1.cu
// 运行: ./ex1_im2col_level1
// 预期输出: "结果: ✓ 全部正确"
//
// im2col_conv.cu 里 im2col 展开和 GEMM 是融合在一起的。
// 本题只练 im2col 展开本身 — 不做 GEMM, 更容易理解。
//
// 输入: image[C_in][H][W]   (CHW 格式)
// 输出: col[out_h * out_w][C_in * Kh * Kw]
//
// 每个输出位置 (oh, ow) 对应一行:
//   for cin in 0..C_in-1:
//     for kh in 0..Kh-1:
//       for kw in 0..Kw-1:
//         ih = oh + kh - pad, iw = ow + kw - pad
//         col[oh*out_w + ow][cin*Kh*Kw + kh*Kw + kw] =
//           (ih >= 0 && ih < H && iw >= 0 && iw < W) ? image[cin*H*W + ih*W + iw] : 0
//
// 每个线程处理一个输出位置 (oh, ow) 的整行
//
// 提示:
//   - idx = blockIdx.x * blockDim.x + threadIdx.x → 对应第 idx 个输出位置
//   - oh = idx / out_w, ow = idx % out_w
//   - 三重循环填 col 的这一行 (长度 = C_in * Kh * Kw)
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// TODO: 实现 im2col kernel
__global__ void im2col_kernel(const float *image, float *col,
                              int C_in, int H, int W,
                              int Kh, int Kw, int pad,
                              int out_h, int out_w) {
    // --- 在这里写你的代码 ---

}

void im2col_cpu(const float *image, float *col,
                int C_in, int H, int W, int Kh, int Kw, int pad,
                int out_h, int out_w) {
    for (int pos = 0; pos < out_h * out_w; pos++) {
        int oh = pos / out_w, ow = pos % out_w;
        for (int cin = 0; cin < C_in; cin++)
            for (int kh = 0; kh < Kh; kh++)
                for (int kw = 0; kw < Kw; kw++) {
                    int ih = oh + kh - pad, iw = ow + kw - pad;
                    int col_idx = pos * (C_in * Kh * Kw) + cin * Kh * Kw + kh * Kw + kw;
                    col[col_idx] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                   ? image[cin * H * W + ih * W + iw] : 0;
                }
    }
}

int main() {
    const int C_in = 3, H = 8, W = 8, Kh = 3, Kw = 3, pad = 1;
    const int out_h = H, out_w = W;
    const int col_rows = out_h * out_w;
    const int col_cols = C_in * Kh * Kw;

    size_t img_bytes = C_in * H * W * sizeof(float);
    size_t col_bytes = col_rows * col_cols * sizeof(float);

    float *h_img = (float *)malloc(img_bytes);
    float *h_col = (float *)malloc(col_bytes);
    float *h_ref = (float *)malloc(col_bytes);

    srand(42);
    for (int i = 0; i < C_in * H * W; i++) h_img[i] = (float)(rand() % 100);

    im2col_cpu(h_img, h_ref, C_in, H, W, Kh, Kw, pad, out_h, out_w);

    float *d_img, *d_col;
    CUDA_CHECK(cudaMalloc(&d_img, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_col, col_bytes));
    CUDA_CHECK(cudaMemcpy(d_img, h_img, img_bytes, cudaMemcpyHostToDevice));

    int total_pos = out_h * out_w;
    int block = 256, grid = (total_pos + block - 1) / block;
    im2col_kernel<<<grid, block>>>(d_img, d_col, C_in, H, W, Kh, Kw, pad, out_h, out_w);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_col, d_col, col_bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < col_rows * col_cols; i++)
        if (h_col[i] != h_ref[i]) { ok = false; printf("@ %d: %.0f vs %.0f\n", i, h_col[i], h_ref[i]); break; }
    printf("结果: %s\n", ok ? "✓ 全部正确" : "✗ 有错误");

    cudaFree(d_img); cudaFree(d_col);
    free(h_img); free(h_col); free(h_ref);
    return 0;
}
