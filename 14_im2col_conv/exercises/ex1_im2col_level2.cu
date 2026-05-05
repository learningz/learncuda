// ============================================================
// 练习 1: im2col 展开 kernel
// 难度: Level 2 (kernel + host 端都要自己写)
//
// 编译: nvcc -O2 -o ex1_im2col_level2 ex1_im2col_level2.cu
// 运行: ./ex1_im2col_level2
// 预期输出: "结果: ✓ 全部正确"
//
// 要求:
//   1. 实现 im2col_kernel: 把 image[C_in][H][W] 展平成 col[out_h*out_w][C_in*Kh*Kw]
//   2. 在 main 中完成 CPU 参考实现、GPU 内存管理、数据传输、验证
//
// im2col 展开规则 (每个线程处理一个输出位置的整行):
//   对位置 (oh, ow):
//     for cin in 0..C_in-1:
//       for kh in 0..Kh-1:
//         for kw in 0..Kw-1:
//           ih = oh + kh - pad, iw = ow + kw - pad  (注意: stride = 1)
//           col[pos][cin*Kh*Kw + kh*Kw + kw] =
//             (ih>=0 && ih<H && iw>=0 && iw<W) ? image[cin*H*W + ih*W + iw] : 0
//
// 提示:
//   - idx = blockIdx.x * blockDim.x + threadIdx.x
//   - pos 从 0 到 out_h*out_w-1
//   - oh = idx / out_w, ow = idx % out_w
//   - 内层三重循环填这一行的所有列
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
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
    const int C_in = 3, H = 16, W = 16, Kh = 3, Kw = 3, pad = 1;
    const int out_h = H, out_w = W;
    const int col_rows = out_h * out_w;
    const int col_cols = C_in * Kh * Kw;

    // ============================================================
    // TODO: 完成以下步骤
    //   1. malloc CPU 内存 (h_img, h_col, h_ref)
    //   2. 初始化 h_img (随机整数), 用 im2col_cpu 算 h_ref
    //   3. cudaMalloc GPU 内存 (d_img, d_col)
    //   4. cudaMemcpy H2D (h_img → d_img)
    //   5. launch im2col_kernel, cudaMemcpy D2H
    //   6. 逐元素对比 h_col 和 h_ref
    //   7. 打印结果 + cudaFree + free
    // ============================================================

    // --- 在这里写你的代码 ---

    return 0;
}
