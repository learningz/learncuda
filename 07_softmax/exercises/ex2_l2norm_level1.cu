// ============================================================
// 练习 2: L2 Normalize — y[i] = x[i] / sqrt(Σ x[j]²)
// 难度: Level 1 (只需填写 kernel 函数体)
//
// 编译: nvcc -O2 -o ex2_l2norm_level1 ex2_l2norm_level1.cu
// 运行: ./ex2_l2norm_level1
// 预期输出: "结果: ✓ 全部正确"
//
// L2 Normalize 和 Softmax 结构几乎一样:
//   Softmax:    y[i] = exp(x[i] - max) / Σ exp(x[j] - max)   (2-3 pass)
//   L2 Norm:    y[i] = x[i] / sqrt(Σ x[j]²)                  (2 pass)
//
// 算法 (每个 Block 处理一行):
//   Pass 1: 求 norm² = Σ x[j]² (用 SMEM 归约)
//   Pass 2: y[i] = x[i] * rsqrtf(norm²)    (rsqrtf = 1/sqrt)
//
// 注意: rsqrtf(0) = Inf, 实际中可以加一个 eps: rsqrtf(norm² + 1e-12f)
//
// 提示:
//   - Pass 1 和 reduce 完全一样, 只是累加的是 x[i]*x[i]
//   - Pass 2 和 softmax 的归一化一样, 只是除数变了
//   - 两个 pass 之间需要 __syncthreads 确保 norm 已经归约完毕
//   - 用 smem[0] 保存归约结果, 让所有线程都能读到
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

// ============================================================
// TODO: 实现这个 kernel
// 配置: <<<rows, blockSize, blockSize * sizeof(float)>>>
// ============================================================
__global__ void l2norm_kernel(const float *input, float *output,
                              int rows, int cols) {
    // --- 在这里写你的代码 ---

}

int main() {
    const int rows = 1024, cols = 4096;
    size_t bytes = rows * cols * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);

    srand(42);
    for (int i = 0; i < rows * cols; i++)
        h_in[i] = (float)(rand() % 200 - 100) / 10.0f;

    for (int r = 0; r < rows; r++) {
        float norm_sq = 0;
        for (int c = 0; c < cols; c++) norm_sq += h_in[r * cols + c] * h_in[r * cols + c];
        float inv_norm = 1.0f / sqrtf(norm_sq + 1e-12f);
        for (int c = 0; c < cols; c++) h_ref[r * cols + c] = h_in[r * cols + c] * inv_norm;
    }

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    l2norm_kernel<<<rows, blockSize, blockSize * sizeof(float)>>>(
        d_in, d_out, rows, cols);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < rows * cols; i++) {
        if (fabsf(h_out[i] - h_ref[i]) > 1e-4f) {
            printf("验证失败 @ %d: got %.6f, expected %.6f\n", i, h_out[i], h_ref[i]);
            ok = false; break;
        }
    }
    printf("结果: %s\n", ok ? "✓ 全部正确" : "✗ 有错误");

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out); free(h_ref);
    return 0;
}
