// ============================================================
// 练习 1: LogSumExp — log(Σ exp(x_i - max)) + max
// 难度: Level 1 (只需填写 kernel 函数体)
//
// 编译: nvcc -O2 -o ex1_logsumexp_level1 ex1_logsumexp_level1.cu
// 运行: ./ex1_logsumexp_level1
// 预期输出: "结果: ✓ 全部正确"
//
// LogSumExp 是 Softmax 的前半段, 也是数值稳定的 log-softmax 的核心。
// 公式: LSE(x) = log(Σ exp(x_i - max(x))) + max(x)
//
// 和 softmax_3pass 的关系:
//   Softmax 的 3 个 pass: 求 max → 求 sum(exp) → 归一化
//   LogSumExp 只需前 2 个 pass: 求 max → 求 sum(exp) → 取 log + max
//   每个 Block 处理一行, 输出一个标量 (不是一整行)
//
// 算法 (每个 Block 处理一行):
//   Pass 1: 用 SMEM 归约求这一行的 max
//   Pass 2: 用 SMEM 归约求 sum(exp(x_i - max))
//   线程 0: output[row] = logf(sum) + max
//
// 提示:
//   - 用 extern __shared__ float smem[] (动态分配, launch 时传大小)
//   - 归约结构和 softmax_3pass 完全一样: 写 smem → __syncthreads → 对半折叠
//   - logf(x) 计算自然对数
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
//   每个 Block 处理一行, 输出 output[blockIdx.x] = LSE(该行)
// ============================================================
__global__ void logsumexp_kernel(const float *input, float *output,
                                 int rows, int cols) {
    // --- 在这里写你的代码 ---

}

int main() {
    const int rows = 1024, cols = 4096;
    size_t bytes_in = rows * cols * sizeof(float);
    size_t bytes_out = rows * sizeof(float);

    float *h_in = (float *)malloc(bytes_in);
    float *h_out = (float *)malloc(bytes_out);
    float *h_ref = (float *)malloc(bytes_out);

    srand(42);
    for (int i = 0; i < rows * cols; i++)
        h_in[i] = (float)(rand() % 200 - 100) / 10.0f;

    for (int r = 0; r < rows; r++) {
        float mx = -INFINITY;
        for (int c = 0; c < cols; c++) mx = fmaxf(mx, h_in[r * cols + c]);
        float sum = 0;
        for (int c = 0; c < cols; c++) sum += expf(h_in[r * cols + c] - mx);
        h_ref[r] = logf(sum) + mx;
    }

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes_in));
    CUDA_CHECK(cudaMalloc(&d_out, bytes_out));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes_in, cudaMemcpyHostToDevice));

    int blockSize = 256;
    logsumexp_kernel<<<rows, blockSize, blockSize * sizeof(float)>>>(
        d_in, d_out, rows, cols);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes_out, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int r = 0; r < rows; r++) {
        if (fabsf(h_out[r] - h_ref[r]) > 0.01f) {
            printf("验证失败 @ row %d: got %.4f, expected %.4f\n", r, h_out[r], h_ref[r]);
            ok = false; break;
        }
    }
    printf("结果: %s\n", ok ? "✓ 全部正确" : "✗ 有错误");

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out); free(h_ref);
    return 0;
}
