// ============================================================
// 练习 3: Fused Multiply-Add — d[i] = a[i] * b[i] + c[i]
// 难度: Level 1 (只需填写 kernel 函数体)
//
// 编译: nvcc -O2 -o ex3_fma_level1 ex3_fma_level1.cu
// 运行: ./ex3_fma_level1
// 预期输出: "结果: ✓ 全部正确"
//
// FMA (Fused Multiply-Add) 是 GPU 上最基本的运算之一。
// 和 vector_add 的区别:
//   1. 三个输入数组 a, b, c + 一个输出数组 d
//   2. kernel 参数更多了, 但每个线程做的事还是一样简单
//
// 提示:
//   - kernel 签名已经写好, 你只需要写函数体
//   - 和 vector_add 一样: 算 idx, 检查边界, 做计算
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
// 每个线程计算: d[idx] = a[idx] * b[idx] + c[idx]
// ============================================================
__global__ void fma_kernel(const float *a, const float *b, const float *c,
                           float *d, int n) {
    // --- 在这里写你的代码 ---

}

int main() {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    float *h_d = (float *)malloc(bytes);
    float *h_d_ref = (float *)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i % 100) / 10.0f;
        h_b[i] = static_cast<float>(i % 77) / 7.0f;
        h_c[i] = static_cast<float>(i % 50) / 5.0f;
        h_d_ref[i] = h_a[i] * h_b[i] + h_c[i];
    }

    float *d_a, *d_b, *d_c, *d_d;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMalloc(&d_d, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c, bytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    fma_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, d_d, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_d, d_d, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_d[i] - h_d_ref[i]) > 1e-4f) {
            printf("验证失败 @ i=%d: got %.6f, expected %.6f\n", i, h_d[i], h_d_ref[i]);
            ok = false;
            break;
        }
    }
    printf("结果: %s\n", ok ? "✓ 全部正确" : "✗ 有错误");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d);
    free(h_d_ref);
    return 0;
}
