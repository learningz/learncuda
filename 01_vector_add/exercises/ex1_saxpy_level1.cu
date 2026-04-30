// ============================================================
// 练习 1: SAXPY — y[i] = a * x[i] + y[i]
// 难度: Level 1 (只需填写 kernel 函数体)
//
// 编译: nvcc -O2 -o ex1_saxpy_level1 ex1_saxpy_level1.cu
// 运行: ./ex1_saxpy_level1
// 预期输出: "结果: ✓ 全部正确"
//
// SAXPY 是经典的 BLAS 操作 (Single-precision A*X Plus Y)。
// 和 vector_add 的区别:
//   1. 多了一个标量参数 a (不是数组, 是一个 float 值)
//   2. 结果写回 y 本身 (原地修改), 不是写到新数组 c
//
// 提示:
//   - 标量参数直接按值传给 kernel, 和传 int n 一样
//   - 每个线程: 读 x[idx] 和 y[idx], 计算 a*x[idx]+y[idx], 写回 y[idx]
//   - 别忘了边界检查 if (idx < n)
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
// 每个线程计算: y[idx] = a * x[idx] + y[idx]
// ============================================================
__global__ void saxpy_kernel(float a, const float *x, float *y, int n) {
    // --- 在这里写你的代码 ---
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        y[tid] = a * x[tid] + y[tid];
    }
}

int main() {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);
    const float a = 2.5f;

    float *h_x = (float *)malloc(bytes);
    float *h_y = (float *)malloc(bytes);
    float *h_y_ref = (float *)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_x[i] = static_cast<float>(i % 100) / 10.0f;
        h_y[i] = static_cast<float>(i % 50) / 5.0f;
        h_y_ref[i] = a * h_x[i] + h_y[i];
    }

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    saxpy_kernel<<<gridSize, blockSize>>>(a, d_x, d_y, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_y[i] - h_y_ref[i]) > 1e-5f) {
            printf("验证失败 @ i=%d: got %.6f, expected %.6f\n", i, h_y[i], h_y_ref[i]);
            ok = false;
            break;
        }
    }
    printf("结果: %s\n", ok ? "✓ 全部正确" : "✗ 有错误");

    cudaFree(d_x);
    cudaFree(d_y);
    free(h_x);
    free(h_y);
    free(h_y_ref);
    return 0;
}
