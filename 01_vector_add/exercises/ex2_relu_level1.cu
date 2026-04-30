// ============================================================
// 练习 2: ReLU — y[i] = max(x[i], 0)
// 难度: Level 1 (只需填写 kernel 函数体)
//
// 编译: nvcc -O2 -o ex2_relu_level1 ex2_relu_level1.cu
// 运行: ./ex2_relu_level1
// 预期输出: "结果: ✓ 全部正确"
//
// ReLU 是深度学习里最常用的激活函数。
// 和 vector_add 的区别:
//   1. 只有一个输入数组 x, 一个输出数组 y (不是两个输入)
//   2. kernel 里需要做条件判断 (x >= 0 ? x : 0)
//      也可以用 fmaxf(x, 0.0f) 代替 if/else
//
// 提示:
//   - fmaxf(a, b) 返回 a 和 b 中的较大值
//   - 或者用三元运算符: (x >= 0) ? x : 0.0f
//   - 两种写法结果一样, 你都可以试试
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
// 每个线程计算: y[idx] = max(x[idx], 0)
// ============================================================
__global__ void relu_kernel(const float *x, float *y, int n) {
    // --- 在这里写你的代码 ---

}

int main() {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    float *h_x = (float *)malloc(bytes);
    float *h_y = (float *)malloc(bytes);
    float *h_y_ref = (float *)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_x[i] = static_cast<float>(i % 200 - 100) / 10.0f;
        h_y_ref[i] = h_x[i] > 0.0f ? h_x[i] : 0.0f;
    }

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    relu_kernel<<<gridSize, blockSize>>>(d_x, d_y, N);
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
