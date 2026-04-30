// ============================================================
// 练习 2: 矩阵加法 — C[i][j] = A[i][j] + B[i][j]
// 难度: Level 1 (只需填写 kernel 函数体)
//
// 编译: nvcc -O2 -o ex2_matadd_level1 ex2_matadd_level1.cu
// 运行: ./ex2_matadd_level1
// 预期输出: "结果: ✓ 全部正确"
//
// 本质上就是 "2D 版的 vector_add"。
// 考点:
//   1. 用 2D 索引算出 row 和 col
//   2. 用 row * N + col 转成线性索引读写数组
//   3. 边界检查 row < M && col < N
//
// 提示:
//   - 和 ex1 转置一样算 row, col
//   - 读: A[row * N + col] 和 B[row * N + col]
//   - 写: C[row * N + col] = A[...] + B[...]
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE 16

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
// 每个线程: C[row][col] = A[row][col] + B[row][col]
// ============================================================
__global__ void matadd_kernel(const float *A, const float *B, float *C,
                              int M, int N) {
    // --- 在这里写你的代码 ---

}

int main() {
    const int M = 512, N = 1024;
    size_t bytes = M * N * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    float *h_C_ref = (float *)malloc(bytes);

    for (int i = 0; i < M * N; i++) {
        h_A[i] = static_cast<float>(i % 100) / 10.0f;
        h_B[i] = static_cast<float>(i % 77) / 7.0f;
        h_C_ref[i] = h_A[i] + h_B[i];
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matadd_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < M * N; i++) {
        if (fabsf(h_C[i] - h_C_ref[i]) > 1e-5f) {
            printf("验证失败 @ %d: got %.6f, expected %.6f\n", i, h_C[i], h_C_ref[i]);
            ok = false;
            break;
        }
    }
    printf("结果: %s\n", ok ? "✓ 全部正确" : "✗ 有错误");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    return 0;
}
