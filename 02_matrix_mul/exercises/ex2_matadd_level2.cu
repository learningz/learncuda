// ============================================================
// 练习 2: 矩阵加法 — C[i][j] = A[i][j] + B[i][j]
// 难度: Level 2 (kernel + host 端都要自己写)
//
// 编译: nvcc -O2 -o ex2_matadd_level2 ex2_matadd_level2.cu
// 运行: ./ex2_matadd_level2
// 预期输出: "结果: ✓ 全部正确"
//
// 要求:
//   1. 实现 matadd_kernel
//   2. 在 main 中:
//      - cudaMalloc 分配 3 块 GPU 内存 (d_A, d_B, d_C)
//      - cudaMemcpy 把 A, B 传到 GPU (2 次 H2D)
//      - 用 dim3 配置 2D Grid/Block, 启动 kernel
//      - cudaMemcpy 把 C 传回 CPU (1 次 D2H)
//      - cudaFree 释放
//
// 想想:
//   - dim3 block 取多大? 16×16 = 256 是个好选择
//   - dim3 grid 怎么算? 要覆盖 M×N 的矩阵
//   - grid 的 x 对应列 (N), y 对应行 (M) — 和 matmul.cu 一致
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

// TODO: 实现 matadd_kernel
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

    // ============================================================
    // TODO: 完成以下步骤
    //   1. cudaMalloc 分配 d_A, d_B, d_C (3 块, 每块 bytes 大小)
    //   2. cudaMemcpy 把 h_A, h_B 传到 GPU
    //   3. 配置 dim3 block 和 dim3 grid, 启动 kernel
    //   4. cudaMemcpy 把 d_C 传回 h_C
    //   5. cudaFree 释放 GPU 内存
    // ============================================================

    // --- 在这里写你的代码 ---

    bool ok = true;
    for (int i = 0; i < M * N; i++) {
        if (fabsf(h_C[i] - h_C_ref[i]) > 1e-5f) {
            printf("验证失败 @ %d: got %.6f, expected %.6f\n", i, h_C[i], h_C_ref[i]);
            ok = false;
            break;
        }
    }
    printf("结果: %s\n", ok ? "✓ 全部正确" : "✗ 有错误");

    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    return 0;
}
