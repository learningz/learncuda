// ============================================================
// 练习 1: 矩阵转置 — B[j][i] = A[i][j]
// 难度: Level 2 (kernel + host 端都要自己写)
//
// 编译: nvcc -O2 -o ex1_transpose_level2 ex1_transpose_level2.cu
// 运行: ./ex1_transpose_level2
// 预期输出: "结果: ✓ 全部正确"
//
// 要求:
//   1. 实现 transpose_kernel
//   2. 在 main 中:
//      - cudaMalloc 分配 GPU 内存
//      - cudaMemcpy 把 A 传到 GPU
//      - 用 dim3 配置 2D Grid 和 Block, 启动 kernel
//      - cudaMemcpy 把 B 传回 CPU
//      - cudaFree 释放
//
// A 是 M×N (512×1024), B 是 N×M (1024×512)。
// 想想:
//   - 你的 Grid 应该覆盖 A 的哪个维度? (提示: 让每个线程处理一个 A 元素)
//   - dim3 grid(?, ?) 里两个数应该是多少?
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

// TODO: 实现 transpose_kernel
__global__ void transpose_kernel(const float *A, float *B, int M, int N) {
    // --- 在这里写你的代码 ---

}

int main() {
    const int M = 512, N = 1024;
    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * M * sizeof(float);

    float *h_A = (float *)malloc(sizeA);
    float *h_B = (float *)malloc(sizeB);
    float *h_B_ref = (float *)malloc(sizeB);

    for (int i = 0; i < M * N; i++) h_A[i] = static_cast<float>(i);
    for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++)
            h_B_ref[c * M + r] = h_A[r * N + c];

    // ============================================================
    // TODO: 完成以下步骤
    //   1. cudaMalloc 分配 d_A, d_B
    //   2. cudaMemcpy 把 h_A 传到 d_A
    //   3. 配置 dim3 block 和 dim3 grid (2D!)
    //   4. 启动 transpose_kernel
    //   5. cudaMemcpy 把 d_B 传回 h_B
    //   6. cudaFree 释放 GPU 内存
    // ============================================================

    // --- 在这里写你的代码 ---

    bool ok = true;
    for (int i = 0; i < N * M; i++) {
        if (h_B[i] != h_B_ref[i]) {
            int r = i / M, c = i % M;
            printf("验证失败 @ B[%d][%d]: got %.0f, expected %.0f\n",
                   r, c, h_B[i], h_B_ref[i]);
            ok = false;
            break;
        }
    }
    printf("结果: %s\n", ok ? "✓ 全部正确" : "✗ 有错误");

    free(h_A); free(h_B); free(h_B_ref);
    return 0;
}
