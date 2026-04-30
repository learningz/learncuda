// ============================================================
// 练习 1: 矩阵转置 — B[j][i] = A[i][j]
// 难度: Level 1 (只需填写 kernel 函数体)
//
// 编译: nvcc -O2 -o ex1_transpose_level1 ex1_transpose_level1.cu
// 运行: ./ex1_transpose_level1
// 预期输出: "结果: ✓ 全部正确"
//
// 和 matmul 的关系:
//   这是最简单的 2D kernel — 不需要 Shared Memory, 不需要 __syncthreads。
//   目的是让你熟悉:
//     1. dim3 和 2D Grid/Block 配置
//     2. 用 threadIdx.x/y + blockIdx.x/y 算出 row 和 col
//     3. row-major 矩阵的索引公式: A[row][col] = A[row * numCols + col]
//
// A 是 M×N 矩阵, B 是 N×M 矩阵, 满足 B[col][row] = A[row][col]
//
// 提示:
//   - 和 matmul_naive 一样算 row 和 col
//   - 读: A[row * N + col]  (M×N 矩阵, N 列)
//   - 写: B[col * M + row]  (N×M 矩阵, M 列)
//   - 别忘了边界检查: row < M && col < N
// ============================================================

#include <cstdio>
#include <cstdlib>
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
// 每个线程: 读 A[row][col], 写到 B[col][row]
// ============================================================
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

    float *d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    transpose_kernel<<<grid, block>>>(d_A, d_B, M, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_B, d_B, sizeB, cudaMemcpyDeviceToHost));

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

    cudaFree(d_A); cudaFree(d_B);
    free(h_A); free(h_B); free(h_B_ref);
    return 0;
}
