// ============================================================
// 练习 2: SMEM 矩阵转置 — 对比有/无 padding 的性能
// 难度: Level 2 (kernel + host 端都要自己写)
//
// 编译: nvcc -O2 -o ex2_transpose_smem_level2 ex2_transpose_smem_level2.cu
// 运行: ./ex2_transpose_smem_level2
//
// 任务:
//   1. 实现两个 SMEM 转置 kernel (一个有 padding, 一个无)
//   2. 在 main 中:
//      - cudaMalloc, cudaMemcpy
//      - dim3 配置 + 启动两个 kernel
//      - 验证正确性 + 比较性能 (用 cudaEvent 计时)
//
// SMEM 转置算法:
//   每个 Block 处理一个 TILE × TILE 子矩阵
//   1. 读 A: tile[ty][tx] = A[row][col]
//   2. __syncthreads()
//   3. 写 B (转置着读 SMEM, 合并写 B):
//      把 (blockIdx.x, blockIdx.y) 互换, 然后
//      B[new_row][new_col] = tile[tx][ty]   ← 这一步可能产生 Bank Conflict
//   padded 版本: tile[TILE][TILE+1] → 消除冲突
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define TILE 32

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// TODO: 实现 SMEM 转置, 不加 padding
__global__ void transpose_smem_kernel(const float *A, float *B, int M, int N) {
    // --- 在这里写你的代码 ---

}

// TODO: 实现 SMEM 转置 + padding (tile[TILE][TILE+1])
__global__ void transpose_smem_padded_kernel(const float *A, float *B,
                                             int M, int N) {
    // --- 在这里写你的代码 ---

}

int main() {
    const int M = 1024, N = 1024;
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
    //   1. cudaMalloc d_A, d_B
    //   2. cudaMemcpy h_A → d_A
    //   3. 配置 dim3 block(TILE, TILE), dim3 grid(...)
    //   4. 启动 transpose_smem_kernel, 验证 → 计时
    //   5. 启动 transpose_smem_padded_kernel, 验证 → 计时
    //   6. 打印加速比
    //   7. cudaFree
    // ============================================================

    // --- 在这里写你的代码 ---

    free(h_A); free(h_B); free(h_B_ref);
    return 0;
}
