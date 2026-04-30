// ============================================================
// 练习 2: SMEM 矩阵转置 — 对比有/无 padding 的性能
// 难度: Level 1 (只需填写两个 kernel 的函数体)
//
// 编译: nvcc -O2 -o ex2_transpose_smem_level1 ex2_transpose_smem_level1.cu
// 运行: ./ex2_transpose_smem_level1
// 预期输出: 正确性通过, 且 padded 版本更快
//
// 背景:
//   01 的练习中你写过朴素矩阵转置: 直接从全局显存读 A[row][col], 写 B[col][row]。
//   写 B 时, 相邻线程的地址不连续 (stride = M) → 非合并写入!
//
//   改进: 先把 A 的一个 TILE 读到 SMEM, 然后从 SMEM 转置着读, 再写到 B。
//   这样写 B 也是合并的! 但 SMEM 中的转置读取可能产生 Bank Conflict。
//   解决: 给 SMEM 加 padding。
//
// TODO:
//   1. transpose_smem_kernel: SMEM 转置, 不加 padding (可能有冲突)
//   2. transpose_smem_padded_kernel: SMEM 转置 + padding (消除冲突)
//
// 提示:
//   每个 Block 处理一个 TILE × TILE 的子矩阵:
//     a. 协作读取 A 到 SMEM: tile[threadIdx.y][threadIdx.x] = A[row][col]
//     b. __syncthreads()
//     c. 从 SMEM 转置读出, 合并写 B:
//        新的 row/col 对应输出矩阵的位置
//        B[new_row][new_col] = tile[threadIdx.x][threadIdx.y]  ← 注意 x/y 交换!
//     d. padded 版本: tile 声明为 tile[TILE][TILE+1], 读写都用 TILE+1 的 stride
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

// ============================================================
// TODO: 实现 SMEM 转置, 不加 padding
// ============================================================
__global__ void transpose_smem_kernel(const float *A, float *B, int M, int N) {
    // --- 在这里写你的代码 ---

}

// ============================================================
// TODO: 实现 SMEM 转置 + padding
// 唯一区别: __shared__ float tile[TILE][TILE+1] (多 1 列)
// ============================================================
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

    float *d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    transpose_smem_kernel<<<grid, block>>>(d_A, d_B, M, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_B, d_B, sizeB, cudaMemcpyDeviceToHost));
    bool ok1 = true;
    for (int i = 0; i < N * M; i++)
        if (h_B[i] != h_B_ref[i]) { ok1 = false; break; }

    transpose_smem_padded_kernel<<<grid, block>>>(d_A, d_B, M, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_B, d_B, sizeB, cudaMemcpyDeviceToHost));
    bool ok2 = true;
    for (int i = 0; i < N * M; i++)
        if (h_B[i] != h_B_ref[i]) { ok2 = false; break; }

    printf("正确性: 无padding=%s  有padding=%s\n",
           ok1 ? "✓" : "✗", ok2 ? "✓" : "✗");

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    auto bench = [&](auto kernel_fn) {
        kernel_fn<<<grid, block>>>(d_A, d_B, M, N); cudaDeviceSynchronize();
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) kernel_fn<<<grid, block>>>(d_A, d_B, M, N);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        return ms / 100.0f;
    };

    float t1 = bench(transpose_smem_kernel);
    float t2 = bench(transpose_smem_padded_kernel);

    printf("性能: 无padding=%.3f ms  有padding=%.3f ms  加速=%.2fx\n", t1, t2, t1 / t2);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B);
    free(h_A); free(h_B); free(h_B_ref);
    return 0;
}
