// ============================================================
// 练习 3: GEMV 矩阵-向量乘 (Shared Memory 归约) — y[i] = Σ_k A[i][k] * x[k]
// 难度: Level 1 (只需填写 kernel 函数体)
//
// 编译: nvcc -O2 -o ex3_gemv_tiled_level1 ex3_gemv_tiled_level1.cu
// 运行: ./ex3_gemv_tiled_level1
// 预期输出: "结果: ✓ 全部正确"
//
// 这道题完整复刻 matmul_tiled 的核心模式:
//   __shared__ + 协作写入 + __syncthreads() + 归约读取
// 但比 matmul 简单 — 输出是 1D 的 y[M], 结构更小更容易上手。
//
// 配置: <<<M, BLOCK_SIZE>>>
//   每个 Block 负责 1 行: blockIdx.x = row
//   Block 内 BLOCK_SIZE 个线程一起算这一行的点积 Σ A[row][k] * x[k]
//
// 算法 (两个阶段):
//
//   阶段 1: 每个线程算自己的部分和 (不需要 SMEM)
//     线程 tid 负责 k = tid, tid+BLOCK_SIZE, tid+2*BLOCK_SIZE, ...
//     循环累加 A[row*K + k] * x[k] 到寄存器变量 local_sum
//
//   阶段 2: Block 内所有线程的 local_sum 用 SMEM 归约
//     a. 每个线程把 local_sum 写到 __shared__ float partial[BLOCK_SIZE]
//     b. __syncthreads()  ← 等所有线程都写完!
//     c. 逐步对半折叠:
//        for (int s = BLOCK_SIZE/2; s > 0; s >>= 1) {
//            if (tid < s) partial[tid] += partial[tid + s];
//            __syncthreads();  ← 必须在 if 外面! 否则死锁!
//        }
//     d. 线程 0 把 partial[0] 写到 y[row]
//
// 提示:
//   - BLOCK_SIZE = 256, 是 2 的幂 → 可以一直对半折叠
//   - 阶段 1 用 grid-stride: for (int k = tid; k < K; k += BLOCK_SIZE)
//   - 阶段 2 的 __syncthreads() 一定要放在 if 外面
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

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
// ============================================================
__global__ void gemv_kernel(const float *A, const float *x, float *y,
                            int M, int K) {
    // --- 在这里写你的代码 ---

}

int main() {
    const int M = 512, K = 2048;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeX = K * sizeof(float);
    size_t sizeY = M * sizeof(float);

    float *h_A = (float *)malloc(sizeA);
    float *h_x = (float *)malloc(sizeX);
    float *h_y = (float *)malloc(sizeY);
    float *h_y_ref = (float *)malloc(sizeY);

    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K; i++) h_x[i] = (float)(rand() % 100) / 100.0f;

    for (int i = 0; i < M; i++) {
        float sum = 0;
        for (int k = 0; k < K; k++) sum += h_A[i * K + k] * h_x[k];
        h_y_ref[i] = sum;
    }

    float *d_A, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_x, sizeX));
    CUDA_CHECK(cudaMalloc(&d_y, sizeY));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, sizeX, cudaMemcpyHostToDevice));

    gemv_kernel<<<M, BLOCK_SIZE>>>(d_A, d_x, d_y, M, K);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_y, d_y, sizeY, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < M; i++) {
        if (fabsf(h_y[i] - h_y_ref[i]) > 0.1f) {
            printf("验证失败 @ y[%d]: got %.4f, expected %.4f\n", i, h_y[i], h_y_ref[i]);
            ok = false;
            break;
        }
    }
    printf("结果: %s\n", ok ? "✓ 全部正确" : "✗ 有错误");

    cudaFree(d_A); cudaFree(d_x); cudaFree(d_y);
    free(h_A); free(h_x); free(h_y); free(h_y_ref);
    return 0;
}
