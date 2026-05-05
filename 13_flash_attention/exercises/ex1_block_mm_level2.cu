// ============================================================
// 练习 1: 分块矩阵乘 (Block Tiled MM) — FlashAttention 的子问题
// 难度: Level 2 (kernel + host 端都要自己写)
//
// 编译: nvcc -O2 -o ex1_block_mm_level2 ex1_block_mm_level2.cu
// 运行: ./ex1_block_mm_level2
// 预期输出: "结果: ✓ 全部正确" + Tiled 版比朴素版快
//
// 要求:
//   1. 实现 matmul_naive (每个线程从全局内存直接读, 不 Tiling)
//   2. 实现 matmul_tiled (Shared Memory Tiling, TILE=16)
//   3. 在 main 中完成内存管理、数据传输、cudaEvent 计时、验证
//
// Tiled 版步骤:
//   - Block 大小: dim3(TILE, TILE) = 256 线程
//   - Grid 大小:  dim3(ceil(N/TILE), ceil(M/TILE))
//   - 每个线程计算 C 的 1 个元素, 用 SMEM tiling 减少全局内存访问
//
// 提示:
//   - As[threadIdx.y][threadIdx.x] = A[row*K + t*TILE + threadIdx.x]
//   - Bs[threadIdx.y][threadIdx.x] = B[(t*TILE+threadIdx.y)*N + col]
//   - 边界检查: row < M, col < N, t*TILE+tx < K 等
//   - 两次 __syncthreads(): 搬运完 → 同步 → 计算 → 同步
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

#define TILE 16

// TODO: 实现朴素 GEMM — 直接从全局内存读
__global__ void matmul_naive(const float *A, const float *B, float *C,
                              int M, int K, int N) {
    // --- 在这里写你的代码 ---

}

// TODO: 实现 Tiled GEMM — 用 Shared Memory Tiling
__global__ void matmul_tiled(const float *A, const float *B, float *C,
                              int M, int K, int N) {
    // --- 在这里写你的代码 ---

}

void cpu_matmul(const float *A, const float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < K; k++) s += A[i*K+k] * B[k*N+j];
            C[i*N+j] = s;
        }
}

int main() {
    const int M = 512, K = 512, N = 512;

    // ============================================================
    // TODO: 完成以下步骤
    //   1. malloc CPU 内存 (hA, hB, hC, hRef)
    //   2. 初始化 hA, hB (随机值), 用 cpu_matmul 算 hRef
    //   3. cudaMalloc GPU 内存 (dA, dB, dC)
    //   4. cudaMemcpy H2D (A, B)
    //   5. 分别 launch naive 和 tiled, cudaEvent 计时
    //   6. 每次 cudaMemcpy D2H, 验证正确性 (max error < 0.1)
    //   7. 打印耗时对比
    //   8. cudaFree + free
    // ============================================================

    // --- 在这里写你的代码 ---

    return 0;
}
