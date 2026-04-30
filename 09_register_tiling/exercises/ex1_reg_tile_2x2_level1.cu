// ============================================================
// 练习 1: 2×2 Register Tiled GEMM — 简化版 Register Blocking
// 难度: Level 1 (只需填写 kernel 函数体)
//
// 编译: nvcc -O2 -o ex1_reg_tile_2x2_level1 ex1_reg_tile_2x2_level1.cu
// 运行: ./ex1_reg_tile_2x2_level1
// 预期输出: "结果: ✓ 全部正确"
//
// gemm_register.cu 用 4×4 的 Thread Tile, 本题简化成 2×2。
// 每个线程计算 C 的 2×2 个元素, 需要从 SMEM 加载 2+2=4 个值,
// 做 2×2=4 次 FMA, 数据复用率 = 4/(2+2) = 1.0 (vs 1×1 的 0.5)。
//
// 算法 (和 gemm_register.cu 一样, 只是 TM=TN=2):
//   每个 Block 处理 C 的一个 BM×BN 子块
//   K-loop 每步:
//     1. Block 内线程协作把 A[BM×BK] 和 B[BK×BN] 搬到 SMEM
//     2. __syncthreads()
//     3. 内层 k-loop: 每个线程从 SMEM 加载 A 的 2 个元素 + B 的 2 个元素
//        做 2×2 外积累加到寄存器 c[2][2]
//     4. __syncthreads()
//   最后: 把 c[2][2] 写回 C
//
// 提示:
//   - BM=BN=64, BK=16, TM=TN=2
//   - Block 大小 = (BN/TN) × (BM/TM) = 32 × 32 = 1024 线程
//   - 太多了! 改成 BM=BN=32, 那 Block = 16×16 = 256 线程
//   - 或者保持 BM=BN=64 但 Block = 32×32, 每线程算 2×2
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define BM 32
#define BN 32
#define BK 16
#define TM 2
#define TN 2

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
// TODO: 实现 2×2 register tiled GEMM kernel
//
// Block 配置: dim3 block(BN/TN, BM/TM) = (16, 16) = 256 线程
// Grid 配置:  dim3 grid(ceil(N/BN), ceil(M/BM))
//
// 每个线程:
//   float c[TM][TN] = {0};   // 2×2 累加器
//   K-loop (每步 BK):
//     搬 A tile [BM][BK] 和 B tile [BK][BN] 到 SMEM
//     __syncthreads()
//     内层 for k in 0..BK-1:
//       float a[TM], b[TN];  // 从 SMEM 加载
//       a[0] = As[threadIdx.y * TM + 0][k];
//       a[1] = As[threadIdx.y * TM + 1][k];
//       b[0] = Bs[k][threadIdx.x * TN + 0];
//       b[1] = Bs[k][threadIdx.x * TN + 1];
//       for i,j: c[i][j] += a[i] * b[j];   // 2×2 外积
//     __syncthreads()
//   写回 C
// ============================================================
__global__ void gemm_reg2x2(const float *A, const float *B, float *C,
                            int M, int K, int N) {
    // --- 在这里写你的代码 ---

}

void cpu_gemm(const float *A, const float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < K; k++) s += A[i*K+k] * B[k*N+j];
            C[i*N+j] = s;
        }
}

int main() {
    const int M = 256, K = 256, N = 256;
    size_t sA = M*K*sizeof(float), sB = K*N*sizeof(float), sC = M*N*sizeof(float);

    float *hA = (float*)malloc(sA), *hB = (float*)malloc(sB);
    float *hC = (float*)malloc(sC), *hC_ref = (float*)malloc(sC);
    srand(42);
    for (int i = 0; i < M*K; i++) hA[i] = (rand()%100)/100.f;
    for (int i = 0; i < K*N; i++) hB[i] = (rand()%100)/100.f;
    cpu_gemm(hA, hB, hC_ref, M, K, N);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, sA));
    CUDA_CHECK(cudaMalloc(&dB, sB));
    CUDA_CHECK(cudaMalloc(&dC, sC));
    CUDA_CHECK(cudaMemcpy(dA, hA, sA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sB, cudaMemcpyHostToDevice));

    dim3 block(BN/TN, BM/TM);
    dim3 grid((N+BN-1)/BN, (M+BM-1)/BM);
    gemm_reg2x2<<<grid, block>>>(dA, dB, dC, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(hC, dC, sC, cudaMemcpyDeviceToHost));

    float maxerr = 0;
    for (int i = 0; i < M*N; i++) maxerr = fmaxf(maxerr, fabsf(hC[i]-hC_ref[i]));
    printf("最大误差: %.2e\n", maxerr);
    printf("结果: %s\n", maxerr < 0.1f ? "✓ 全部正确" : "✗ 有错误");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC); free(hC_ref);
    return 0;
}
