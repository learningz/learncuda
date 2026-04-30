#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ============================================================
// 09: 手写 Tiled GEMM with Register Blocking
//
// 配合理论: theory/05_operator_development.md 5.5 节 (GEMM 优化层级)
//           theory/05_operator_development.md 5.9 节 (Register Tiling 手推)
//
// 在 02_matrix_mul 的基础上进一步优化:
//   02 版本: 每线程计算 C 的 1 个元素 (Thread Tile = 1×1)
//   本版本: 每线程计算 C 的 TM×TN 个元素 (Thread Tile = 4×4)
//
// 为什么更快?
//   从 Shared Memory 加载 TM+TN 个元素 → 做 TM×TN 次乘加
//   数据复用率 = TM×TN / (TM+TN) = 4×4 / (4+4) = 2.0
//   vs 1×1 版本的复用率 = 1×1 / (1+1) = 0.5
//   → Shared Memory 读取量减少 4× → 性能更高!
//
// 新概念:
//   Register Blocking / Thread Tile:
//     每个线程在寄存器中维护一个 TM×TN 的累加器数组。
//     K-loop 的每一步, 从 Shared Memory 加载 A 的 TM 个元素和 B 的 TN 个元素,
//     然后做 TM×TN 次 FMA (全在寄存器中, 不额外访问任何内存)。
//
//   外积 (Outer Product):
//     A_frag[TM] × B_frag[TN] → C_acc[TM][TN] 的累加
//     这就是 "Register Tile" 的计算核心。
// ============================================================

#define BM 64       // Block Tile 行
#define BN 64       // Block Tile 列
#define BK 8        // K 方向的 Tile
#define TM 4        // Thread Tile 行 (每线程计算 4 行)
#define TN 4        // Thread Tile 列 (每线程计算 4 列)

// Block 内线程排列: (BM/TM) × (BN/TN) = 16 × 16 = 256 线程
#define BLOCK_DIM_X (BN / TN)  // 16
#define BLOCK_DIM_Y (BM / TM)  // 16

#define CUDA_CHECK(call) do {                                              \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                cudaGetErrorString(err)); exit(1); }                       \
} while(0)

__global__ void gemm_register_tiled(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    int M, int K, int N) {

    // Shared Memory: 存放 A 和 B 的当前 tile
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // 每线程的累加器: TM×TN 个寄存器 (= 4×4 = 16 个 float)
    float c_reg[TM][TN] = {0.0f};
    // 每线程的 A/B 片段 (从 Shared Memory 加载)
    float a_frag[TM];
    float b_frag[TN];

    // 线程在 Block 内的位置
    int tx = threadIdx.x;  // 0..15 (列方向)
    int ty = threadIdx.y;  // 0..15 (行方向)

    // 本 Block 负责的 C 子矩阵的起始位置
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // 沿 K 维度循环
    for (int k_tile = 0; k_tile < K; k_tile += BK) {

        // ---- 协作加载 A[BM×BK] 和 B[BK×BN] 到 Shared Memory ----
        // 每个线程加载多个元素 (因为 BM*BK / blockDim = 64*8/256 = 2)
        for (int load_idx = ty * BLOCK_DIM_X + tx;
             load_idx < BM * BK;
             load_idx += BLOCK_DIM_X * BLOCK_DIM_Y) {
            int li = load_idx / BK;
            int lj = load_idx % BK;
            int gi = block_row + li;
            int gj = k_tile + lj;
            As[li][lj] = (gi < M && gj < K) ? A[gi * K + gj] : 0.0f;
        }
        for (int load_idx = ty * BLOCK_DIM_X + tx;
             load_idx < BK * BN;
             load_idx += BLOCK_DIM_X * BLOCK_DIM_Y) {
            int li = load_idx / BN;
            int lj = load_idx % BN;
            int gi = k_tile + li;
            int gj = block_col + lj;
            Bs[li][lj] = (gi < K && gj < N) ? B[gi * N + gj] : 0.0f;
        }
        __syncthreads();

        // ---- 在 Shared Memory 上做计算 (Register Tiling) ----
        for (int k = 0; k < BK; k++) {
            // 从 Shared Memory 加载 A 的 TM 个元素到寄存器
            for (int i = 0; i < TM; i++) {
                a_frag[i] = As[ty * TM + i][k];
            }
            // 从 Shared Memory 加载 B 的 TN 个元素到寄存器
            for (int j = 0; j < TN; j++) {
                b_frag[j] = Bs[k][tx * TN + j];
            }
            // 外积累加: TM × TN 次 FMA, 全在寄存器中!
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    c_reg[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }
        __syncthreads();
    }

    // ---- 写回结果 ----
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            int gi = block_row + ty * TM + i;
            int gj = block_col + tx * TN + j;
            if (gi < M && gj < N) {
                C[gi * N + gj] = c_reg[i][j];
            }
        }
    }
}

// 朴素版本 (对比用)
__global__ void gemm_naive(const float *A, const float *B, float *C,
                            int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0;
        for (int k = 0; k < K; k++) sum += A[row*K+k] * B[k*N+col];
        C[row*N+col] = sum;
    }
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
    const int M = 1024, K = 1024, NN = 1024;
    size_t sA = M*K*sizeof(float), sB = K*NN*sizeof(float), sC = M*NN*sizeof(float);

    float *hA=(float*)malloc(sA), *hB=(float*)malloc(sB);
    float *hC_ref=(float*)malloc(sC), *hC=(float*)malloc(sC);
    srand(42);
    for(int i=0;i<M*K;i++) hA[i]=(rand()%100)/100.f;
    for(int i=0;i<K*NN;i++) hB[i]=(rand()%100)/100.f;
    cpu_gemm(hA, hB, hC_ref, M, K, NN);

    float *dA,*dB,*dC;
    CUDA_CHECK(cudaMalloc(&dA,sA));
    CUDA_CHECK(cudaMalloc(&dB,sB));
    CUDA_CHECK(cudaMalloc(&dC,sC));
    CUDA_CHECK(cudaMemcpy(dA,hA,sA,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB,hB,sB,cudaMemcpyHostToDevice));

    printf("GEMM %d×%d × %d×%d (Register Tiling: TM=%d, TN=%d)\n\n", M,K,K,NN,TM,TN);

    #define BENCH(name, kernel, grid_cfg, block_cfg, ...) {                 \
        kernel<<<grid_cfg, block_cfg>>>(__VA_ARGS__); cudaDeviceSynchronize(); \
        CUDA_CHECK(cudaMemcpy(hC,dC,sC,cudaMemcpyDeviceToHost));           \
        float maxerr=0; for(int i=0;i<M*NN;i++) maxerr=fmaxf(maxerr,fabsf(hC[i]-hC_ref[i])); \
        cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);     \
        cudaEventRecord(t0);                                                \
        for(int _=0;_<100;_++){ kernel<<<grid_cfg, block_cfg>>>(__VA_ARGS__); } \
        cudaEventRecord(t1); cudaEventSynchronize(t1);                      \
        float ms; cudaEventElapsedTime(&ms,t0,t1); ms/=100;                \
        double gflops=2.0*M*K*NN/(ms*1e6);                                 \
        printf("  %-25s: %.3f ms  %.1f GFLOPS  maxerr=%.2e\n",name,ms,gflops,maxerr); \
        cudaEventDestroy(t0); cudaEventDestroy(t1); }

    dim3 naive_block(16,16);
    dim3 naive_grid((NN+15)/16, (M+15)/16);
    BENCH("Naive (1×1 per thread)",
          gemm_naive, naive_grid, naive_block,
          dA,dB,dC,M,K,NN);

    dim3 reg_block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 reg_grid((NN+BN-1)/BN, (M+BM-1)/BM);
    BENCH("Register Tiled (4×4)",
          gemm_register_tiled, reg_grid, reg_block,
          dA,dB,dC,M,K,NN);

    printf("\n为什么 Register Tiled 更快?\n");
    printf("  每线程从 SMEM 加载 TM+TN=%d 个值 → 做 TM×TN=%d 次 FMA\n", TM+TN, TM*TN);
    printf("  数据复用率 = %d/%d = %.1f (vs 1×1 的 0.5)\n", TM*TN, TM+TN, (float)(TM*TN)/(TM+TN));
    printf("  SMEM 读取量减少 → 计算/访存比提高 → 性能提升!\n");
    printf("\n更进一步: + 向量化加载 + 双缓冲 + Tensor Core → 见 theory/06\n");

    #undef BENCH
    cudaFree(dA);cudaFree(dB);cudaFree(dC);
    free(hA);free(hB);free(hC_ref);free(hC);
    return 0;
}
