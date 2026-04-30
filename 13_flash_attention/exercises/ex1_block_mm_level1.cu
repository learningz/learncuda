// ============================================================
// 练习 1: 分块矩阵乘 (Block Tiled MM) — FlashAttention 的子问题
// 难度: Level 1 (只需填写 kernel 函数体)
//
// 编译: nvcc -O2 -o ex1_block_mm_level1 ex1_block_mm_level1.cu
// 运行: ./ex1_block_mm_level1
// 预期输出: "结果: ✓ 全部正确"
//
// FlashAttention 的核心操作之一: 把 Q 的一块 × K^T 的一块 做小矩阵乘,
// 结果存在 Shared Memory 或寄存器中。
//
// 本题: 每个 Block 计算 C[TILE_M × TILE_N] = A[TILE_M × K] × B[K × TILE_N]
//   Block(x) 确定 N 方向位置, Block(y) 确定 M 方向位置
//   在 K 维度上分 tile, 每次搬 A 和 B 的一小块到 SMEM, 计算后累加
//
// 和 matmul_tiled 完全一样, 但这里让你自己从零写一遍。
//
// 提示:
//   - __shared__ float As[TILE][TILE], Bs[TILE][TILE]
//   - 外层: for t in 0..ceil(K/TILE)-1
//   - 搬运: As[ty][tx] = A[row*K + t*TILE + tx]  (注意边界)
//   - 计算: for k in 0..TILE-1: sum += As[ty][k] * Bs[k][tx]
//   - 两次 __syncthreads (搬完等 + 算完等)
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

// TODO: 实现 tiled matmul kernel (和 matmul_tiled 相同结构)
__global__ void block_mm_kernel(const float *A, const float *B, float *C,
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
    const int M = 256, K = 512, N = 256;
    size_t sA = M*K*4, sB = K*N*4, sC = M*N*4;

    float *hA=(float*)malloc(sA), *hB=(float*)malloc(sB);
    float *hC=(float*)malloc(sC), *hRef=(float*)malloc(sC);
    srand(42);
    for (int i=0;i<M*K;i++) hA[i]=(rand()%100)/100.f;
    for (int i=0;i<K*N;i++) hB[i]=(rand()%100)/100.f;
    cpu_matmul(hA,hB,hRef,M,K,N);

    float *dA,*dB,*dC;
    CUDA_CHECK(cudaMalloc(&dA,sA));
    CUDA_CHECK(cudaMalloc(&dB,sB));
    CUDA_CHECK(cudaMalloc(&dC,sC));
    CUDA_CHECK(cudaMemcpy(dA,hA,sA,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB,hB,sB,cudaMemcpyHostToDevice));

    dim3 block(TILE,TILE);
    dim3 grid((N+TILE-1)/TILE,(M+TILE-1)/TILE);
    block_mm_kernel<<<grid,block>>>(dA,dB,dC,M,K,N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(hC,dC,sC,cudaMemcpyDeviceToHost));

    float maxerr=0;
    for(int i=0;i<M*N;i++) maxerr=fmaxf(maxerr,fabsf(hC[i]-hRef[i]));
    printf("最大误差: %.2e\n", maxerr);
    printf("结果: %s\n", maxerr<0.1f ? "✓ 全部正确" : "✗ 有错误");

    cudaFree(dA);cudaFree(dB);cudaFree(dC);
    free(hA);free(hB);free(hC);free(hRef);
    return 0;
}
