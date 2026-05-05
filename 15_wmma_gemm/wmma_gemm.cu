#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// ============================================================
// 15: WMMA 入门 — 你的第一个 Tensor Core 程序
//
// 配合理论: theory/06_tensor_core.md 6.3 节 (WMMA API)
//
// WMMA (Warp Matrix Multiply-Accumulate):
//   CUDA 提供的 C++ 级 Tensor Core API。
//   一个 Warp (32 线程) 协作完成一个小矩阵乘: D = A × B + C
//
// 本程序从零演示:
//   1. 分配 FP16 矩阵 A, B 和 FP32 矩阵 C (累加器)
//   2. 加载到 WMMA Fragment (矩阵被切碎分散到 32 个线程的寄存器中)
//   3. 调用 wmma::mma_sync 执行矩阵乘
//   4. 把结果存回内存, 对比 CPU 验证正确性
//
// 关键概念:
//   Fragment: WMMA 的核心数据类型。一个 16×16 的矩阵不是存在一个数组里,
//     而是分散在一个 Warp 的 32 个线程的寄存器中。每个线程持有矩阵的几个元素。
//     你不需要知道哪个线程持有哪个元素——WMMA API 自动处理。
//
//   wmma::load_matrix_sync: 从全局/Shared Memory 加载矩阵到 Fragment
//   wmma::mma_sync: 执行 D = A × B + C (一条 Tensor Core 指令!)
//   wmma::store_matrix_sync: 把 Fragment 结果存回内存
//
// 编译: nvcc -arch=sm_70 -o wmma_gemm wmma_gemm.cu  (需要 Volta+)
// ============================================================

#define CUDA_CHECK(call) do {                                              \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                cudaGetErrorString(err)); exit(1); }                       \
} while(0)

// WMMA 的 tile 大小: 16×16×16
// A: 16×16 (FP16), B: 16×16 (FP16), C/D: 16×16 (FP32)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// 矩阵大小 (必须是 WMMA tile 大小的倍数)
// 用 512 可以比 64 更好地展示 Tensor Core 的优势
#define M 512
#define N 512
#define K 512

// ---- 对比用: FP32 naive GEMM ----
__global__ void gemm_fp32_naive(const float *A, const float *B, float *C,
                                  int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;
    float sum = 0.0f;
    for (int i = 0; i < k; i++) sum += A[row * k + i] * B[i * n + col];
    C[row * n + col] = sum;
}

// ---- 对比用: FP16 naive GEMM (CUDA Core) ----
__global__ void gemm_fp16_naive(const __half *A, const __half *B, float *C,
                                  int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;
    float sum = 0.0f;
    for (int i = 0; i < k; i++)
        sum += __half2float(A[row * k + i]) * __half2float(B[i * n + col]);
    C[row * n + col] = sum;
}

// ---- 对比用: FP32 Shared Memory Tiled GEMM ----
#define TILE_SIZE 16
__global__ void gemm_fp32_tiled(const float *A, const float *B, float *C,
                                  int m, int n, int k) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < k; t += TILE_SIZE) {
        As[threadIdx.y][threadIdx.x] = (row < m && t + threadIdx.x < k)
            ? A[row * k + t + threadIdx.x] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (t + threadIdx.y < k && col < n)
            ? B[(t + threadIdx.y) * n + col] : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE_SIZE; i++) sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        __syncthreads();
    }
    if (row < m && col < n) C[row * n + col] = sum;
}

// ---- WMMA Kernel ----
// 每个 Warp 计算 C 的一个 16×16 tile
// Grid/Block 配置: 每个 Block 有多个 Warp, 每个 Warp 独立处理一个 C tile
__global__ void wmma_gemm(const half *A, const half *B, float *C,
                           int m, int n, int k) {
    // 计算本 Warp 负责 C 的哪个 16×16 tile
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    // warpM: 第几行 tile (0-based), warpN: 第几列 tile

    if (warpM * WMMA_M >= m || warpN * WMMA_N >= n) return;

    // ---- 声明 Fragment ----
    // Fragment 是 "分散在 32 个线程寄存器中的矩阵块"
    // 你不能直接索引它的元素 (element[i][j])
    // 只能通过 load/store/mma 操作整个 Fragment

    // A Fragment: 16×16 FP16, 行主序 (row_major)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;

    // B Fragment: 16×16 FP16, 列主序 (col_major)
    // 注意: B 用列主序! 这是因为 Tensor Core 的硬件要求。
    // 如果你的 B 是行主序, 需要转置或用 wmma::row_major 并转置 B。
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;

    // C/D Fragment: 16×16 FP32 (累加器)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // 初始化累加器为 0
    wmma::fill_fragment(c_frag, 0.0f);

    // ---- 沿 K 维度循环, 做分块矩阵乘 ----
    for (int ki = 0; ki < k; ki += WMMA_K) {
        // 从全局内存加载 A 的 [warpM*16 : warpM*16+16, ki : ki+16] tile
        wmma::load_matrix_sync(a_frag,
            A + warpM * WMMA_M * k + ki,   // 基址: A[warpM*16][ki]
            k);                              // leading dimension (行宽)

        // 从全局内存加载 B 的 [ki : ki+16, warpN*16 : warpN*16+16] tile
        // 注意: B 是列主序! 所以 leading dimension 是 k (不是 n)
        wmma::load_matrix_sync(b_frag,
            B + ki + warpN * WMMA_N * k,    // 列主序基址
            k);                              // leading dimension

        // 矩阵乘累加: C += A × B
        // 这一条调用在 GPU 上编译为 HMMA 指令 → Tensor Core 执行!
        // 16×16×16 的乘加 = 8192 FLOP, 在几个周期内完成 (vs CUDA Core 要 ~128 周期)
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // ---- 把结果存回全局内存 ----
    wmma::store_matrix_sync(
        C + warpM * WMMA_M * n + warpN * WMMA_N,  // C[warpM*16][warpN*16]
        c_frag,
        n,                      // leading dimension
        wmma::mem_row_major);   // 输出用行主序
}

// CPU 参考 (FP32, 对比正确性)
void cpu_gemm(const half *A, const half *B, float *C, int m, int n, int k) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int ki = 0; ki < k; ki++)
                sum += __half2float(A[i*k+ki]) * __half2float(B[ki+j*k]); // B 列主序!
            C[i*n+j] = sum;
        }
}

int main() {
    printf("WMMA GEMM: C[%d×%d] = A[%d×%d] × B[%d×%d]  (FP16 input, FP32 accumulate)\n\n",
           M, N, M, K, K, N);

    size_t sA = M * K * sizeof(half);
    size_t sB = K * N * sizeof(half);
    size_t sC = M * N * sizeof(float);

    // 主机端分配 + 初始化
    half *hA = (half*)malloc(sA);
    half *hB = (half*)malloc(sB);
    float *hC_ref = (float*)malloc(sC);
    float *hC_gpu = (float*)malloc(sC);

    srand(42);
    for (int i = 0; i < M*K; i++) hA[i] = __float2half((rand()%10 - 5) / 5.0f);
    for (int i = 0; i < K*N; i++) hB[i] = __float2half((rand()%10 - 5) / 5.0f);

    cpu_gemm(hA, hB, hC_ref, M, N, K);

    // GPU 显存分配
    half *dA, *dB; float *dC;
    CUDA_CHECK(cudaMalloc(&dA, sA));
    CUDA_CHECK(cudaMalloc(&dB, sB));
    CUDA_CHECK(cudaMalloc(&dC, sC));
    CUDA_CHECK(cudaMemcpy(dA, hA, sA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sB, cudaMemcpyHostToDevice));

    // 辅助: 计时 Lambda
    auto time_kernel = [&](const char *label, auto launch_fn, int warmup, int iters) {
        for (int r = 0; r < warmup; r++) launch_fn();
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0));
        for (int r = 0; r < iters; r++) launch_fn();
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        ms /= iters;
        double gflops = 2.0 * M * N * K / (ms * 1e6);
        double tflops_peak = 19.5; // A100 FP32 peak
        printf("  %-40s %8.3f ms  %8.1f GFLOPS  (%5.1f%% peak)\n",
               label, ms, gflops, gflops / (tflops_peak * 1e3) * 100);
        CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    };

    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("性能对比: 不同 GEMM 实现 (M=N=K=%d)\n", M);
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // ---- 1. FP32 naive ----
    {
        size_t b32 = M * K * sizeof(float);
        float *hA32 = (float*)malloc(b32), *hB32 = (float*)malloc(b32);
        for (int i = 0; i < M*K; i++) hA32[i] = (rand()%10-5)/5.0f;
        for (int i = 0; i < K*N; i++) hB32[i] = (rand()%10-5)/5.0f;
        float *dA32, *dB32, *dC32;
        CUDA_CHECK(cudaMalloc(&dA32, b32));
        CUDA_CHECK(cudaMalloc(&dB32, b32));
        CUDA_CHECK(cudaMalloc(&dC32, M*N*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA32, hA32, b32, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB32, hB32, b32, cudaMemcpyHostToDevice));
        dim3 b1(16, 16), g1((N+15)/16, (M+15)/16);
        time_kernel("1. FP32 naive (CUDA Core, no tiling)",
            [&]{ gemm_fp32_naive<<<g1, b1>>>(dA32, dB32, dC32, M, N, K); }, 5, 20);
        CUDA_CHECK(cudaFree(dA32)); CUDA_CHECK(cudaFree(dB32)); CUDA_CHECK(cudaFree(dC32));
        free(hA32); free(hB32);
    }

    // ---- 2. FP32 tiled (Shared Memory) ----
    {
        size_t b32 = M * K * sizeof(float);
        float *hA32 = (float*)malloc(b32), *hB32 = (float*)malloc(b32);
        for (int i = 0; i < M*K; i++) hA32[i] = (rand()%10-5)/5.0f;
        for (int i = 0; i < K*N; i++) hB32[i] = (rand()%10-5)/5.0f;
        float *dA32, *dB32, *dC32;
        CUDA_CHECK(cudaMalloc(&dA32, b32));
        CUDA_CHECK(cudaMalloc(&dB32, b32));
        CUDA_CHECK(cudaMalloc(&dC32, M*N*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA32, hA32, b32, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB32, hB32, b32, cudaMemcpyHostToDevice));
        dim3 b2(TILE_SIZE, TILE_SIZE), g2((N+TILE_SIZE-1)/TILE_SIZE, (M+TILE_SIZE-1)/TILE_SIZE);
        time_kernel("2. FP32 tiled (Shared Memory, CUDA Core)",
            [&]{ gemm_fp32_tiled<<<g2, b2>>>(dA32, dB32, dC32, M, N, K); }, 5, 20);
        CUDA_CHECK(cudaFree(dA32)); CUDA_CHECK(cudaFree(dB32)); CUDA_CHECK(cudaFree(dC32));
        free(hA32); free(hB32);
    }

    // ---- 3. FP16 naive (CUDA Core) ----
    {
        size_t b16 = M * K * sizeof(half);
        half *hA16 = (half*)malloc(b16), *hB16 = (half*)malloc(b16);
        for (int i = 0; i < M*K; i++) hA16[i] = __float2half((rand()%10-5)/5.0f);
        for (int i = 0; i < K*N; i++) hB16[i] = __float2half((rand()%10-5)/5.0f);
        half *dA16, *dB16; float *dC16;
        CUDA_CHECK(cudaMalloc(&dA16, b16));
        CUDA_CHECK(cudaMalloc(&dB16, b16));
        CUDA_CHECK(cudaMalloc(&dC16, M*N*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA16, hA16, b16, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB16, hB16, b16, cudaMemcpyHostToDevice));
        dim3 b3(16, 16), g3((N+15)/16, (M+15)/16);
        time_kernel("3. FP16 naive (CUDA Core, no tiling)",
            [&]{ gemm_fp16_naive<<<g3, b3>>>(dA16, dB16, dC16, M, N, K); }, 5, 20);
        CUDA_CHECK(cudaFree(dA16)); CUDA_CHECK(cudaFree(dB16)); CUDA_CHECK(cudaFree(dC16));
        free(hA16); free(hB16);
    }

    // ---- 4. FP16 Tensor Core (WMMA) ----
    {
        dim3 block(128, 1);
        dim3 grid_wmma(N / WMMA_N / (128/32), M / WMMA_M);
        time_kernel("4. FP16 Tensor Core (WMMA, 16×16×16)",
            [&]{ wmma_gemm<<<grid_wmma, block>>>(dA, dB, dC, M, N, K); }, 10, 50);
    }

    printf("\n注意: FP32 peak = 19.5 TFLOPS (A100). FP16 Tensor Core peak = 312 TFLOPS.\n");
    printf("小矩阵受限于 launch overhead + 内存延迟, 大矩阵 (4096+) 才能接近峰值.\n");

    // ---- 验证 WMMA 正确性 ----
    printf("\n正确性验证:\n");
    wmma_gemm<<<dim3(N/WMMA_N/(128/32), M/WMMA_M), dim3(128,1)>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(hC_gpu, dC, sC, cudaMemcpyDeviceToHost));

    float max_err = 0;
    for (int i = 0; i < M*N; i++)
        max_err = fmaxf(max_err, fabsf(hC_ref[i] - hC_gpu[i]));
    printf("  FP16 WMMA vs CPU 参考: 最大误差 = %.4f  %s\n",
           max_err, max_err < 0.5f ? "(FP16 精度正常)" : "✗ 异常!");

    printf("\n关键点:\n");
    printf("  1. 代码中没有手写乘法循环! wmma::mma_sync 一条调用完成 16×16×16 乘加\n");
    printf("  2. Fragment 是分散在 32 线程中的 — 你不需要知道具体分布\n");
    printf("  3. FP16 输入 + FP32 累加 → Tensor Core 的标准使用模式\n");
    printf("  4. B 用列主序 → Tensor Core 的硬件要求\n");
    printf("  5. 对比 CUDA Core GEMM, 看 Tensor Core 的吞吐优势\n");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC_ref); free(hC_gpu);
    return 0;
}
