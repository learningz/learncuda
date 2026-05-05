#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ============================================================
// 17: 混合精度实战 — FP32 vs FP16 vs BF16 GEMM 全面对比
//
// 配合理论: theory/06_tensor_core.md 6.1-6.2 节
//
// 现代深度学习训练和推理几乎都使用混合精度。
// 本程序从零演示:
//   1. FP32 GEMM (naive, 作为基准)
//   2. FP16 GEMM (CUDA Core, 利用 FP16 ALU 的 2× 吞吐)
//   3. BF16 GEMM (CUDA Core, 和 FP16 同吞吐但范围大)
//   4. FP16 Tensor Core GEMM (WMMA, ~16× vs FP32)
//
// 你会看到:
//   - FP16/BF16 相比 FP32 的吞吐提升 (2× on Ampere)
//   - Tensor Core 相比 CUDA Core 的巨大优势 (~8×+)
//   - BF16 vs FP16 的数值精度对比
//   - 相同的内存带宽, 不同的计算吞吐 → Arithmetic Intensity 的体现
//
// 关键概念:
//   FP16: 1 sign + 5 exp + 10 mantissa = 范围±65504, 精度~3-4位
//   BF16: 1 sign + 8 exp + 7 mantissa  = 范围和FP32相同, 精度~2-3位
//   两者都是 2 bytes, 内存带宽需求减半, 但 BF16 不需要 loss scaling
//
// 编译: nvcc -arch=sm_80 -o mixed_precision mixed_precision.cu  (需要 Ampere+)
// ============================================================

#define CUDA_CHECK(call) do {                                              \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                cudaGetErrorString(err)); exit(1); }                       \
} while(0)

// ---- FP32 naive GEMM (基准) ----
// 每个线程算 C 的一个元素, 从 Global Memory 反复读取 A 和 B
__global__ void gemm_fp32_naive(const float *A, const float *B, float *C,
                                  int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k++) sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
}

// ---- FP16 naive GEMM (CUDA Core) ----
// 用 half 类型存储, 但计算在 CUDA Core 上 (不是 Tensor Core)
// FP16 ALU 在 Ampere 上比 FP32 快 2×
__global__ void gemm_fp16_naive(const __half *A, const __half *B, float *C,
                                  int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
    }
    C[row * N + col] = sum;
}

// ---- BF16 naive GEMM (CUDA Core) ----
// BF16: 和 FP16 同样的 2 bytes, 同样的内存带宽节省
// 但范围更大 (和 FP32 相同的 8-bit exponent)
__global__ void gemm_bf16_naive(const __nv_bfloat16 *A, const __nv_bfloat16 *B,
                                  float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += __bfloat162float(A[row * K + k]) * __bfloat162float(B[k * N + col]);
    }
    C[row * N + col] = sum;
}

// ---- FP16 Tensor Core GEMM (WMMA) ----
// 真正的 Tensor Core 加速: 一条 HMMA 指令做 16×16×16 乘加
#include <mma.h>
using namespace nvcuda;
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void gemm_fp16_tensorcore(const __half *A, const __half *B, float *C,
                                       int M, int N, int K) {
    int warpM = blockIdx.y * blockDim.y + threadIdx.y;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int ki = 0; ki < K; ki += WMMA_K) {
        wmma::load_matrix_sync(a_frag, A + warpM * WMMA_M * K + ki, K);
        wmma::load_matrix_sync(b_frag, B + ki + warpN * WMMA_N * K, K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    wmma::store_matrix_sync(C + warpM * WMMA_M * N + warpN * WMMA_N,
                            c_frag, N, wmma::mem_row_major);
}

// ---- CPU 参考 ----
void cpu_gemm_fp32(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) sum += A[i*K+k] * B[k*N+j];
            C[i*N+j] = sum;
        }
}

void cpu_gemm_fp16(const __half *A, const __half *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++)
                sum += __half2float(A[i*K+k]) * __half2float(B[k*N+j]);
            C[i*N+j] = sum;
        }
}

void cpu_gemm_bf16(const __nv_bfloat16 *A, const __nv_bfloat16 *B, float *C,
                    int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++)
                sum += __bfloat162float(A[i*K+k]) * __bfloat162float(B[k*N+j]);
            C[i*N+j] = sum;
        }
}

int main() {
    // 矩阵大小: 足够小让 CPU 参考能跑, 足够大能看到 GPU 差异
    const int M = 512, N = 512, K = 512;

    printf("混合精度 GEMM 对比: C[%d×%d] = A[%d×%d] × B[%d×%d]\n\n", M, N, M, K, K, N);
    printf("%-35s %8s %10s %12s\n", "实现", "耗时(ms)", "GFLOPS", "vs FP32");
    printf("-------------------------------------------------------------------\n");

    // ================================================================
    // 1. FP32 naive — 基准
    // ================================================================
    {
        size_t bytes = M * K * sizeof(float);
        float *hA = (float*)malloc(bytes);
        float *hB = (float*)malloc(K * N * sizeof(float));
        float *hC = (float*)malloc(M * N * sizeof(float));
        srand(42);
        for (int i = 0; i < M*K; i++) hA[i] = (rand()%200-100)/100.0f;
        for (int i = 0; i < K*N; i++) hB[i] = (rand()%200-100)/100.0f;

        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, bytes));
        CUDA_CHECK(cudaMalloc(&dB, K*N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, M*N*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB, K*N*sizeof(float), cudaMemcpyHostToDevice));

        dim3 block(16, 16);
        dim3 grid((N+15)/16, (M+15)/16);

        // warmup
        gemm_fp32_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0));
        for (int r = 0; r < 50; r++) gemm_fp32_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        ms /= 50;
        double gflops = 2.0 * M * N * K / (ms * 1e6);
        printf("%-35s %8.3f %10.1f %12s\n", "FP32 naive (CUDA Core)", ms, gflops, "1.0×");

        CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        free(hA); free(hB); free(hC);
    }

    // ================================================================
    // 2. FP16 naive (CUDA Core) — 2× 内存带宽节省 + 2× ALU 吞吐
    // ================================================================
    {
        size_t bytes = M * K * sizeof(__half);
        __half *hA = (__half*)malloc(bytes);
        __half *hB = (__half*)malloc(K * N * sizeof(__half));
        float *hC = (float*)malloc(M * N * sizeof(float));
        srand(42);
        for (int i = 0; i < M*K; i++) hA[i] = __float2half((rand()%200-100)/100.0f);
        for (int i = 0; i < K*N; i++) hB[i] = __float2half((rand()%200-100)/100.0f);

        __half *dA, *dB; float *dC;
        CUDA_CHECK(cudaMalloc(&dA, bytes));
        CUDA_CHECK(cudaMalloc(&dB, K*N*sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&dC, M*N*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB, K*N*sizeof(__half), cudaMemcpyHostToDevice));

        dim3 block(16, 16);
        dim3 grid((N+15)/16, (M+15)/16);
        gemm_fp16_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0));
        for (int r = 0; r < 50; r++) gemm_fp16_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        ms /= 50;
        double gflops = 2.0 * M * N * K / (ms * 1e6);
        printf("%-35s %8.3f %10.1f %12s\n", "FP16 naive (CUDA Core)", ms, gflops, "—");

        CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        free(hA); free(hB); free(hC);
    }

    // ================================================================
    // 3. BF16 naive (CUDA Core) — 同 FP16 的内存带宽, 更大的数值范围
    // ================================================================
    {
        size_t bytes = M * K * sizeof(__nv_bfloat16);
        __nv_bfloat16 *hA = (__nv_bfloat16*)malloc(bytes);
        __nv_bfloat16 *hB = (__nv_bfloat16*)malloc(K * N * sizeof(__nv_bfloat16));
        float *hC = (float*)malloc(M * N * sizeof(float));
        srand(42);
        for (int i = 0; i < M*K; i++) hA[i] = __float2bfloat16((rand()%200-100)/100.0f);
        for (int i = 0; i < K*N; i++) hB[i] = __float2bfloat16((rand()%200-100)/100.0f);

        __nv_bfloat16 *dA, *dB; float *dC;
        CUDA_CHECK(cudaMalloc(&dA, bytes));
        CUDA_CHECK(cudaMalloc(&dB, K*N*sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&dC, M*N*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB, K*N*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

        dim3 block(16, 16);
        dim3 grid((N+15)/16, (M+15)/16);
        gemm_bf16_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0));
        for (int r = 0; r < 50; r++) gemm_bf16_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        ms /= 50;
        double gflops = 2.0 * M * N * K / (ms * 1e6);
        printf("%-35s %8.3f %10.1f %12s\n", "BF16 naive (CUDA Core)", ms, gflops, "—");

        CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        free(hA); free(hB); free(hC);
    }

    // ================================================================
    // 4. FP16 Tensor Core (WMMA) — 真正的硬件加速
    // ================================================================
    {
        size_t bytes = M * K * sizeof(__half);
        __half *hA = (__half*)malloc(bytes);
        __half *hB = (__half*)malloc(K * N * sizeof(__half));
        float *hC = (float*)malloc(M * N * sizeof(float));
        srand(42);
        for (int i = 0; i < M*K; i++) hA[i] = __float2half((rand()%200-100)/100.0f);
        for (int i = 0; i < K*N; i++) hB[i] = __float2half((rand()%200-100)/100.0f);

        __half *dA, *dB; float *dC;
        CUDA_CHECK(cudaMalloc(&dA, bytes));
        CUDA_CHECK(cudaMalloc(&dB, K*N*sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&dC, M*N*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB, K*N*sizeof(__half), cudaMemcpyHostToDevice));

        // WMMA: 每个 Warp 处理一个 16×16 tile
        // blockDim 设为 128 = 4 Warps, 注意 grid 要除以 Warp 数
        dim3 block(128, 1);
        dim3 grid(N / WMMA_N / (128/32), M / WMMA_M);

        gemm_fp16_tensorcore<<<grid, block>>>(dA, dB, dC, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0));
        for (int r = 0; r < 200; r++) gemm_fp16_tensorcore<<<grid, block>>>(dA, dB, dC, M, N, K);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        ms /= 200;
        double gflops = 2.0 * M * N * K / (ms * 1e6);
        printf("%-35s %8.3f %10.1f %12s\n", "FP16 Tensor Core (WMMA)", ms, gflops, "—");

        CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        free(hA); free(hB); free(hC);
    }

    // ================================================================
    // 数值精度对比: FP16 vs BF16 vs FP32
    // ================================================================
    printf("\n数值精度对比 (小矩阵 64×64, 便于观察):\n");
    printf("%-30s %12s %15s\n", "格式", "最大误差 vs FP32", "说明");
    printf("-----------------------------------------------------------------\n");

    {
        const int SM = 64, SN = 64, SK = 64;
        // FP32 reference
        float *hA32 = (float*)malloc(SM*SK*sizeof(float));
        float *hB32 = (float*)malloc(SK*SN*sizeof(float));
        float *hC32 = (float*)malloc(SM*SN*sizeof(float));
        srand(123);
        for (int i = 0; i < SM*SK; i++) hA32[i] = (rand()%200-100)/100.0f;
        for (int i = 0; i < SK*SN; i++) hB32[i] = (rand()%200-100)/100.0f;
        cpu_gemm_fp32(hA32, hB32, hC32, SM, SN, SK);

        // FP16: convert → compute → compare
        __half *hA16 = (__half*)malloc(SM*SK*sizeof(__half));
        __half *hB16 = (__half*)malloc(SK*SN*sizeof(__half));
        float *hC16 = (float*)malloc(SM*SN*sizeof(float));
        for (int i = 0; i < SM*SK; i++) hA16[i] = __float2half(hA32[i]);
        for (int i = 0; i < SK*SN; i++) hB16[i] = __float2half(hB32[i]);
        cpu_gemm_fp16(hA16, hB16, hC16, SM, SN, SK);
        float max_err_fp16 = 0;
        for (int i = 0; i < SM*SN; i++)
            max_err_fp16 = fmaxf(max_err_fp16, fabsf(hC32[i] - hC16[i]));
        printf("%-30s %12.2e %15s\n", "FP16 → FP32 累加", max_err_fp16,
               max_err_fp16 < 1.0f ? "✓ 可接受" : "✗ 误差较大");

        // BF16: convert → compute → compare
        __nv_bfloat16 *hABF = (__nv_bfloat16*)malloc(SM*SK*sizeof(__nv_bfloat16));
        __nv_bfloat16 *hBBF = (__nv_bfloat16*)malloc(SK*SN*sizeof(__nv_bfloat16));
        float *hCBF = (float*)malloc(SM*SN*sizeof(float));
        for (int i = 0; i < SM*SK; i++) hABF[i] = __float2bfloat16(hA32[i]);
        for (int i = 0; i < SK*SN; i++) hBBF[i] = __float2bfloat16(hB32[i]);
        cpu_gemm_bf16(hABF, hBBF, hCBF, SM, SN, SK);
        float max_err_bf16 = 0;
        for (int i = 0; i < SM*SN; i++)
            max_err_bf16 = fmaxf(max_err_bf16, fabsf(hC32[i] - hCBF[i]));
        printf("%-30s %12.2e %15s\n", "BF16 → FP32 累加", max_err_bf16,
               max_err_bf16 < 10.0f ? "✓ 可接受" : "✗ 误差较大");

        free(hA32); free(hB32); free(hC32);
        free(hA16); free(hB16); free(hC16);
        free(hABF); free(hBBF); free(hCBF);
    }

    // ================================================================
    // Loss Scaling 演示 (仅 FP16 需要)
    // ================================================================
    printf("\nLoss Scaling 演示 (FP16 小梯度问题):\n");
    {
        float small_grad = 1e-7f;  // 一个很小的梯度
        __half grad_half = __float2half(small_grad);
        float grad_back = __half2float(grad_half);
        printf("  原始梯度:       %.10f\n", small_grad);
        printf("  FP16 存储后:    %.10f  ← 变成 0! (FP16 最小正数 ≈ 6e-8)\n", grad_back);

        float scale = 1024.0f;
        __half scaled = __float2half(small_grad * scale);
        float recovered = __half2float(scaled) / scale;
        printf("  Loss Scale %g:  %.10f  ← 保存成功!\n", scale, recovered);

        __nv_bfloat16 grad_bf16 = __float2bfloat16(small_grad);
        float bf16_back = __bfloat162float(grad_bf16);
        printf("  BF16 存储后:    %.10f  ← 不需要 Loss Scaling!\n", bf16_back);
    }

    printf("\n关键结论:\n");
    printf("  1. FP16/BF16 内存带宽需求是 FP32 的一半 → Memory Bound kernel 收益大\n");
    printf("  2. FP16 ALU 在 Ampere 上 2× vs FP32, 但 Tensor Core 可达 16×\n");
    printf("  3. BF16 范围 = FP32 (8-bit exponent), 不需要 Loss Scaling\n");
    printf("  4. FP16 精度略高 (10-bit mantissa vs 7-bit), 推理通常用 FP16\n");
    printf("  5. 训练推荐: BF16 (免 Loss Scaling) 或 FP16 + Loss Scaling\n");
    printf("  6. 始终用 FP32 做累加器和权重更新! 否则小梯度会丢失\n");

    return 0;
}
