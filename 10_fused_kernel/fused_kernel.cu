#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ============================================================
// 10: 算子融合实战 — 亲眼看到融合的威力
//
// 配合理论: theory/08_advanced_optimization.md 8.2 节 (算子融合)
//           theory/05_operator_development.md 5.11 Q&A (融合 vs 编译器优化)
//
// 场景: 对一个向量做 ReLU → Scale → Bias 三个操作
//   y[i] = max(x[i], 0) * scale + bias
//
// 三个版本:
//   版本 A (未融合): 3 个独立 kernel, 中间结果写回显存再读出
//     kernel1: tmp1[i] = max(x[i], 0)         → 写 tmp1 到显存
//     kernel2: tmp2[i] = tmp1[i] * scale       → 读 tmp1, 写 tmp2
//     kernel3: y[i]    = tmp2[i] + bias        → 读 tmp2, 写 y
//     总显存访问: 读 x + 写 tmp1 + 读 tmp1 + 写 tmp2 + 读 tmp2 + 写 y = 6N
//
//   版本 B (融合): 1 个 kernel, 中间结果留在寄存器
//     kernel:  y[i] = max(x[i], 0) * scale + bias
//     总显存访问: 读 x + 写 y = 2N
//     → 显存访问减少 3×!
//
//   版本 C (融合 + float4 向量化): 在版本 B 基础上用 128-bit 加载
//     每线程一次加载 4 个 float → 减少 LD/ST 指令数
//
// 核心概念:
//   大多数 elementwise 算子是 Memory Bound (瓶颈在显存带宽, 不在计算)。
//   融合的本质: 减少显存读写次数 → 直接提升性能。
//   中间结果不写回显存, 而是留在寄存器 (0 cycle 延迟) → "免费"的计算!
// ============================================================

#define N (1 << 24)  // 16M 元素 = 64 MB
#define BLOCK_SIZE 256

#define CUDA_CHECK(call) do {                                              \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                cudaGetErrorString(err)); exit(1); }                       \
} while(0)

// ---- 版本 A: 3 个独立 kernel (未融合) ----

__global__ void relu_kernel(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fmaxf(in[idx], 0.0f);
}

__global__ void scale_kernel(const float *in, float *out, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * scale;
}

__global__ void bias_kernel(const float *in, float *out, float bias, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] + bias;
}

// ---- 版本 B: 1 个融合 kernel ----

__global__ void fused_relu_scale_bias(const float *in, float *out,
                                       float scale, float bias, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];           // 1 次显存读
        val = fmaxf(val, 0.0f);        // ReLU  (寄存器内, 0 额外访存)
        val = val * scale;              // Scale  (寄存器内)
        val = val + bias;               // Bias   (寄存器内)
        out[idx] = val;                 // 1 次显存写
    }
}

// ---- 版本 C: 融合 + float4 向量化 ----

__global__ void fused_relu_scale_bias_vec4(const float *in, float *out,
                                            float scale, float bias, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < n) {
        // 一次读 4 个 float = 128 bit → 减少 LD/ST 指令数
        float4 v = reinterpret_cast<const float4*>(in)[idx / 4];
        v.x = fmaxf(v.x, 0.0f) * scale + bias;
        v.y = fmaxf(v.y, 0.0f) * scale + bias;
        v.z = fmaxf(v.z, 0.0f) * scale + bias;
        v.w = fmaxf(v.w, 0.0f) * scale + bias;
        reinterpret_cast<float4*>(out)[idx / 4] = v;
    }
}

bool verify(const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) > 1e-5f) {
            printf("  不匹配 @ %d: %.6f vs %.6f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

int main() {
    float scale = 0.5f, bias = 1.0f;
    size_t bytes = N * sizeof(float);

    float *h_in = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; i++) h_in[i] = (float)(rand() % 200 - 100) / 10.0f;

    // CPU 参考
    for (int i = 0; i < N; i++) h_ref[i] = fmaxf(h_in[i], 0.0f) * scale + bias;

    float *d_in, *d_out, *d_tmp1, *d_tmp2;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp1, bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp2, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid4 = (N / 4 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("算子融合性能对比 (N = %d = %.0f MB)\n\n", N, bytes / 1e6);

    #define BENCH(name, ...) {                                             \
        __VA_ARGS__; cudaDeviceSynchronize();                               \
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost)); \
        printf("    正确性: %s\n", verify(h_ref, h_out, N) ? "✓" : "✗");   \
        cudaEvent_t t0, t1;                                                 \
        cudaEventCreate(&t0); cudaEventCreate(&t1);                         \
        cudaEventRecord(t0);                                                \
        for (int _r = 0; _r < 100; _r++) { __VA_ARGS__; }                  \
        cudaEventRecord(t1); cudaEventSynchronize(t1);                      \
        float ms; cudaEventElapsedTime(&ms, t0, t1); ms /= 100;            \
        float gbps = (float)bytes * 2 / (ms * 1e6);                        \
        printf("  %-30s: %.3f ms  有效带宽: %6.1f GB/s\n", name, ms, gbps); \
        cudaEventDestroy(t0); cudaEventDestroy(t1);                         \
    }

    BENCH("A: 3个独立kernel (未融合)",
          relu_kernel<<<grid, BLOCK_SIZE>>>(d_in, d_tmp1, N);
          scale_kernel<<<grid, BLOCK_SIZE>>>(d_tmp1, d_tmp2, scale, N);
          bias_kernel<<<grid, BLOCK_SIZE>>>(d_tmp2, d_out, bias, N));

    BENCH("B: 1个融合kernel",
          fused_relu_scale_bias<<<grid, BLOCK_SIZE>>>(d_in, d_out, scale, bias, N));

    BENCH("C: 融合 + float4 向量化",
          fused_relu_scale_bias_vec4<<<grid4, BLOCK_SIZE>>>(d_in, d_out, scale, bias, N));

    printf("\n分析:\n");
    printf("  版本 A: 6N 次显存访问 (3次读 + 3次写, 含中间 tmp)\n");
    printf("  版本 B: 2N 次显存访问 (1次读 + 1次写) → 理论快 3×\n");
    printf("  版本 C: 同 B 的访存量, 但 LD/ST 指令减少 4× → 可能更快\n");
    printf("\n  有效带宽的计算: 只算有用的数据量 (读 x + 写 y = 2N × 4B)\n");
    printf("  版本 A 的有效带宽看起来低, 是因为实际传输了 6N 但我们只算 2N\n");
    printf("\n  这就是为什么 torch.compile / Triton 的核心优化就是算子融合!\n");

    #undef BENCH
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_tmp1); cudaFree(d_tmp2);
    free(h_in); free(h_ref); free(h_out);
    return 0;
}
