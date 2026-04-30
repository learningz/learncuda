// ============================================================
// Ch3 练习 1: 合并访问探索 — stride=4 + float4 向量化
// 配合: theory/03_memory_hierarchy.md 练习 1
//
// 编译: nvcc -O2 -o ch03_ex1_coalescing ch03_ex1_coalescing.cu
// 运行: ./ch03_ex1_coalescing
//
// 在 coalescing.cu 的基础上增加 stride=4 + float4 向量化两个测试。
// ============================================================

#include <cstdio>
#include <cuda_runtime.h>

#define N (1 << 22)
#define BLOCK_SIZE 256

#define CUDA_CHECK(call) do {                                              \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n",   \
        __FILE__, __LINE__, cudaGetErrorString(err)); exit(1); }           \
} while(0)

__global__ void access_stride1(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0;
    for (int i = idx; i < n; i += stride) sum += in[i];
    if (idx == 0) out[0] = sum;
}

__global__ void access_stride4(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0;
    int n4 = n / 4;
    for (int i = idx; i < n4; i += stride) sum += in[i * 4];
    if (idx == 0) out[0] = sum;
}

__global__ void access_float4(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0;
    const float4 *in4 = reinterpret_cast<const float4*>(in);
    int n4 = n / 4;
    for (int i = idx; i < n4; i += stride) {
        float4 v = in4[i];
        sum += v.x + v.y + v.z + v.w;
    }
    if (idx == 0) out[0] = sum;
}

int main() {
    size_t bytes = N * sizeof(float);
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    float *h = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(d_in, h, bytes, cudaMemcpyHostToDevice));

    int grid = 256;
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);

    auto bench = [&](const char *name, auto fn) {
        fn(); cudaDeviceSynchronize();
        cudaEventRecord(t0);
        for (int r = 0; r < 50; r++) fn();
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);
        float gb = (float)N * 4 / 1e9;
        printf("%-20s: %.3f ms  有效带宽: %.1f GB/s\n", name, ms/50, gb/(ms/50/1000));
    };

    printf("合并访问探索 (N=%d)\n\n", N);
    bench("stride=1 (合并)", [&]{ access_stride1<<<grid, BLOCK_SIZE>>>(d_in, d_out, N); });
    bench("stride=4 (部分合并)", [&]{ access_stride4<<<grid, BLOCK_SIZE>>>(d_in, d_out, N); });
    bench("float4 向量化", [&]{ access_float4<<<grid, BLOCK_SIZE>>>(d_in, d_out, N); });

    printf("\n观察: stride=4 的带宽大约是 stride=1 的 1/4 (效率 ~25%%)\n");
    printf("float4 向量化可能比 stride=1 还快 (减少了 LD/ST 指令数)\n");

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_in); cudaFree(d_out); free(h);
    return 0;
}
