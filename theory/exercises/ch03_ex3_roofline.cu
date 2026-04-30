// ============================================================
// Ch3 练习 3: Roofline 分析 — 向量加法的理论极限
// 配合: theory/03_memory_hierarchy.md 练习 3
//
// 编译: nvcc -O2 -o ch03_ex3_roofline ch03_ex3_roofline.cu
// 运行: ./ch03_ex3_roofline
//
// 自动计算向量加法的 AI, 实测带宽, 和理论极限对比。
// ============================================================

#include <cstdio>
#include <cuda_runtime.h>

#define N (1 << 24)
#define CUDA_CHECK(call) do {                                              \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n",   \
        __FILE__, __LINE__, cudaGetErrorString(err)); exit(1); }           \
} while(0)

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    size_t bytes = N * sizeof(float);
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    float *h = (float*)malloc(bytes);
    for (int i=0; i<N; i++) h[i]=1.0f;
    CUDA_CHECK(cudaMemcpy(d_a, h, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h, bytes, cudaMemcpyHostToDevice));

    int block = 256, grid = (N + block - 1) / block;
    vector_add<<<grid, block>>>(d_a, d_b, d_c, N); cudaDeviceSynchronize();

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int r = 0; r < 100; r++) vector_add<<<grid, block>>>(d_a, d_b, d_c, N);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1); ms /= 100;

    double total_bytes = 3.0 * N * 4;
    double total_flop = (double)N;
    double ai = total_flop / total_bytes;
    double bw_gb = total_bytes / (ms * 1e6);

    printf("Roofline 分析: 向量加法 (N = %d)\n\n", N);
    printf("算术强度 (AI)   = %.4f FLOP/Byte\n", ai);
    printf("(AI << Ridge Point → Memory Bound)\n\n");
    printf("总数据量        = %.1f MB (读 2N + 写 1N)\n", total_bytes / 1e6);
    printf("实测耗时        = %.3f ms\n", ms);
    printf("有效带宽        = %.1f GB/s\n", bw_gb);
    printf("\n和你 GPU 的理论峰值带宽比较, 达到了百分之几?\n");

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); free(h);
    return 0;
}
