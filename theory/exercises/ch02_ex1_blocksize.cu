// ============================================================
// Ch2 练习 1: blockSize 对性能的影响
// 配合: theory/02_cuda_programming_model.md 练习 1
//
// 编译: nvcc -O2 -o ch02_ex1_blocksize ch02_ex1_blocksize.cu
// 运行: ./ch02_ex1_blocksize
//
// 本程序自动测试 blockSize = 32, 64, 128, 256, 512, 1024 六种配置,
// 打印每种配置下向量加法的耗时和有效带宽。
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
    float *h_a = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_a[i] = 1.0f;

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_a, bytes, cudaMemcpyHostToDevice));

    int sizes[] = {32, 64, 128, 256, 512, 1024};
    printf("blockSize  gridSize     耗时(ms)   有效带宽(GB/s)\n");
    printf("─────────────────────────────────────────────────\n");

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    for (int s = 0; s < 6; s++) {
        int bs = sizes[s];
        int gs = (N + bs - 1) / bs;
        vector_add<<<gs, bs>>>(d_a, d_b, d_c, N); cudaDeviceSynchronize();
        cudaEventRecord(t0);
        for (int r = 0; r < 50; r++) vector_add<<<gs, bs>>>(d_a, d_b, d_c, N);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1); ms /= 50;
        float bw = 3.0f * N * 4 / (ms * 1e6);
        printf("%5d      %7d     %.3f      %.1f\n", bs, gs, ms, bw);
    }

    printf("\n观察: 哪个 blockSize 最快? 为什么 32 可能比 256 慢?\n");
    printf("(提示: blockSize 太小 → Occupancy 低, 延迟隐藏不够)\n");

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); free(h_a);
    return 0;
}
