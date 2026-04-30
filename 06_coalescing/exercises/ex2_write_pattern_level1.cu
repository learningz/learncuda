// ============================================================
// 练习 2: Vector Scale — 合并写 vs 非合并写
// 难度: Level 1 (只需填写两个 kernel 函数体)
//
// 编译: nvcc -O2 --extended-lambda -o ex2_write_pattern_level1 ex2_write_pattern_level1.cu
// 运行: ./ex2_write_pattern_level1
// 预期输出: 连续写版本更快
//
// 背景:
//   coalescing.cu 演示了"读"的合并/不合并。这道题演示"写"的合并/不合并。
//   output[idx] = input[idx] * scale → 读合并 + 写合并 → 快
//   output[idx * STRIDE] = input[idx] * scale → 写不合并 → 慢
//
// TODO: 填写两个 kernel
//   1. scale_coalesced: output[idx] = input[idx] * scale
//   2. scale_strided:   output[idx * STRIDE] = input[idx] * scale
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define STRIDE 16

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// 合并写: 每个线程写 output[idx] (连续地址)
__global__ void scale_coalesced(const float *input, float *output,
                                float scale, int n) {
    // --- 在这里写你的代码 ---

}

// 非合并写: 每个线程写 output[idx * STRIDE] (跨步地址)
__global__ void scale_strided(const float *input, float *output,
                              float scale, int n) {
    // --- 在这里写你的代码 ---

}

int main() {
    const int N = 1 << 22;
    const float scale = 3.14f;

    float *d_in, *d_out_coal, *d_out_stride;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_coal, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_stride, (size_t)N * STRIDE * sizeof(float)));

    float *h_in = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (float)i;
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    int block = 256, grid = (N + block - 1) / block;

    auto bench = [&](const char *name, auto fn) {
        fn(); cudaDeviceSynchronize();
        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);
        for (int r = 0; r < 50; r++) fn();
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);
        printf("%-20s: %.3f ms\n", name, ms / 50);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    };

    bench("合并写 (stride=1)", [&]{
        scale_coalesced<<<grid, block>>>(d_in, d_out_coal, scale, N); });
    bench("非合并写 (stride=16)", [&]{
        scale_strided<<<grid, block>>>(d_in, d_out_stride, scale, N); });

    printf("\n合并写应该更快 — 写入也需要合并访问!\n");

    cudaFree(d_in); cudaFree(d_out_coal); cudaFree(d_out_stride);
    free(h_in);
    return 0;
}
