// ============================================================
// 练习 1: 用 Warp-uniform 条件消除分歧
// 难度: Level 1 (只需填写 kernel 函数体)
//
// 编译: nvcc -O2 -o ex1_warp_uniform_level1 ex1_warp_uniform_level1.cu
// 运行: ./ex1_warp_uniform_level1
// 预期输出: warp-uniform 版本和无分歧版本速度接近
//
// warp_divergence.cu 中展示了:
//   threadIdx.x % 2 == 0 → 50% 分歧 (同一 Warp 内奇偶线程走不同路径)
//
// 修复思路: 让分支条件以 Warp 为单位变化, 而不是以线程为单位
//   if (threadIdx.x / 32 % 2 == 0) → 整个 Warp 全走 if 或全走 else → 无分歧!
//
// TODO:
//   实现 warp_uniform_kernel: 和 half_divergence 做同样的计算,
//   但用 (threadIdx.x / 32) % 2 做分支条件, 消除 Warp 内分歧
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define N (1 << 22)
#define BLOCK_SIZE 256

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// 对照组: 有 50% Warp Divergence
__global__ void half_divergence(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = input[idx];
    if (threadIdx.x % 2 == 0)
        output[idx] = val * val + val;
    else
        output[idx] = val * 0.5f - 1.0f;
}

// ============================================================
// TODO: 用 Warp-uniform 条件做同样的计算
// 条件: (threadIdx.x / 32) % 2 == 0  → 整个 Warp 走同一分支!
// ============================================================
__global__ void warp_uniform_kernel(const float *input, float *output, int n) {
    // --- 在这里写你的代码 ---

}

int main() {
    size_t bytes = N * sizeof(float);
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    float *h_in = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = (float)(i % 100) / 10.0f;
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);

    auto bench = [&](const char *name, auto kernel) {
        kernel<<<grid, BLOCK_SIZE>>>(d_in, d_out, N); cudaDeviceSynchronize();
        cudaEventRecord(t0);
        for (int r = 0; r < 100; r++) kernel<<<grid, BLOCK_SIZE>>>(d_in, d_out, N);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);
        printf("  %-25s: %.3f ms\n", name, ms / 100);
    };

    printf("Warp Divergence 消除实验\n\n");
    bench("50%% 分歧 (thread%%2)", half_divergence);
    bench("Warp-uniform (warp%%2)", warp_uniform_kernel);

    printf("\nWarp-uniform 版本应该更快 — 同一 Warp 全走同一分支, 无串行化!\n");

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_in); cudaFree(d_out); free(h_in);
    return 0;
}
