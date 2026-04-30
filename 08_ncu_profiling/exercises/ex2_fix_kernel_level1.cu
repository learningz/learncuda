// ============================================================
// 练习 2: 找出并修复 kernel 的性能问题
// 难度: Level 1 (读懂问题, 修改几行代码)
//
// 编译: nvcc -O2 -o ex2_fix_kernel_level1 ex2_fix_kernel_level1.cu
// 运行: ./ex2_fix_kernel_level1
// 预期输出: 修复后的 kernel 比原始版快 2x+, 正确性通过
//
// 下面的 kernel_buggy 有两个性能问题:
//   问题 1: 读输入用了 stride-2 (非合并访问, 50% 带宽浪费)
//   问题 2: 写输出也用了 stride-2 (同上)
//
// 修复方法:
//   改成 stride-1 的访问模式, 但保持计算结果不变。
//   原本: out[idx*2] = in[idx*2] * 3.0 + in[idx*2 + 1] * 2.0
//   修复: 把数据重新组织, 让每个线程读写连续地址
//
// TODO:
//   实现 kernel_fixed, 做同样的计算, 但用合并的访问模式。
//   提示: 每个线程处理 2 个连续输出:
//     out[idx*2]     = in[idx*2] * 3.0 + in[idx*2+1] * 2.0
//     out[idx*2 + 1] = in[idx*2+1] * 3.0 + in[idx*2] * 2.0
//   但让相邻线程处理相邻的对:
//     线程 0: 处理 in[0],in[1] → out[0],out[1]
//     线程 1: 处理 in[2],in[3] → out[2],out[3]
//   这样读写都是连续的!
//
//   更好的写法: 用 float2 向量化, 一条指令读两个 float
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define N (1 << 24)
#define BLOCK 256

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// 有问题的版本: stride-2 访问
__global__ void kernel_buggy(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n / 2) {
        out[idx * 2] = in[idx * 2] * 3.0f + in[idx * 2 + 1] * 2.0f;
    }
}

// ============================================================
// TODO: 实现修复后的版本
// 做同样的计算, 但改成合并访问
// 每个线程处理一对连续元素:
//   base = idx * 2
//   out[base]     = in[base] * 3.0f + in[base + 1] * 2.0f
//   out[base + 1] = in[base + 1] * 3.0f + in[base] * 2.0f
// (让相邻线程的 base 相差 2, 读写都是连续的)
// ============================================================
__global__ void kernel_fixed(const float *in, float *out, int n) {
    // --- 在这里写你的代码 ---

}

int main() {
    size_t bytes = N * sizeof(float);
    float *d_in, *d_out_buggy, *d_out_fixed;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out_buggy, bytes));
    CUDA_CHECK(cudaMalloc(&d_out_fixed, bytes));

    float *h_in = (float *)malloc(bytes);
    float *h_out_b = (float *)malloc(bytes);
    float *h_out_f = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = (float)(i % 100) / 10.0f;
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out_buggy, 0, bytes));
    CUDA_CHECK(cudaMemset(d_out_fixed, 0, bytes));

    int grid_half = (N / 2 + BLOCK - 1) / BLOCK;

    kernel_buggy<<<grid_half, BLOCK>>>(d_in, d_out_buggy, N);
    kernel_fixed<<<grid_half, BLOCK>>>(d_in, d_out_fixed, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_out_b, d_out_buggy, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_f, d_out_fixed, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < N; i += 2) {
        float expected = h_in[i] * 3.0f + h_in[i + 1] * 2.0f;
        if (fabsf(h_out_f[i] - expected) > 1e-3f) { ok = false; break; }
    }
    printf("正确性 (fixed kernel out[even]): %s\n", ok ? "✓" : "✗");

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    kernel_buggy<<<grid_half, BLOCK>>>(d_in, d_out_buggy, N); cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int r = 0; r < 100; r++) kernel_buggy<<<grid_half, BLOCK>>>(d_in, d_out_buggy, N);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms_b; cudaEventElapsedTime(&ms_b, t0, t1);

    kernel_fixed<<<grid_half, BLOCK>>>(d_in, d_out_fixed, N); cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int r = 0; r < 100; r++) kernel_fixed<<<grid_half, BLOCK>>>(d_in, d_out_fixed, N);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms_f; cudaEventElapsedTime(&ms_f, t0, t1);

    printf("buggy: %.3f ms  fixed: %.3f ms  加速: %.2fx\n", ms_b/100, ms_f/100, ms_b/ms_f);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_in); cudaFree(d_out_buggy); cudaFree(d_out_fixed);
    free(h_in); free(h_out_b); free(h_out_f);
    return 0;
}
