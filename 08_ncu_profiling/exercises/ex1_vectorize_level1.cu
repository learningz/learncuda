// ============================================================
// 练习 1: 把标量 kernel 改成 float4 向量化版本
// 难度: Level 1 (只需填写 kernel 函数体)
//
// 编译: nvcc -O2 -o ex1_vectorize_level1 ex1_vectorize_level1.cu
// 运行: ./ex1_vectorize_level1
// 预期输出: 正确性通过 + float4 版本更快
//
// ncu_demo.cu 的 kernel_B 是标量合并版, kernel_C 是 float4 版。
// 这道题让你自己写 float4 版本。
//
// 公式 (和 kernel_B 一样): out[i] = in[i] * 2.0 + 1.0
//
// TODO: 实现 kernel_vec4
//   - 每个线程处理 4 个连续 float
//   - 用 reinterpret_cast<const float4*>(in)[idx] 读
//   - 对 v.x/v.y/v.z/v.w 各做 *2+1
//   - 用 reinterpret_cast<float4*>(out)[idx] = v 写
//   - idx = blockIdx.x * blockDim.x + threadIdx.x (注意: 这个 idx 是 float4 粒度的!)
//   - 边界: if (idx < n/4)
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

// 对照组: 标量版 (和 kernel_B 一样)
__global__ void kernel_scalar(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * 2.0f + 1.0f;
}

// ============================================================
// TODO: 实现 float4 向量化版本
// ============================================================
__global__ void kernel_vec4(const float *in, float *out, int n) {
    // --- 在这里写你的代码 ---

}

int main() {
    size_t bytes = N * sizeof(float);
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = (float)i;
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int grid = (N + BLOCK - 1) / BLOCK;
    int grid4 = (N / 4 + BLOCK - 1) / BLOCK;

    kernel_scalar<<<grid, BLOCK>>>(d_in, d_out, N);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    bool ok_scalar = true;
    for (int i = 0; i < N; i++)
        if (fabsf(h_out[i] - (h_in[i] * 2.0f + 1.0f)) > 1e-3f) { ok_scalar = false; break; }

    kernel_vec4<<<grid4, BLOCK>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    bool ok_vec4 = true;
    for (int i = 0; i < N; i++)
        if (fabsf(h_out[i] - (h_in[i] * 2.0f + 1.0f)) > 1e-3f) { ok_vec4 = false; break; }

    printf("正确性: scalar=%s  vec4=%s\n", ok_scalar ? "✓" : "✗", ok_vec4 ? "✓" : "✗");

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    kernel_scalar<<<grid, BLOCK>>>(d_in, d_out, N); cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int r = 0; r < 100; r++) kernel_scalar<<<grid, BLOCK>>>(d_in, d_out, N);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms_s; cudaEventElapsedTime(&ms_s, t0, t1);

    kernel_vec4<<<grid4, BLOCK>>>(d_in, d_out, N); cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int r = 0; r < 100; r++) kernel_vec4<<<grid4, BLOCK>>>(d_in, d_out, N);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms_v; cudaEventElapsedTime(&ms_v, t0, t1);

    printf("scalar: %.3f ms  vec4: %.3f ms  加速: %.2fx\n", ms_s/100, ms_v/100, ms_s/ms_v);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    return 0;
}
