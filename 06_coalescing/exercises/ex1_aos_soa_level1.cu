// ============================================================
// 练习 1: AoS → SoA 改写 — 消除非合并访问
// 难度: Level 1 (只需填写 kernel 函数体)
//
// 编译: nvcc -O2 -o ex1_aos_soa_level1 ex1_aos_soa_level1.cu
// 运行: ./ex1_aos_soa_level1
// 预期输出: 正确性通过 + SoA 版本明显更快
//
// 背景:
//   coalescing.md 讲过: AoS (Array of Structures) 是最常见的非合并陷阱。
//   struct Particle { float x, y, z, w; };  // 16B/粒子
//   读 particles[tid].x → 相邻线程地址间隔 16B → 效率仅 25%!
//
//   修复: SoA (Structure of Arrays)
//   float *px, *py, *pz, *pw;
//   读 px[tid] → 相邻线程地址连续 → 效率 100%!
//
// 本题:
//   下面有两个 kernel, host 端已写好, 你只需填 kernel 函数体:
//   1. scale_aos_kernel: 读 AoS 数组, 把 x 分量乘以 2.0 写到输出
//   2. scale_soa_kernel: 读 SoA 数组, 做同样的事
//
// 提示:
//   - AoS: 输入 particles 是 float4*, 读 particles[idx].x
//   - SoA: 输入 px 是 float*, 读 px[idx]
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// AoS 版: 从 float4 数组读 .x, 乘以 2.0 写到 output
__global__ void scale_aos_kernel(const float4 *particles, float *output, int n) {
    // --- 在这里写你的代码 ---
    // 提示: int idx = ...; if (idx < n) output[idx] = particles[idx].x * 2.0f;

}

// SoA 版: 从 float* 数组读, 乘以 2.0 写到 output
__global__ void scale_soa_kernel(const float *px, float *output, int n) {
    // --- 在这里写你的代码 ---
    // 提示: int idx = ...; if (idx < n) output[idx] = px[idx] * 2.0f;

}

int main() {
    const int N = 1 << 22;

    float4 *h_aos = (float4 *)malloc(N * sizeof(float4));
    float *h_soa_x = (float *)malloc(N * sizeof(float));
    float *h_out = (float *)malloc(N * sizeof(float));
    float *h_ref = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        float val = (float)(i % 1000) / 10.0f;
        h_aos[i] = make_float4(val, val + 1, val + 2, val + 3);
        h_soa_x[i] = val;
        h_ref[i] = val * 2.0f;
    }

    float4 *d_aos; float *d_soa_x, *d_out;
    CUDA_CHECK(cudaMalloc(&d_aos, N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_soa_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_aos, h_aos, N * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_soa_x, h_soa_x, N * sizeof(float), cudaMemcpyHostToDevice));

    int block = 256, grid = (N + block - 1) / block;

    auto bench = [&](const char *name, auto fn) {
        fn(); cudaDeviceSynchronize();
        CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = true;
        for (int i = 0; i < N; i++)
            if (fabsf(h_out[i] - h_ref[i]) > 1e-5f) { ok = false; break; }

        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);
        for (int r = 0; r < 50; r++) fn();
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);
        printf("%-12s: %.3f ms  %s\n", name, ms / 50, ok ? "✓" : "✗");
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    };

    bench("AoS (慢)", [&]{ scale_aos_kernel<<<grid, block>>>(d_aos, d_out, N); });
    bench("SoA (快)", [&]{ scale_soa_kernel<<<grid, block>>>(d_soa_x, d_out, N); });

    printf("\nSoA 应该更快 — 因为相邻线程读连续地址, 完美合并!\n");

    cudaFree(d_aos); cudaFree(d_soa_x); cudaFree(d_out);
    free(h_aos); free(h_soa_x); free(h_out); free(h_ref);
    return 0;
}
