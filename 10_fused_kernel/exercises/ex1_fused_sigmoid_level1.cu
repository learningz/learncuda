// ============================================================
// 练习 1: Sigmoid + Scale 融合 — y[i] = sigmoid(x[i]) * scale
// 难度: Level 1 (只需填写 kernel 函数体)
//
// 编译: nvcc -O2 -o ex1_fused_sigmoid_level1 ex1_fused_sigmoid_level1.cu
// 运行: ./ex1_fused_sigmoid_level1
// 预期输出: 正确性通过 + 融合版比未融合版快
//
// 未融合: kernel1: tmp[i] = sigmoid(x[i])  → 写 tmp 到显存
//         kernel2: y[i] = tmp[i] * scale   → 读 tmp, 写 y
//         显存访问: 4N (读 x + 写 tmp + 读 tmp + 写 y)
//
// 融合:   kernel:  y[i] = sigmoid(x[i]) * scale  → 全在寄存器
//         显存访问: 2N (读 x + 写 y)
//
// TODO: 实现 fused_sigmoid_scale_kernel
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
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

__global__ void sigmoid_kernel(const float *x, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = 1.0f / (1.0f + expf(-x[idx]));
}

__global__ void scale_kernel(const float *in, float *out, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * scale;
}

// ============================================================
// TODO: 一个 kernel 搞定: y[i] = sigmoid(x[i]) * scale
// ============================================================
__global__ void fused_sigmoid_scale_kernel(const float *x, float *y,
                                           float scale, int n) {
    // --- 在这里写你的代码 ---

}

int main() {
    const float scale = 2.5f;
    size_t bytes = N * sizeof(float);

    float *h_x = (float *)malloc(bytes);
    float *h_y = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; i++) {
        h_x[i] = (float)(rand() % 200 - 100) / 10.0f;
        h_ref[i] = 1.0f / (1.0f + expf(-h_x[i])) * scale;
    }

    float *d_x, *d_y, *d_tmp;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);

    sigmoid_kernel<<<grid, BLOCK_SIZE>>>(d_x, d_tmp, N);
    scale_kernel<<<grid, BLOCK_SIZE>>>(d_tmp, d_y, scale, N);
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int r = 0; r < 50; r++) {
        sigmoid_kernel<<<grid, BLOCK_SIZE>>>(d_x, d_tmp, N);
        scale_kernel<<<grid, BLOCK_SIZE>>>(d_tmp, d_y, scale, N);
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms_sep; cudaEventElapsedTime(&ms_sep, t0, t1);

    fused_sigmoid_scale_kernel<<<grid, BLOCK_SIZE>>>(d_x, d_y, scale, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < N; i++)
        if (fabsf(h_y[i] - h_ref[i]) > 1e-4f) { ok = false; break; }

    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int r = 0; r < 50; r++)
        fused_sigmoid_scale_kernel<<<grid, BLOCK_SIZE>>>(d_x, d_y, scale, N);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms_fused; cudaEventElapsedTime(&ms_fused, t0, t1);

    printf("正确性: %s\n", ok ? "✓" : "✗");
    printf("未融合: %.3f ms  融合: %.3f ms  加速: %.2fx\n",
           ms_sep/50, ms_fused/50, ms_sep/ms_fused);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_tmp);
    free(h_x); free(h_y); free(h_ref);
    return 0;
}
