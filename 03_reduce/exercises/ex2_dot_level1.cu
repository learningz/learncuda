// ============================================================
// 练习 2: 点积 (Dot Product) — sum(a[i] * b[i])
// 难度: Level 1 (只需填写 kernel 函数体)
//
// 编译: nvcc -O2 -o ex2_dot_level1 ex2_dot_level1.cu
// 运行: ./ex2_dot_level1
// 预期输出: "结果: ✓ 全部正确"
//
// 点积 = elementwise 乘法 + 求和归约，综合了两个操作。
//
// 思路 (和 reduce_v2 几乎一样):
//   1. Grid-Stride Loop: 每个线程累加 a[i]*b[i] 到 local_sum
//   2. 写到 SMEM, __syncthreads()
//   3. SMEM 归约 (对半折叠)
//   4. tid==0 写 output[blockIdx.x]
//
// 和 reduce_v2 唯一的区别: 循环体里是 a[i]*b[i] 而不是 input[i]
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define GRID_SIZE 256

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ============================================================
// TODO: 实现这个 kernel
// 每个 Block 算出局部点积, 写到 output[blockIdx.x]
// 然后 CPU 端汇总这些局部结果
// ============================================================
__global__ void dot_kernel(const float *a, const float *b,
                           float *output, int n) {
    // --- 在这里写你的代码 ---

}

int main() {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)(rand() % 100) / 100.0f;
        h_b[i] = (float)(rand() % 100) / 100.0f;
    }

    double ref_dot = 0;
    for (int i = 0; i < N; i++) ref_dot += (double)h_a[i] * h_b[i];

    float *d_a, *d_b, *d_partial;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_partial, GRID_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    dot_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_partial, N);
    CUDA_CHECK(cudaGetLastError());

    float *h_partial = (float *)malloc(GRID_SIZE * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    double gpu_dot = 0;
    for (int i = 0; i < GRID_SIZE; i++) gpu_dot += h_partial[i];

    bool ok = (fabs(gpu_dot - ref_dot) / fabs(ref_dot) < 1e-4);
    printf("CPU dot = %.4f, GPU dot = %.4f\n", ref_dot, gpu_dot);
    printf("结果: %s\n", ok ? "✓ 全部正确" : "✗ 有错误");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_partial);
    free(h_a); free(h_b); free(h_partial);
    return 0;
}
