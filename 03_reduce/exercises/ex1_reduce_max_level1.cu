// ============================================================
// 练习 1: Reduce Max — 求数组中的最大值
// 难度: Level 1 (只需填写 kernel 函数体)
//
// 编译: nvcc -O2 -o ex1_reduce_max_level1 ex1_reduce_max_level1.cu
// 运行: ./ex1_reduce_max_level1
// 预期输出: "结果: ✓ 全部正确"
//
// 和 reduce_v1 的区别:
//   只需把 += 换成 max 操作，归约结构完全一样。
//
// 提示:
//   - fmaxf(a, b) 返回 a 和 b 中的较大值
//   - 初始值不能用 0 (数组可能全是负数!) → 用 -INFINITY
//   - SMEM 初始化: sdata[tid] = (idx < n) ? input[idx] : -INFINITY
//   - 归约循环: sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride])
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

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

// ============================================================
// TODO: 实现这个 kernel
// 每个 Block 归约出局部最大值, 写到 output[blockIdx.x]
// 结构和 reduce_v1 完全一样, 只是把 += 换成 fmaxf
// ============================================================
__global__ void reduce_max_kernel(const float *input, float *output, int n) {
    // --- 在这里写你的代码 ---

}

int main() {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    float *h_input = (float *)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; i++)
        h_input[i] = (float)(rand() % 10000 - 5000) / 10.0f;

    float ref_max = -FLT_MAX;
    for (int i = 0; i < N; i++)
        if (h_input[i] > ref_max) ref_max = h_input[i];

    float *d_input, *d_partial;
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_partial, gridSize * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    reduce_max_kernel<<<gridSize, BLOCK_SIZE>>>(d_input, d_partial, N);
    CUDA_CHECK(cudaGetLastError());

    float *h_partial = (float *)malloc(gridSize * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, gridSize * sizeof(float), cudaMemcpyDeviceToHost));

    float gpu_max = -FLT_MAX;
    for (int i = 0; i < gridSize; i++)
        if (h_partial[i] > gpu_max) gpu_max = h_partial[i];

    bool ok = (fabsf(gpu_max - ref_max) < 1e-5f);
    printf("CPU max = %.4f, GPU max = %.4f\n", ref_max, gpu_max);
    printf("结果: %s\n", ok ? "✓ 全部正确" : "✗ 有错误");

    cudaFree(d_input); cudaFree(d_partial);
    free(h_input); free(h_partial);
    return 0;
}
