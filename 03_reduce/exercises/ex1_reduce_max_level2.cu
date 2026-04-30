// ============================================================
// 练习 1: Reduce Max — 求数组中的最大值
// 难度: Level 2 (kernel + host 端都要自己写)
//
// 编译: nvcc -O2 -o ex1_reduce_max_level2 ex1_reduce_max_level2.cu
// 运行: ./ex1_reduce_max_level2
// 预期输出: "结果: ✓ 全部正确"
//
// 要求:
//   1. 实现 reduce_max_kernel (SMEM 归约, 用 fmaxf)
//   2. 在 main 中: cudaMalloc, cudaMemcpy, 启动 kernel, 回传 partial 结果
//   3. CPU 端对 partial 结果做最终 max
//
// 注意:
//   - SMEM 初始化用 -INFINITY (不能用 0!)
//   - gridSize = ceil(N / BLOCK_SIZE), 输出数组大小 = gridSize
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

// TODO: 实现 reduce_max_kernel
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

    // ============================================================
    // TODO: 完成以下步骤
    //   1. cudaMalloc 分配 d_input 和 d_partial
    //   2. cudaMemcpy 把 h_input 传到 GPU
    //   3. 启动 reduce_max_kernel
    //   4. cudaMemcpy 把 d_partial 传回 h_partial
    //   5. CPU 端遍历 h_partial 求最终 max
    //   6. cudaFree
    // ============================================================

    float gpu_max = -FLT_MAX;

    // --- 在这里写你的代码 ---

    bool ok = (fabsf(gpu_max - ref_max) < 1e-5f);
    printf("CPU max = %.4f, GPU max = %.4f\n", ref_max, gpu_max);
    printf("结果: %s\n", ok ? "✓ 全部正确" : "✗ 有错误");

    free(h_input);
    return 0;
}
