// ============================================================
// 练习 3: 计数大于阈值的元素 — count(x[i] > threshold)
// 难度: Level 2 (kernel + host 端都要自己写)
//
// 编译: nvcc -O2 -o ex3_count_level2 ex3_count_level2.cu
// 运行: ./ex3_count_level2
// 预期输出: "结果: ✓ 全部正确"
//
// 要求:
//   1. 实现 count_above_kernel (条件计数 + SMEM 归约 + atomicAdd)
//   2. 在 main 中:
//      - cudaMalloc 分配 d_input 和 d_count
//      - cudaMemcpy 把 h_input 传到 GPU
//      - cudaMemset 把 d_count 清零! (atomicAdd 是累加的, 必须从 0 开始)
//      - 启动 kernel
//      - cudaMemcpy 把 d_count 传回 (只有 1 个 int!)
//      - cudaFree
//
// 注意:
//   - 这次输出只有 1 个 int, 不需要 partial 数组!
//   - cudaMemset(d_count, 0, sizeof(int)) 必须在 launch 之前
//   - 用 GRID_SIZE = 256 (Grid-Stride Loop 处理任意大的 N)
// ============================================================

#include <cstdio>
#include <cstdlib>
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

// TODO: 实现 count_above_kernel
__global__ void count_above_kernel(const float *input, int *count,
                                   float threshold, int n) {
    // --- 在这里写你的代码 ---

}

int main() {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);
    const float threshold = 0.0f;

    float *h_input = (float *)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; i++)
        h_input[i] = (float)(rand() % 2000 - 1000) / 100.0f;

    int ref_count = 0;
    for (int i = 0; i < N; i++)
        if (h_input[i] > threshold) ref_count++;

    // ============================================================
    // TODO: 完成以下步骤
    //   1. cudaMalloc: d_input (bytes), d_count (sizeof(int))
    //   2. cudaMemcpy: h_input → d_input
    //   3. cudaMemset(d_count, 0, sizeof(int))  ← 别忘了!
    //   4. 启动 count_above_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(...)
    //   5. cudaMemcpy: d_count → &gpu_count (只有 1 个 int!)
    //   6. cudaFree
    // ============================================================

    int gpu_count = 0;

    // --- 在这里写你的代码 ---

    bool ok = (gpu_count == ref_count);
    printf("CPU count = %d, GPU count = %d\n", ref_count, gpu_count);
    printf("结果: %s\n", ok ? "✓ 全部正确" : "✗ 有错误");

    free(h_input);
    return 0;
}
