// ============================================================
// 练习 2: 点积 (Dot Product) — sum(a[i] * b[i])
// 难度: Level 2 (kernel + host 端都要自己写)
//
// 编译: nvcc -O2 -o ex2_dot_level2 ex2_dot_level2.cu
// 运行: ./ex2_dot_level2
// 预期输出: "结果: ✓ 全部正确"
//
// 要求:
//   1. 实现 dot_kernel (Grid-Stride 乘加 + SMEM 归约)
//   2. 在 main 中: 分配 d_a, d_b, d_partial → 传数据 → 启动 kernel → 回传
//   3. CPU 端汇总 partial 结果
//
// 注意:
//   - 需要 3 块 GPU 内存: d_a, d_b, d_partial (大小 = GRID_SIZE)
//   - GRID_SIZE 取 256 (不是 ceil(N/BLOCK_SIZE), 用 Grid-Stride Loop)
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

// TODO: 实现 dot_kernel
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

    // ============================================================
    // TODO: 完成以下步骤
    //   1. cudaMalloc: d_a (bytes), d_b (bytes), d_partial (GRID_SIZE * 4)
    //   2. cudaMemcpy: h_a → d_a, h_b → d_b
    //   3. 启动 dot_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_partial, N)
    //   4. cudaMemcpy: d_partial → h_partial
    //   5. CPU 汇总 h_partial[0..GRID_SIZE-1]
    //   6. cudaFree
    // ============================================================

    double gpu_dot = 0;

    // --- 在这里写你的代码 ---

    bool ok = (fabs(gpu_dot - ref_dot) / fabs(ref_dot) < 1e-4);
    printf("CPU dot = %.4f, GPU dot = %.4f\n", ref_dot, gpu_dot);
    printf("结果: %s\n", ok ? "✓ 全部正确" : "✗ 有错误");

    free(h_a); free(h_b);
    return 0;
}
