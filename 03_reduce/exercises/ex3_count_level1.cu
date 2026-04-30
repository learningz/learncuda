// ============================================================
// 练习 3: 计数大于阈值的元素 — count(x[i] > threshold)
// 难度: Level 1 (只需填写 kernel 函数体)
//
// 编译: nvcc -O2 -o ex3_count_level1 ex3_count_level1.cu
// 运行: ./ex3_count_level1
// 预期输出: "结果: ✓ 全部正确"
//
// 综合: 条件判断 + 归约 + atomicAdd
//
// 思路:
//   1. Grid-Stride Loop: 每个线程遍历自己负责的元素
//      如果 input[i] > threshold, local_count++
//   2. SMEM 归约: 把 Block 内所有线程的 local_count 汇总
//   3. tid==0 用 atomicAdd 把 Block 的结果加到全局计数器
//      → atomicAdd(&output[0], block_count)
//      → 这样只需要一次 kernel 调用, 不需要 CPU 汇总!
//
// 提示:
//   - local_count 用 int 类型 (不是 float)
//   - SMEM 也用 int: __shared__ int scount[BLOCK_SIZE]
//   - atomicAdd 对 int 同样有效
//   - 记得在 launch 前 cudaMemset(d_count, 0, sizeof(int))
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

// ============================================================
// TODO: 实现这个 kernel
// 统计 input 中大于 threshold 的元素个数, 用 atomicAdd 写到 count[0]
// ============================================================
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

    float *d_input;
    int *d_count;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

    count_above_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_count, threshold, N);
    CUDA_CHECK(cudaGetLastError());

    int gpu_count = 0;
    CUDA_CHECK(cudaMemcpy(&gpu_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));

    bool ok = (gpu_count == ref_count);
    printf("CPU count = %d, GPU count = %d\n", ref_count, gpu_count);
    printf("结果: %s\n", ok ? "✓ 全部正确" : "✗ 有错误");

    cudaFree(d_input); cudaFree(d_count);
    free(h_input);
    return 0;
}
