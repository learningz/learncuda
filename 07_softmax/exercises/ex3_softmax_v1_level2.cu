// ============================================================
// 练习 3: 朴素 3-pass Softmax — 从零实现 V1 版本
// 难度: Level 2 (kernel + host 端都要自己写)
//
// 编译: nvcc -O2 -o ex3_softmax_v1_level2 ex3_softmax_v1_level2.cu
// 运行: ./ex3_softmax_v1_level2
// 预期输出: "结果: ✓ 全部正确"
//
// 这是最完整的练习: 从零写 softmax_3pass, 包含:
//   Pass 1: SMEM 归约求 max
//   Pass 2: SMEM 归约求 sum(exp(x - max))
//   Pass 3: 归一化 y[i] = exp(x[i] - max) / sum
//
// 配置: <<<rows, blockSize, blockSize * sizeof(float)>>>
//   每个 Block 处理一行
//
// 提示:
//   - 每个 pass 的归约结构都一样: 写 smem → __syncthreads → 对半折叠
//   - Pass 1 用 fmaxf, Pass 2 用 +=
//   - 两个 pass 之间需要 __syncthreads 隔开
//   - 归约完成后, smem[0] 保存结果 (max 或 sum), 广播给所有线程
//   - Pass 3 不需要归约, 每个线程独立算 y[i] = expf(x[i] - max) / sum
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

// TODO: 实现 3-pass softmax kernel
__global__ void softmax_3pass_kernel(const float *input, float *output,
                                     int rows, int cols) {
    // --- 在这里写你的代码 ---

}

void softmax_cpu(const float *input, float *output, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float mx = -INFINITY;
        for (int c = 0; c < cols; c++) mx = fmaxf(mx, input[r * cols + c]);
        float sum = 0;
        for (int c = 0; c < cols; c++) sum += expf(input[r * cols + c] - mx);
        for (int c = 0; c < cols; c++)
            output[r * cols + c] = expf(input[r * cols + c] - mx) / sum;
    }
}

int main() {
    const int rows = 1024, cols = 4096;
    size_t bytes = rows * cols * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);

    srand(42);
    for (int i = 0; i < rows * cols; i++)
        h_in[i] = (float)(rand() % 200 - 100) / 10.0f;

    softmax_cpu(h_in, h_ref, rows, cols);

    // ============================================================
    // TODO: 完成以下步骤
    //   1. cudaMalloc: d_in, d_out
    //   2. cudaMemcpy: h_in → d_in
    //   3. 启动 softmax_3pass_kernel<<<rows, blockSize, smem_bytes>>>
    //      blockSize = 256, smem_bytes = blockSize * sizeof(float)
    //   4. cudaMemcpy: d_out → h_out
    //   5. cudaFree
    // ============================================================

    // --- 在这里写你的代码 ---

    bool ok = true;
    for (int i = 0; i < rows * cols; i++) {
        if (fabsf(h_out[i] - h_ref[i]) > 1e-5f) {
            printf("验证失败 @ %d: got %.6f, expected %.6f\n", i, h_out[i], h_ref[i]);
            ok = false; break;
        }
    }
    printf("结果: %s\n", ok ? "✓ 全部正确" : "✗ 有错误");

    free(h_in); free(h_out); free(h_ref);
    return 0;
}
