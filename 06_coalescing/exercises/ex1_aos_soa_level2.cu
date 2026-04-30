// ============================================================
// 练习 1: AoS → SoA 改写
// 难度: Level 2 (kernel + host 端都要自己写)
//
// 编译: nvcc -O2 --extended-lambda -o ex1_aos_soa_level2 ex1_aos_soa_level2.cu
// 运行: ./ex1_aos_soa_level2
//
// 任务:
//   1. 实现 scale_aos_kernel 和 scale_soa_kernel
//   2. 在 main 中: 分配 AoS + SoA 数据, 传到 GPU, 分别启动两个 kernel, 回传 + 验证
//   3. 用 cudaEvent 计时, 对比两者的性能差异
//
// 注意:
//   - AoS: cudaMalloc N * sizeof(float4) → 每个元素 16B
//   - SoA: cudaMalloc N * sizeof(float) → 只需要 x 分量
//   - 两个 kernel 输出到同一个 d_out (每次用完验证再下一个)
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

// TODO: 实现 scale_aos_kernel
__global__ void scale_aos_kernel(const float4 *particles, float *output, int n) {
    // --- 在这里写你的代码 ---

}

// TODO: 实现 scale_soa_kernel
__global__ void scale_soa_kernel(const float *px, float *output, int n) {
    // --- 在这里写你的代码 ---

}

int main() {
    const int N = 1 << 22;

    float4 *h_aos = (float4 *)malloc(N * sizeof(float4));
    float *h_soa_x = (float *)malloc(N * sizeof(float));
    float *h_ref = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        float val = (float)(i % 1000) / 10.0f;
        h_aos[i] = make_float4(val, val + 1, val + 2, val + 3);
        h_soa_x[i] = val;
        h_ref[i] = val * 2.0f;
    }

    // ============================================================
    // TODO: 完成以下步骤
    //   1. cudaMalloc: d_aos, d_soa_x, d_out
    //   2. cudaMemcpy: h_aos → d_aos, h_soa_x → d_soa_x
    //   3. 启动 scale_aos_kernel, 验证 + 计时
    //   4. 启动 scale_soa_kernel, 验证 + 计时
    //   5. 打印对比, cudaFree
    // ============================================================

    // --- 在这里写你的代码 ---

    free(h_aos); free(h_soa_x); free(h_ref);
    return 0;
}
