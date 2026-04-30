// ============================================================
// 练习 2: Vector Scale — 合并写 vs 非合并写
// 难度: Level 2 (kernel + host 端都要自己写)
//
// 编译: nvcc -O2 --extended-lambda -o ex2_write_pattern_level2 ex2_write_pattern_level2.cu
// 运行: ./ex2_write_pattern_level2
//
// 任务:
//   1. 实现 scale_coalesced 和 scale_strided 两个 kernel
//   2. 在 main 中: 分配 GPU 内存, 启动 kernel, 用 cudaEvent 计时对比
//
// 注意:
//   - strided 版本的输出数组需要 N * STRIDE * sizeof(float) 的空间!
//   - 两个 kernel 都只处理 N 个元素, 区别只在写入的地址模式
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define STRIDE 16

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// TODO: 合并写版本
__global__ void scale_coalesced(const float *input, float *output,
                                float scale, int n) {
    // --- 在这里写你的代码 ---

}

// TODO: 非合并写版本 (stride=STRIDE)
__global__ void scale_strided(const float *input, float *output,
                              float scale, int n) {
    // --- 在这里写你的代码 ---

}

int main() {
    const int N = 1 << 22;
    const float scale = 3.14f;

    float *h_in = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (float)i;

    // ============================================================
    // TODO: 完成以下步骤
    //   1. cudaMalloc: d_in (N), d_out_coal (N), d_out_stride (N * STRIDE)
    //   2. cudaMemcpy: h_in → d_in
    //   3. 启动 scale_coalesced, 用 cudaEvent 计时
    //   4. 启动 scale_strided, 用 cudaEvent 计时
    //   5. 打印对比, cudaFree
    // ============================================================

    // --- 在这里写你的代码 ---

    free(h_in);
    return 0;
}
