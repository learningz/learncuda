// ============================================================
// 练习 3: Fused Multiply-Add — d[i] = a[i] * b[i] + c[i]
// 难度: Level 2 (kernel + host 端都要自己写)
//
// 编译: nvcc -O2 -o ex3_fma_level2 ex3_fma_level2.cu
// 运行: ./ex3_fma_level2
// 预期输出: "结果: ✓ 全部正确"
//
// 要求:
//   1. 实现 fma_kernel
//   2. 在 main 中完成 GPU 内存分配、数据传输、kernel 启动、结果回传
//
// 注意:
//   - 这次有 3 个输入数组 + 1 个输出数组 → 需要 4 块 GPU 内存
//   - 3 个输入都要传到 GPU, 只有 1 个输出需要传回 CPU
//   - 想想: 4 次 cudaMalloc, 3 次 H2D cudaMemcpy, 1 次 D2H cudaMemcpy
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

// TODO: 实现 fma_kernel
__global__ void fma_kernel(const float *a, const float *b, const float *c,
                           float *d, int n) {
    // --- 在这里写你的代码 ---

}

int main() {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    float *h_d = (float *)malloc(bytes);
    float *h_d_ref = (float *)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i % 100) / 10.0f;
        h_b[i] = static_cast<float>(i % 77) / 7.0f;
        h_c[i] = static_cast<float>(i % 50) / 5.0f;
        h_d_ref[i] = h_a[i] * h_b[i] + h_c[i];
    }

    // ============================================================
    // TODO: 完成以下步骤
    //   1. cudaMalloc 分配 4 块 GPU 内存 (d_a, d_b, d_c, d_d)
    //   2. cudaMemcpy 把 h_a, h_b, h_c 传到 GPU (3 次 H2D)
    //   3. 计算 gridSize 和 blockSize, 启动 kernel
    //   4. cudaMemcpy 把 d_d 传回 h_d (1 次 D2H)
    //   5. cudaFree 释放 4 块 GPU 内存
    // ============================================================

    // --- 在这里写你的代码 ---

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_d[i] - h_d_ref[i]) > 1e-4f) {
            printf("验证失败 @ i=%d: got %.6f, expected %.6f\n", i, h_d[i], h_d_ref[i]);
            ok = false;
            break;
        }
    }
    printf("结果: %s\n", ok ? "✓ 全部正确" : "✗ 有错误");

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d);
    free(h_d_ref);
    return 0;
}
