// ============================================================
// 练习 1: SAXPY — y[i] = a * x[i] + y[i]
// 难度: Level 2 (kernel + host 端都要自己写)
//
// 编译: nvcc -O2 -o ex1_saxpy_level2 ex1_saxpy_level2.cu
// 运行: ./ex1_saxpy_level2
// 预期输出: "结果: ✓ 全部正确"
//
// 要求:
//   1. 实现 saxpy_kernel
//   2. 在 main 中完成 GPU 内存分配、数据传输、kernel 启动、结果回传
//
// 注意:
//   - y 既是输入也是输出 (原地修改)
//   - 传到 GPU 之前, h_y 里已经有初始值了
//   - 回传之后, h_y 应该变成 a*x+y 的结果
//   - 想想: 你需要几块 GPU 内存? 需要几次 cudaMemcpy?
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

// TODO: 实现 saxpy_kernel
__global__ void saxpy_kernel(float a, const float *x, float *y, int n) {
    // --- 在这里写你的代码 ---

}

int main() {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);
    const float a = 2.5f;

    float *h_x = (float *)malloc(bytes);
    float *h_y = (float *)malloc(bytes);
    float *h_y_ref = (float *)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_x[i] = static_cast<float>(i % 100) / 10.0f;
        h_y[i] = static_cast<float>(i % 50) / 5.0f;
        h_y_ref[i] = a * h_x[i] + h_y[i];
    }

    // ============================================================
    // TODO: 完成以下步骤
    //   1. cudaMalloc 分配 GPU 内存 (需要几块?)
    //   2. cudaMemcpy 把 h_x 和 h_y 传到 GPU
    //   3. 计算 gridSize 和 blockSize, 启动 kernel
    //   4. cudaMemcpy 把结果从 GPU 传回 h_y
    //   5. 最后别忘了 cudaFree
    // ============================================================

    // --- 在这里写你的代码 ---

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_y[i] - h_y_ref[i]) > 1e-5f) {
            printf("验证失败 @ i=%d: got %.6f, expected %.6f\n", i, h_y[i], h_y_ref[i]);
            ok = false;
            break;
        }
    }
    printf("结果: %s\n", ok ? "✓ 全部正确" : "✗ 有错误");

    free(h_x);
    free(h_y);
    free(h_y_ref);
    return 0;
}
