// ============================================================
// 练习 3: GEMV 矩阵-向量乘 (Shared Memory 归约)
// 难度: Level 2 (kernel + host 端都要自己写)
//
// 编译: nvcc -O2 -o ex3_gemv_tiled_level2 ex3_gemv_tiled_level2.cu
// 运行: ./ex3_gemv_tiled_level2
// 预期输出: "结果: ✓ 全部正确"
//
// 公式: y[i] = Σ_k A[i][k] * x[k]
//   A: M×K 矩阵, x: 长度 K 的向量, y: 长度 M 的向量
//
// 要求:
//   1. 实现 gemv_kernel (用 Shared Memory 做 Block 内归约)
//   2. 在 main 中:
//      - cudaMalloc 分配 d_A, d_x, d_y
//      - cudaMemcpy 把 A 和 x 传到 GPU (2 次 H2D)
//      - 启动 kernel: gemv_kernel<<<M, BLOCK_SIZE>>>(...)
//      - cudaMemcpy 把 y 传回 (1 次 D2H)
//      - cudaFree
//
// kernel 算法 (两个阶段, 详见提示):
//   阶段 1: 每个线程算自己的部分和 local_sum
//           (用 grid-stride: for k = tid; k < K; k += BLOCK_SIZE)
//   阶段 2: SMEM 归约 — 写 partial[tid], __syncthreads(), 对半折叠求和
//
// 注意:
//   - __syncthreads() 必须放在 if 外面 (放 if 里会死锁)
//   - 启动配置: <<<M, BLOCK_SIZE>>> — M 个 Block, 每个 Block 算 1 行
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
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

// TODO: 实现 gemv_kernel
__global__ void gemv_kernel(const float *A, const float *x, float *y,
                            int M, int K) {
    // --- 在这里写你的代码 ---

}

int main() {
    const int M = 512, K = 2048;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeX = K * sizeof(float);
    size_t sizeY = M * sizeof(float);

    float *h_A = (float *)malloc(sizeA);
    float *h_x = (float *)malloc(sizeX);
    float *h_y = (float *)malloc(sizeY);
    float *h_y_ref = (float *)malloc(sizeY);

    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K; i++) h_x[i] = (float)(rand() % 100) / 100.0f;

    for (int i = 0; i < M; i++) {
        float sum = 0;
        for (int k = 0; k < K; k++) sum += h_A[i * K + k] * h_x[k];
        h_y_ref[i] = sum;
    }

    // ============================================================
    // TODO: 完成以下步骤
    //   1. cudaMalloc 分配 d_A, d_x, d_y
    //   2. cudaMemcpy 把 h_A 和 h_x 传到 GPU
    //   3. 启动 gemv_kernel<<<M, BLOCK_SIZE>>>(d_A, d_x, d_y, M, K);
    //   4. cudaMemcpy 把 d_y 传回 h_y
    //   5. cudaFree
    // ============================================================

    // --- 在这里写你的代码 ---

    bool ok = true;
    for (int i = 0; i < M; i++) {
        if (fabsf(h_y[i] - h_y_ref[i]) > 0.1f) {
            printf("验证失败 @ y[%d]: got %.4f, expected %.4f\n", i, h_y[i], h_y_ref[i]);
            ok = false;
            break;
        }
    }
    printf("结果: %s\n", ok ? "✓ 全部正确" : "✗ 有错误");

    free(h_A); free(h_x); free(h_y); free(h_y_ref);
    return 0;
}
