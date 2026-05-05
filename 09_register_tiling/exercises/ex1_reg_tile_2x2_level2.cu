// ============================================================
// 练习 1: 2×2 Register Tiled GEMM
// 难度: Level 2 (kernel + host 端都要自己写)
//
// 编译: nvcc -O2 -o ex1_reg_tile_2x2_level2 ex1_reg_tile_2x2_level2.cu
// 运行: ./ex1_reg_tile_2x2_level2
// 预期输出: "结果: ✓ 全部正确" + Register Tiled 版比朴素版快
//
// Level 1 只要写 kernel, 本题要求全流程自己写。
//
// 要求:
//   1. 实现 gemm_naive (每个线程算 1 个 C 元素)
//   2. 实现 gemm_reg2x2 (每线程算 2×2 的 C 子块, 用 Register Blocking)
//   3. 在 main 中完成: CPU 内存分配 → GPU 内存分配 → 数据传输 →
//      kernel 启动 (两个版本) → cudaEvent 计时 → 结果回传 → 验证
//
// 参数: M=K=N=256, BM=BN=32, BK=16, TM=TN=2
// Block 配置: dim3(BN/TN, BM/TM) = (16, 16) = 256 线程
//
// 提示:
//   - 朴素版: float sum = 0; for k: sum += A[row*K+k] * B[k*N+col]
//   - Register Tiled: 参考 09_register_tiling/gemm_register.cu 的结构
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

#define BM 32
#define BN 32
#define BK 16
#define TM 2
#define TN 2

// TODO: 实现朴素 GEMM kernel
__global__ void gemm_naive(const float *A, const float *B, float *C,
                           int M, int K, int N) {
    // --- 在这里写你的代码 ---

}

// TODO: 实现 2×2 Register Tiled GEMM kernel
// Block: dim3(BN/TN, BM/TM) = (16, 16)
// Grid:  dim3(ceil(N/BN), ceil(M/BM))
__global__ void gemm_reg2x2(const float *A, const float *B, float *C,
                             int M, int K, int N) {
    // --- 在这里写你的代码 ---

}

void cpu_gemm(const float *A, const float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < K; k++) s += A[i*K+k] * B[k*N+j];
            C[i*N+j] = s;
        }
}

int main() {
    const int M = 256, K = 256, N = 256;

    // ============================================================
    // TODO: 完成以下步骤
    //   1. malloc CPU 内存 (hA, hB, hC, hRef)
    //   2. 初始化 hA, hB (随机值), 用 cpu_gemm 算 hRef
    //   3. cudaMalloc GPU 内存 (dA, dB, dC)
    //   4. cudaMemcpy H2D (A, B)
    //   5. 分别 launch gemm_naive 和 gemm_reg2x2, cudaEvent 计时
    //   6. 每次 cudaMemcpy D2H, 验证正确性 (max error < 0.1)
    //   7. 打印耗时和 GFLOPS
    //      GFLOPS = 2.0 * M * K * N / (ms * 1e6)
    //   8. cudaFree + free
    // ============================================================

    // --- 在这里写你的代码 ---

    return 0;
}
