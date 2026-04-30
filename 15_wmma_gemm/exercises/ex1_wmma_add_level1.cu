// ============================================================
// 练习 1: WMMA 矩阵加法 — 最简单的 Fragment 操作
// 难度: Level 1 (只需填写 kernel 函数体)
//
// 编译: nvcc -O2 -arch=sm_75 -o ex1_wmma_add_level1 ex1_wmma_add_level1.cu
// 运行: ./ex1_wmma_add_level1
// 预期输出: "结果: ✓ 全部正确"
//
// wmma_gemm.cu 做的是 D = A×B+C (矩阵乘)。
// 本题只做 D = C + C_bias, 不做乘法 — 最小化地练 Fragment 的 load/store。
//
// 步骤:
//   1. 声明两个 accumulator fragment (float, 16×16)
//   2. load_matrix_sync 从全局显存加载 C 和 C_bias
//   3. 手动遍历 fragment 的 .x[] 数组, 逐元素相加
//   4. store_matrix_sync 写回
//
// 提示:
//   - fragment 的类型: wmma::fragment<wmma::accumulator, 16, 16, 16, float>
//   - frag.num_elements 告诉你这个线程持有多少个元素
//   - frag.x[i] 可以直接读写
//   - load: wmma::load_matrix_sync(frag, ptr, ldm, wmma::mem_row_major)
//   - store: wmma::store_matrix_sync(ptr, frag, ldm, wmma::mem_row_major)
//   - 一个 Warp 协作处理一个 16×16 块
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define M_TILE 16
#define N_TILE 16

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
// TODO: 实现 WMMA 矩阵加法 kernel
// 配置: <<<1, 32>>>  (1 个 Warp 处理 1 个 16×16 块)
// ============================================================
__global__ void wmma_add_kernel(const float *C, const float *bias, float *D,
                                int M, int N) {
    // --- 在这里写你的代码 ---
    // 1. 声明 fragment<accumulator, 16,16,16, float> frag_c, frag_bias
    // 2. wmma::load_matrix_sync(frag_c, C, N, wmma::mem_row_major)
    // 3. wmma::load_matrix_sync(frag_bias, bias, N, wmma::mem_row_major)
    // 4. for (int i = 0; i < frag_c.num_elements; i++)
    //        frag_c.x[i] += frag_bias.x[i];
    // 5. wmma::store_matrix_sync(D, frag_c, N, wmma::mem_row_major)

}

int main() {
    const int M = 16, N = 16;
    size_t bytes = M * N * sizeof(float);

    float *h_C = (float *)malloc(bytes);
    float *h_bias = (float *)malloc(bytes);
    float *h_D = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);

    for (int i = 0; i < M * N; i++) {
        h_C[i] = (float)(i % 10);
        h_bias[i] = (float)(i % 3);
        h_ref[i] = h_C[i] + h_bias[i];
    }

    float *d_C, *d_bias, *d_D;
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMalloc(&d_bias, bytes));
    CUDA_CHECK(cudaMalloc(&d_D, bytes));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias, bytes, cudaMemcpyHostToDevice));

    wmma_add_kernel<<<1, 32>>>(d_C, d_bias, d_D, M, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_D, d_D, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < M * N; i++)
        if (fabsf(h_D[i] - h_ref[i]) > 1e-3f) {
            printf("@ %d: got %.1f expected %.1f\n", i, h_D[i], h_ref[i]);
            ok = false; break;
        }
    printf("结果: %s\n", ok ? "✓ 全部正确" : "✗ 有错误");

    cudaFree(d_C); cudaFree(d_bias); cudaFree(d_D);
    free(h_C); free(h_bias); free(h_D); free(h_ref);
    return 0;
}
