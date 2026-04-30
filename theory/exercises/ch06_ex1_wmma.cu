// ============================================================
// Ch6 练习 1: WMMA 16×16 矩阵乘法
// 配合: theory/06_tensor_core.md 练习 1
//
// 编译: nvcc -O2 -arch=sm_75 -o ch06_ex1_wmma ch06_ex1_wmma.cu
// 运行: ./ch06_ex1_wmma
// 预期输出: "结果: ✓ 全部正确"
//
// TODO: 实现 wmma_matmul_kernel (只填 kernel 函数体)
//   用 WMMA API 做 D = A × B + C (16×16, half 输入, float 累加)
//
// 提示:
//   wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
//   wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
//   wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
//   wmma::fill_fragment(c_frag, 0.0f);
//   wmma::load_matrix_sync(a_frag, d_A, 16);
//   wmma::load_matrix_sync(b_frag, d_B, 16);
//   wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
//   wmma::store_matrix_sync(d_C, c_frag, 16, wmma::mem_row_major);
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){   \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}} while(0)

// TODO: 实现 WMMA 16×16 矩阵乘 kernel
// 配置: <<<1, 32>>> (1 个 Warp)
__global__ void wmma_matmul_kernel(const half *A, const half *B, float *C) {
    // --- 在这里写你的代码 ---

}

int main() {
    const int M = 16, K = 16, N = 16;

    half h_A[M*K], h_B[K*N];
    float h_C[M*N], h_ref[M*N];

    for (int i = 0; i < M*K; i++) h_A[i] = __float2half((float)(i % 5));
    for (int i = 0; i < K*N; i++) h_B[i] = __float2half((float)(i % 3));

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < K; k++)
                s += __half2float(h_A[i*K+k]) * __half2float(h_B[k*N+j]);
            h_ref[i*N+j] = s;
        }

    half *d_A, *d_B; float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M*K*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, K*N*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, M*N*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M*K*sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K*N*sizeof(half), cudaMemcpyHostToDevice));

    wmma_matmul_kernel<<<1, 32>>>(d_A, d_B, d_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost));

    float maxerr = 0;
    for (int i = 0; i < M*N; i++) maxerr = fmaxf(maxerr, fabsf(h_C[i] - h_ref[i]));
    printf("最大误差: %.2e\n", maxerr);
    printf("结果: %s\n", maxerr < 1.0f ? "✓ 全部正确" : "✗ 有错误");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
