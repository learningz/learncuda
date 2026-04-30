// ============================================================
// Ch6 练习 2: TF32 精度对比 (纯 CUDA 版)
// 配合: theory/06_tensor_core.md 练习 2
//
// 编译: nvcc -O2 -arch=sm_75 -o ch06_ex2_tf32 ch06_ex2_tf32.cu
// 运行: ./ch06_ex2_tf32
//
// 对比 FP32 CUDA Core 矩阵乘和 FP16 Tensor Core 矩阵乘的精度差异。
// (真正的 TF32 需要 Ampere+, 这里用 FP16 Tensor Core 做近似演示)
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

__global__ void matmul_fp32(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float s = 0;
        for (int k = 0; k < N; k++) s += A[row*N+k] * B[k*N+col];
        C[row*N+col] = s;
    }
}

__global__ void matmul_wmma_16x16(const half *A, const half *B, float *C) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    wmma::load_matrix_sync(a, A, 16);
    wmma::load_matrix_sync(b, B, 16);
    wmma::mma_sync(c, a, b, c);
    wmma::store_matrix_sync(C, c, 16, wmma::mem_row_major);
}

int main() {
    const int N = 16;

    float h_Af[N*N], h_Bf[N*N], h_Cf32[N*N], h_Ctc[N*N];
    half h_Ah[N*N], h_Bh[N*N];
    srand(42);
    for (int i = 0; i < N*N; i++) {
        h_Af[i] = (float)(rand() % 100) / 10.0f;
        h_Bf[i] = (float)(rand() % 100) / 10.0f;
        h_Ah[i] = __float2half(h_Af[i]);
        h_Bh[i] = __float2half(h_Bf[i]);
    }

    float *dAf, *dBf, *dCf32, *dCtc;
    half *dAh, *dBh;
    CUDA_CHECK(cudaMalloc(&dAf, N*N*4)); CUDA_CHECK(cudaMalloc(&dBf, N*N*4));
    CUDA_CHECK(cudaMalloc(&dCf32, N*N*4)); CUDA_CHECK(cudaMalloc(&dCtc, N*N*4));
    CUDA_CHECK(cudaMalloc(&dAh, N*N*2)); CUDA_CHECK(cudaMalloc(&dBh, N*N*2));
    CUDA_CHECK(cudaMemcpy(dAf, h_Af, N*N*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBf, h_Bf, N*N*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dAh, h_Ah, N*N*2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBh, h_Bh, N*N*2, cudaMemcpyHostToDevice));

    dim3 block(16,16), grid(1,1);
    matmul_fp32<<<grid,block>>>(dAf,dBf,dCf32,N);
    matmul_wmma_16x16<<<1,32>>>(dAh,dBh,dCtc);
    CUDA_CHECK(cudaMemcpy(h_Cf32, dCf32, N*N*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Ctc, dCtc, N*N*4, cudaMemcpyDeviceToHost));

    float maxdiff = 0, maxrel = 0;
    for (int i = 0; i < N*N; i++) {
        float d = fabsf(h_Cf32[i] - h_Ctc[i]);
        if (d > maxdiff) maxdiff = d;
        float r = d / fmaxf(fabsf(h_Cf32[i]), 1e-7f);
        if (r > maxrel) maxrel = r;
    }

    printf("FP32 CUDA Core vs FP16 Tensor Core 精度对比 (16×16)\n\n");
    printf("最大绝对误差: %.4f\n", maxdiff);
    printf("最大相对误差: %.4f%%\n", maxrel * 100);
    printf("\n对深度学习训练: 这个误差通常可以接受 (梯度本身就有噪声)。\n");

    cudaFree(dAf); cudaFree(dBf); cudaFree(dCf32);
    cudaFree(dCtc); cudaFree(dAh); cudaFree(dBh);
    return 0;
}
