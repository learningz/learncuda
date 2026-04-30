// ============================================================
// Ch3 练习 2: Bank Conflict — stride=3 测试 + 转置 padding
// 配合: theory/03_memory_hierarchy.md 练习 2
//
// 编译: nvcc -O2 -o ch03_ex2_bank ch03_ex2_bank.cu
// 运行: ./ch03_ex2_bank
//
// 测试 stride=3 是否有 Bank Conflict (GCD(3,32)=1 → 无冲突),
// 然后做一个 32×32 矩阵转置, 对比有/无 padding 的性能差异。
// ============================================================

#include <cstdio>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define REPEAT 1000

template <int STRIDE>
__global__ void bank_test(float *output) {
    __shared__ float smem[32 * 32 + 32];
    int tid = threadIdx.x;
    float sum = 0;
    for (int r = 0; r < REPEAT; r++) {
        smem[tid * STRIDE] = (float)tid;
        __syncthreads();
        sum += smem[tid * STRIDE];
        __syncthreads();
    }
    if (tid == 0) output[0] = sum;
}

__global__ void transpose_no_pad(const float *A, float *B) {
    __shared__ float tile[32][32];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * 32 + ty, col = blockIdx.x * 32 + tx;
    tile[ty][tx] = A[row * 32 + col];
    __syncthreads();
    row = blockIdx.x * 32 + ty; col = blockIdx.y * 32 + tx;
    B[row * 32 + col] = tile[tx][ty];
}

__global__ void transpose_pad(const float *A, float *B) {
    __shared__ float tile[32][33];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * 32 + ty, col = blockIdx.x * 32 + tx;
    tile[ty][tx] = A[row * 32 + col];
    __syncthreads();
    row = blockIdx.x * 32 + ty; col = blockIdx.y * 32 + tx;
    B[row * 32 + col] = tile[tx][ty];
}

int main() {
    float *d_out;
    cudaMalloc(&d_out, 256 * sizeof(float));
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);

    auto bench_bank = [&](const char *name, auto fn) {
        fn<<<1, BLOCK_SIZE>>>(d_out); cudaDeviceSynchronize();
        cudaEventRecord(t0);
        for (int i = 0; i < 100; i++) fn<<<1, BLOCK_SIZE>>>(d_out);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);
        printf("%-25s: %.3f ms\n", name, ms/100);
    };

    printf("Bank Conflict stride 实验\n");
    bench_bank("stride=1 (无冲突)", bank_test<1>);
    bench_bank("stride=2 (2-way)", bank_test<2>);
    bench_bank("stride=3 (无冲突!)", bank_test<3>);
    bench_bank("stride=32 (32-way)", bank_test<32>);
    printf("\nstride=3: GCD(3,32)=1 → 无冲突, 速度应接近 stride=1\n\n");

    float *d_A, *d_B;
    cudaMalloc(&d_A, 32*32*sizeof(float));
    cudaMalloc(&d_B, 32*32*sizeof(float));
    float h[32*32]; for (int i=0;i<32*32;i++) h[i]=(float)i;
    cudaMemcpy(d_A, h, 32*32*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(32,32), grid(1,1);

    auto bench_trans = [&](const char *name, auto fn) {
        fn<<<grid, block>>>(d_A, d_B); cudaDeviceSynchronize();
        cudaEventRecord(t0);
        for (int i = 0; i < 1000; i++) fn<<<grid, block>>>(d_A, d_B);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);
        printf("%-25s: %.4f ms\n", name, ms/1000);
    };

    printf("32x32 矩阵转置 (SMEM)\n");
    bench_trans("无 padding", transpose_no_pad);
    bench_trans("有 padding [32][33]", transpose_pad);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_out); cudaFree(d_A); cudaFree(d_B);
    return 0;
}
