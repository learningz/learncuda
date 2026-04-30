// ============================================================
// 练习 1: 用 Padding 消除 Bank Conflict
// 难度: Level 1 (只需修改 SMEM 声明和索引)
//
// 编译: nvcc -O2 -o ex1_padding_level1 ex1_padding_level1.cu
// 运行: ./ex1_padding_level1
// 预期输出: stride=32 加 padding 后的耗时接近 stride=1 (无冲突)
//
// bank_conflict.cu 展示了 stride=32 时的 32-way Bank Conflict。
// 修复方法: 声明 SMEM 时多加 1 列 (padding), 使 stride=32 的访问不再
// 全部落到同一个 Bank。
//
// 原理:
//   32 个 Bank, 每 Bank 宽 4B。地址 → Bank 映射: bank = (addr/4) % 32
//   stride=32: 地址 0, 128, 256, ... → bank 0, 0, 0, ... → 32-way 冲突!
//   加 1 列 padding: smem[tid * (32 + 1)]
//     地址 0, 132, 264, ... → bank 0, 1, 2, ... → 无冲突!
//
// TODO:
//   下面的 kernel 已经写好了 stride=32 的无 padding 版本。
//   你需要修改 SMEM 声明和访问索引, 加上 padding, 消除冲突。
//
// 提示:
//   - 把 smem[32 * 32] 改成 smem[32 * (32 + 1)]
//   - 把 tid * 32 改成 tid * (32 + 1)
//   - 只需改 2 处!
// ============================================================

#include <cstdio>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define REPEAT 1000

// 无 padding 版本 (有 32-way Bank Conflict) — 对照组
__global__ void conflict_no_padding(float *output) {
    __shared__ float smem[32 * 32];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int r = 0; r < REPEAT; r++) {
        smem[tid * 32] = (float)tid;
        __syncthreads();
        sum += smem[tid * 32];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sum;
}

// ============================================================
// TODO: 修改 SMEM 声明和索引, 加 padding 消除 Bank Conflict
// ============================================================
__global__ void conflict_with_padding(float *output) {
    // TODO: 把这行的 smem 大小改成 32 * (32 + 1)
    __shared__ float smem[32 * 32];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int r = 0; r < REPEAT; r++) {
        // TODO: 把 tid * 32 改成 tid * (32 + 1)
        smem[tid * 32] = (float)tid;
        __syncthreads();
        // TODO: 这里也改
        sum += smem[tid * 32];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sum;
}

float benchmark(void (*kernel)(float*), float *d_out) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    kernel<<<1, BLOCK_SIZE>>>(d_out);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) kernel<<<1, BLOCK_SIZE>>>(d_out);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms / 100.0f;
}

int main() {
    float *d_out;
    cudaMalloc(&d_out, 256 * sizeof(float));

    printf("Bank Conflict Padding 实验\n\n");

    float t_conflict = benchmark(conflict_no_padding, d_out);
    printf("stride=32 无 padding (32-way 冲突): %.3f ms\n", t_conflict);

    float t_padded = benchmark(conflict_with_padding, d_out);
    printf("stride=32 有 padding (无冲突):       %.3f ms\n", t_padded);

    printf("\n加速比: %.2fx\n", t_conflict / t_padded);
    printf("(如果你正确加了 padding, 加速比应该接近 bank_conflict.cu 中 stride=1 的水平)\n");

    cudaFree(d_out);
    return 0;
}
