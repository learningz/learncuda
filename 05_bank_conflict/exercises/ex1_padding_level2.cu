// ============================================================
// 练习 1: 用 Padding 消除 Bank Conflict
// 难度: Level 2 (从零写一个 padded SMEM kernel)
//
// 编译: nvcc -O2 -o ex1_padding_level2 ex1_padding_level2.cu
// 运行: ./ex1_padding_level2
//
// 任务:
//   写一个 kernel, 用 stride=32 访问 Shared Memory, 但通过 padding 避免冲突。
//   然后用 main 中的 benchmark 测它的速度, 应该和 stride=1 接近。
//
// 提示:
//   - SMEM 大小: __shared__ float smem[32 * (32 + 1)]
//   - 索引: smem[tid * (32 + 1)]
//   - 内层循环结构和 bank_conflict.cu 一样:
//       for r = 0..REPEAT: 写 → __syncthreads → 读 → __syncthreads
//   - 最后 if (tid == 0) output[blockIdx.x] = sum
// ============================================================

#include <cstdio>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define REPEAT 1000

// TODO: 实现一个用 padding 避免 Bank Conflict 的 kernel
__global__ void padded_kernel(float *output) {
    // --- 在这里写你的代码 ---

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

    float t = benchmark(padded_kernel, d_out);
    printf("padded_kernel 耗时: %.3f ms\n", t);
    printf("(对比 bank_conflict.cu 中 stride=32 = ~快 10x 才算对)\n");

    cudaFree(d_out);
    return 0;
}
