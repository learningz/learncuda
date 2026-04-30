#include <cstdio>
#include <cuda_runtime.h>

// ============================================================
// 实验: 亲眼看到 Shared Memory Bank Conflict 的性能差异
//
// 配合理论: theory/03_memory_hierarchy.md 3.3 节 (Bank Conflict)
//
// Shared Memory 有 32 个 Bank, 每 Bank 宽 4 字节。
// 同一 Warp 的多个线程访问同一 Bank 的不同地址 → Bank Conflict → 变慢。
//
// 本程序对比 3 种访问模式:
//   stride=1:  无冲突 (每线程访问不同 Bank)
//   stride=2:  2-way 冲突 (Thread 0 和 Thread 16 访问同一 Bank)
//   stride=32: 32-way 冲突 (所有线程访问同一 Bank, 最坏情况!)
// ============================================================

#define BLOCK_SIZE 256
#define REPEAT 1000

// 通过 stride 参数控制访问模式
template <int STRIDE>
__global__ void bank_conflict_test(float *output) {
    // 声明一块 Shared Memory (足够大以容纳各种 stride)
    __shared__ float smem[32 * 32];

    int tid = threadIdx.x;
    float sum = 0.0f;

    // 写入 Shared Memory (用 stride 访问)
    for (int r = 0; r < REPEAT; r++) {
        smem[tid * STRIDE] = (float)tid;
        __syncthreads();

        // 读取 Shared Memory (用 stride 访问)
        sum += smem[tid * STRIDE];
        __syncthreads();
    }

    // 防止编译器优化掉循环
    if (tid == 0) output[blockIdx.x] = sum;
}

float benchmark(void (*kernel)(float*), float *d_out) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热
    kernel<<<1, BLOCK_SIZE>>>(d_out);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        kernel<<<1, BLOCK_SIZE>>>(d_out);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / 100.0f;
}

int main() {
    float *d_out;
    cudaMalloc(&d_out, 256 * sizeof(float));

    printf("Shared Memory Bank Conflict 性能对比\n");
    printf("每次测试重复 %d 次读写, 取 100 次平均\n\n", REPEAT);

    float t1 = benchmark(bank_conflict_test<1>, d_out);
    printf("stride=1  (无冲突):     %.3f ms\n", t1);

    float t2 = benchmark(bank_conflict_test<2>, d_out);
    printf("stride=2  (2-way冲突):  %.3f ms  (%.1fx slower)\n", t2, t2/t1);

    float t32 = benchmark(bank_conflict_test<32>, d_out);
    printf("stride=32 (32-way冲突): %.3f ms  (%.1fx slower)\n", t32, t32/t1);

    printf("\n理论预测: stride=2 慢 ~2x, stride=32 慢 ~32x\n");
    printf("实际差异可能小于理论值 (因为编译器优化和其他开销)\n");
    printf("\n用 ncu 可以看到具体的 bank conflict 次数:\n");
    printf("  ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared ./bank_conflict\n");

    cudaFree(d_out);
    return 0;
}
