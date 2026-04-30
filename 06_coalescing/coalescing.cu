#include <cstdio>
#include <cuda_runtime.h>

// ============================================================
// 实验: 亲眼看到合并访问 vs 非合并访问的性能差异
//
// 配合理论: theory/03_memory_hierarchy.md 3.4 节 (合并访问)
//
// 合并访问: 同一 Warp 的 32 个线程访问连续地址 → 合并为 1 次内存事务
// 非合并:   每个线程访问的地址相隔很远 → 多次内存事务 → 浪费带宽
//
// 本程序对比:
//   连续访问 (stride=1): data[tid]           → 完美合并
//   跨步访问 (stride=32): data[tid * 32]     → 每个线程在不同 cache line
//   随机访问: data[random_index[tid]]        → 完全随机
// ============================================================

#define N (1 << 22)  // 4M 元素
#define BLOCK_SIZE 256

__global__ void access_coalesced(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0;
    // 连续访问: Thread 0 读 data[0], Thread 1 读 data[1], ...
    // 同一 Warp 的 32 线程读 32 个连续 float = 128 bytes = 1 次事务!
    for (int i = idx; i < n; i += stride) {
        sum += input[i];
    }
    if (idx == 0) output[0] = sum;
}

__global__ void access_strided(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0;
    int n_strided = n / 32;
    // 跨步访问: Thread 0 读 data[0], Thread 1 读 data[32], ...
    // 同一 Warp 的 32 线程读 32 个相隔 128 字节的位置 = 32 次事务!
    for (int i = idx; i < n_strided; i += stride) {
        sum += input[i * 32];
    }
    if (idx == 0) output[0] = sum;
}

__global__ void setup_random_indices(int *indices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // 简单的伪随机: 用线性同余生成散列索引
    for (int i = idx; i < n; i += stride) {
        indices[i] = (i * 2654435761u) % n;
    }
}

__global__ void access_random(const float *input, const int *indices,
                               float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0;
    // 随机访问: 每个线程的地址完全不可预测
    for (int i = idx; i < n; i += stride) {
        sum += input[indices[i]];
    }
    if (idx == 0) output[0] = sum;
}

int main() {
    float *d_input, *d_output;
    int *d_indices;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    cudaMalloc(&d_indices, N * sizeof(int));

    // 初始化
    float *h_input = new float[N];
    for (int i = 0; i < N; i++) h_input[i] = 1.0f;
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_input;

    int gridSize = 256;
    setup_random_indices<<<gridSize, BLOCK_SIZE>>>(d_indices, N);
    cudaDeviceSynchronize();

    // 计时宏
    #define BENCH(name, ...) {                                            \
        cudaEvent_t start, stop;                                          \
        cudaEventCreate(&start); cudaEventCreate(&stop);                  \
        __VA_ARGS__; cudaDeviceSynchronize();                             \
        cudaEventRecord(start);                                           \
        for (int i = 0; i < 50; i++) { __VA_ARGS__; }                    \
        cudaEventRecord(stop); cudaEventSynchronize(stop);               \
        float ms; cudaEventElapsedTime(&ms, start, stop);                \
        printf("%-20s: %.3f ms  ", name, ms/50);                         \
        float gb = (float)N * 4 / 1e9;                                   \
        printf("有效带宽: %.1f GB/s\n", gb / (ms/50/1000));              \
        cudaEventDestroy(start); cudaEventDestroy(stop);                  \
    }

    printf("全局内存访问模式性能对比 (N = %d 个 float = %.1f MB)\n\n", N, N*4.0f/1e6);

    BENCH("连续 (stride=1)", access_coalesced<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, N));
    BENCH("跨步 (stride=32)", access_strided<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, N));
    BENCH("随机", access_random<<<gridSize, BLOCK_SIZE>>>(d_input, d_indices, d_output, N));

    printf("\n观察:\n");
    printf("  连续访问的有效带宽接近该卡的显存峰值 (A100 常见可到 ~1800 GB/s 量级)\n");
    printf("  跨步和随机访问的有效带宽急剧下降 — 这就是合并访问的重要性!\n");
    printf("\n用 ncu 看 Global Load Efficiency:\n");
    printf("  ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld ./coalescing\n");

    cudaFree(d_input); cudaFree(d_output); cudaFree(d_indices);
    return 0;
}
