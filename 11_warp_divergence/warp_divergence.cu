#include <cstdio>
#include <cuda_runtime.h>

// ============================================================
// 实验: Warp Divergence 的性能影响
//
// 配合理论: theory/04_warp_and_sync.md 4.1 节
//
// Warp Divergence: 同一 Warp (32线程) 中的线程走了不同的 if/else 分支。
// GPU 必须串行执行两条路径 → 性能下降。
//
// 本程序对比 3 种场景:
//   无分歧:   每个 Warp 的所有线程走同一分支
//   50% 分歧: 每个 Warp 中一半线程走 if, 一半走 else
//   条件计算:  用无分支算术替代 if/else (优化手法)
// ============================================================

#define N (1 << 22)
#define BLOCK_SIZE 256

// 场景 1: 无分歧 — 以 Warp 为边界做分支
// Thread 0-31 (Warp 0) 全走 if, Thread 32-63 (Warp 1) 全走 else, ...
// 同一 Warp 的线程走相同路径 → 无分歧!
__global__ void no_divergence(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int warp_id = threadIdx.x / 32;
    float val = input[idx];
    if (warp_id % 2 == 0) {
        // 所有 Warp 0, 2, 4, 6 走这里
        output[idx] = val * val + val;
    } else {
        // 所有 Warp 1, 3, 5, 7 走这里
        output[idx] = val * 0.5f - 1.0f;
    }
}

// 场景 2: 50% 分歧 — 以奇偶线程做分支
// 每个 Warp 中 Thread 0,2,4... 走 if, Thread 1,3,5... 走 else
// → 每个 Warp 都有分歧 → 两条路径串行执行!
__global__ void half_divergence(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val = input[idx];
    if (threadIdx.x % 2 == 0) {
        // 偶数线程走这里
        output[idx] = val * val + val;
    } else {
        // 奇数线程走这里
        output[idx] = val * 0.5f - 1.0f;
    }
}

// 场景 3: 无分支优化 — 用算术替代 if/else
// fmaxf, 条件乘法等让编译器生成谓词指令而不是分支跳转
__global__ void branchless(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val = input[idx];
    float path_a = val * val + val;
    float path_b = val * 0.5f - 1.0f;
    // 用条件选择替代 if/else — 编译器用谓词指令 (SEL/CSEL), 无分支跳转
    output[idx] = (threadIdx.x % 2 == 0) ? path_a : path_b;
}

int main() {
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    float *h_in = new float[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)(i % 100) / 10.0f;
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_in;

    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    #define BENCH(name, ...) {                                             \
        __VA_ARGS__; cudaDeviceSynchronize();                               \
        cudaEvent_t start, stop;                                            \
        cudaEventCreate(&start); cudaEventCreate(&stop);                    \
        cudaEventRecord(start);                                             \
        for (int i = 0; i < 100; i++) { __VA_ARGS__; }                     \
        cudaEventRecord(stop); cudaEventSynchronize(stop);                  \
        float ms; cudaEventElapsedTime(&ms, start, stop);                  \
        printf("  %-20s: %.3f ms\n", name, ms / 100);                     \
    }

    printf("Warp Divergence 性能对比 (N = %d)\n\n", N);

    BENCH("无分歧", no_divergence<<<grid, BLOCK_SIZE>>>(d_in, d_out, N));
    BENCH("50%%分歧", half_divergence<<<grid, BLOCK_SIZE>>>(d_in, d_out, N));
    BENCH("无分支算术", branchless<<<grid, BLOCK_SIZE>>>(d_in, d_out, N));

    printf("\n观察:\n");
    printf("  '50%%分歧' 应该比 '无分歧' 慢 (两条路径串行执行)\n");
    printf("  '无分支算术' 应该接近 '无分歧' 的速度 (编译器用谓词指令)\n");
    printf("\n这个例子中差异可能不大, 因为两条路径都很短 (编译器自动谓词化)。\n");
    printf("分歧代价在路径很长 (几十条指令) 时更明显。\n");
    printf("\n用 ncu 看 Warp Execution Efficiency:\n");
    printf("  ncu --metrics smsp__thread_inst_executed_per_inst_executed ./warp_divergence\n");

    #undef BENCH
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
