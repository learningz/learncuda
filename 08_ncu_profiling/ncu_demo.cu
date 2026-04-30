#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ============================================================
// 08: ncu Profiling 实战 — 从输出到诊断的完整流程
//
// 配合理论: theory/08_advanced_optimization.md 8.5 节 (Nsight Compute)
//           theory/03_memory_hierarchy.md 3.8 节 (Roofline)
//
// 本程序包含 3 个故意写了不同"病"的 kernel:
//   kernel_A: 未合并访问 (stride 访问)
//   kernel_B: 合并访问但无向量化
//   kernel_C: 合并 + float4 向量化 (最优)
//
// 目的: 让你用 ncu 逐个分析, 看到真实的性能指标差异。
//
// 使用方法:
//   1. 先正常编译运行, 看到三个版本的耗时:
//      nvcc -O2 -arch=sm_80 -o ncu_demo ncu_demo.cu && ./ncu_demo
//
//   2. 用 ncu 采集详细数据:
//      ncu --set full -o profile_report ./ncu_demo
//
//   3. 用 ncu-ui 打开报告 (如果有 GUI):
//      ncu-ui profile_report.ncu-rep
//
//   4. 或者直接在命令行看关键指标:
//      ncu --metrics \
//        sm__throughput.avg.pct_of_peak_sustained_elapsed,\
//        gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
//        l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
//        l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
//        ./ncu_demo
//
// 下面逐步解释你会看到什么。
// ============================================================

#define N (1 << 24)  // 16M 元素
#define BLOCK 256

#define CUDA_CHECK(call) do {                                              \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                cudaGetErrorString(err)); exit(1); }                       \
} while(0)

// ---- kernel A: 故意写的 stride 访问 (不合并!) ----
// 每个线程读 input[tid * 2], 相邻线程的地址间隔 8 字节
// → 同一 Warp 的 32 线程触及 2 条 Cache Line → 50% 带宽浪费
__global__ void kernel_A_strided(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_idx = idx * 2;  // 故意 stride-2
    if (stride_idx < n) {
        out[idx] = in[stride_idx] * 2.0f + 1.0f;
    }
}

// ---- kernel B: 合并访问但标量加载 ----
__global__ void kernel_B_coalesced(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * 2.0f + 1.0f;
    }
}

// ---- kernel C: 合并 + float4 向量化 (最优) ----
__global__ void kernel_C_vectorized(const float *in, float *out, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < n) {
        float4 v = reinterpret_cast<const float4*>(in)[idx / 4];
        v.x = v.x * 2.0f + 1.0f;
        v.y = v.y * 2.0f + 1.0f;
        v.z = v.z * 2.0f + 1.0f;
        v.w = v.w * 2.0f + 1.0f;
        reinterpret_cast<float4*>(out)[idx / 4] = v;
    }
}

int main() {
    size_t bytes = N * sizeof(float);
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    float *h_in = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = (float)i;
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int grid = (N + BLOCK - 1) / BLOCK;
    int grid4 = (N/4 + BLOCK - 1) / BLOCK;

    printf("ncu Profiling 教学 (N = %d = %.0f MB)\n", N, bytes/1e6);
    printf("========================================\n\n");

    #define BENCH(name, ...) {                                             \
        __VA_ARGS__; cudaDeviceSynchronize();                               \
        cudaEvent_t t0, t1;                                                 \
        cudaEventCreate(&t0); cudaEventCreate(&t1);                         \
        cudaEventRecord(t0);                                                \
        for (int i = 0; i < 100; i++) { __VA_ARGS__; }                     \
        cudaEventRecord(t1); cudaEventSynchronize(t1);                      \
        float ms; cudaEventElapsedTime(&ms, t0, t1); ms /= 100;            \
        float gbps = bytes * 2.0f / (ms * 1e6);                            \
        printf("%-25s: %.3f ms  有效带宽: %7.1f GB/s\n", name, ms, gbps);  \
        cudaEventDestroy(t0); cudaEventDestroy(t1);                         \
    }

    BENCH("A: stride-2 (不合并)", kernel_A_strided<<<grid, BLOCK>>>(d_in, d_out, N));
    BENCH("B: 合并 (标量)",       kernel_B_coalesced<<<grid, BLOCK>>>(d_in, d_out, N));
    BENCH("C: 合并 + float4",    kernel_C_vectorized<<<grid4, BLOCK>>>(d_in, d_out, N));
    #undef BENCH

    printf("\n========================================\n");
    printf("现在用 ncu 分析, 你会看到以下差异:\n");
    printf("========================================\n\n");

    printf("1. GPU Speed Of Light (SOL) — 整体利用率\n");
    printf("   ┌──────────────────────────────────────────────┐\n");
    printf("   │ Kernel │ Compute(SM) │ Memory(DRAM)          │\n");
    printf("   │ A      │ ~10%%        │ ~40%%  ← 内存没吃满!  │\n");
    printf("   │ B      │ ~15%%        │ ~75%%  ← 接近带宽峰值 │\n");
    printf("   │ C      │ ~15%%        │ ~85%%  ← 最高!        │\n");
    printf("   └──────────────────────────────────────────────┘\n");
    printf("   → 三个 kernel 都是 Memory Bound (Memory >> Compute)\n");
    printf("   → A 的 Memory 利用率低 = 带宽被浪费 (stride 访问)\n\n");

    printf("2. Memory Workload Analysis — 内存效率\n");
    printf("   关键指标: Global Load Efficiency (有效字节 / 实际传输字节)\n");
    printf("   ┌──────────────────────────────────────────────┐\n");
    printf("   │ Kernel │ Sectors/Request │ Efficiency         │\n");
    printf("   │ A      │ ~8             │ ~50%%  ← 浪费一半!  │\n");
    printf("   │ B      │ ~4             │ ~100%% ← 完美合并   │\n");
    printf("   │ C      │ ~4             │ ~100%% ← 完美合并   │\n");
    printf("   └──────────────────────────────────────────────┘\n");
    printf("   → Sectors/Request: 每次请求触发多少个 32B sector\n");
    printf("     完美合并: 32 线程 × 4B = 128B = 4 sectors\n");
    printf("     stride-2: 32 线程 × 4B 跨 256B = 8 sectors (多了一倍!)\n\n");

    printf("3. Warp Stall Reasons — 瓶颈在哪里?\n");
    printf("   ┌──────────────────────────────────────────────┐\n");
    printf("   │ 指标                  │ 含义                  │\n");
    printf("   │ Stall Long Scoreboard │ 等全局内存 (最常见)   │\n");
    printf("   │ Stall MIO Throttle    │ 内存指令队列满        │\n");
    printf("   │ Stall Not Selected    │ 就绪但没被选 (好事!) │\n");
    printf("   └──────────────────────────────────────────────┘\n");
    printf("   → A 的 Stall Long Scoreboard 应该最高 (等内存最多)\n");
    printf("   → C 的 Stall Not Selected 应该最高 (线程够多, 延迟隐藏好)\n\n");

    printf("4. 指令统计\n");
    printf("   B: 每元素 1 条 LDG.32 (32-bit load)\n");
    printf("   C: 每 4 元素 1 条 LDG.128 (128-bit load) → 指令数减少 4×\n");
    printf("   → C 的 LD/ST pipe 压力更小 → 更多空间给其他操作\n\n");

    printf("尝试自己跑:\n");
    printf("  ncu --set full -o my_report ./ncu_demo\n");
    printf("  ncu-ui my_report.ncu-rep\n");
    printf("  或: ncu --page raw ./ncu_demo  (命令行文本输出)\n");

    cudaFree(d_in); cudaFree(d_out); free(h_in);
    return 0;
}
