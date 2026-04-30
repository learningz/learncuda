#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ============================================================
// 16: CUDA Stream 与异步执行
//
// 本示例展示 CUDA 异步编程的三个核心概念:
//
//   实验 1: 单 Stream vs 多 Stream
//           → 多 Stream 让计算和传输重叠，总时间 < 各部分之和
//
//   实验 2: Pinned Memory vs Pageable Memory
//           → Pinned Memory 通常是实现稳定异步传输和重叠的前提
//
//   实验 3: cudaEvent 精确计时
//           → 在 GPU 时间线上打点，比 CPU 计时更准确
//
// 配合理论: theory/02_cuda_programming_model.md 2.4-2.5 节
// ============================================================

// ---- 错误检查宏 ----
#define CUDA_CHECK(call) do {                                              \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                cudaGetErrorString(err)); exit(1); }                       \
} while(0)

// ---- 一个故意"慢"的 kernel，方便观察重叠效果 ----
// 每个线程做大量计算（循环 iter 次 sin/cos），模拟计算密集型工作
// 这样 kernel 执行时间足够长，能和数据传输产生可观察的重叠
__global__ void heavy_compute(float *data, int n, int iter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        // 做 iter 次三角函数运算，纯粹为了消耗 GPU 时间
        for (int i = 0; i < iter; i++) {
            val = sinf(val) * cosf(val) + 0.001f;
        }
        data[idx] = val;
    }
}

// ============================================================
// 实验 1: 单 Stream — 一切串行
//
// 默认的 CUDA 编程模型: 所有操作在 "默认 Stream" (stream 0) 上。
// 同一 Stream 内的操作严格按顺序执行:
//
//   时间线:  [--- H2D 传输 ---][--- kernel 计算 ---][--- D2H 传输 ---]
//   总时间 = T_copy_in + T_compute + T_copy_out
//
// 问题: 传输时 GPU 计算单元空闲，计算时 PCIe 总线空闲。
// ============================================================
float experiment_single_stream(float *h_data, float *d_data, int n, int iter) {
    size_t bytes = n * sizeof(float);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    // 全部在默认 stream 上: 传入 → 计算 → 传出，严格串行
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    heavy_compute<<<(n + 255) / 256, 256>>>(d_data, n, iter);
    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

// ============================================================
// 实验 2: 多 Stream + 分块传输 — 计算与传输重叠
//
// 核心思想: 把数据分成 num_streams 块，每块用独立的 Stream 处理。
// 不同 Stream 上的操作可以并行执行!
//
// 2 个 Stream 的时间线 (理想情况):
//   Stream 0: [H2D chunk0][compute chunk0][D2H chunk0]
//   Stream 1:             [H2D chunk1    ][compute chunk1][D2H chunk1]
//                         ↑ 重叠! H2D_1 和 compute_0 同时进行
//
// 4 个 Stream 的时间线:
//   Stream 0: [H2D_0][compute_0        ][D2H_0]
//   Stream 1:        [H2D_1][compute_1        ][D2H_1]
//   Stream 2:               [H2D_2][compute_2        ][D2H_2]
//   Stream 3:                      [H2D_3][compute_3        ][D2H_3]
//                    ↑ 大量重叠! 传输和计算交错进行
//
// 前提条件:
//   1. Host 内存最好是 Pinned Memory (cudaMallocHost)
//      → 这样 cudaMemcpyAsync 才能稳定实现真正异步，并和 kernel 重叠
//   2. 使用 cudaMemcpyAsync 而不是 cudaMemcpy
//   3. 每个 Stream 的操作之间仍然是串行的 (同一 Stream 内保证顺序)
//
// 硬件基础:
//   GPU 有独立的硬件引擎:
//   - Copy Engine (H2D): 负责 Host → Device 传输
//   - Copy Engine (D2H): 负责 Device → Host 传输
//   - Compute Engine:     负责 kernel 执行
//   这三个引擎可以同时工作! 多 Stream 就是让它们同时忙起来。
// ============================================================
float experiment_multi_stream(float *h_pinned, float *d_data, int n, int iter,
                               int num_streams) {
    size_t bytes = n * sizeof(float);
    int chunk_size = n / num_streams;
    size_t chunk_bytes = chunk_size * sizeof(float);

    // 创建多个 Stream
    cudaStream_t *streams = new cudaStream_t[num_streams];
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_size;
        float *h_ptr = h_pinned + offset;
        float *d_ptr = d_data + offset;

        // 异步传输: 立即返回，不等传输完成
        // 使用 Pinned Memory 时，才最容易观察到稳定的异步和传输/计算重叠效果。
        CUDA_CHECK(cudaMemcpyAsync(d_ptr, h_ptr, chunk_bytes,
                                    cudaMemcpyHostToDevice, streams[i]));

        // 在同一 Stream 上启动 kernel
        // 保证: 这个 kernel 一定在上面的 H2D 传输完成后才开始
        //       (同一 Stream 内的操作是严格有序的)
        // 但: 它可以和其他 Stream 的 H2D 传输并行!
        heavy_compute<<<(chunk_size + 255) / 256, 256, 0, streams[i]>>>(
            d_ptr, chunk_size, iter);

        // 异步传出
        CUDA_CHECK(cudaMemcpyAsync(h_ptr, d_ptr, chunk_bytes,
                                    cudaMemcpyDeviceToHost, streams[i]));
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    for (int i = 0; i < num_streams; i++)
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    delete[] streams;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

// ============================================================
// 实验 3: Pinned Memory vs Pageable Memory 传输速度对比
//
// Pageable Memory (普通 malloc):
//   → OS 可以把这块内存交换到磁盘 (page out)
//   → CUDA 驱动在传输前必须先拷到一个内部的 pinned staging buffer
//   → 实际路径: malloc → staging buffer → GPU  (2 次拷贝!)
//   → 且无法和 kernel 重叠 (必须等 staging 拷贝完)
//
// Pinned Memory (cudaMallocHost / cudaHostAlloc):
//   → OS 保证这块内存不会被交换出去 (锁定在物理内存)
//   → GPU 可以通过 DMA 直接访问，无需 staging buffer
//   → 实际路径: pinned mem → GPU  (1 次拷贝, DMA 直传!)
//   → 可以和 kernel 异步执行
//
// 代价:
//   → Pinned Memory 会占用物理内存，不能被 OS 调度
//   → 分配太多会导致系统内存紧张
//   → 建议: 只对需要频繁传输的 buffer 使用 pinned memory
// ============================================================
void experiment_pinned_vs_pageable(int n) {
    size_t bytes = n * sizeof(float);

    // 分配两种内存
    float *h_pageable = (float *)malloc(bytes);
    float *h_pinned;
    CUDA_CHECK(cudaMallocHost(&h_pinned, bytes));

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    // 初始化
    for (int i = 0; i < n; i++) {
        h_pageable[i] = (float)i;
        h_pinned[i] = (float)i;
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int repeat = 20;

    // ---- Pageable Memory 传输 ----
    // 预热
    CUDA_CHECK(cudaMemcpy(d_data, h_pageable, bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++) {
        CUDA_CHECK(cudaMemcpy(d_data, h_pageable, bytes, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float pageable_ms;
    CUDA_CHECK(cudaEventElapsedTime(&pageable_ms, start, stop));
    pageable_ms /= repeat;
    float pageable_bw = bytes / (pageable_ms * 1e6);

    // ---- Pinned Memory 传输 ----
    CUDA_CHECK(cudaMemcpy(d_data, h_pinned, bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++) {
        CUDA_CHECK(cudaMemcpy(d_data, h_pinned, bytes, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float pinned_ms;
    CUDA_CHECK(cudaEventElapsedTime(&pinned_ms, start, stop));
    pinned_ms /= repeat;
    float pinned_bw = bytes / (pinned_ms * 1e6);

    printf("  Pageable Memory: %.2f ms, %.1f GB/s\n", pageable_ms, pageable_bw);
    printf("  Pinned Memory:   %.2f ms, %.1f GB/s\n", pinned_ms, pinned_bw);
    printf("  加速比: %.2fx\n", pageable_ms / pinned_ms);

    // 清理
    free(h_pageable);
    CUDA_CHECK(cudaFreeHost(h_pinned));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// ============================================================
int main() {
    // 查询 GPU 信息
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("CUDA Stream 与异步执行实验\n");
    printf("GPU: %s\n", prop.name);
    printf("异步引擎数: %d (>1 表示支持双向传输+计算同时进行)\n\n", prop.asyncEngineCount);

    // ============================================================
    // 参数说明:
    //   N = 4M 个 float = 16MB 数据
    //   iter = 500 次循环 → 每个线程做大量计算，让 kernel 运行足够久
    //   这样传输时间和计算时间在同一量级，重叠效果最明显
    // ============================================================
    int N = 4 * 1024 * 1024;  // 4M 元素
    int iter = 500;            // 每线程计算迭代次数
    size_t bytes = N * sizeof(float);

    // ---- 分配 Pinned Memory (多 Stream 实验的前提) ----
    float *h_pinned;
    CUDA_CHECK(cudaMallocHost(&h_pinned, bytes));
    for (int i = 0; i < N; i++) h_pinned[i] = (float)(i % 1000) / 1000.0f;

    // ---- 分配 Pageable Memory (单 Stream 实验用) ----
    float *h_pageable = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) h_pageable[i] = h_pinned[i];

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    // ============================================================
    printf("实验 1: 单 Stream vs 多 Stream\n");
    printf("  数据量: %d MB, 每线程 %d 次迭代\n", (int)(bytes / 1024 / 1024), iter);
    printf("──────────────────────────────\n");

    float t_single = experiment_single_stream(h_pageable, d_data, N, iter);
    printf("  单 Stream (串行):        %.2f ms\n", t_single);

    float t_2stream = experiment_multi_stream(h_pinned, d_data, N, iter, 2);
    printf("  2 Stream (重叠):         %.2f ms  (加速 %.2fx)\n",
           t_2stream, t_single / t_2stream);

    float t_4stream = experiment_multi_stream(h_pinned, d_data, N, iter, 4);
    printf("  4 Stream (更多重叠):     %.2f ms  (加速 %.2fx)\n",
           t_4stream, t_single / t_4stream);

    float t_8stream = experiment_multi_stream(h_pinned, d_data, N, iter, 8);
    printf("  8 Stream:                %.2f ms  (加速 %.2fx)\n",
           t_8stream, t_single / t_8stream);

    printf("\n  为什么加速有上限?\n");
    printf("  → 当 Stream 足够多时，传输已经完全被计算遮盖\n");
    printf("  → 总时间 ≈ max(T_compute, T_transfer) 而不是两者之和\n");
    printf("  → 继续加 Stream 不会更快 (传输引擎只有 %d 个)\n", prop.asyncEngineCount);

    // ============================================================
    printf("\n实验 2: Pinned Memory vs Pageable Memory\n");
    printf("  数据量: %d MB\n", (int)(bytes / 1024 / 1024));
    printf("──────────────────────────────\n");
    experiment_pinned_vs_pageable(N);

    printf("\n  为什么 Pinned Memory 更快?\n");
    printf("  → Pageable: CPU→staging buffer→GPU (2次拷贝)\n");
    printf("  → Pinned:   CPU→GPU (DMA直传, 1次拷贝)\n");
    printf("  → Pinned 还支持异步传输 (cudaMemcpyAsync)\n");

    // ============================================================
    printf("\n实验 3: 验证异步性 — cudaMemcpyAsync 的行为\n");
    printf("──────────────────────────────\n");

    cudaEvent_t ev_start, ev_h2d, ev_kernel, ev_d2h;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_h2d));
    CUDA_CHECK(cudaEventCreate(&ev_kernel));
    CUDA_CHECK(cudaEventCreate(&ev_d2h));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 在同一 stream 上依次提交三个操作，用 Event 精确测量每段时间
    CUDA_CHECK(cudaEventRecord(ev_start, stream));

    CUDA_CHECK(cudaMemcpyAsync(d_data, h_pinned, bytes,
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaEventRecord(ev_h2d, stream));

    heavy_compute<<<(N + 255) / 256, 256, 0, stream>>>(d_data, N, iter);
    CUDA_CHECK(cudaEventRecord(ev_kernel, stream));

    CUDA_CHECK(cudaMemcpyAsync(h_pinned, d_data, bytes,
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(ev_d2h, stream));

    CUDA_CHECK(cudaEventSynchronize(ev_d2h));

    float t_h2d, t_kern, t_d2h;
    CUDA_CHECK(cudaEventElapsedTime(&t_h2d, ev_start, ev_h2d));
    CUDA_CHECK(cudaEventElapsedTime(&t_kern, ev_h2d, ev_kernel));
    CUDA_CHECK(cudaEventElapsedTime(&t_d2h, ev_kernel, ev_d2h));

    printf("  H2D 传输:  %.2f ms\n", t_h2d);
    printf("  Kernel:    %.2f ms\n", t_kern);
    printf("  D2H 传输:  %.2f ms\n", t_d2h);
    printf("  总计:      %.2f ms (= %.2f + %.2f + %.2f, 串行时)\n",
           t_h2d + t_kern + t_d2h, t_h2d, t_kern, t_d2h);
    printf("  → 多 Stream 的加速来自让这三段重叠执行\n");

    // 清理
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_h2d));
    CUDA_CHECK(cudaEventDestroy(ev_kernel));
    CUDA_CHECK(cudaEventDestroy(ev_d2h));
    CUDA_CHECK(cudaFreeHost(h_pinned));
    free(h_pageable);
    CUDA_CHECK(cudaFree(d_data));

    return 0;
}
