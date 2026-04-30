// ============================================================
// 练习 1: 双 Stream 流水线 — 传输和计算重叠
// 难度: Level 1 (只需填写 host 端的 Stream 操作)
//
// 编译: nvcc -O2 -o ex1_dual_stream_level1 ex1_dual_stream_level1.cu
// 运行: ./ex1_dual_stream_level1
// 预期输出: 双 Stream 比单 Stream 快
//
// streams.cu 展示了自动分 4/8 个 Stream 的完整实验。
// 本题简化: 只用 2 个 Stream, 手动写出流水线逻辑。
//
// 数据分成两半: chunk0 和 chunk1
//
// 单 Stream (串行):
//   [H2D_all] → [compute_all] → [D2H_all]
//
// 双 Stream (重叠):
//   Stream 0: [H2D_0] → [compute_0] → [D2H_0]
//   Stream 1:          [H2D_1] → [compute_1] → [D2H_1]
//   → H2D_1 和 compute_0 重叠! compute_1 和 D2H_0 重叠!
//
// TODO: 在标记 TODO 的地方填写 cudaMemcpyAsync + kernel launch 代码
//
// 提示:
//   - cudaMemcpyAsync(dst, src, bytes, direction, stream)
//   - kernel<<<grid, block, 0, stream>>>(args)
//   - 两个 Stream 已经创建好了: streams[0] 和 streams[1]
//   - h_pinned 是 pinned memory (已分配)
//   - chunk_size = N / 2, chunk_bytes = chunk_size * sizeof(float)
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

__global__ void heavy_compute(float *data, int n, int iter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < iter; i++)
            val = sinf(val) * cosf(val) + 0.001f;
        data[idx] = val;
    }
}

int main() {
    const int N = 4 * 1024 * 1024;
    const int iter = 500;
    size_t bytes = N * sizeof(float);
    int chunk_size = N / 2;
    size_t chunk_bytes = chunk_size * sizeof(float);

    float *h_pinned;
    CUDA_CHECK(cudaMallocHost(&h_pinned, bytes));
    for (int i = 0; i < N; i++) h_pinned[i] = (float)(i % 1000) / 1000.0f;

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    cudaStream_t streams[2];
    CUDA_CHECK(cudaStreamCreate(&streams[0]));
    CUDA_CHECK(cudaStreamCreate(&streams[1]));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ---- 单 Stream 基准 ----
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_data, h_pinned, bytes, cudaMemcpyHostToDevice));
    heavy_compute<<<(N+255)/256, 256>>>(d_data, N, iter);
    CUDA_CHECK(cudaMemcpy(h_pinned, d_data, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_single;
    CUDA_CHECK(cudaEventElapsedTime(&ms_single, start, stop));

    // ---- 双 Stream ----
    CUDA_CHECK(cudaEventRecord(start));

    // ============================================================
    // TODO: 对每个 chunk (i=0 和 i=1):
    //   1. cudaMemcpyAsync: h_pinned + offset → d_data + offset, chunk_bytes, H2D, streams[i]
    //   2. heavy_compute<<<grid, 256, 0, streams[i]>>>(d_data + offset, chunk_size, iter)
    //   3. cudaMemcpyAsync: d_data + offset → h_pinned + offset, chunk_bytes, D2H, streams[i]
    //
    // offset = i * chunk_size
    // grid = (chunk_size + 255) / 256
    // ============================================================

    for (int i = 0; i < 2; i++) {
        int offset = i * chunk_size;
        int grid = (chunk_size + 255) / 256;
        (void)offset; (void)grid;

        // --- 在这里写你的代码 (3 行) ---

    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_dual;
    CUDA_CHECK(cudaEventElapsedTime(&ms_dual, start, stop));

    printf("单 Stream: %.2f ms\n", ms_single);
    printf("双 Stream: %.2f ms\n", ms_dual);
    printf("加速比: %.2fx\n", ms_single / ms_dual);

    CUDA_CHECK(cudaStreamDestroy(streams[0]));
    CUDA_CHECK(cudaStreamDestroy(streams[1]));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFreeHost(h_pinned));
    CUDA_CHECK(cudaFree(d_data));
    return 0;
}
