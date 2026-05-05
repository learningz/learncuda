// ============================================================
// 练习 1: 双 Stream 流水线 — 传输和计算重叠
// 难度: Level 2 (kernel + host 端都要自己写)
//
// 编译: nvcc -O2 -o ex1_dual_stream_level2 ex1_dual_stream_level2.cu
// 运行: ./ex1_dual_stream_level2
// 预期输出: 双 Stream 比单 Stream 快
//
// 要求:
//   1. 实现 heavy_compute kernel (对 data 做多次 sin/cos 计算)
//   2. 在 main 中分别实现单 Stream 和双 Stream 流水线
//   3. 对比耗时, 验证加速比
//
// 数据分两块: chunk0 (前半) 和 chunk1 (后半)
// 双 Stream 流水线:
//   Stream 0: [H2D_0] → [compute_0] → [D2H_0]
//   Stream 1:          [H2D_1] → [compute_1] → [D2H_1]
//
// 关键:
//   - 用 cudaMallocHost 分配 Pinned Memory (DMA 直传)
//   - 用 cudaMemcpyAsync 异步传输 (指定 stream)
//   - kernel<<<grid, block, 0, stream>>>(args) 指定 stream
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
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

// TODO: 实现计算 kernel — 对 data 做 iter 次 sin*cos 运算
__global__ void heavy_compute(float *data, int n, int iter) {
    // --- 在这里写你的代码 ---

}

int main() {
    const int N = 4 * 1024 * 1024;
    const int iter = 500;
    const int chunk_size = N / 2;

    // ============================================================
    // TODO: 完成以下步骤
    //   1. cudaMallocHost 分配 pinned CPU 内存 (h_pinned)
    //   2. 初始化 h_pinned
    //   3. cudaMalloc GPU 内存 (d_data)
    //   4. 创建 2 个 cudaStream, 2 个 cudaEvent
    //
    //   5. 单 Stream 基准:
    //      cudaMemcpy H2D (全部) → kernel → cudaMemcpy D2H (全部)
    //      cudaEvent 计时
    //
    //   6. 双 Stream 流水线:
    //      for i in 0,1:
    //        cudaMemcpyAsync(h_pinned+i*chunk → d_data+i*chunk, chunk_size, H2D, stream[i])
    //        kernel<<<(chunk_size+255)/256, 256, 0, stream[i]>>>(d_data+i*chunk, chunk_size, iter)
    //        cudaMemcpyAsync(d_data+i*chunk → h_pinned+i*chunk, chunk_size, D2H, stream[i])
    //      cudaDeviceSynchronize(); 等所有 stream 完成
    //      cudaEvent 计时
    //
    //   7. 打印单/双 Stream 耗时和加速比
    //   8. cudaStreamDestroy + cudaEventDestroy + cudaFreeHost + cudaFree
    // ============================================================

    // --- 在这里写你的代码 ---

    return 0;
}
