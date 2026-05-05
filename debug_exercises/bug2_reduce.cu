#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ============================================================
// 调试练习 2: 并行归约 — 结果不稳定, 每次跑可能不一样
//
// 这个程序对 N 个数求和。CPU 参考值是确定的,
// 但 GPU 结果每次运行可能不同, 且和参考值有较大偏差。
//
// 你的任务: 找到 bug 并修复它, 让 GPU 结果和 CPU 一致。
//
// 提示: 数据竞争 (Race Condition) 是结果不稳定的典型原因。
//       想想 Shared Memory 的读写顺序。
// ============================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define BLOCK_SIZE 256

__global__ void buggy_reduce(const float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? input[idx] : 0.0f;

    // ====== BUG IS HERE ======
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
    }
    // =========================

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

float cpu_reduce(const float *data, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) sum += data[i];
    return (float)sum;
}

int main() {
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float *h_data = (float *)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; i++)
        h_data[i] = (float)(rand() % 100) / 100.0f;

    float ref = cpu_reduce(h_data, N);

    float *d_input, *d_partial;
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_partial, gridSize * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_data, bytes, cudaMemcpyHostToDevice));

    float *h_partial = (float *)malloc(gridSize * sizeof(float));

    printf("运行 5 次, 观察结果是否稳定:\n");
    for (int run = 0; run < 5; run++) {
        buggy_reduce<<<gridSize, BLOCK_SIZE>>>(d_input, d_partial, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_partial, d_partial, gridSize * sizeof(float),
                              cudaMemcpyDeviceToHost));
        float gpu_sum = cpu_reduce(h_partial, gridSize);
        float err = fabsf(gpu_sum - ref) / fmaxf(fabsf(ref), 1e-7f);
        printf("  第 %d 次: GPU=%.2f  CPU=%.2f  相对误差=%.1e %s\n",
               run + 1, gpu_sum, ref, err, err < 1e-3 ? "✓" : "✗");
    }

    cudaFree(d_input); cudaFree(d_partial);
    free(h_data); free(h_partial);
    return 0;
}

// ============================================================
// BUG ANSWER (先自己找!):
//
// 第 36-40 行: 归约循环中缺少 __syncthreads()!
//
//   for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
//       if (tid < stride) {
//           sdata[tid] += sdata[tid + stride];
//       }
//       // 这里缺少 __syncthreads();
//   }
//
//   后果: 假设 stride=128, 线程 0 正在读 sdata[128]。
//   但线程 128 可能还没完成上一轮对 sdata[128] 的写入!
//   → 线程 0 读到旧值或新值取决于执行顺序 → 结果不确定。
//
//   更隐蔽的是: 在某些 GPU 上, 由于硬件调度顺序碰巧一致,
//   bug 可能不出现。换一块 GPU 或数据量变大就会暴露。
//
// 修复: 在 if 块之后、循环体末尾添加 __syncthreads();
//
//   for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
//       if (tid < stride) {
//           sdata[tid] += sdata[tid + stride];
//       }
//       __syncthreads();  // 确保所有线程写完再进入下一轮!
//   }
// ============================================================
