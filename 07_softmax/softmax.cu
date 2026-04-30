#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ============================================================
// 实验: Softmax 的 3 个版本 — 从朴素到极致
//
// 配合理论: theory/07_classic_operators.md 7.1 节
//
// 数学: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
//
// 三个版本:
//   V1 (3-pass): 先求 max → 再求 sum(exp) → 最后归一化。3 次遍历数据。
//   V2 (2-pass Online): 一次遍历同时求 max 和 sum → 一次遍历归一化。
//   V3 (Warp-level): 当一行 ≤ 32 个元素时, 一个 Warp 处理一行, 全用 Shuffle。
//
// 核心概念:
//   - 数值稳定性: 为什么要减 max? 因为 exp(100) 会溢出!
//   - Online 算法: 不需要先看完所有数据就能开始计算
//   - Warp Shuffle: 同一 Warp 的线程直接交换寄存器值 (不经过 Shared Memory)
//   - Memory Bound: Softmax 的算术强度很低 → 减少内存遍历次数 = 直接提速
//
//   常见错误 (写 Softmax 时特别容易踩的坑):
//      ✗ 忘了减 max → exp 溢出 → 输出全是 NaN/Inf
//      ✗ 在 block 内做 reduce 时忘了 __syncthreads → max/sum 不对 → 结果随机
//      ✗ Online 算法的修正因子写反 → exp(new-old) 应该是 exp(old-new)!
//      ✗ 只对 local 变量做了 Warp Reduce, 忘了汇总多个 Warp 的结果
//      ✗ 用 expf 而非 __expf → 精度更高但慢 (~对 ML 场景通常用 expf 就够)
// ============================================================

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                               \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

// ---- V1: 3-pass (朴素但正确) ----
// Pass 1: 求每行最大值 (为了数值稳定性)
// Pass 2: 求 sum(exp(x - max))
// Pass 3: 归一化 y = exp(x - max) / sum
__global__ void softmax_3pass(const float *input, float *output,
                               int rows, int cols) {
    // 每个 Block 处理一行
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float *x = input + row * cols;
    float *y = output + row * cols;

    // Pass 1: 求 max
    float local_max = -INFINITY;
    for (int i = tid; i < cols; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }
    smem[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float max_val = smem[0];
    __syncthreads();

    // Pass 2: 求 sum(exp(x - max))
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        local_sum += expf(x[i] - max_val);
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float sum_val = smem[0];
    __syncthreads();

    // Pass 3: 归一化
    for (int i = tid; i < cols; i += blockDim.x) {
        y[i] = expf(x[i] - max_val) / sum_val;
    }
}

// ---- V2: 2-pass Online Softmax ----
// Pass 1: 一次遍历, 同时追踪 running max 和 running sum
//   关键公式: 当 max 更新时, 之前的 sum 需要乘以修正因子 exp(old_max - new_max)
// Pass 2: 归一化
__global__ void softmax_online(const float *input, float *output,
                                int rows, int cols) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float *x = input + row * cols;
    float *y = output + row * cols;

    // Pass 1: Online 同时计算 max 和 sum
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = x[i];
        float old_max = local_max;
        local_max = fmaxf(local_max, val);
        // 修正之前的 sum: 因为 max 变了, 之前算的 exp(x-old_max) 要变成 exp(x-new_max)
        // exp(x-old_max) × exp(old_max-new_max) = exp(x-new_max) ✓
        local_sum = local_sum * expf(old_max - local_max) + expf(val - local_max);
    }

    // Block 级 Online Reduce (需要同时 reduce max 和 sum)
    smem[tid] = local_max;
    smem[tid + blockDim.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float m1 = smem[tid], d1 = smem[tid + blockDim.x];
            float m2 = smem[tid + s], d2 = smem[tid + s + blockDim.x];
            float new_max = fmaxf(m1, m2);
            smem[tid] = new_max;
            smem[tid + blockDim.x] =
                d1 * expf(m1 - new_max) + d2 * expf(m2 - new_max);
        }
        __syncthreads();
    }
    float max_val = smem[0];
    float sum_val = smem[blockDim.x];
    __syncthreads();

    // Pass 2: 归一化
    for (int i = tid; i < cols; i += blockDim.x) {
        y[i] = expf(x[i] - max_val) / sum_val;
    }
}

// ---- V3: Warp-level Softmax (适用于 cols ≤ 32) ----
// 一个 Warp 处理一行, 完全用 Shuffle, 无 Shared Memory!
__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return __shfl_sync(0xffffffff, val, 0);
}

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return __shfl_sync(0xffffffff, val, 0);
}

__global__ void softmax_warp(const float *input, float *output,
                              int rows, int cols) {
    // 每个 Warp 处理一行
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= rows) return;

    const float *x = input + warp_id * cols;
    float *y = output + warp_id * cols;

    float val = (lane < cols) ? x[lane] : -INFINITY;
    float m = warp_reduce_max(val);
    float e = (lane < cols) ? expf(val - m) : 0.0f;
    float s = warp_reduce_sum(e);
    if (lane < cols) y[lane] = e / s;
}

// ---- CPU 参考实现 ----
void softmax_cpu(const float *input, float *output, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float *x = input + r * cols;
        float *y = output + r * cols;
        float m = -INFINITY;
        for (int i = 0; i < cols; i++) m = fmaxf(m, x[i]);
        float s = 0;
        for (int i = 0; i < cols; i++) s += expf(x[i] - m);
        for (int i = 0; i < cols; i++) y[i] = expf(x[i] - m) / s;
    }
}

bool verify(const float *ref, const float *test, int n, float tol = 1e-5f) {
    for (int i = 0; i < n; i++) {
        if (fabsf(ref[i] - test[i]) > tol) {
            printf("  不匹配 @ %d: ref=%.6f test=%.6f\n", i, ref[i], test[i]);
            return false;
        }
    }
    return true;
}

int main() {
    // 测试两种场景: 长行 (cols=4096) 和短行 (cols=32)
    printf("========== 场景 1: 长行 (rows=1024, cols=4096) ==========\n");
    {
        int rows = 1024, cols = 4096;
        size_t bytes = rows * cols * sizeof(float);
        float *h_in = (float*)malloc(bytes);
        float *h_ref = (float*)malloc(bytes);
        float *h_out = (float*)malloc(bytes);
        srand(42);
        for (int i = 0; i < rows * cols; i++)
            h_in[i] = (float)(rand() % 200 - 100) / 10.0f;

        softmax_cpu(h_in, h_ref, rows, cols);

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, bytes));
        CUDA_CHECK(cudaMalloc(&d_out, bytes));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

        int blockSize = 256;
        size_t smem = 2 * blockSize * sizeof(float);

        #define BENCH(name, ...) {                                         \
            __VA_ARGS__; cudaDeviceSynchronize();                           \
            CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost)); \
            bool ok = verify(h_ref, h_out, rows * cols);                   \
            cudaEvent_t start, stop;                                       \
            cudaEventCreate(&start); cudaEventCreate(&stop);               \
            cudaEventRecord(start);                                        \
            for (int i = 0; i < 100; i++) { __VA_ARGS__; }                \
            cudaEventRecord(stop); cudaEventSynchronize(stop);             \
            float ms; cudaEventElapsedTime(&ms, start, stop);              \
            printf("  %-20s: %.3f ms  %s\n", name, ms/100, ok?"✓":"✗"); \
            cudaEventDestroy(start); cudaEventDestroy(stop);               \
        }

        BENCH("V1 (3-pass)", softmax_3pass<<<rows, blockSize, smem>>>(d_in, d_out, rows, cols));
        BENCH("V2 (online)", softmax_online<<<rows, blockSize, smem>>>(d_in, d_out, rows, cols));

        printf("  (V3 Warp-level 不适用于 cols=4096, 跳过)\n");

        cudaFree(d_in); cudaFree(d_out);
        free(h_in); free(h_ref); free(h_out);
    }

    printf("\n========== 场景 2: 短行 (rows=32768, cols=32) ==========\n");
    {
        int rows = 32768, cols = 32;
        size_t bytes = rows * cols * sizeof(float);
        float *h_in = (float*)malloc(bytes);
        float *h_ref = (float*)malloc(bytes);
        float *h_out = (float*)malloc(bytes);
        srand(42);
        for (int i = 0; i < rows * cols; i++)
            h_in[i] = (float)(rand() % 200 - 100) / 10.0f;

        softmax_cpu(h_in, h_ref, rows, cols);

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, bytes));
        CUDA_CHECK(cudaMalloc(&d_out, bytes));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

        int blockSize = 256;
        size_t smem = 2 * blockSize * sizeof(float);

        BENCH("V1 (3-pass)", softmax_3pass<<<rows, blockSize, smem>>>(d_in, d_out, rows, cols));
        BENCH("V2 (online)", softmax_online<<<rows, blockSize, smem>>>(d_in, d_out, rows, cols));
        int warps_per_block = 8;
        int grid = (rows + warps_per_block - 1) / warps_per_block;
        BENCH("V3 (warp)", softmax_warp<<<grid, warps_per_block * 32>>>(d_in, d_out, rows, cols));

        #undef BENCH

        printf("\n  观察: V3 (Warp-level) 在短行场景下应该最快 — 零 Shared Memory, 零 __syncthreads!\n");

        cudaFree(d_in); cudaFree(d_out);
        free(h_in); free(h_ref); free(h_out);
    }

    printf("\n理论分析 (详见 theory/07_classic_operators.md 7.1 节):\n");
    printf("  V1→V2: 3次遍历→2次遍历 = 减少33%%内存访问 → 期望加速 ~1.3x\n");
    printf("  V2→V3: 消除 Shared Memory + __syncthreads → 短行场景大幅加速\n");

    return 0;
}
