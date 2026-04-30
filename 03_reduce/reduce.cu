#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ============================================================
// 03: 并行归约 (Reduce) — 7 版演进，从朴素到极致
//
// 配合理论: theory/05_operator_development.md 5.3 节 (Reduce 7 版演进)
//           theory/04_warp_and_sync.md 4.2 节 (Warp Shuffle)
//
// 归约 (Reduce): 把 N 个数汇总成 1 个值。
//   例: sum([3, 1, 4, 1, 5]) = 14
//   CPU 做法: 一个 for 循环顺序累加 → O(N) 时间, 0 并行度。
//   GPU 做法: 树形归约 → O(log N) 轮, 每轮大量并行。
//
// 7 个版本 (V0-V6) 展示了 GPU 归约优化的完整路径:
//
//   V0: 朴素交错归约 — 有 Warp Divergence
//   V1: 连续线程归约 — 消除 Divergence
//   V2: Grid-Stride Loop — 每线程处理多元素，减少 Block 数
//   V3: Warp Shuffle — 消除大部分 __syncthreads
//   V4: Warp Shuffle + 完全展开最后 Warp
//   V5: float4 向量化加载 — 减少指令数
//   V6: atomicAdd 单 kernel — 无需 CPU 做最终汇总
//
// 每个版本的性能提升来源和硬件原理都在注释中详细解释。
// ============================================================

#define CUDA_CHECK(call) do {                                              \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                cudaGetErrorString(err)); exit(1); }                       \
} while(0)

#define BLOCK_SIZE 256

// ============================================================
// V0: 朴素交错归约 — 最直观但有 Warp Divergence
//
// 问题: stride 从 1 开始，每轮只有偶数号线程工作
//   第 1 轮: stride=1, tid 0,2,4,6... 工作 → 同一 Warp 内奇偶线程走不同分支!
//   → Warp Divergence: 两条路径串行执行 → 性能减半
//
// 另一个问题: 非连续线程访问 Shared Memory
//   tid 0 读 sdata[0] 和 sdata[1]
//   tid 2 读 sdata[2] 和 sdata[3]  ← 间隔的! 可能有 Bank Conflict
// ============================================================
__global__ void reduce_v0(const float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // 交错归约: stride 从 1 到 blockDim.x/2
    // 第 1 轮: sdata[0] += sdata[1], sdata[2] += sdata[3], ...
    // 第 2 轮: sdata[0] += sdata[2], sdata[4] += sdata[6], ...
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        // 问题: 只有 tid % (2*stride) == 0 的线程工作
        // → 同一 Warp 内一半线程空闲 → Divergence!
        if (tid % (2 * stride) == 0) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// ============================================================
// V1: 连续线程归约 — 消除 Warp Divergence
//
// 改进: stride 从 blockDim.x/2 开始递减
//   第 1 轮: stride=128, 线程 0-127 工作，加 sdata[tid+128]
//   第 2 轮: stride=64,  线程 0-63 工作
//   ...
//
// 为什么没有 Divergence?
//   第 1 轮: Warp 0-3 (线程 0-127) 全部工作 → 无分歧
//            Warp 4-7 (线程 128-255) 全部不工作 → 也无分歧
//   → 分歧只可能发生在 stride < 32 时（最后几轮）
// ============================================================
__global__ void reduce_v1(const float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// ============================================================
// V2: Grid-Stride Loop — 每线程处理多个元素，减少 Block 数
//   gridSize 从 16384 降到 256 → 调度开销减少
//   局部累加在寄存器中 → 零同步开销
// ============================================================
__global__ void reduce_v2(const float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float local_sum = 0.0f;
    for (int i = idx; i < n; i += stride) local_sum += input[i];
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// ---- Warp Shuffle 辅助: 32 线程寄存器直接交换求和 ----
__device__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

// ============================================================
// V3: Warp Shuffle — 只需 1 次 __syncthreads (V1 需要 8 次)
// ============================================================
__global__ void reduce_v3(const float *input, float *output, int n) {
    __shared__ float warp_sums[BLOCK_SIZE / 32];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float local_sum = 0.0f;
    for (int i = idx; i < n; i += stride) local_sum += input[i];
    local_sum = warp_reduce_sum(local_sum);
    int warp_id = tid / 32, lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0) {
        local_sum = (tid < (BLOCK_SIZE / 32)) ? warp_sums[tid] : 0.0f;
        local_sum = warp_reduce_sum(local_sum);
        if (tid == 0) output[blockIdx.x] = local_sum;
    }
}

// ============================================================
// V4: Warp Shuffle + 4 路循环展开 (ILP 隐藏内存延迟)
//   4 条 LDG 背靠背发射 → 等待期间其他加载已在路上
// ============================================================
__global__ void reduce_v4(const float *input, float *output, int n) {
    __shared__ float warp_sums[BLOCK_SIZE / 32];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float local_sum = 0.0f;
    int i = idx;
    for (; i + 3 * stride < n; i += 4 * stride) {
        local_sum += input[i] + input[i + stride]
                   + input[i + 2 * stride] + input[i + 3 * stride];
    }
    for (; i < n; i += stride) local_sum += input[i];
    local_sum = warp_reduce_sum(local_sum);
    int warp_id = tid / 32, lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0) {
        local_sum = (tid < (BLOCK_SIZE / 32)) ? warp_sums[tid] : 0.0f;
        local_sum = warp_reduce_sum(local_sum);
        if (tid == 0) output[blockIdx.x] = local_sum;
    }
}

// ============================================================
// V5: float4 向量化加载 — 1 条 LDG.128 替代 4 条 LDG.32
// ============================================================
__global__ void reduce_v5(const float *input, float *output, int n) {
    __shared__ float warp_sums[BLOCK_SIZE / 32];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float local_sum = 0.0f;
    int n4 = n / 4;
    const float4 *in4 = reinterpret_cast<const float4 *>(input);
    for (int i = idx; i < n4; i += stride) {
        float4 v = in4[i];
        local_sum += v.x + v.y + v.z + v.w;
    }
    for (int i = n4 * 4 + tid; i < n; i += blockDim.x) local_sum += input[i];
    local_sum = warp_reduce_sum(local_sum);
    int warp_id = tid / 32, lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0) {
        local_sum = (tid < (BLOCK_SIZE / 32)) ? warp_sums[tid] : 0.0f;
        local_sum = warp_reduce_sum(local_sum);
        if (tid == 0) output[blockIdx.x] = local_sum;
    }
}

// ============================================================
// V6: atomicAdd 单 kernel — 无需第二阶段汇总
//   输出必须预先 cudaMemset 为 0!
// ============================================================
__global__ void reduce_v6(const float *input, float *output, int n) {
    __shared__ float warp_sums[BLOCK_SIZE / 32];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float local_sum = 0.0f;
    int n4 = n / 4;
    const float4 *in4 = reinterpret_cast<const float4 *>(input);
    for (int i = idx; i < n4; i += stride) {
        float4 v = in4[i];
        local_sum += v.x + v.y + v.z + v.w;
    }
    for (int i = n4 * 4 + tid; i < n; i += blockDim.x) local_sum += input[i];
    local_sum = warp_reduce_sum(local_sum);
    int warp_id = tid / 32, lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0) {
        local_sum = (tid < (BLOCK_SIZE / 32)) ? warp_sums[tid] : 0.0f;
        local_sum = warp_reduce_sum(local_sum);
        if (tid == 0) atomicAdd(output, local_sum);
    }
}

// ---- CPU 参考 ----
float cpu_reduce(const float *data, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) sum += data[i];
    return (float)sum;
}

int main() {
    const int N = 1 << 22;
    size_t bytes = N * sizeof(float);
    float *h_data = (float *)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; i++) h_data[i] = (float)(rand() % 100) / 100.0f;
    float ref = cpu_reduce(h_data, N);

    float *d_input, *d_partial, *d_single;
    int gridFull = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int gridSmall = 256;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_partial, gridFull * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_single, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_data, bytes, cudaMemcpyHostToDevice));
    float *h_partial = (float *)malloc(gridFull * sizeof(float));

    printf("并行归约 7 版演进: N = %d → 求总和 (ref=%.2f)\n\n", N, ref);

    auto bench = [&](const char *name, int grid, auto launch_fn, bool atomic) {
        if (!atomic) {
            launch_fn(grid);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_partial, d_partial, grid * sizeof(float), cudaMemcpyDeviceToHost));
            float s = cpu_reduce(h_partial, grid);
            float e = fabsf(s - ref) / fmaxf(fabsf(ref), 1e-7f);
            printf("%-33s sum=%.2f err=%.1e %s", name, s, e, e<1e-3?"✓":"✗");
        } else {
            CUDA_CHECK(cudaMemset(d_single, 0, sizeof(float)));
            launch_fn(grid);
            CUDA_CHECK(cudaDeviceSynchronize());
            float s; CUDA_CHECK(cudaMemcpy(&s, d_single, sizeof(float), cudaMemcpyDeviceToHost));
            float e = fabsf(s - ref) / fmaxf(fabsf(ref), 1e-7f);
            printf("%-33s sum=%.2f err=%.1e %s", name, s, e, e<1e-3?"✓":"✗");
        }
        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0));
        for (int r = 0; r < 100; r++) {
            if (atomic) CUDA_CHECK(cudaMemset(d_single, 0, sizeof(float)));
            launch_fn(grid);
        }
        CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1)); ms /= 100;
        printf("  %.3f ms  %.0f GB/s\n", ms, bytes / (ms * 1e6));
        CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    };

    bench("V0: 朴素交错 (Divergence!)", gridFull,
        [&](int g){ reduce_v0<<<g, BLOCK_SIZE>>>(d_input, d_partial, N); }, false);
    bench("V1: 连续线程 (无Divergence)", gridFull,
        [&](int g){ reduce_v1<<<g, BLOCK_SIZE>>>(d_input, d_partial, N); }, false);
    bench("V2: Grid-Stride Loop", gridSmall,
        [&](int g){ reduce_v2<<<g, BLOCK_SIZE>>>(d_input, d_partial, N); }, false);
    bench("V3: Warp Shuffle", gridSmall,
        [&](int g){ reduce_v3<<<g, BLOCK_SIZE>>>(d_input, d_partial, N); }, false);
    bench("V4: Shuffle+4x展开(ILP)", gridSmall,
        [&](int g){ reduce_v4<<<g, BLOCK_SIZE>>>(d_input, d_partial, N); }, false);
    bench("V5: float4 向量化", gridSmall,
        [&](int g){ reduce_v5<<<g, BLOCK_SIZE>>>(d_input, d_partial, N); }, false);
    bench("V6: atomicAdd 单kernel", gridSmall,
        [&](int g){ reduce_v6<<<g, BLOCK_SIZE>>>(d_input, d_single, N); }, true);

    printf("\n优化路径:\n");
    printf("  V0→V1: 消除 Warp Divergence\n");
    printf("  V1→V2: Grid-Stride Loop (每线程多元素)\n");
    printf("  V2→V3: Warp Shuffle (零同步归约)\n");
    printf("  V3→V4: 循环展开 (ILP 隐藏延迟)\n");
    printf("  V4→V5: float4 (减少指令数)\n");
    printf("  V5→V6: atomicAdd (单 kernel)\n");

    cudaFree(d_input); cudaFree(d_partial); cudaFree(d_single);
    free(h_data); free(h_partial);
    return 0;
}
