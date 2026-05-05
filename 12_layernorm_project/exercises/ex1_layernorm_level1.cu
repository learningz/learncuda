// ============================================================
// 练习 1: 单行 LayerNorm 前向 — mean+rstd 归约
// 难度: Level 1 (只需填写 kernel 函数体)
//
// 编译: nvcc -O2 -o ex1_layernorm_level1 ex1_layernorm_level1.cu
// 运行: ./ex1_layernorm_level1
// 预期输出: "结果: ✓ 全部正确"
//
// 12_layernorm_project 的 starter code 做的是完整的 [rows, cols] LayerNorm。
// 本题大幅简化: 只处理一行, 任务是算出这行的 mean 和 rstd。
//
// 步骤 (参考 03_reduce 的 Warp+SMEM 归约):
//   1. 每个线程把自己负责的元素加起来 (共享 sum)
//   2. Warp 内 Shuffle 归约 (32→1)
//   3. 每 Warp 的 lane 0 写入 SMEM[warp_id]
//   4. __syncthreads()
//   5. warp 0 从 SMEM 读出, 再 Shuffle 归约一次 → 全 Block 的 sum
//   6. 同样算 sum_sq (平方和), 得到 mean = sum/N, rstd = 1/sqrt(var+eps)
//
// 参数: N=256, blockDim=256, 所有线程在同一个 Block 内
//
// 提示:
//   - 用 __shfl_down_sync(0xffffffff, val, offset) 做 Warp 归约
//   - warp_id = threadIdx.x / 32, lane = threadIdx.x % 32
//   - num_warps = blockDim.x / 32
//   - 归约结果: 每个线程输出 mean 和 rstd (全部相同)
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

// TODO: 实现单行 LayerNorm 的统计量计算
// 输入: x[N], 输出: d_mean[N] 和 d_rstd[N] (每线程写自己的元素, 实际值都相同)
__global__ void layernorm_stats_kernel(const float *x, float *mean, float *rstd,
                                        int N, float eps) {
    // --- 在这里写你的代码 ---
    // 1. 计算 sum 和 sum_sq (每个线程贡献一个元素)
    // 2. Warp 内 Shuffle 归约
    // 3. Warp 间 Shared Memory 归约
    // 4. 计算 mean = sum/N, var = sum_sq/N - mean*mean, rstd = rsqrtf(var+eps)
    // 5. 每个线程把结果写到自己的位置

}

int main() {
    const int N = 256;
    const float eps = 1e-5f;
    const size_t bytes = N * sizeof(float);

    float *h_x = (float *)malloc(bytes);
    float *h_mean = (float *)malloc(bytes);
    float *h_rstd = (float *)malloc(bytes);

    srand(42);
    float sum = 0, sum_sq = 0;
    for (int i = 0; i < N; i++) {
        h_x[i] = (float)(rand() % 200 - 100) / 20.0f;
        sum += h_x[i];
        sum_sq += h_x[i] * h_x[i];
    }
    float ref_mean = sum / N;
    float ref_var = sum_sq / N - ref_mean * ref_mean;
    float ref_rstd = 1.0f / sqrtf(ref_var + eps);

    float *d_x, *d_mean, *d_rstd;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_mean, bytes));
    CUDA_CHECK(cudaMalloc(&d_rstd, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

    layernorm_stats_kernel<<<1, N>>>(d_x, d_mean, d_rstd, N, eps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_mean, d_mean, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rstd, d_rstd, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_mean[i] - ref_mean) > 1e-3f) {
            printf("mean @ %d: got %f, expected %f\n", i, h_mean[i], ref_mean);
            ok = false; break;
        }
        if (fabsf(h_rstd[i] - ref_rstd) > 1e-3f) {
            printf("rstd @ %d: got %f, expected %f\n", i, h_rstd[i], ref_rstd);
            ok = false; break;
        }
    }
    printf("ref mean=%.4f rstd=%.4f\n", ref_mean, ref_rstd);
    printf("结果: %s\n", ok ? "✓ 全部正确" : "✗ 有错误");

    cudaFree(d_x); cudaFree(d_mean); cudaFree(d_rstd);
    free(h_x); free(h_mean); free(h_rstd);
    return 0;
}
