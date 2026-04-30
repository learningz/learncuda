#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ============================================================
// 12: LayerNorm Starter Code — 填空完成一个完整算子
//
// 这个文件已经搭好了框架, 你需要填写标记了 TODO 的部分。
// 完成后运行会自动对比 CPU 参考实现验证正确性。
//
// 参考资料:
//   theory/07_classic_operators.md 7.2 节 (LayerNorm + Welford)
//   theory/04_warp_and_sync.md 4.2 节 (Warp Shuffle)
//   03_reduce/reduce.cu (Warp Shuffle 归约的完整实现)
// ============================================================

#define CUDA_CHECK(call) do {                                              \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                cudaGetErrorString(err)); exit(1); }                       \
} while(0)

// ==== Warp Shuffle 辅助函数 (已提供) ====

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ==== TODO: 你需要实现这个 kernel ====
// 每个 Block 处理一行 (cols 个元素)
// 输入: x [rows, cols], gamma [cols], beta [cols]
// 输出: y [rows, cols]
// 公式: y[i] = gamma[i] * (x[i] - mean) / sqrt(variance + eps) + beta[i]
__global__ void layernorm_forward(
    const float *x,       // 输入 [rows, cols]
    const float *gamma,   // 缩放参数 [cols]
    const float *beta,    // 偏移参数 [cols]
    float *y,             // 输出 [rows, cols]
    int rows, int cols, float eps) {

    int row = blockIdx.x;  // 每个 Block 处理第 row 行
    int tid = threadIdx.x;
    const float *x_row = x + row * cols;
    float *y_row = y + row * cols;

    // ---- Step 1: 计算均值 mean ----
    // 提示: 用 Grid-Stride Loop 让每个线程累加多个元素
    //       然后用 warp_reduce_sum + Shared Memory 做 Block 级归约
    //       最后 mean = total_sum / cols

    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        local_sum += x_row[i];
    }

    // TODO: Block 级归约得到 mean
    // 提示: 参考 03_reduce/reduce.cu 的 V2 版本
    //   1. warp_reduce_sum(local_sum) → 每 Warp 的 lane 0 有部分和
    //   2. lane 0 写入 Shared Memory
    //   3. __syncthreads()
    //   4. 第一个 Warp 再 reduce → 得到总和
    //   5. mean = total / cols

    __shared__ float s_mean, s_var;
    // ... 你的归约代码 ...
    // s_mean = ???;

    // ---- Step 2: 计算方差 variance ----
    // variance = (1/cols) × Σ (x[i] - mean)²
    //
    // 提示: 和 Step 1 类似, 但累加的是 (x[i] - mean)²

    // TODO: 计算 variance 并存入 s_var
    // float local_var = 0.0f;
    // for (int i = tid; i < cols; i += blockDim.x) {
    //     float diff = x_row[i] - s_mean;
    //     local_var += diff * diff;
    // }
    // ... 归约 ...
    // s_var = ???;

    // ---- Step 3: 归一化 ----
    // y[i] = gamma[i] * (x[i] - mean) * rsqrt(var + eps) + beta[i]

    // TODO: 取消下面的注释并确保 s_mean, s_var 正确
    // __syncthreads();
    // float inv_std = rsqrtf(s_var + eps);
    // for (int i = tid; i < cols; i += blockDim.x) {
    //     y_row[i] = gamma[i] * (x_row[i] - s_mean) * inv_std + beta[i];
    // }
}

// ==== CPU 参考实现 (已提供) ====
void layernorm_cpu(const float *x, const float *gamma, const float *beta,
                    float *y, int rows, int cols, float eps) {
    for (int r = 0; r < rows; r++) {
        const float *xr = x + r * cols;
        float *yr = y + r * cols;
        double mean = 0;
        for (int i = 0; i < cols; i++) mean += xr[i];
        mean /= cols;
        double var = 0;
        for (int i = 0; i < cols; i++) { double d = xr[i] - mean; var += d*d; }
        var /= cols;
        double inv_std = 1.0 / sqrt(var + eps);
        for (int i = 0; i < cols; i++)
            yr[i] = (float)(gamma[i] * (xr[i] - mean) * inv_std + beta[i]);
    }
}

int main() {
    int rows = 1024, cols = 768;  // 典型的 Transformer hidden_size
    float eps = 1e-5f;
    size_t x_bytes = rows * cols * sizeof(float);
    size_t p_bytes = cols * sizeof(float);

    printf("LayerNorm: [%d, %d], eps=%e\n\n", rows, cols, eps);

    float *h_x = (float*)malloc(x_bytes);
    float *h_gamma = (float*)malloc(p_bytes);
    float *h_beta = (float*)malloc(p_bytes);
    float *h_y_ref = (float*)malloc(x_bytes);
    float *h_y_gpu = (float*)malloc(x_bytes);

    srand(42);
    for (int i = 0; i < rows * cols; i++) h_x[i] = (rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < cols; i++) { h_gamma[i] = 1.0f; h_beta[i] = 0.0f; }

    layernorm_cpu(h_x, h_gamma, h_beta, h_y_ref, rows, cols, eps);

    float *d_x, *d_gamma, *d_beta, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, x_bytes));
    CUDA_CHECK(cudaMalloc(&d_gamma, p_bytes));
    CUDA_CHECK(cudaMalloc(&d_beta, p_bytes));
    CUDA_CHECK(cudaMalloc(&d_y, x_bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, x_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma, p_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta, p_bytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    layernorm_forward<<<rows, blockSize>>>(d_x, d_gamma, d_beta, d_y, rows, cols, eps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_y_gpu, d_y, x_bytes, cudaMemcpyDeviceToHost));

    float max_err = 0;
    for (int i = 0; i < rows * cols; i++)
        max_err = fmaxf(max_err, fabsf(h_y_ref[i] - h_y_gpu[i]));

    printf("最大误差: %.2e  %s\n", max_err, max_err < 1e-4 ? "✓ 通过!" : "✗ 未通过 (检查你的实现)");

    if (max_err >= 1e-4) {
        printf("\n提示:\n");
        printf("  1. 先完成 Step 1 (mean) 的归约, 确认 mean 正确\n");
        printf("  2. 再完成 Step 2 (var), 确认 var 正确\n");
        printf("  3. 最后取消 Step 3 的注释\n");
        printf("  4. 归约参考 03_reduce/reduce.cu 的 V2 版本\n");
    }

    cudaFree(d_x); cudaFree(d_gamma); cudaFree(d_beta); cudaFree(d_y);
    free(h_x); free(h_gamma); free(h_beta); free(h_y_ref); free(h_y_gpu);
    return 0;
}
