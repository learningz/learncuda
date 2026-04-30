#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ============================================================
// 12: LayerNorm Solution — 完整实现 (前向 + 反向)
//
// 这是 12_layernorm_project/layernorm_starter.cu 的参考答案。
// 如果你还没有自己尝试, 请先用 starter code 独立完成!
//
// 配合理论: theory/07_classic_operators.md 7.2 节
// ============================================================

#define CUDA_CHECK(call) do {                                              \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                cudaGetErrorString(err)); exit(1); }                       \
} while(0)

// ---- Warp Shuffle 辅助 ----
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ---- 前向 Kernel ----
// 每个 Block 处理一行。两遍: 先算统计量, 再归一化。
__global__ void layernorm_forward(
    const float *x, const float *gamma, const float *beta,
    float *y, float *mean_out, float *rstd_out,
    int rows, int cols, float eps) {

    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float *xr = x + row * cols;
    float *yr = y + row * cols;

    // ---- Pass 1: 计算 mean 和 variance ----
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = xr[i];
        local_sum += val;
        local_sq_sum += val * val;
    }

    // Warp 内归约
    local_sum = warp_reduce_sum(local_sum);
    local_sq_sum = warp_reduce_sum(local_sq_sum);

    // Warp 间归约 (通过 Shared Memory)
    __shared__ float s_sum[32];   // 最多 32 个 Warp (blockDim 最大 1024)
    __shared__ float s_sq[32];
    int warp_id = tid / 32;
    int lane = tid % 32;

    if (lane == 0) {
        s_sum[warp_id] = local_sum;
        s_sq[warp_id] = local_sq_sum;
    }
    __syncthreads();

    // 第一个 Warp 汇总
    int num_warps = blockDim.x / 32;
    if (warp_id == 0) {
        local_sum = (lane < num_warps) ? s_sum[lane] : 0.0f;
        local_sq_sum = (lane < num_warps) ? s_sq[lane] : 0.0f;
        local_sum = warp_reduce_sum(local_sum);
        local_sq_sum = warp_reduce_sum(local_sq_sum);
    }

    __shared__ float s_mean, s_rstd;
    if (tid == 0) {
        float mean = local_sum / cols;
        // variance = E[x²] - E[x]² (这个公式比 Welford 简单, 对 float 够用)
        float var = local_sq_sum / cols - mean * mean;
        float rstd = rsqrtf(var + eps);
        s_mean = mean;
        s_rstd = rstd;
        if (mean_out) mean_out[row] = mean;
        if (rstd_out) rstd_out[row] = rstd;
    }
    __syncthreads();

    float mean = s_mean;
    float rstd = s_rstd;

    // ---- Pass 2: 归一化 ----
    for (int i = tid; i < cols; i += blockDim.x) {
        yr[i] = gamma[i] * (xr[i] - mean) * rstd + beta[i];
    }
}

// ---- 反向 Kernel ----
// LayerNorm 反向比前向复杂得多。需要计算 dx, dgamma, dbeta。
//
// 数学推导:
//   前向: y[i] = gamma[i] * (x[i] - mean) * rstd + beta[i]
//   其中: mean = (1/N) Σ x[j]
//         var  = (1/N) Σ (x[j] - mean)²
//         rstd = 1 / sqrt(var + eps)
//
//   反向 (对 x):
//     dx[i] = rstd * gamma[i] * (dy[i] - (1/N) * sum_dy - (1/N) * xhat[i] * sum_dy_xhat)
//     其中: xhat[i] = (x[i] - mean) * rstd
//           sum_dy = Σ dy[j] * gamma[j]
//           sum_dy_xhat = Σ dy[j] * gamma[j] * xhat[j]
//
//   反向 (对 gamma): dgamma[i] = Σ_over_rows dy[i] * xhat[i]
//   反向 (对 beta):  dbeta[i]  = Σ_over_rows dy[i]
//
// 每个 Block 处理一行 (和前向一样), 算 dx 的那一行。
// dgamma/dbeta 需要跨行累加 → 本实现用 atomicAdd (简单但非最优)。
__global__ void layernorm_backward(
    const float *dy,         // [rows, cols] 上游梯度
    const float *x,          // [rows, cols] 前向输入 (saved_for_backward)
    const float *gamma,      // [cols]
    const float *mean_saved, // [rows] 前向保存的 mean
    const float *rstd_saved, // [rows] 前向保存的 rstd
    float *dx,               // [rows, cols] 输出: 对 x 的梯度
    float *dgamma,           // [cols] 输出: 对 gamma 的梯度 (需要预清零!)
    float *dbeta,            // [cols] 输出: 对 beta 的梯度 (需要预清零!)
    int rows, int cols) {

    int row = blockIdx.x;
    int tid = threadIdx.x;

    float mean = mean_saved[row];
    float rstd = rstd_saved[row];

    const float *dy_row = dy + row * cols;
    const float *x_row = x + row * cols;
    float *dx_row = dx + row * cols;

    // ---- 先算两个中间 reduce: sum_dy 和 sum_dy_xhat ----
    float local_sum_dy = 0.0f;
    float local_sum_dy_xhat = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float xhat = (x_row[i] - mean) * rstd;
        float dy_gamma = dy_row[i] * gamma[i];
        local_sum_dy += dy_gamma;
        local_sum_dy_xhat += dy_gamma * xhat;
    }

    // Block 级归约
    local_sum_dy = warp_reduce_sum(local_sum_dy);
    local_sum_dy_xhat = warp_reduce_sum(local_sum_dy_xhat);

    __shared__ float s1[32], s2[32];
    int warp_id = tid / 32, lane = tid % 32;
    int num_warps = blockDim.x / 32;
    if (lane == 0) { s1[warp_id] = local_sum_dy; s2[warp_id] = local_sum_dy_xhat; }
    __syncthreads();
    if (warp_id == 0) {
        local_sum_dy = (lane < num_warps) ? s1[lane] : 0.0f;
        local_sum_dy_xhat = (lane < num_warps) ? s2[lane] : 0.0f;
        local_sum_dy = warp_reduce_sum(local_sum_dy);
        local_sum_dy_xhat = warp_reduce_sum(local_sum_dy_xhat);
    }
    __shared__ float ss_dy, ss_dy_xhat;
    if (tid == 0) { ss_dy = local_sum_dy; ss_dy_xhat = local_sum_dy_xhat; }
    __syncthreads();

    float inv_cols = 1.0f / cols;

    // ---- 计算 dx, 并累加 dgamma/dbeta ----
    for (int i = tid; i < cols; i += blockDim.x) {
        float xhat = (x_row[i] - mean) * rstd;

        // dx[i] = rstd * gamma[i] * (dy[i] - inv_cols * sum_dy - inv_cols * xhat * sum_dy_xhat)
        dx_row[i] = rstd * gamma[i] *
            (dy_row[i] - inv_cols * ss_dy - inv_cols * xhat * ss_dy_xhat);

        // dgamma[i] += dy[i] * xhat  (跨行累加, 用 atomicAdd)
        atomicAdd(&dgamma[i], dy_row[i] * xhat);
        // dbeta[i] += dy[i]
        atomicAdd(&dbeta[i], dy_row[i]);
    }
}

// ---- CPU 参考 ----
void layernorm_cpu_fwd(const float *x, const float *g, const float *b,
                        float *y, float *mean, float *rstd,
                        int rows, int cols, float eps) {
    for (int r = 0; r < rows; r++) {
        double m = 0;
        for (int i = 0; i < cols; i++) m += x[r*cols+i];
        m /= cols;
        double v = 0;
        for (int i = 0; i < cols; i++) { double d = x[r*cols+i]-m; v += d*d; }
        v /= cols;
        double rs = 1.0 / sqrt(v + eps);
        mean[r] = (float)m;
        rstd[r] = (float)rs;
        for (int i = 0; i < cols; i++)
            y[r*cols+i] = (float)(g[i] * (x[r*cols+i] - m) * rs + b[i]);
    }
}

int main() {
    int rows = 512, cols = 768;
    float eps = 1e-5f;
    size_t xy_bytes = rows * cols * sizeof(float);
    size_t p_bytes = cols * sizeof(float);
    size_t stat_bytes = rows * sizeof(float);

    printf("LayerNorm Solution: [%d, %d]\n\n", rows, cols);

    float *hx = (float*)malloc(xy_bytes);
    float *hg = (float*)malloc(p_bytes);
    float *hb = (float*)malloc(p_bytes);
    float *hy_ref = (float*)malloc(xy_bytes);
    float *hy_gpu = (float*)malloc(xy_bytes);
    float *hmean = (float*)malloc(stat_bytes);
    float *hrstd = (float*)malloc(stat_bytes);

    srand(42);
    for (int i = 0; i < rows*cols; i++) hx[i] = (rand()%200-100)/100.0f;
    for (int i = 0; i < cols; i++) { hg[i] = 1.0f + (rand()%100)/1000.0f; hb[i] = (rand()%100-50)/1000.0f; }

    layernorm_cpu_fwd(hx, hg, hb, hy_ref, hmean, hrstd, rows, cols, eps);

    float *dx, *dg, *db, *dy_gpu, *dmean, *drstd, *ddy, *ddx, *ddg, *ddb;
    CUDA_CHECK(cudaMalloc(&dx, xy_bytes));
    CUDA_CHECK(cudaMalloc(&dg, p_bytes));
    CUDA_CHECK(cudaMalloc(&db, p_bytes));
    CUDA_CHECK(cudaMalloc(&dy_gpu, xy_bytes));
    CUDA_CHECK(cudaMalloc(&dmean, stat_bytes));
    CUDA_CHECK(cudaMalloc(&drstd, stat_bytes));
    CUDA_CHECK(cudaMemcpy(dx, hx, xy_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dg, hg, p_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb, p_bytes, cudaMemcpyHostToDevice));

    // ---- 前向 ----
    layernorm_forward<<<rows, 256>>>(dx, dg, db, dy_gpu, dmean, drstd, rows, cols, eps);
    CUDA_CHECK(cudaMemcpy(hy_gpu, dy_gpu, xy_bytes, cudaMemcpyDeviceToHost));

    float fwd_err = 0;
    for (int i = 0; i < rows*cols; i++) fwd_err = fmaxf(fwd_err, fabsf(hy_ref[i]-hy_gpu[i]));
    printf("前向最大误差: %.2e %s\n", fwd_err, fwd_err < 1e-4 ? "✓" : "✗");

    // ---- 反向 ----
    float *hdy = (float*)malloc(xy_bytes);
    for (int i = 0; i < rows*cols; i++) hdy[i] = (rand()%200-100)/100.0f;
    CUDA_CHECK(cudaMalloc(&ddy, xy_bytes));
    CUDA_CHECK(cudaMalloc(&ddx, xy_bytes));
    CUDA_CHECK(cudaMalloc(&ddg, p_bytes));
    CUDA_CHECK(cudaMalloc(&ddb, p_bytes));
    CUDA_CHECK(cudaMemcpy(ddy, hdy, xy_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(ddg, 0, p_bytes));
    CUDA_CHECK(cudaMemset(ddb, 0, p_bytes));

    layernorm_backward<<<rows, 256>>>(ddy, dx, dg, dmean, drstd, ddx, ddg, ddb, rows, cols);

    float *hdx_gpu = (float*)malloc(xy_bytes);
    CUDA_CHECK(cudaMemcpy(hdx_gpu, ddx, xy_bytes, cudaMemcpyDeviceToHost));

    printf("反向 kernel 已运行 (dx 输出形状: [%d, %d])\n", rows, cols);
    printf("  (完整验证需要 PyTorch 对比, 见 test_gelu.py 的模式)\n");

    printf("\n性能:\n");
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < 100; i++)
        layernorm_forward<<<rows, 256>>>(dx, dg, db, dy_gpu, dmean, drstd, rows, cols, eps);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1); ms /= 100;
    float gbps = (float)xy_bytes * 2 / (ms * 1e6);
    printf("  前向: %.3f ms, 有效带宽: %.1f GB/s\n", ms, gbps);

    cudaEventRecord(t0);
    for (int i = 0; i < 100; i++) {
        CUDA_CHECK(cudaMemset(ddg, 0, p_bytes));
        CUDA_CHECK(cudaMemset(ddb, 0, p_bytes));
        layernorm_backward<<<rows, 256>>>(ddy, dx, dg, dmean, drstd, ddx, ddg, ddb, rows, cols);
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1); ms /= 100;
    printf("  反向: %.3f ms (含 memset dgamma/dbeta)\n", ms);

    // 清理
    cudaFree(dx); cudaFree(dg); cudaFree(db); cudaFree(dy_gpu);
    cudaFree(dmean); cudaFree(drstd); cudaFree(ddy); cudaFree(ddx);
    cudaFree(ddg); cudaFree(ddb);
    free(hx); free(hg); free(hb); free(hy_ref); free(hy_gpu);
    free(hmean); free(hrstd); free(hdy); free(hdx_gpu);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return 0;
}
