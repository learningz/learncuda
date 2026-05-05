#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ============================================================
// 调试练习 3: Softmax — 大输入时输出全是 NaN
//
// 这个程序对每一行做 Softmax。小输入 (值在 -10~10) 时正确,
// 但输入值较大 (如 50~150) 时, 输出全是 NaN 或 Inf。
//
// 你的任务: 找到 bug 并修复它, 让大输入也能正确计算。
//
// 提示: exp() 函数的溢出条件是什么? Softmax 的标准写法
//       是怎么避免溢出的?
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

// ====== BUG IS HERE ======
__global__ void buggy_softmax(const float *input, float *output,
                               int rows, int cols) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float *x = input + row * cols;
    float *y = output + row * cols;

    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        local_sum += expf(x[i]);
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float sum_val = smem[0];
    __syncthreads();

    for (int i = tid; i < cols; i += blockDim.x) {
        y[i] = expf(x[i]) / sum_val;
    }
}
// =========================

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
        if (isnan(test[i]) || isinf(test[i])) {
            printf("  NaN/Inf @ %d: test=%.6f\n", i, test[i]);
            return false;
        }
        if (fabsf(ref[i] - test[i]) > tol) {
            printf("  不匹配 @ %d: ref=%.6f test=%.6f\n", i, ref[i], test[i]);
            return false;
        }
    }
    return true;
}

int main() {
    int rows = 256, cols = 512;
    size_t bytes = rows * cols * sizeof(float);
    float *h_in = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);

    srand(42);

    printf("===== 测试 1: 小输入 (值在 -10 ~ 10) =====\n");
    for (int i = 0; i < rows * cols; i++)
        h_in[i] = (float)(rand() % 200 - 100) / 10.0f;

    softmax_cpu(h_in, h_ref, rows, cols);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    size_t smem = blockSize * sizeof(float);
    buggy_softmax<<<rows, blockSize, smem>>>(d_in, d_out, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("  结果: %s\n", verify(h_ref, h_out, rows * cols) ? "✓ 正确" : "✗ 有错误");

    printf("\n===== 测试 2: 大输入 (值在 50 ~ 150) =====\n");
    for (int i = 0; i < rows * cols; i++)
        h_in[i] = 50.0f + (float)(rand() % 1000) / 10.0f;

    softmax_cpu(h_in, h_ref, rows, cols);
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    buggy_softmax<<<rows, blockSize, smem>>>(d_in, d_out, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("  结果: %s\n", verify(h_ref, h_out, rows * cols) ? "✓ 正确" : "✗ 有错误");

    printf("\n如果测试 1 通过但测试 2 失败 (NaN/Inf), 说明有数值稳定性问题!\n");

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_ref); free(h_out);
    return 0;
}

// ============================================================
// BUG ANSWER (先自己找!):
//
// 第 38-50 行: Softmax 计算时没有减去最大值!
//
//   buggy 版本直接算 exp(x[i]), 但 float 的 exp 在 x > 88 时
//   就会溢出为 +Inf。所以当输入值较大时:
//     exp(100) = +Inf
//     Inf / Inf = NaN
//
//   而 CPU 参考实现正确地先求了 max, 然后算 exp(x[i] - max)。
//   减去 max 后, 最大的指数变成 exp(0)=1, 永远不会溢出。
//
// 修复: 在求 sum 之前, 先做一次归约求每行的最大值, 然后
//       用 exp(x[i] - max_val) 替代 exp(x[i])。
//
//   具体修复 (参考 07_softmax/softmax.cu 的 V1):
//
//   // 新增: 先求 max
//   float local_max = -INFINITY;
//   for (int i = tid; i < cols; i += blockDim.x)
//       local_max = fmaxf(local_max, x[i]);
//   smem[tid] = local_max;
//   __syncthreads();
//   for (int s = blockDim.x / 2; s > 0; s >>= 1) {
//       if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
//       __syncthreads();
//   }
//   float max_val = smem[0];
//   __syncthreads();
//
//   // 修改: exp(x[i]) → exp(x[i] - max_val)
//   float local_sum = 0.0f;
//   for (int i = tid; i < cols; i += blockDim.x)
//       local_sum += expf(x[i] - max_val);
//   ...
//   y[i] = expf(x[i] - max_val) / sum_val;
// ============================================================
