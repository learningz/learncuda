// ============================================================
// Ch7 练习 2: Warp-level Softmax 支持 cols ≤ 64
// 配合: theory/07_classic_operators.md 练习 2
//
// 编译: nvcc -O2 -o ch07_ex2_warp_softmax64 ch07_ex2_warp_softmax64.cu
// 运行: ./ch07_ex2_warp_softmax64
// 预期输出: "结果: ✓ 全部正确"
//
// 07_softmax/softmax.cu 的 V3 只处理 cols ≤ 32 (每线程 1 个元素)。
// 本题扩展到 cols ≤ 64: 每线程处理 2 个元素, 先局部处理再 Warp Shuffle 归约。
//
// TODO: 实现 softmax_warp64_kernel (只填 kernel 函数体)
//
// 提示:
//   - 每线程加载 x[lane] 和 x[lane+32] (注意边界!)
//   - 局部 max = fmaxf(val0, val1)
//   - Warp Shuffle 归约求全局 max
//   - 局部 sum = expf(val0 - global_max) + expf(val1 - global_max)
//   - Warp Shuffle 归约求全局 sum
//   - 写回: y[lane] = expf(val0 - max) / sum, y[lane+32] = ...
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){   \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}} while(0)

__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// TODO: 实现支持 cols ≤ 64 的 Warp-level Softmax
// 配置: <<<rows, 32>>>  (每行 1 个 Warp)
__global__ void softmax_warp64_kernel(const float *input, float *output,
                                      int rows, int cols) {
    // --- 在这里写你的代码 ---

}

void softmax_cpu(const float *in, float *out, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float mx = -INFINITY;
        for (int c = 0; c < cols; c++) mx = fmaxf(mx, in[r*cols+c]);
        float s = 0;
        for (int c = 0; c < cols; c++) s += expf(in[r*cols+c] - mx);
        for (int c = 0; c < cols; c++) out[r*cols+c] = expf(in[r*cols+c] - mx) / s;
    }
}

int main() {
    const int rows = 4096, cols = 64;
    size_t bytes = rows * cols * sizeof(float);

    float *h_in = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < rows*cols; i++) h_in[i] = (float)(rand()%200-100)/10.f;
    softmax_cpu(h_in, h_ref, rows, cols);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes)); CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    softmax_warp64_kernel<<<rows, 32>>>(d_in, d_out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < rows*cols; i++)
        if (fabsf(h_out[i] - h_ref[i]) > 1e-4f) {
            printf("@ %d: %.6f vs %.6f\n", i, h_out[i], h_ref[i]);
            ok = false; break;
        }
    printf("结果: %s\n", ok ? "✓ 全部正确" : "✗ 有错误");

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out); free(h_ref);
    return 0;
}
