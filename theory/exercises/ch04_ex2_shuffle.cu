// ============================================================
// Ch4 练习 2: Warp Shuffle — __shfl_xor_sync 蝶形归约 + warp_reduce_max
// 配合: theory/04_warp_and_sync.md 练习 2
//
// 编译: nvcc -O2 -o ch04_ex2_shuffle ch04_ex2_shuffle.cu
// 运行: ./ch04_ex2_shuffle
// 预期输出: 两种归约方式结果一致 + warp max 正确
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){   \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}} while(0)

__device__ float warp_reduce_sum_down(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

__device__ float warp_reduce_sum_xor(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

__device__ float warp_reduce_max(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 8));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    return val;
}

__global__ void test_shuffle(const float *input, float *out_down, float *out_xor,
                             float *out_max, int n) {
    int idx = threadIdx.x;
    float val = (idx < n) ? input[idx] : 0.0f;

    float sum_down = warp_reduce_sum_down(val);
    float sum_xor  = warp_reduce_sum_xor(val);
    float mx       = warp_reduce_max(val);

    if (idx == 0) { out_down[0] = sum_down; out_xor[0] = sum_xor; out_max[0] = mx; }
}

int main() {
    const int N = 32;
    float h_in[32];
    srand(42);
    float ref_sum = 0, ref_max = -FLT_MAX;
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)(rand() % 100);
        ref_sum += h_in[i];
        if (h_in[i] > ref_max) ref_max = h_in[i];
    }

    float *d_in, *d_sd, *d_sx, *d_mx;
    CUDA_CHECK(cudaMalloc(&d_in, 32*4));
    CUDA_CHECK(cudaMalloc(&d_sd, 4)); CUDA_CHECK(cudaMalloc(&d_sx, 4)); CUDA_CHECK(cudaMalloc(&d_mx, 4));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, 32*4, cudaMemcpyHostToDevice));

    test_shuffle<<<1, 32>>>(d_in, d_sd, d_sx, d_mx, N);
    CUDA_CHECK(cudaGetLastError());

    float gpu_sd, gpu_sx, gpu_mx;
    CUDA_CHECK(cudaMemcpy(&gpu_sd, d_sd, 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&gpu_sx, d_sx, 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&gpu_mx, d_mx, 4, cudaMemcpyDeviceToHost));

    printf("shfl_down sum = %.1f (ref = %.1f) %s\n", gpu_sd, ref_sum, fabsf(gpu_sd-ref_sum)<1 ? "✓":"✗");
    printf("shfl_xor  sum = %.1f (ref = %.1f) %s\n", gpu_sx, ref_sum, fabsf(gpu_sx-ref_sum)<1 ? "✓":"✗");
    printf("shfl_xor  max = %.1f (ref = %.1f) %s\n", gpu_mx, ref_max, gpu_mx==ref_max ? "✓":"✗");
    printf("\n注意: shfl_xor 的结果所有 lane 都能看到 (不只是 lane 0)!\n");

    cudaFree(d_in); cudaFree(d_sd); cudaFree(d_sx); cudaFree(d_mx);
    return 0;
}
