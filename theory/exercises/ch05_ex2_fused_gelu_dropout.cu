// ============================================================
// Ch5 练习 2: GELU + Dropout 融合 vs 未融合
// 配合: theory/05_operator_development.md 练习 2
//
// 编译: nvcc -O2 -o ch05_ex2_fused_gelu_dropout ch05_ex2_fused_gelu_dropout.cu
// 运行: ./ch05_ex2_fused_gelu_dropout
//
// TODO: 实现 fused_gelu_dropout_kernel (只填 kernel 函数体)
// 提示: output[idx] = (mask > threshold) ? gelu(input[idx]) : 0
//       用简单的 hash 做伪随机 mask: mask = (idx * 2654435761u) % 1000
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define N (1 << 22)
#define BLOCK 256
#define DROP_PROB 0.1f
#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){   \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}} while(0)

__device__ float gelu_fn(float x) {
    return x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f*x*x*x)));
}

__global__ void gelu_kernel(const float *in, float *out, int n) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < n) out[idx] = gelu_fn(in[idx]);
}

__global__ void dropout_kernel(const float *in, float *out, int n, float scale) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < n) {
        unsigned mask = (idx * 2654435761u) % 1000;
        out[idx] = (mask >= (unsigned)(DROP_PROB * 1000)) ? in[idx] * scale : 0.0f;
    }
}

// ============================================================
// TODO: 实现融合版 — 一个 kernel 同时做 GELU + Dropout
// ============================================================
__global__ void fused_gelu_dropout_kernel(const float *in, float *out,
                                          int n, float scale) {
    // --- 在这里写你的代码 ---

}

int main() {
    size_t bytes = N * sizeof(float);
    float scale = 1.0f / (1.0f - DROP_PROB);
    float *d_in, *d_out, *d_tmp;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp, bytes));
    float *h = (float*)malloc(bytes);
    for(int i=0;i<N;i++) h[i]=(float)(i%200-100)/10.f;
    CUDA_CHECK(cudaMemcpy(d_in, h, bytes, cudaMemcpyHostToDevice));

    int grid = (N+BLOCK-1)/BLOCK;
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);

    gelu_kernel<<<grid,BLOCK>>>(d_in,d_tmp,N);
    dropout_kernel<<<grid,BLOCK>>>(d_tmp,d_out,N,scale);
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for(int r=0;r<50;r++){
        gelu_kernel<<<grid,BLOCK>>>(d_in,d_tmp,N);
        dropout_kernel<<<grid,BLOCK>>>(d_tmp,d_out,N,scale);
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms_sep; cudaEventElapsedTime(&ms_sep,t0,t1);

    fused_gelu_dropout_kernel<<<grid,BLOCK>>>(d_in,d_out,N,scale);
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for(int r=0;r<50;r++)
        fused_gelu_dropout_kernel<<<grid,BLOCK>>>(d_in,d_out,N,scale);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms_fused; cudaEventElapsedTime(&ms_fused,t0,t1);

    printf("GELU + Dropout 融合实验\n\n");
    printf("未融合: %.3f ms  融合: %.3f ms  加速: %.2fx\n",
           ms_sep/50, ms_fused/50, ms_sep/ms_fused);
    printf("\n(融合版省了一次中间数组的写回+读出 → 带宽节省 ~50%%)\n");

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_tmp); free(h);
    return 0;
}
