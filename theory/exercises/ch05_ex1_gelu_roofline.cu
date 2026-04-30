// ============================================================
// Ch5 练习 1: GELU Roofline — 实测带宽 vs 理论峰值
// 配合: theory/05_operator_development.md 练习 1
//
// 编译: nvcc -O2 -o ch05_ex1_gelu_roofline ch05_ex1_gelu_roofline.cu
// 运行: ./ch05_ex1_gelu_roofline
// ============================================================

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define N (10 * 1024 * 1024)
#define BLOCK 256
#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){   \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}} while(0)

__global__ void gelu_scalar(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x*x*x)));
        out[idx] = x * cdf;
    }
}

__global__ void gelu_vec4(const float *in, float *out, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < n) {
        float4 v = reinterpret_cast<const float4*>(in)[idx/4];
        auto g = [](float x) {
            return x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f*x*x*x)));
        };
        v.x = g(v.x); v.y = g(v.y); v.z = g(v.z); v.w = g(v.w);
        reinterpret_cast<float4*>(out)[idx/4] = v;
    }
}

int main() {
    size_t bytes = N * sizeof(float);
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes)); CUDA_CHECK(cudaMalloc(&d_out, bytes));
    float *h = (float*)malloc(bytes);
    for(int i=0;i<N;i++) h[i]=(float)(i%200-100)/10.f;
    CUDA_CHECK(cudaMemcpy(d_in, h, bytes, cudaMemcpyHostToDevice));

    int grid_s = (N+BLOCK-1)/BLOCK;
    int grid_v = (N/4+BLOCK-1)/BLOCK;
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);

    auto bench = [&](const char *name, auto fn, int g) {
        fn<<<g,BLOCK>>>(d_in,d_out,N); cudaDeviceSynchronize();
        cudaEventRecord(t0);
        for(int r=0;r<100;r++) fn<<<g,BLOCK>>>(d_in,d_out,N);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms,t0,t1); ms/=100;
        double total = 2.0*N*4;
        printf("%-15s: %.3f ms  有效带宽: %.1f GB/s\n", name, ms, total/(ms*1e6));
    };

    double ai = 15.0 / 8.0;
    printf("GELU Roofline 分析 (N = %dM)\n", N/1024/1024);
    printf("算术强度 AI ≈ %.2f FLOP/Byte (Memory Bound)\n\n", ai);
    bench("scalar", gelu_scalar, grid_s);
    bench("float4", gelu_vec4, grid_v);
    printf("\nfloat4 版本减少了 LD/ST 指令, 带宽利用率应该更高。\n");

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_in); cudaFree(d_out); free(h);
    return 0;
}
