// ============================================================
// Ch8 练习 1: ILP 实验 — 无展开 vs 4 路展开
// 配合: theory/08_advanced_optimization.md 练习 1
//
// 编译: nvcc -O2 --extended-lambda -o ch08_ex1_ilp ch08_ex1_ilp.cu
// 运行: ./ch08_ex1_ilp
// ============================================================

#include <cstdio>
#include <cuda_runtime.h>

#define N (1 << 24)
#define BLOCK 256
#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){   \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}} while(0)

__global__ void no_ilp(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float a = in[i];
        float b = a * 2.0f;
        float c = b + 1.0f;
        out[i] = c;
    }
}

__global__ void with_ilp(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i = idx;
    for (; i + 3*stride < n; i += 4*stride) {
        float a0 = in[i], a1 = in[i+stride], a2 = in[i+2*stride], a3 = in[i+3*stride];
        out[i]          = a0*2.0f+1.0f;
        out[i+stride]   = a1*2.0f+1.0f;
        out[i+2*stride] = a2*2.0f+1.0f;
        out[i+3*stride] = a3*2.0f+1.0f;
    }
    for (; i < n; i += stride) out[i] = in[i]*2.0f+1.0f;
}

int main() {
    size_t bytes = N*sizeof(float);
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes)); CUDA_CHECK(cudaMalloc(&d_out, bytes));
    float *h = (float*)malloc(bytes);
    for(int i=0;i<N;i++) h[i]=(float)i;
    CUDA_CHECK(cudaMemcpy(d_in, h, bytes, cudaMemcpyHostToDevice));

    int grid = 256;
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);

    auto bench = [&](const char *name, auto fn) {
        fn<<<grid,BLOCK>>>(d_in,d_out,N); cudaDeviceSynchronize();
        cudaEventRecord(t0);
        for(int r=0;r<100;r++) fn<<<grid,BLOCK>>>(d_in,d_out,N);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms,t0,t1);
        double bw = 2.0*N*4/(ms/100*1e6);
        printf("%-15s: %.3f ms  有效带宽: %.1f GB/s\n", name, ms/100, bw);
    };

    printf("ILP 实验 (N=%dM)\n\n", N/1024/1024);
    bench("无展开", no_ilp);
    bench("4路展开 (ILP)", with_ilp);
    printf("\n4路展开: 4条 LDG 背靠背发射 → 等待第一条回来时, 后面的已经在路上了。\n");

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_in); cudaFree(d_out); free(h);
    return 0;
}
