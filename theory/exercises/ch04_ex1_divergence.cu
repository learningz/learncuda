// ============================================================
// Ch4 练习 1: 分支分歧探索 — %2 vs %4 vs 长路径
// 配合: theory/04_warp_and_sync.md 练习 1
//
// 编译: nvcc -O2 --extended-lambda -o ch04_ex1_divergence ch04_ex1_divergence.cu
// 运行: ./ch04_ex1_divergence
// ============================================================

#include <cstdio>
#include <cuda_runtime.h>

#define N (1 << 22)
#define BLOCK 256
#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){   \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}} while(0)

__global__ void div2_short(const float *in, float *out, int n) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x; if(idx>=n) return;
    float v = in[idx];
    out[idx] = (threadIdx.x%2==0) ? v*v+v : v*0.5f-1.0f;
}

__global__ void div4_short(const float *in, float *out, int n) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x; if(idx>=n) return;
    float v = in[idx];
    int branch = threadIdx.x % 4;
    if      (branch==0) out[idx] = v*v;
    else if (branch==1) out[idx] = v+1.0f;
    else if (branch==2) out[idx] = v-1.0f;
    else                out[idx] = v*0.5f;
}

__global__ void div2_long(const float *in, float *out, int n) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x; if(idx>=n) return;
    float v = in[idx];
    if (threadIdx.x%2==0) {
        for(int i=0;i<10;i++) v = v*v*1.00001f + 0.5f;
        out[idx] = v;
    } else {
        for(int i=0;i<10;i++) v = v*0.99999f - 0.3f;
        out[idx] = v;
    }
}

int main() {
    size_t bytes = N*sizeof(float);
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes)); CUDA_CHECK(cudaMalloc(&d_out, bytes));
    float *h = (float*)malloc(bytes);
    for(int i=0;i<N;i++) h[i]=(float)(i%100)/10.f;
    CUDA_CHECK(cudaMemcpy(d_in, h, bytes, cudaMemcpyHostToDevice));

    int grid = (N+BLOCK-1)/BLOCK;
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);

    auto bench = [&](const char *name, auto fn) {
        fn<<<grid,BLOCK>>>(d_in,d_out,N); cudaDeviceSynchronize();
        cudaEventRecord(t0);
        for(int r=0;r<100;r++) fn<<<grid,BLOCK>>>(d_in,d_out,N);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms,t0,t1);
        printf("  %-25s: %.3f ms\n", name, ms/100);
    };

    printf("分支分歧探索\n\n");
    bench("%%2 短路径", div2_short);
    bench("%%4 短路径 (4分支)", div4_short);
    bench("%%2 长路径 (10次乘)", div2_long);

    printf("\n观察: 短路径时编译器可能用谓词化消除分歧 → %%2 和 %%4 差不多\n");
    printf("长路径时分歧代价更明显 → %%2 长路径应该最慢\n");

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_in); cudaFree(d_out); free(h);
    return 0;
}
