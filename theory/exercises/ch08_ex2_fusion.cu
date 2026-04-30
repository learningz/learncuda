// ============================================================
// Ch8 练习 2: ReLU + Scale + Bias 三版融合对比
// 配合: theory/08_advanced_optimization.md 练习 2
//
// 编译: nvcc -O2 --extended-lambda -o ch08_ex2_fusion ch08_ex2_fusion.cu
// 运行: ./ch08_ex2_fusion
//
// TODO: 实现 fused_kernel 和 fused_vec4_kernel (只填 kernel 函数体)
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define N (1 << 22)
#define BLOCK 256
#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){   \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}} while(0)

__global__ void relu_k(const float *x, float *o, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=fmaxf(x[i],0.f);
}
__global__ void scale_k(const float *x, float *o, float s, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=x[i]*s;
}
__global__ void bias_k(const float *x, float *o, float b, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=x[i]+b;
}

// TODO: 融合版 — 一个 kernel: out[i] = max(x[i],0) * scale + bias
__global__ void fused_kernel(const float *x, float *out, float scale, float bias, int n) {
    // --- 在这里写你的代码 ---

}

// TODO: 融合 + float4 向量化
__global__ void fused_vec4_kernel(const float *x, float *out, float scale, float bias, int n) {
    // --- 在这里写你的代码 ---

}

int main() {
    const float scale = 2.0f, bias = 0.5f;
    size_t bytes = N * sizeof(float);

    float *h_x=(float*)malloc(bytes), *h_ref=(float*)malloc(bytes), *h_out=(float*)malloc(bytes);
    srand(42);
    for(int i=0;i<N;i++) { h_x[i]=(float)(rand()%200-100)/10.f; h_ref[i]=fmaxf(h_x[i],0.f)*scale+bias; }

    float *d_x,*d_out,*d_t1,*d_t2;
    CUDA_CHECK(cudaMalloc(&d_x,bytes)); CUDA_CHECK(cudaMalloc(&d_out,bytes));
    CUDA_CHECK(cudaMalloc(&d_t1,bytes)); CUDA_CHECK(cudaMalloc(&d_t2,bytes));
    CUDA_CHECK(cudaMemcpy(d_x,h_x,bytes,cudaMemcpyHostToDevice));

    int grid=(N+BLOCK-1)/BLOCK, grid4=(N/4+BLOCK-1)/BLOCK;
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);

    auto bench = [&](const char *name, auto fn) {
        fn(); cudaDeviceSynchronize();
        cudaEventRecord(t0);
        for(int r=0;r<50;r++) fn();
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms,t0,t1);
        double bw=2.0*N*4/(ms/50*1e6);
        printf("%-25s: %.3f ms  有效带宽: %.1f GB/s\n", name, ms/50, bw);
    };

    printf("ReLU + Scale + Bias 融合对比\n\n");
    bench("A: 3 独立 kernel", [&]{
        relu_k<<<grid,BLOCK>>>(d_x,d_t1,N);
        scale_k<<<grid,BLOCK>>>(d_t1,d_t2,scale,N);
        bias_k<<<grid,BLOCK>>>(d_t2,d_out,bias,N);
    });
    bench("B: 1 融合 kernel", [&]{
        fused_kernel<<<grid,BLOCK>>>(d_x,d_out,scale,bias,N);
    });
    bench("C: 融合 + float4", [&]{
        fused_vec4_kernel<<<grid4,BLOCK>>>(d_x,d_out,scale,bias,N);
    });

    CUDA_CHECK(cudaMemcpy(h_out,d_out,bytes,cudaMemcpyDeviceToHost));
    bool ok=true;
    for(int i=0;i<N;i++) if(fabsf(h_out[i]-h_ref[i])>1e-3f){ok=false;break;}
    printf("\n正确性 (融合版C): %s\n", ok?"✓":"✗");

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_x);cudaFree(d_out);cudaFree(d_t1);cudaFree(d_t2);
    free(h_x);free(h_ref);free(h_out);
    return 0;
}
