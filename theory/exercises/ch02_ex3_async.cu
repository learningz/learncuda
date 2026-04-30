// ============================================================
// Ch2 练习 3: 异步执行观察
// 配合: theory/02_cuda_programming_model.md 练习 3
//
// 编译: nvcc -O2 -o ch02_ex3_async ch02_ex3_async.cu
// 运行: ./ch02_ex3_async
//
// 观察 kernel launch 的异步性: CPU 不等 GPU 完成就继续执行。
// ============================================================

#include <cstdio>
#include <cuda_runtime.h>

__global__ void slow_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < 10000; i++) val = sinf(val) + 0.001f;
        data[idx] = val;
    }
}

int main() {
    const int N = 1 << 20;
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemset(d_data, 0, N * sizeof(float));

    printf("[1] 准备启动 kernel...\n");
    slow_kernel<<<(N+255)/256, 256>>>(d_data, N);
    printf("[2] kernel 已启动 (但 GPU 可能还在算!)\n");
    printf("[3] 调用 cudaDeviceSynchronize 等待 GPU...\n");
    cudaDeviceSynchronize();
    printf("[4] GPU 计算完成!\n");

    printf("\n观察: [2] 在 [4] 之前很快就打印了 → kernel launch 是异步的!\n");
    printf("CPU 只是把命令塞进 GPU 队列, 没有等 GPU 执行完。\n");
    printf("cudaDeviceSynchronize() 才是真正等待的地方。\n");

    cudaFree(d_data);
    return 0;
}
