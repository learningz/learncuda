// ============================================================
// Ch4 练习 3: __syncthreads 死锁演示
// 配合: theory/04_warp_and_sync.md 练习 3
//
// 编译: nvcc -O2 -o ch04_ex3_deadlock ch04_ex3_deadlock.cu
// 运行: ./ch04_ex3_deadlock
//
// 警告: 这个程序会故意挂起! 用 Ctrl+C 终止或等待超时。
// 然后取消注释 "修复版" 的代码看正确行为。
// ============================================================

#include <cstdio>
#include <cuda_runtime.h>

__global__ void deadlock_demo() {
    if (threadIdx.x < 128) {
        __syncthreads();
    }
}

__global__ void fixed_demo() {
    if (threadIdx.x < 128) {
        // 做一些只有前 128 个线程需要做的事
    }
    __syncthreads();
    // 现在所有线程都到达了 — 安全!
}

int main() {
    printf("=== __syncthreads 死锁演示 ===\n\n");

    printf("[修复版] 所有线程都到达 __syncthreads...\n");
    fixed_demo<<<1, 256>>>();
    cudaError_t err = cudaDeviceSynchronize();
    printf("结果: %s\n\n", err == cudaSuccess ? "✓ 正常完成" : cudaGetErrorString(err));

    printf("[死锁版] 只有 128 个线程到达 __syncthreads...\n");
    printf("(如果程序挂起, 用 Ctrl+C 终止)\n");
    deadlock_demo<<<1, 256>>>();
    err = cudaDeviceSynchronize();
    printf("结果: %s\n", err == cudaSuccess ? "竟然没死锁? (某些 GPU 可能不挂起但结果未定义)" : cudaGetErrorString(err));

    printf("\n规则: __syncthreads() 必须被 Block 内所有线程执行到!\n");
    printf("放在 if 里 → 部分线程到不了 → 未定义行为 (可能死锁/挂起)\n");
    return 0;
}
