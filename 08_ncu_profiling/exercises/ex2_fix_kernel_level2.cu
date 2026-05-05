// ============================================================
// 练习 2: 修复 kernel 性能问题 — 合并访问 + Bank Conflict
// 难度: Level 2 (host 端自己写)
//
// 编译: nvcc -O2 -lineinfo -o ex2_fix_kernel_level2 ex2_fix_kernel_level2.cu
// 运行: ./ex2_fix_kernel_level2
// 预期输出: 修复后带宽有明显提升
//
// 下面的 kernel 有两个性能问题:
//   1. 合并访问: threadIdx.x 对应行而不是列 → 地址不连续 → 非合并
//   2. SMEM 访问: 没有 Bank Conflict 意识 → 同 Bank 多线程冲突
//
// 任务:
//   1. 修复 bad_kernel, 让它做合并访问和消除 Bank Conflict
//   2. 在 main 中完成内存管理、计时、验证
//
// 提示:
//   - 合并访问: 让 threadIdx.x 对应连续维度 (列)
//   - Bank Conflict: SMEM 读的时候 threadIdx.x 连续 → 遍历连续 Bank
//   - 对比修复前后的带宽变化
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define TILE 16

// 有性能问题的版本 — 每线程计算 output[row][col] 但访问不连续
__global__ void bad_kernel(const float *input, float *output, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float tile[TILE][TILE];

    // 问题 1: 加载时 threadIdx.x 对应 row → 非合并访问
    if (row < M && col < N) {
        int load_row = threadIdx.x;  // 错: x 对应行 → 跨步
        int load_col = threadIdx.y;  // 错
        if (load_row < TILE && load_col < TILE)
            tile[load_row][load_col] = input[row * N + col];
    }
    __syncthreads();

    // 问题 2: 读取时 threadIdx.y 变化 → 都在同一 Bank → Bank Conflict
    if (row < M && col < N) {
        float val = tile[threadIdx.y][threadIdx.x];  // 不对的访问顺序
        output[row * N + col] = val * 2.0f;
    }
}

// TODO: 实现修复后的版本 (合并访问 + 无 Bank Conflict)
__global__ void fixed_kernel(const float *input, float *output, int M, int N) {
    // --- 在这里写你的代码 ---

}

int main() {
    const int M = 512, N = 512;
    const size_t bytes = M * N * sizeof(float);

    // ============================================================
    // TODO: 完成以下步骤
    //   1. malloc CPU 内存
    //   2. 初始化 input
    //   3. cudaMalloc + cudaMemcpy
    //   4. 分别 launch bad_kernel 和 fixed_kernel, cudaEvent 计时
    //   5. 计算有效带宽 (bytes*2*iteration / ms / 1e6 = GB/s)
    //   6. 打印对比结果
    //   7. cudaFree + free
    // ============================================================

    // --- 在这里写你的代码 ---

    return 0;
}
