// ============================================================
// 练习 1: 用 Warp-uniform 条件消除分歧
// 难度: Level 2 (kernel + host 端都要自己写)
//
// 编译: nvcc -O2 -o ex1_warp_uniform_level2 ex1_warp_uniform_level2.cu
// 运行: ./ex1_warp_uniform_level2
// 预期输出: warp-uniform 版比 50% 分歧版快
//
// Level 1 只写 kernel, 本题全部自己写。
//
// 要求:
//   1. 实现 divergent_kernel: 用 threadIdx.x % 2 分支 → 50% Warp 分歧
//   2. 实现 uniform_kernel:   用 (threadIdx.x / 32) % 2 分支 → 无 Warp 分歧
//     计算: if 分支 → val * val + val; else 分支 → val * 0.5f - 1.0f
//   3. 在 main 中完成内存管理、数据传输、cudaEvent 计时 (100 次迭代), 验证
//
// 提示:
//   - divergent: threadIdx.x % 2 → 相邻线程走不同分支 → 同一 Warp 分歧
//   - uniform:   (threadIdx.x/32) % 2 → 前 32 线程全走 if, 后 32 全走 else
//   - 两者计算结果相同但 uniform 更快
// ============================================================

#include <cstdio>
#include <cstdlib>
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

// TODO: 实现有分歧的 kernel — 用 threadIdx.x % 2 分支
__global__ void divergent_kernel(const float *input, float *output, int n) {
    // --- 在这里写你的代码 ---

}

// TODO: 实现无分歧的 kernel — 用 (threadIdx.x / 32) % 2 分支
__global__ void uniform_kernel(const float *input, float *output, int n) {
    // --- 在这里写你的代码 ---

}

int main() {
    const int N = 1 << 22;
    const size_t bytes = N * sizeof(float);
    const int blockSize = 256;

    // ============================================================
    // TODO: 完成以下步骤
    //   1. malloc CPU 内存 (h_in, h_out_div, h_out_uniform, h_ref)
    //   2. 初始化 h_in (随机值), 计算 h_ref (CPU 上算期望输出)
    //   3. cudaMalloc GPU 内存 (d_in, d_out)
    //   4. cudaMemcpy H2D (h_in → d_in)
    //   5. 分别 launch divergent 和 uniform, cudaEvent 计时 (100 次迭代)
    //   6. 每次 cudaMemcpy D2H, 验证正确性
    //   7. 打印耗时和加速比
    //   8. cudaFree + free
    //
    // 注意: divergent 和 uniform 的计算结果应该相同!
    // ============================================================

    // --- 在这里写你的代码 ---

    return 0;
}
