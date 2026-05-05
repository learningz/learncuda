// ============================================================
// 练习 1: WMMA 矩阵加法 — Fragment 的 load/store
// 难度: Level 2 (kernel + host 端都要自己写)
//
// 编译: nvcc -O2 -arch=sm_75 -o ex1_wmma_add_level2 ex1_wmma_add_level2.cu
// 运行: ./ex1_wmma_add_level2
// 预期输出: "结果: ✓ 全部正确"
//
// 要求:
//   1. 实现 wmma_add_kernel: 用 WMMA fragment 做 D = C + bias (16×16)
//   2. 在 main 中完成 CPU 内存管理、GPU 分配、launch、验证
//
// 步骤:
//   1. 声明两个 accumulator fragment<float, 16, 16, 16> frag_c, frag_bias
//   2. load_matrix_sync 从全局显存加载 C 和 bias
//   3. 手动遍历 frag_c.x[i] += frag_bias.x[i]
//   4. store_matrix_sync 写回 D
//
// 提示:
//   - 需要 #include <mma.h> 和 using namespace nvcuda
//   - fragment 的 num_elements 告诉你这个线程有几条数据
//   - load 和 store 都需要 ldm = N (leading dimension)
//   - <<<1, 32>>> 一个 Warp 跑一个 16×16 块
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// TODO: 实现 WMMA 矩阵加法 kernel — D = C + bias
__global__ void wmma_add_kernel(const float *C, const float *bias, float *D,
                                 int M, int N) {
    // --- 在这里写你的代码 ---

}

int main() {
    const int M = 16, N = 16;
    const size_t bytes = M * N * sizeof(float);

    // ============================================================
    // TODO: 完成以下步骤
    //   1. malloc CPU 内存 (h_C, h_bias, h_D, h_ref)
    //   2. 初始化 h_C, h_bias, 算 h_ref = C + bias
    //   3. cudaMalloc GPU 内存 (d_C, d_bias, d_D)
    //   4. cudaMemcpy H2D
    //   5. launch wmma_add_kernel<<<1, 32>>>
    //   6. cudaMemcpy D2H (d_D → h_D)
    //   7. 验证: 逐元素比较 h_D 和 h_ref
    //   8. 打印结果 + cudaFree + free
    // ============================================================

    // --- 在这里写你的代码 ---

    return 0;
}
