// ============================================================
// 练习 1: Sigmoid + Scale 融合
// 难度: Level 2 (kernel + host 端都要自己写)
//
// 编译: nvcc -O2 -o ex1_fused_sigmoid_level2 ex1_fused_sigmoid_level2.cu
// 运行: ./ex1_fused_sigmoid_level2
// 预期输出: 正确性通过 + 融合版比未融合版快
//
// Level 1 只写 kernel, 本题全部自己写。
//
// 要求:
//   1. 实现 sigmoid_kernel: y[i] = 1/(1+exp(-x[i]))
//   2. 实现 scale_kernel:   y[i] = in[i] * scale
//   3. 实现 fused_kernel:   y[i] = 1/(1+exp(-x[i])) * scale  (一个 kernel 搞定!)
//   4. 在 main 中分别跑未融合版 (两个 kernel 串行) 和融合版, 对比性能
//
// 关键: 未融合版需要中间 buffer (d_tmp), 融合版直接在寄存器里做完。
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

// TODO: 实现 sigmoid kernel
__global__ void sigmoid_kernel(const float *x, float *out, int n) {
    // --- 在这里写你的代码 ---

}

// TODO: 实现 scale kernel
__global__ void scale_kernel(const float *in, float *out, float scale, int n) {
    // --- 在这里写你的代码 ---

}

// TODO: 实现融合版 — y[i] = sigmoid(x[i]) * scale, 一个 kernel 搞定
__global__ void fused_kernel(const float *x, float *y, float scale, int n) {
    // --- 在这里写你的代码 ---

}

int main() {
    const int N = 1 << 22;
    const size_t bytes = N * sizeof(float);
    const float scale = 2.5f;
    const int blockSize = 256;

    // ============================================================
    // TODO: 完成以下步骤
    //   1. malloc CPU 内存 (h_x, h_y, h_ref)
    //   2. 初始化 h_x, 计算 h_ref = sigmoid(h_x[i]) * scale
    //   3. cudaMalloc (d_x, d_y, d_tmp)
    //   4. cudaMemcpy H2D (h_x → d_x)
    //   5. 未融合版: sigmoid_kernel → scale_kernel, cudaEvent 计时 (50 次迭代)
    //   6. 融合版:   fused_kernel, cudaEvent 计时 (50 次迭代)
    //   7. cudaMemcpy D2H (d_y → h_y), 验证正确性
    //   8. 打印: 正确性, 未融合/融合耗时, 加速比
    //   9. cudaFree + free
    // ============================================================

    // --- 在这里写你的代码 ---

    return 0;
}
