// ============================================================
// 练习 1: float4 向量化加载 — y[i] = x[i] * scale + bias
// 难度: Level 2 (kernel + host 端都要自己写)
//
// 编译: nvcc -O2 -o ex1_vectorize_level2 ex1_vectorize_level2.cu
// 运行: ./ex1_vectorize_level2
// 预期输出: "结果: ✓ 全部正确" + 向量化版带宽更高
//
// 要求:
//   1. 实现 scalar_kernel: 每个线程处理 1 个 float (普通版)
//   2. 实现 vec4_kernel: 每个线程处理 4 个 float, 用 float4 一次加载
//   3. 在 main 中完成 GPU 内存分配、数据传输、kernel 启动、结果回传、计时
//
// 提示:
//   - float4 要求 N 是 4 的倍数, 用 reinterpret_cast<const float4*>(ptr)
//   - scalar 版: float x = in[idx]
//   - vec4 版:   float4 v = reinterpret_cast<const float4*>(in)[idx];
//                out[idx*4+0] = v.x * scale + bias; ... out[idx*4+3] = v.w * scale + bias;
//   - vec4 grid 只需要 scalar 的 1/4
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

// TODO: 实现 scalar 版 — 每个线程处理 1 个元素
__global__ void scalar_kernel(const float *in, float *out, float scale, float bias, int n) {
    // --- 在这里写你的代码 ---

}

// TODO: 实现 vec4 版 — 每个线程处理 4 个元素, 用 float4 一次加载
__global__ void vec4_kernel(const float *in, float *out, float scale, float bias, int n) {
    // --- 在这里写你的代码 ---

}

int main() {
    const int N = 1 << 22;
    const size_t bytes = N * sizeof(float);
    const float scale = 2.0f, bias = 1.5f;

    // ============================================================
    // TODO: 完成以下步骤
    //   1. malloc CPU 内存 (h_in, h_out_scalar, h_out_vec4, h_ref)
    //   2. 初始化 h_in (随机值), 计算 h_ref = in * scale + bias
    //   3. cudaMalloc GPU 内存 (d_in, d_out)
    //   4. cudaMemcpy H2D
    //   5. 分别 launch scalar 和 vec4 kernel, 用 cudaEvent 计时
    //   6. cudaMemcpy D2H, 验证正确性
    //   7. 打印耗时和有效带宽 (GB/s = bytes*3 / (ms*1e6) )
    //   8. cudaFree + free
    // ============================================================

    // --- 在这里写你的代码 ---

    return 0;
}
