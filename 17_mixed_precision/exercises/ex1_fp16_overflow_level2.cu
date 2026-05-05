// ============================================================
// 练习 1: FP16 溢出实验 — Loss Scaling 的完整实现
// 难度: Level 2 (kernel + host 端都要自己写)
//
// 编译: nvcc -O2 -arch=sm_75 -o ex1_fp16_overflow_level2 ex1_fp16_overflow_level2.cu
// 运行: ./ex1_fp16_overflow_level2
// 预期输出: 展示 FP16 overflow/underflow 和 Loss Scaling 的效果
//
// Level 1 只观察数值, 本题要自己写所有代码。
//
// 要求:
//   1. 实现 test_convert_kernel: 把 float 数组转成 FP16 (用 __float2half),
//      再转回 float (用 __half2float), 存在 out 中。
//      → 溢出变 inf, 下溢变 0
//
//   2. 实现 loss_scale_kernel: y[i] = x[i] * scale (float 乘法)
//      把梯度放大, 这样 __float2half 时不会下溢
//
//   3. 实现 loss_unscale_kernel: y[i] = x[i] / scale (float 除法)
//      恢复原始梯度值
//
//   4. 在 main 中:
//      a. 生成 3 类测试值: 正常值, 大值 (>65504), 小值 (<6e-8)
//      b. 跑 test_convert_kernel 展示溢出/下溢
//      c. 跑 loss_scale → convert → loss_unscale 展示 Loss Scaling 能保护小值
//      d. 打印对比: original vs direct_fp16 vs loss_scaled_recovered
//
// 提示:
//   - 包含 <cuda_fp16.h>
//   - __half 是 FP16 类型, sizeof(half)=2
//   - __float2half(v) 返回 __half (需要 nvcc 编译)
//   - __half2float(h) 返回 float
//   - N 不用太大, 20 个测试值就够了
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// TODO: 实现转换 kernel — x[i] → __float2half → __half2float → out[i]
__global__ void test_convert_kernel(const float *x, float *out, int n) {
    // --- 在这里写你的代码 ---

}

// TODO: 放大 kernel — out[i] = x[i] * scale
__global__ void loss_scale_kernel(const float *x, float *out, float scale, int n) {
    // --- 在这里写你的代码 ---

}

// TODO: 恢复 kernel — out[i] = x[i] / scale
__global__ void loss_unscale_kernel(const float *x, float *out, float scale, int n) {
    // --- 在这里写你的代码 ---

}

int main() {
    const int N = 20;
    const size_t bytes = N * sizeof(float);
    const float scale = 1024.0f;

    // ============================================================
    // TODO: 完成以下步骤
    //   1. malloc CPU 内存 (h_x, h_direct, h_scaled, h_tmp, h_recovered)
    //   2. 初始化 h_x: 前半是正常值, 中间是大值, 后几个是小值
    //      - 正常: 3.14, -1.5, 100.0, 0.001, 65504.0
    //      - 大值: 70000.0, 100000.0, -80000.0
    //      - 小值: 1e-7, 5e-8, 1e-8, 3e-8
    //   3. cudaMalloc (d_x, d_direct, d_scaled, d_recovered)
    //   4. cudaMemcpy H2D
    //   5. launch test_convert_kernel → d_direct (直接转换, 看溢出)
    //   6. launch loss_scale_kernel → loss_convert → loss_unscale
    //      (缩放 → 转换 → 恢复, 看小值是否存活)
    //   7. cudaMemcpy D2H (d_direct, d_recovered → h_direct, h_recovered)
    //   8. 打印表格对比: original | direct FP16 | Loss Scaled | 保护成功?
    //   9. cudaFree + free
    //
    // 输出格式示例:
    //   value      | direct FP16   | Loss Scaled   | OK?
    //   3.14       | 3.140625      | 3.140625      | ✓
    //   70000.0    | inf           | inf           | 溢出, Loss Scaling 也救不了
    //   1e-7       | 0.0           | 9.9999e-08   | ✓ Loss Scaling 救活了!
    // ============================================================

    // --- 在这里写你的代码 ---

    return 0;
}
