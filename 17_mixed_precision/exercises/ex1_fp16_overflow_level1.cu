// ============================================================
// 练习 1: FP16 溢出实战 — 感受 FP16 的数值边界
// 难度: Level 1 (只需填写几个数值观察结果)
//
// 编译: nvcc -O2 -arch=sm_75 -o ex1_fp16_overflow_level1 ex1_fp16_overflow_level1.cu
// 运行: ./ex1_fp16_overflow_level1
// 预期输出: 观察 FP16 overflow/underflow 的行为
//
// 任务: 在标记 TODO 的地方填入你对结果的预测, 然后运行验证。
//
// 17_mixed_precision/mixed_precision.cu 展示了完整的三种精度对比。
// 本题只聚焦 FP16 的溢出行为 — 理解为什么需要 Loss Scaling。
//
// 知识点:
//   - FP16 最大值 ≈ 65504, 最小值正数 ≈ 6e-8
//   - 超过 max → +inf; 小于 min → 0 (underflow)
//   - Loss Scaling: 把小梯度放大到 FP16 能表示的范围
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>

__global__ void fp16_test_kernel(float *results, int *overflow_count) {
    if (threadIdx.x != 0) return;

    // 测试 1: 正常值
    float v1 = 3.14f;
    __half h1 = __float2half(v1);
    results[0] = __half2float(h1);  // = ?

    // 测试 2: FP16 max
    float v2 = 65504.0f;
    __half h2 = __float2half(v2);
    results[1] = __half2float(h2);  // = ?

    // 测试 3: 溢出 (超过 FP16 max)
    float v3 = 70000.0f;
    __half h3 = __float2half(v3);
    results[2] = __half2float(h3);  // = ? (提示: isinf 可检测)

    // 测试 4: 下溢 (小于 FP16 最小正数)
    float v4 = 1e-7f;
    __half h4 = __float2half(v4);
    results[3] = __half2float(h4);  // = ? (变成 0?)

    // 测试 5: Loss Scaling
    float scale = 1024.0f;
    float scaled = v4 * scale;       // 放大到 FP16 能表示
    __half h5 = __float2half(scaled);
    float recovered = __half2float(h5) / scale;  // 恢复
    results[4] = recovered;          // ≈ 1e-7 ?

    // 测试 6: FP16 min positive
    results[5] = __half2float(__float2half(5.96e-8f));   // 略大于 min → ?
    results[6] = __half2float(__float2half(5.0e-8f));    // 略小于 min → ?

    // 统计多少个值溢出了
    *overflow_count = 0;
}

int main() {
    float *d_results; int *d_count;
    cudaMalloc(&d_results, 8 * sizeof(float));
    cudaMalloc(&d_count, sizeof(int));

    fp16_test_kernel<<<1, 1>>>(d_results, d_count);
    cudaDeviceSynchronize();

    float results[8]; int count;
    cudaMemcpy(results, d_results, 8 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    printf("=== FP16 溢出实验 ===\n\n");

    // TODO: 运行前先预测这些值!
    printf("--- 你的预测 (运行前填写) ---\n");
    printf("1. 正常值 3.14 → FP16 → float:  ________\n");
    printf("2. 65504 (FP16 max) → FP16 → float:  ________\n");
    printf("3. 70000 (>FP16 max) → FP16 → float:  ________\n");
    printf("4. 1e-7 (<FP16 min) → FP16 → float:   ________\n");
    printf("5. Loss Scaling 恢复后:                 ________\n\n");

    printf("--- 实际结果 ---\n");
    printf("1. 正常值 3.14 → %.6f\n", results[0]);
    printf("2. 65504.0 → %g\n", results[1]);
    printf("3. 70000.0 → %g (isinf=%d)\n", results[2], isinf(results[2]));
    printf("4. 1e-7 → %g\n", results[3]);
    printf("5. Loss Scaling 恢复: %.10f\n", results[4]);
    printf("6. 5.96e-8 (略大于min) → %g\n", results[5]);
    printf("7. 5.0e-8  (略小于min) → %g\n", results[6]);

    printf("\n结论:\n");
    printf("  - FP16 max = 65504, 超过 → inf\n");
    printf("  - FP16 min ≈ 6e-8, 小于 → 0 (underflow)\n");
    printf("  - Loss Scaling: 放大→存储→恢复, 小梯度存活!\n");
    printf("  - 这就是 FP16 训练需要 GradScaler 的原因\n");

    cudaFree(d_results); cudaFree(d_count);
    return 0;
}
