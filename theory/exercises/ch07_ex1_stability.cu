// ============================================================
// Ch7 练习 1: Softmax 数值稳定性 — 有/无 "减 max" 的对比
// 配合: theory/07_classic_operators.md 练习 1
//
// 编译: nvcc -O2 -o ch07_ex1_stability ch07_ex1_stability.cu
// 运行: ./ch07_ex1_stability
//
// 直观演示: 不减 max 时 exp 溢出 → 输出全是 NaN
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>

int main() {
    const int N = 8;
    float x[8];
    srand(42);
    for (int i = 0; i < N; i++) x[i] = 100.0f + (float)(rand() % 100);

    printf("输入: ");
    for (int i = 0; i < N; i++) printf("%.1f ", x[i]);
    printf("\n\n");

    printf("=== 不减 max (不稳定) ===\n");
    float sum_bad = 0;
    for (int i = 0; i < N; i++) sum_bad += expf(x[i]);
    printf("sum(exp(x)) = %e  (溢出了!)\n", sum_bad);
    printf("softmax[0]  = %e\n\n", expf(x[0]) / sum_bad);

    printf("=== 减 max (稳定) ===\n");
    float mx = x[0];
    for (int i = 1; i < N; i++) mx = fmaxf(mx, x[i]);
    float sum_good = 0;
    for (int i = 0; i < N; i++) sum_good += expf(x[i] - mx);
    printf("max = %.1f\n", mx);
    printf("sum(exp(x - max)) = %f  (正常!)\n", sum_good);
    printf("softmax[0]  = %f\n\n", expf(x[0] - mx) / sum_good);

    printf("结论: exp(100+) ≈ 10^43 → 超出 float 范围 → Inf/NaN\n");
    printf("减 max 后: exp(x - max) ≤ 1 → 永远不溢出!\n");
    return 0;
}
