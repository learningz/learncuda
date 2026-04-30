// ============================================================
// Ch5 练习 3: 判断瓶颈类型 — 自动计算 AI 并和 Ridge Point 对比
// 配合: theory/05_operator_development.md 练习 3
//
// 编译: nvcc -O2 -o ch05_ex3_bottleneck ch05_ex3_bottleneck.cu
// 运行: ./ch05_ex3_bottleneck
//
// 本程序自动输出三个 kernel 的算术强度, 用户对比自己 GPU 的 Ridge Point。
// ============================================================

#include <cstdio>

int main() {
    printf("瓶颈类型判断练习\n\n");

    printf("(a) 向量加法: c[i] = a[i] + b[i]\n");
    printf("    FLOP/element = 1\n");
    printf("    Bytes/element = 12 (read 2 × 4B + write 1 × 4B)\n");
    printf("    AI = 1 / 12 = %.4f FLOP/Byte\n\n", 1.0/12);

    printf("(b) GELU: y = 0.5*x*(1 + tanh(...))\n");
    printf("    FLOP/element ≈ 15\n");
    printf("    Bytes/element = 8 (read 4B + write 4B)\n");
    printf("    AI = 15 / 8 = %.4f FLOP/Byte\n\n", 15.0/8);

    printf("(c) GEMM 1024×1024: C = A × B\n");
    printf("    FLOP = 2 × 1024³ = %.0f\n", 2.0*1024*1024*1024);
    printf("    Bytes = 3 × 1024² × 4 = %.0f\n", 3.0*1024*1024*4);
    printf("    AI = %.1f FLOP/Byte\n\n", 2.0*1024*1024*1024/(3.0*1024*1024*4));

    printf("Ridge Point 参考 (峰值算力 / 峰值带宽):\n");
    printf("  A100:  19.5 TFLOPS / 2.0 TB/s = 9.75 FLOP/Byte\n");
    printf("  V100:  15.7 TFLOPS / 0.9 TB/s = 17.4 FLOP/Byte\n");
    printf("  T4:     8.1 TFLOPS / 0.3 TB/s = 27.0 FLOP/Byte\n\n");

    printf("结论:\n");
    printf("  (a) AI=0.08 << Ridge Point → Memory Bound\n");
    printf("  (b) AI=1.88 < Ridge Point  → Memory Bound\n");
    printf("  (c) AI=170  >> Ridge Point → Compute Bound\n");
    return 0;
}
