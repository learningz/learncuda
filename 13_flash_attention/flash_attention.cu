#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ============================================================
// 13: 简化版 FlashAttention — 理解核心算法
//
// 配合理论: theory/07_classic_operators.md 7.3 节 (FlashAttention)
//
// 标准 Attention:
//   S = Q × K^T / √d       → [N, N] 的注意力矩阵 (巨大!)
//   P = softmax(S)          → [N, N]
//   O = P × V               → [N, d] 输出
//   问题: S 和 P 是 O(N²) 的矩阵, N=4096 时 = 64MB, 必须存在 HBM!
//
// FlashAttention 的核心思想:
//   利用 Online Softmax, 将 Attention 分块计算。
//   S 和 P 的小块在 Shared Memory 中计算, 永远不写回 HBM。
//   只需要 O(N) 额外存储 (每行的 max 和 sum), 不需要 O(N²)。
//
// 本实现是教学简化版:
//   - 单头注意力 (Single Head)
//   - 不用 Tensor Core (纯 FP32 标量, 方便理解)
//   - 不做 causal mask
//   - Block 大小固定
//   重点: 让你看到 Online Softmax 如何和矩阵乘结合
//
// 新概念 (如果你从前面章节过来):
//   - Online Softmax 的"修正因子": 当 max 更新时, 之前的部分和要乘以
//     exp(old_max - new_max) 来修正 → 见 07_softmax/ 的 V2 版本
//   - 这里的关键是: 输出 O 也要乘修正因子 (不只是 sum)
//     O_new = O_old × exp(m_old - m_new) + P_new_block × V_block
//     这就是 FlashAttention 的全部秘密!
// ============================================================

#define CUDA_CHECK(call) do {                                              \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                cudaGetErrorString(err)); exit(1); }                       \
} while(0)

// 参数 (教学用, 不追求极致性能)
#define Br 32   // Q 的分块大小 (每个 Block 处理 Br 行 Q)
#define Bc 32   // K/V 的分块大小 (每步加载 Bc 行 K/V)
#define D  64   // head dimension

// ---- 简化版 FlashAttention Forward ----
// 每个 Block 处理 Br 行 Q, 循环遍历所有 K/V 的块
__global__ void flash_attention_forward(
    const float *Q,     // [N, D]
    const float *K,     // [N, D]
    const float *V,     // [N, D]
    float *O,           // [N, D] 输出
    float *L,           // [N]    logsumexp (保存给反向用)
    int N) {

    int row_start = blockIdx.x * Br;  // 本 Block 处理的 Q 行起始
    int tid = threadIdx.x;             // 线程 ID (0..Br-1)

    if (row_start + tid >= N) return;

    // ---- 每线程维护自己那一行的状态 ----
    // 这些变量存在寄存器中, 不经过 HBM!
    float m_i = -INFINITY;       // running max (数值稳定性)
    float l_i = 0.0f;            // running sum of exp(s - m)
    float o_i[D];                // running output (D 个寄存器)
    for (int d = 0; d < D; d++) o_i[d] = 0.0f;

    // 加载 Q 的一行到寄存器 (只读一次!)
    float q_i[D];
    for (int d = 0; d < D; d++) {
        q_i[d] = Q[(row_start + tid) * D + d];
    }

    // ---- 外循环: 遍历 K/V 的所有块 ----
    for (int j = 0; j < N; j += Bc) {

        // 计算 S_ij = q_i × K_j^T / √d (一行 Q 和 Bc 行 K 的点积)
        // 结果: Bc 个分数 (存在寄存器中, 不写 HBM!)
        float s[Bc];
        for (int c = 0; c < Bc && j + c < N; c++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += q_i[d] * K[(j + c) * D + d];
            }
            s[c] = dot / sqrtf((float)D);
        }

        // Online Softmax 更新
        // Step 1: 找这一块的 max
        float m_block = -INFINITY;
        for (int c = 0; c < Bc && j + c < N; c++) {
            m_block = fmaxf(m_block, s[c]);
        }

        // Step 2: 新的全局 max
        float m_new = fmaxf(m_i, m_block);

        // Step 3: 修正因子 — 这是 FlashAttention 的核心!
        // 之前算的 l_i 和 o_i 是基于 old max (m_i) 的。
        // 现在 max 变了 (m_new), 需要修正:
        //   exp(x - m_old) × exp(m_old - m_new) = exp(x - m_new) ✓
        float correction = expf(m_i - m_new);

        // Step 4: 修正之前的 sum 和 output
        l_i = l_i * correction;   // 旧的 sum 乘修正因子
        for (int d = 0; d < D; d++) {
            o_i[d] = o_i[d] * correction;  // 旧的 output 也乘修正因子!
        }

        // Step 5: 计算新块的 exp(s - m_new) 并累加到 sum 和 output
        for (int c = 0; c < Bc && j + c < N; c++) {
            float p = expf(s[c] - m_new);  // softmax 的分子 (基于 m_new)
            l_i += p;                       // 累加到 sum

            // O += p × V[j+c] — 这就是 "online 更新 output"!
            for (int d = 0; d < D; d++) {
                o_i[d] += p * V[(j + c) * D + d];
            }
        }

        m_i = m_new;  // 更新 max
    }

    // ---- 最终归一化: O = O / l ----
    for (int d = 0; d < D; d++) {
        O[(row_start + tid) * D + d] = o_i[d] / l_i;
    }

    // 保存 logsumexp (反向传播需要)
    L[row_start + tid] = m_i + logf(l_i);
}

// ---- 标准 Attention (对比用) ----
// 构建完整的 N×N 矩阵 → 巨大的内存占用!
__global__ void standard_attention(
    const float *Q, const float *K, const float *V,
    float *O, int N) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    // Step 1: 计算 S[row, :] = Q[row] · K^T
    float s[1024];  // 假设 N ≤ 1024 (教学简化)
    float m = -INFINITY;
    for (int j = 0; j < N; j++) {
        float dot = 0;
        for (int d = 0; d < D; d++) dot += Q[row*D+d] * K[j*D+d];
        s[j] = dot / sqrtf((float)D);
        m = fmaxf(m, s[j]);
    }

    // Step 2: softmax
    float sum = 0;
    for (int j = 0; j < N; j++) {
        s[j] = expf(s[j] - m);
        sum += s[j];
    }
    for (int j = 0; j < N; j++) s[j] /= sum;

    // Step 3: O[row] = P[row] · V
    for (int d = 0; d < D; d++) {
        float val = 0;
        for (int j = 0; j < N; j++) val += s[j] * V[j*D+d];
        O[row*D+d] = val;
    }
}

int main() {
    // 用多个 N 来展示 FlashAttention 的内存优势随序列长度增长
    int N_values[] = {256, 512, 1024};
    int num_tests = sizeof(N_values) / sizeof(N_values[0]);

    printf("简化版 FlashAttention (D=%d) — 性能与内存对比\n\n", D);

    for (int t = 0; t < num_tests; t++) {
        int N = N_values[t];
        size_t qkv_bytes = N * D * sizeof(float);
        size_t l_bytes = N * sizeof(float);

        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("N=%d, D=%d\n", N, D);

        float *hQ = (float*)malloc(qkv_bytes);
        float *hK = (float*)malloc(qkv_bytes);
        float *hV = (float*)malloc(qkv_bytes);
        float *hO_std = (float*)malloc(qkv_bytes);
        float *hO_flash = (float*)malloc(qkv_bytes);
        srand(42);
        for (int i = 0; i < N*D; i++) {
            hQ[i] = (rand() % 200 - 100) / 100.0f;
            hK[i] = (rand() % 200 - 100) / 100.0f;
            hV[i] = (rand() % 200 - 100) / 100.0f;
        }

        float *dQ, *dK, *dV, *dO, *dL;
        CUDA_CHECK(cudaMalloc(&dQ, qkv_bytes));
        CUDA_CHECK(cudaMalloc(&dK, qkv_bytes));
        CUDA_CHECK(cudaMalloc(&dV, qkv_bytes));
        CUDA_CHECK(cudaMalloc(&dO, qkv_bytes));
        CUDA_CHECK(cudaMalloc(&dL, l_bytes));
        CUDA_CHECK(cudaMemcpy(dQ, hQ, qkv_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dK, hK, qkv_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dV, hV, qkv_bytes, cudaMemcpyHostToDevice));

        // 标准 Attention — 正确性 + 计时
        float ms_std = 0;
        {
            standard_attention<<<(N+255)/256, 256>>>(dQ, dK, dV, dO, N);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(hO_std, dO, qkv_bytes, cudaMemcpyDeviceToHost));

            cudaEvent_t t0, t1;
            CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
            CUDA_CHECK(cudaEventRecord(t0));
            int iters = (N <= 512) ? 100 : 20;
            for (int r = 0; r < iters; r++)
                standard_attention<<<(N+255)/256, 256>>>(dQ, dK, dV, dO, N);
            CUDA_CHECK(cudaEventRecord(t1));
            CUDA_CHECK(cudaEventSynchronize(t1));
            CUDA_CHECK(cudaEventElapsedTime(&ms_std, t0, t1));
            ms_std /= iters;
            CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
        }

        // Flash Attention — 正确性 + 计时
        float ms_flash = 0;
        {
            flash_attention_forward<<<(N+Br-1)/Br, Br>>>(dQ, dK, dV, dO, dL, N);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(hO_flash, dO, qkv_bytes, cudaMemcpyDeviceToHost));

            cudaEvent_t t0, t1;
            CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
            CUDA_CHECK(cudaEventRecord(t0));
            int iters = (N <= 512) ? 200 : 50;
            for (int r = 0; r < iters; r++)
                flash_attention_forward<<<(N+Br-1)/Br, Br>>>(dQ, dK, dV, dO, dL, N);
            CUDA_CHECK(cudaEventRecord(t1));
            CUDA_CHECK(cudaEventSynchronize(t1));
            CUDA_CHECK(cudaEventElapsedTime(&ms_flash, t0, t1));
            ms_flash /= iters;
            CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
        }

        // 验证
        float max_err = 0;
        for (int i = 0; i < N * D; i++)
            max_err = fmaxf(max_err, fabsf(hO_std[i] - hO_flash[i]));

        // 内存
        float mem_std_mb = (float)N * N * sizeof(float) / 1e6;
        float mem_flash_kb = (float)N * sizeof(float) / 1e3;

        printf("  正确性:   max|std - flash| = %.2e  %s\n",
               max_err, max_err < 1e-4 ? "✓" : "✗");
        printf("  耗时:     标准=%.4f ms,  Flash=%.4f ms,  加速比=%.2f×\n",
               ms_std, ms_flash, ms_std / ms_flash);
        printf("  内存:     标准 S/P 矩阵 = %.1f MB,  Flash O(1) 状态 = %.1f KB\n",
               mem_std_mb, mem_flash_kb);
        printf("  内存节省: %.0f×\n", (float)N * N * 4 / (N * 4));
        printf("\n");

        CUDA_CHECK(cudaFree(dQ)); CUDA_CHECK(cudaFree(dK));
        CUDA_CHECK(cudaFree(dV)); CUDA_CHECK(cudaFree(dO)); CUDA_CHECK(cudaFree(dL));
        free(hQ); free(hK); free(hV); free(hO_std); free(hO_flash);
    }

    printf("核心发现:\n");
    printf("  1. FlashAttention 永远不构建 N×N 的 S 矩阵 → O(N) 内存\n");
    printf("  2. N=1024 时标准 Attention 需要 4MB S/P, N=8192 时需要 256MB!\n");
    printf("  3. FlashAttention 内存与 N 线性增长, 可以处理长序列\n");
    printf("  4. Online Softmax 修正因子: exp(old_max - new_max) 同时修正 sum 和 O\n");

    return 0;
}
