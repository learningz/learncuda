#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ============================================================
// 02: 矩阵乘法 — 理解 Shared Memory 和 Tiling
//
// 配合理论: tutorial.md Part 2
//           theory/03_memory_hierarchy.md 3.3 节 (Shared Memory)
//           theory/05_operator_development.md 5.5 节 (GEMM 优化)
//
// 矩阵乘法 C = A × B, 其中:
//   A: M×K 矩阵
//   B: K×N 矩阵
//   C: M×N 矩阵
//   C[i][j] = Σ(k=0..K-1) A[i][k] × B[k][j]
//
// 本程序包含两个版本并对比性能:
//
//   朴素版: 每个线程独立计算 C 的一个元素。
//     问题: 每个线程需要读 A 的一行 (K 个元素) + B 的一列 (K 个元素)。
//     同一个 Block 内相邻的线程需要读 A 的相同行 → 从全局显存重复读!
//     全局显存很慢 (~500 cycles), 这些重复读取严重浪费带宽。
//
//   Tiled版 (分块): 利用 Shared Memory 消除重复读取。
//     思路: 将 A 和 B 各切成 TILE_SIZE×TILE_SIZE 的小块。
//     每一步, Block 内的所有线程协作将一小块 A 和 B 从全局显存搬到 Shared Memory。
//     然后从 Shared Memory (快 ~100×) 中读取数据做计算。
//     同一块数据被 Block 内的多个线程共用 → 全局显存读取量减少 TILE_SIZE 倍!
//
// 新概念:
//   dim3: CUDA 的三维向量类型, 用于指定 Grid 和 Block 的多维大小。
//         矩阵乘法用 2D Block: dim3(TILE_SIZE, TILE_SIZE) = 16×16 = 256 个线程
//         threadIdx.x 对应列方向, threadIdx.y 对应行方向
//
//   __syncthreads(): Block 内的线程同步屏障。
//         确保所有线程都完成了 Shared Memory 的写入后, 再开始读取。
//         如果不同步 → 读到其他线程还没写完的数据 → 结果错误!
//
//   初学者常见错误:
//      ✗ 只放 1 个 __syncthreads → 需要 2 个! (搬运前等 + 计算后等)
//      ✗ threadIdx.x/y 弄反 → 合并访问变成跨步访问 → 慢 10 倍
//      ✗ As/Bs 的索引写错 → 结果不对但不崩溃, 很难排查
//      ✗ 忘了边界检查 → 矩阵不是 TILE_SIZE 的倍数时越界
//      ✗ TILE_SIZE 取太大 → Shared Memory 超限 → kernel launch 失败
// ============================================================

#define TILE_SIZE 16

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---- 朴素版本 ----
// 每个线程计算 C[row][col] 的一个元素
// 使用 2D Grid + 2D Block:
//   blockIdx.y, threadIdx.y → 行方向
//   blockIdx.x, threadIdx.x → 列方向
//   (为什么 x 对应列? 因为同一 Warp 的 32 个线程 threadIdx.x 连续,
//    让它们读 B 矩阵的连续列地址 → 合并访问! 见 Ch3.4)
__global__ void matmul_naive(const float *A, const float *B, float *C,
                             int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // 每个线程独立读 A[row][k] 和 B[k][col]
            // 问题: 相邻线程 (threadIdx.x 不同但 threadIdx.y 相同) 读同一行 A
            //        → 重复从全局显存读取! 浪费带宽。
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ---- Shared Memory 分块版本 (Tiled) ----
// 核心思想: 将全局显存的重复读取替换为 Shared Memory 的快速读取
//
// 算法:
//   沿 K 维度分成多个 TILE_SIZE 大小的块
//   每一步:
//     1. Block 内所有线程协作将 A 的一小块 和 B 的一小块 搬到 Shared Memory
//     2. __syncthreads() — 确保搬完了
//     3. 从 Shared Memory 读数据做矩阵乘 (TILE_SIZE 次乘加)
//     4. __syncthreads() — 确保大家都算完了, 再搬下一块
__global__ void matmul_tiled(const float *A, const float *B, float *C,
                             int M, int K, int N) {
    // As, Bs: Shared Memory 中的两个 TILE×TILE 缓冲区
    // __shared__ 声明: 这块存储在 SM 的片上 SRAM 中, Block 内所有线程可读写
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    // 沿 K 维度循环, 每次处理 TILE_SIZE 个 K
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 步骤 1: 每个线程搬运一个元素到 Shared Memory
        // 线程 (threadIdx.y, threadIdx.x) 负责 As[threadIdx.y][threadIdx.x]
        // 和 Bs[threadIdx.y][threadIdx.x]
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        // 边界检查: 超出矩阵范围的位置填 0
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        // 步骤 2: 同步! 等所有线程都写完 Shared Memory
        __syncthreads();

        // 步骤 3: 从 Shared Memory (快!) 读数据做 TILE_SIZE 次乘加
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        // 步骤 4: 再次同步, 确保大家都算完了, 再覆盖 Shared Memory
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ---- CPU 参考实现 (用于验证正确性) ----
void cpu_matmul(const float *A, const float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < K; k++) s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

bool verify(const float *ref, const float *test, int n, float tol = 1e-3f) {
    for (int i = 0; i < n; i++) {
        float diff = fabsf(ref[i] - test[i]);
        if (diff > tol) {
            printf("  不匹配 @ %d: ref=%.4f test=%.4f diff=%.4f\n", i, ref[i], test[i], diff);
            return false;
        }
    }
    return true;
}

int main() {
    const int M = 512, K = 512, N = 512;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float *h_A = (float *)malloc(sizeA);
    float *h_B = (float *)malloc(sizeB);
    float *h_C_naive = (float *)malloc(sizeC);
    float *h_C_tiled = (float *)malloc(sizeC);
    float *h_C_ref   = (float *)malloc(sizeC);

    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 100) / 100.0f;

    printf("矩阵乘法: C[%d×%d] = A[%d×%d] × B[%d×%d]\n\n", M, N, M, K, K, N);

    cpu_matmul(h_A, h_B, h_C_ref, M, K, N);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // 2D Block: TILE_SIZE × TILE_SIZE 个线程 (16×16 = 256)
    dim3 block(TILE_SIZE, TILE_SIZE);
    // 2D Grid: 覆盖整个 C 矩阵
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, sizeC, cudaMemcpyDeviceToHost));

    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaMemcpy(h_C_tiled, d_C, sizeC, cudaMemcpyDeviceToHost));

    printf("正确性:\n");
    printf("  朴素版本: %s\n", verify(h_C_ref, h_C_naive, M * N) ? "✓" : "✗");
    printf("  Tiled版本: %s\n", verify(h_C_ref, h_C_tiled, M * N) ? "✓" : "✗");

    // ---- 性能对比 ----
    // cudaEvent 是 GPU 端的计时器, 比 CPU 的 clock() 更精确 (精度 ~0.5μs)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_naive;
    cudaEventElapsedTime(&ms_naive, start, stop);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_tiled;
    cudaEventElapsedTime(&ms_tiled, start, stop);

    printf("\n性能 (100次平均):\n");
    printf("  朴素版本: %.3f ms\n", ms_naive / 100);
    printf("  Tiled版本: %.3f ms\n", ms_tiled / 100);
    printf("  加速比: %.2fx\n", ms_naive / ms_tiled);

    // 计算达到的 FLOPS
    double flops = 2.0 * M * K * N;  // 每个 C 元素: K 次乘 + K 次加 = 2K FLOP
    printf("\n  朴素 GFLOPS: %.1f\n", flops / (ms_naive / 100 * 1e6));
    printf("  Tiled GFLOPS: %.1f\n", flops / (ms_tiled / 100 * 1e6));
    printf("\n为什么 Tiled 版快?\n");
    printf("  朴素版: 每个线程从全局显存读 %d 次 A + %d 次 B = %d 次慢读\n", K, K, 2*K);
    printf("  Tiled版: 每 TILE (%d次) 只从全局显存读 1 次, 剩下从 Shared Memory 读\n", TILE_SIZE);
    printf("  全局显存读取减少 ~%dx → 这就是加速的来源!\n", TILE_SIZE);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_naive); free(h_C_tiled); free(h_C_ref);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
