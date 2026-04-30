#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ============================================================
// 01: CUDA 向量加法 — 你的第一个 GPU 并行程序
//
// 配合理论: tutorial.md Part 1
//           theory/02_cuda_programming_model.md 2.1 节 (线程层级)
//
// 这个程序做的事: c[i] = a[i] + b[i], 对 100 万个元素。
//
// 核心概念 (每个都在下面的代码中用注释详细解释):
//
//   1. Host vs Device
//      Host = CPU 和它的内存 (你用 malloc 分配的)
//      Device = GPU 和它的显存 (用 cudaMalloc 分配的)
//      它们是两块物理上分开的内存! 数据不能自动共享。
//
//   2. Kernel = 在 GPU 上执行的函数
//      用 __global__ 修饰。调用时写 kernel<<<gridSize, blockSize>>>(参数)。
//      GPU 会启动 gridSize × blockSize 个线程, 每个线程独立执行这个函数。
//
//   3. 线程编号
//      每个线程通过 blockIdx.x (我在第几个 Block) 和 threadIdx.x (我在 Block 内第几号)
//      算出自己的全局编号: idx = blockIdx.x * blockDim.x + threadIdx.x
//      然后用 idx 去处理数组的第 idx 个元素。
//
//   4. 编程流程 (5 步):
//      CPU 分配+初始化 → GPU 分配 → CPU→GPU 拷贝 → GPU 计算 → GPU→CPU 拷贝
//
//   5. 初学者常见错误 (对照检查!):
//      ✗ 忘了 cudaMemcpy → GPU 上全是垃圾数据, 输出乱七八糟
//      ✗ 去掉 if (idx < n) → 越界访问, 可能崩溃或结果错误
//      ✗ cudaMemcpy 方向写反 → 数据没搬到 GPU, kernel 读到全 0
//      ✗ 忘了 cudaFree → 显存泄漏, 多次运行后 OOM
//      ✗ 在 CPU 端解引用 GPU 指针 → 段错误 (两块内存不互通!)
// ============================================================

// ---- GPU kernel ----
// __global__ 告诉编译器: 这个函数由 CPU 调用, 在 GPU 上执行
// 返回类型必须是 void (GPU 函数不能返回值给 CPU)
__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    // 计算 "我是全局第几号线程"
    // blockIdx.x:  我在第几个 Block (0, 1, 2, ...)
    // blockDim.x:  每个 Block 有多少线程 (= blockSize, 这里是 256)
    // threadIdx.x: 我在 Block 内是第几号 (0, 1, ..., 255)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查: 总线程数 (gridSize × blockSize) 通常大于 N
    // 多出来的线程必须什么都不做, 否则会访问数组越界 → 程序崩溃!
    if (idx < n) {
        c[idx] = a[idx] + b[idx];  // 每个线程只负责 1 个元素
    }
}

// 辅助宏: 检查每个 CUDA 调用是否成功
// CUDA 函数返回 cudaError_t 类型的错误码, 必须检查!
// 否则 GPU 显存分配失败、拷贝失败等问题会被静默忽略。
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

int main() {
    const int N = 1 << 20;  // 1,048,576 个元素 (1M)
    const size_t bytes = N * sizeof(float);  // 4MB

    // ---- 步骤 1: 在 Host (CPU) 上分配并初始化数据 ----
    // h_ 前缀表示 host (CPU) 内存
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);  // 用来接收 GPU 的计算结果

    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    // ---- 步骤 2: 在 Device (GPU) 上分配显存 ----
    // d_ 前缀表示 device (GPU) 显存
    // cudaMalloc 的用法类似 malloc, 但分配的是 GPU 显存
    // 注意: d_a 是一个指针, 但它指向 GPU 显存, CPU 代码不能直接读写它!
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // ---- 步骤 3: 把数据从 Host 拷到 Device ----
    // cudaMemcpy(目标, 来源, 字节数, 方向)
    // cudaMemcpyHostToDevice = 从 CPU 内存 → GPU 显存
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // ---- 步骤 4: 启动 GPU kernel ----
    // blockSize: 每个 Block 有多少线程。256 是常用值 (必须是 32 的倍数, 最大 1024)
    // gridSize:  需要多少个 Block。向上取整确保总线程数 ≥ N
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;  // = ceil(N / blockSize)
    // gridSize = ceil(1048576 / 256) = 4096
    // 总线程数 = 4096 × 256 = 1,048,576 = N (刚好!)

    printf("启动 kernel: gridSize=%d, blockSize=%d, N=%d\n", gridSize, blockSize, N);

    // <<<gridSize, blockSize>>> 是 CUDA 的特殊语法, 指定并行度
    // 这行代码会让 GPU 启动 4096 × 256 = 100 万个线程!
    // CPU 不等 GPU 执行完就继续往下执行 (异步!)
    vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    // 检查 kernel 启动是否有错 (如 blockSize > 1024 等配置错误)
    CUDA_CHECK(cudaGetLastError());

    // ---- 步骤 5: 把结果从 Device 拷回 Host ----
    // cudaMemcpyDeviceToHost = 从 GPU 显存 → CPU 内存
    // 这个调用会等 GPU 执行完再拷贝 (隐式同步)
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // ---- 验证结果 ----
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];
        if (h_c[i] != expected) {
            printf("验证失败 @ i=%d: %.1f != %.1f\n", i, h_c[i], expected);
            ok = false;
            break;
        }
    }
    printf("结果: %s\n", ok ? "✓ 全部正确" : "✗ 有错误");

    // ---- 清理: 释放 GPU 显存和 CPU 内存 ----
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
