# 10 分钟快速上手 CUDA

这是整个教程的"最短路径"——10 分钟写个 CUDA 程序跑起来，感受一下到底是什么感觉。

> 完整教程：[`tutorial.md`](./tutorial.md)（3-7 天）
> 学习路线：[`LEARNING_PATH.md`](./LEARNING_PATH.md)

---

## 1. 环境检查（1 分钟）

```bash
nvidia-smi        # 确认 GPU 正常（能看到型号和驱动版本）
nvcc --version    # 确认 CUDA 编译器正常（能看到版本号）
```

两个命令都不报错就可以继续。如果报错，参考 [`tutorial.md Part 0`](./tutorial.md#part-0-环境准备--5-分钟检查清单) 的安装说明。

---

## 2. 第一个 CUDA 程序（3 分钟）

**新建 `hello.cu`**：

```cuda
#include <cstdio>

// 在 GPU 上执行的函数 (__global__ = CPU 调用, GPU 执行)
__global__ void hello_kernel() {
    // threadIdx.x = 线程在 Block 内的编号
    // blockIdx.x  = Block 在 Grid 内的编号
    // blockDim.x  = 每个 Block 有多少线程
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from thread %d\n", tid);
}

int main() {
    // 启动 4 个 Block, 每个 8 个线程 = 32 个总线程
    hello_kernel<<<4, 8>>>();
    cudaDeviceSynchronize();  // 等 GPU 执行完
    return 0;
}
```

**编译运行**：

```bash
nvcc -O2 -o hello hello.cu && ./hello
```

你会看到 32 行输出，每行来自一个不同的线程。

---

## 3. 真正干活：向量加法（5 分钟）

CUDA 强在大量并行计算。下面让 1000 万个线程同时算加法：

```cuda
#include <cstdio>

// 每个线程处理一个元素
__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 全局线程编号
    if (i < n) c[i] = a[i] + b[i];  // 边界检查，多余的线程不干活
}

int main() {
    int n = 10000000;  // 1000 万元素
    size_t bytes = n * sizeof(float);

    // 申请 CPU 内存
    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c = new float[n];
    for (int i = 0; i < n; i++) { h_a[i] = i * 1.0f; h_b[i] = i * 2.0f; }

    // 申请 GPU 显存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // CPU→GPU 拷贝
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Launch kernel: 39100 个 Block × 256 线程 ≈ 1000 万
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    // GPU→CPU 拷贝
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // 验证
    float max_err = 0;
    for (int i = 0; i < n; i++)
        max_err = max(max_err, h_a[i] + h_b[i] - h_c[i]);
    printf("最大误差: %f  %s\n", max_err, max_err < 1e-5 ? "✓" : "✗");

    // 清理
    delete[] h_a; delete[] h_b; delete[] h_c;
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
```

**编译运行**：

```bash
nvcc -O2 -o vector_add vector_add.cu && ./vector_add
```

### 这段代码教会你三件事

```
1. __ global__  = 函数在 GPU 上执行
2. <<<grid, block>>>  = 启动 grid×block 个线程
3. cudaMalloc / cudaMemcpy = GPU 有自己的显存，需要手动搬数据
```

**到此你已写出了真正的 CUDA 程序。**

---

## 4. 然后学什么

| 方向 | 路径 |
|------|------|
| 完整学习 | [`tutorial.md`](./tutorial.md) — 从零到 LayerNorm |
| 理论深度 | [`theory/`](./theory/) — 硬件架构到高级优化 |
| 综合项目 | [`12_layernorm_project/`](./12_layernorm_project/) — 手写 LayerNorm + PyTorch |
| 面试准备 | [`INTERVIEW_QUESTIONS.md`](./INTERVIEW_QUESTIONS.md) — 各章面试题 |
| 调试急救 | [`DEBUG_AND_OPTIMIZE.md`](./DEBUG_AND_OPTIMIZE.md) — 常见错误 + ncu |
