#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ============================================================
// 调试练习 1: 向量加法 — 结果全是 0 或垃圾值
//
// 这个程序做 c[i] = a[i] + b[i], 但输出结果全是 0。
// 程序不会崩溃, 编译没有警告, 但结果就是不对。
//
// 你的任务: 找到 bug 并修复它, 让输出显示 "✓ 全部正确"。
//
// 提示: 仔细检查每一步数据流向。数据真的到达 GPU 了吗?
// ============================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // ====== BUG IS HERE ======
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyDeviceToHost));
    // =========================

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];
        if (h_c[i] != expected) {
            if (errors < 5)
                printf("错误 @ i=%d: 得到 %.1f, 期望 %.1f\n", i, h_c[i], expected);
            errors++;
        }
    }
    printf("结果: %s (共 %d 个错误)\n",
           errors == 0 ? "✓ 全部正确" : "✗ 有错误", errors);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}

// ============================================================
// BUG ANSWER (先自己找!):
//
// 第 51-52 行: cudaMemcpy 的方向写反了!
//   cudaMemcpyDeviceToHost 应该改为 cudaMemcpyHostToDevice
//
//   写反的后果: 这个调用试图从 GPU 显存拷到 CPU 内存
//   (覆盖了 h_a/h_b), 而 GPU 上的 d_a/d_b 仍然是
//   cudaMalloc 后的未初始化垃圾值。
//   kernel 读到垃圾值做加法, 结果当然不对。
//
//   更隐蔽的是: cudaMemcpy 不会报错! 方向参数是枚举值,
//   两个方向都是合法操作, CUDA 不知道你"想"往哪个方向拷。
//
// 修复: 把 cudaMemcpyDeviceToHost 改成 cudaMemcpyHostToDevice
// ============================================================
