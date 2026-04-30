#include <torch/extension.h>
#include <cuda_runtime.h>

// ============================================================
// 04: PyTorch 自定义 CUDA 算子 — 从 kernel 到框架集成
//
// 配合理论: theory/05_operator_development.md 5.8 节 (PyTorch 接入)
//           tutorial.md (前 3 个示例让你会写 kernel)
//
// 这个文件展示了一个完整的自定义算子:
//   1. CUDA kernel (GPU 上的计算核心)
//   2. C++ 封装 (参数检查 + kernel 调用)
//   3. pybind11 导出 (暴露给 Python)
//
// 为什么需要自定义算子?
//   PyTorch 内置了大多数常用算子, 但你可能需要:
//   - 融合多个操作到一个 kernel (减少内存访问, 见 10_fused_kernel/)
//   - 实现 PyTorch 没有的新算子 (如自定义注意力变体)
//   - 对特定场景做极致优化 (如量化推理)
//
// 工程结构:
//   gelu_cuda.cu  ← 你正在看的文件 (kernel + C++ 封装 + pybind11)
//   setup.py      ← 编译配置 (告诉 pip 怎么编译这个 .cu 文件)
//   test_gelu.py  ← Python 测试 (autograd.Function + 正确性/性能验证)
//
// 安装: cd 04_pytorch_extension && pip install -e .
// 测试: python test_gelu.py
// ============================================================


// ---- CUDA kernel: GELU 前向 ----
// GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
// 这是 tanh 近似版本, 和 PyTorch 的 F.gelu() 默认实现一致。
//
// 为什么要手写? 可以和其他操作融合 (如 GELU+Dropout), 或做向量化优化。
// 实际中 GELU 是 Memory Bound (AI ≈ 1.9 FLOP/Byte < Ridge Point),
// 所以优化方向是减少内存访问 (融合) 而不是减少计算。
__global__ void gelu_forward_kernel(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        // 0.7978845608 = √(2/π)
        // tanh 近似公式比精确的 erf 快, 且误差 < 0.001
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        output[idx] = x * cdf;
    }
}

// ---- CUDA kernel: GELU 反向 ----
// 反向传播: 给定 grad_output (loss 对 y 的梯度), 求 grad_input (loss 对 x 的梯度)
// 链式法则: grad_input = grad_output × dGELU/dx
//
// dGELU/dx = cdf + x × pdf
//   其中 cdf = 0.5(1 + tanh(inner))
//        pdf = 0.5(1 - tanh²(inner)) × √(2/π) × (1 + 3×0.044715×x²)
//        inner = √(2/π) × (x + 0.044715x³)
//
// 注意: 反向需要原始输入 x (不是 y), 所以前向时要 save_for_backward(input)
__global__ void gelu_backward_kernel(const float *grad_output, const float *input,
                                     float *grad_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        float tanh_val = tanhf(inner);
        float cdf = 0.5f * (1.0f + tanh_val);
        // (1 - tanh²) 是 tanh 的导数
        float pdf = 0.5f * (1.0f - tanh_val * tanh_val) *
                    0.7978845608f * (1.0f + 3.0f * 0.044715f * x * x);
        // 链式法则: grad_input = grad_output × (cdf + x × pdf)
        grad_input[idx] = grad_output[idx] * (cdf + x * pdf);
    }
}


// ---- C++ 封装: 参数检查 + kernel 调用 ----
// PyTorch 的 tensor 可能在 CPU 上、可能不连续、可能是 double 不是 float...
// 这一层负责检查所有这些前提条件, 然后调用 CUDA kernel。

torch::Tensor gelu_forward(torch::Tensor input) {
    // TORCH_CHECK: 如果条件不满足, 抛出 Python 可以捕获的异常 (不是 segfault!)
    TORCH_CHECK(input.is_cuda(), "输入 tensor 必须在 GPU 上 (用 .cuda() 转换)");
    TORCH_CHECK(input.is_contiguous(), "输入 tensor 必须是连续存储的 (用 .contiguous())");
    // 注意: 这里只支持 float32。生产代码需要用 AT_DISPATCH_FLOATING_TYPES 支持多类型。

    auto output = torch::empty_like(input);  // 分配和 input 相同 shape/dtype 的输出
    int n = input.numel();                    // 总元素数 (自动处理任意 shape)
    int block = 256;
    int grid = (n + block - 1) / block;

    // 调用 kernel。input.data_ptr<float>() 获取底层的 float* 指针。
    gelu_forward_kernel<<<grid, block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);

    return output;
}

torch::Tensor gelu_backward(torch::Tensor grad_output, torch::Tensor input) {
    TORCH_CHECK(grad_output.is_cuda() && input.is_cuda(), "输入必须在 GPU 上");

    auto grad_input = torch::empty_like(input);
    int n = input.numel();
    int block = 256;
    int grid = (n + block - 1) / block;

    gelu_backward_kernel<<<grid, block>>>(
        grad_output.data_ptr<float>(), input.data_ptr<float>(),
        grad_input.data_ptr<float>(), n);

    return grad_input;
}


// ---- pybind11: 暴露给 Python ----
// TORCH_EXTENSION_NAME 是 setup.py 中 CUDAExtension 的 name 参数
// m.def("函数名", &C++函数指针, "描述") → Python 中用 custom_gelu.forward(...) 调用
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU forward (CUDA)");
    m.def("backward", &gelu_backward, "GELU backward (CUDA)");
}
