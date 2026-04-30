// ============================================================
// 练习 1: Sigmoid — 自定义 PyTorch CUDA 算子 (仅 forward)
//
// 安装: cd 04_pytorch_extension/exercises/ex1_sigmoid && pip install -e .
// 测试: python test_sigmoid.py
// 预期输出: "结果: ✓ 通过"
//
// 公式: sigmoid(x) = 1 / (1 + exp(-x))
//
// TODO:
//   1. 实现 sigmoid_forward_kernel 的函数体
//   2. C++ 封装和 pybind11 已经写好，不需要改
//
// 提示:
//   - expf(-x) 计算 e^(-x)
//   - 每个线程: output[idx] = 1.0f / (1.0f + expf(-input[idx]))
// ============================================================

#include <torch/extension.h>
#include <cuda_runtime.h>

// TODO: 实现 sigmoid forward kernel
__global__ void sigmoid_forward_kernel(const float *input, float *output, int n) {
    // --- 在这里写你的代码 ---

}

torch::Tensor sigmoid_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "输入必须在 GPU 上");
    TORCH_CHECK(input.is_contiguous(), "输入必须连续存储");

    auto output = torch::empty_like(input);
    int n = input.numel();
    int block = 256;
    int grid = (n + block - 1) / block;

    sigmoid_forward_kernel<<<grid, block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sigmoid_forward, "Sigmoid forward (CUDA)");
}
