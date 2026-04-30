// ============================================================
// 练习 2: LeakyReLU — 自定义 PyTorch CUDA 算子 (forward + backward)
//
// 安装: cd 04_pytorch_extension/exercises/ex2_leaky_relu && pip install -e .
// 测试: python test_leaky_relu.py
// 预期输出: "结果: ✓ 通过"
//
// 公式:
//   forward:  y = x > 0 ? x : alpha * x
//   backward: grad_input = grad_output * (x > 0 ? 1 : alpha)
//
// TODO:
//   1. 实现 leaky_relu_forward_kernel 的函数体
//   2. 实现 leaky_relu_backward_kernel 的函数体
//   C++ 封装和 pybind11 已写好。
//
// 提示:
//   - forward: 一行代码: output[idx] = (x > 0) ? x : alpha * x
//   - backward: grad_input[idx] = grad_output[idx] * ((x > 0) ? 1.0f : alpha)
//   - alpha 是标量参数, 按值传给 kernel (和 SAXPY 一样)
//   - 反向需要原始输入 x, 所以 Python 端用 save_for_backward(input)
// ============================================================

#include <torch/extension.h>
#include <cuda_runtime.h>

// TODO: 实现 forward kernel
__global__ void leaky_relu_forward_kernel(const float *input, float *output,
                                          float alpha, int n) {
    // --- 在这里写你的代码 ---

}

// TODO: 实现 backward kernel
__global__ void leaky_relu_backward_kernel(const float *grad_output,
                                           const float *input,
                                           float *grad_input,
                                           float alpha, int n) {
    // --- 在这里写你的代码 ---

}

torch::Tensor leaky_relu_forward(torch::Tensor input, float alpha) {
    TORCH_CHECK(input.is_cuda(), "输入必须在 GPU 上");
    TORCH_CHECK(input.is_contiguous(), "输入必须连续存储");

    auto output = torch::empty_like(input);
    int n = input.numel();
    int block = 256;
    int grid = (n + block - 1) / block;

    leaky_relu_forward_kernel<<<grid, block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), alpha, n);

    return output;
}

torch::Tensor leaky_relu_backward(torch::Tensor grad_output,
                                   torch::Tensor input, float alpha) {
    TORCH_CHECK(grad_output.is_cuda() && input.is_cuda(), "输入必须在 GPU 上");

    auto grad_input = torch::empty_like(input);
    int n = input.numel();
    int block = 256;
    int grid = (n + block - 1) / block;

    leaky_relu_backward_kernel<<<grid, block>>>(
        grad_output.data_ptr<float>(), input.data_ptr<float>(),
        grad_input.data_ptr<float>(), alpha, n);

    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward (CUDA)");
    m.def("backward", &leaky_relu_backward, "LeakyReLU backward (CUDA)");
}
