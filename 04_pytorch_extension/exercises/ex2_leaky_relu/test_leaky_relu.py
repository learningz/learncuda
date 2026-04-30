"""
练习 2: 自定义 LeakyReLU CUDA 算子测试 (forward + backward)

安装:
    cd 04_pytorch_extension/exercises/ex2_leaky_relu
    pip install -e .

运行: python test_leaky_relu.py
预期: "结果: ✓ 通过"
"""
import torch
import custom_leaky_relu


class CustomLeakyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return custom_leaky_relu.forward(input, alpha)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return custom_leaky_relu.backward(grad_output.contiguous(), input, ctx.alpha), None


def main():
    torch.manual_seed(42)
    alpha = 0.01
    x = torch.randn(1024, 1024, device="cuda", requires_grad=True)

    y_custom = CustomLeakyReLU.apply(x, alpha)
    y_ref = torch.nn.functional.leaky_relu(x, negative_slope=alpha)

    max_diff_fwd = (y_custom - y_ref).abs().max().item()
    print(f"前向最大误差: {max_diff_fwd:.2e}")

    loss_custom = y_custom.sum()
    loss_custom.backward()
    grad_custom = x.grad.clone()

    x.grad = None
    y_ref2 = torch.nn.functional.leaky_relu(x, negative_slope=alpha)
    y_ref2.sum().backward()
    grad_ref = x.grad

    max_diff_bwd = (grad_custom - grad_ref).abs().max().item()
    print(f"反向最大误差: {max_diff_bwd:.2e}")

    ok = max_diff_fwd < 1e-5 and max_diff_bwd < 1e-5
    print(f"结果: {'✓ 通过' if ok else '✗ 失败'}")


if __name__ == "__main__":
    main()
