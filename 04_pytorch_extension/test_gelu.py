"""
第4课: 使用自定义 CUDA GELU 算子

安装:
    cd 04_pytorch_extension
    pip install -e .

使用: python test_gelu.py
"""
import torch
import custom_gelu


class CustomGELU(torch.autograd.Function):
    """用 autograd.Function 封装，使其支持反向传播"""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return custom_gelu.forward(input)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return custom_gelu.backward(grad_output.contiguous(), input)


def main():
    torch.manual_seed(42)
    x = torch.randn(1024, 1024, device="cuda", requires_grad=True)

    # 自定义 CUDA 算子
    y_custom = CustomGELU.apply(x)

    # PyTorch 内置 GELU 作为参考
    y_ref = torch.nn.functional.gelu(x)

    # 对比前向
    max_diff = (y_custom - y_ref).abs().max().item()
    print(f"前向最大误差: {max_diff:.2e}")

    # 对比反向
    loss_custom = y_custom.sum()
    loss_custom.backward()
    grad_custom = x.grad.clone()

    x.grad = None
    y_ref2 = torch.nn.functional.gelu(x)
    y_ref2.sum().backward()
    grad_ref = x.grad

    grad_diff = (grad_custom - grad_ref).abs().max().item()
    print(f"反向最大误差: {grad_diff:.2e}")
    print(f"结果: {'✓ 通过' if max_diff < 1e-3 and grad_diff < 1e-3 else '✗ 失败'}")


if __name__ == "__main__":
    main()
