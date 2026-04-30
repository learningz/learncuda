"""
练习 1: 自定义 Sigmoid CUDA 算子测试

安装:
    cd 04_pytorch_extension/exercises/ex1_sigmoid
    pip install -e .

运行: python test_sigmoid.py
预期: "结果: ✓ 通过"
"""
import torch
import custom_sigmoid


def main():
    torch.manual_seed(42)
    x = torch.randn(1024, 1024, device="cuda")

    # 你的自定义实现
    y_custom = custom_sigmoid.forward(x)

    # PyTorch 内置 sigmoid 作为参考
    y_ref = torch.sigmoid(x)

    max_diff = (y_custom - y_ref).abs().max().item()
    print(f"前向最大误差: {max_diff:.2e}")
    print(f"结果: {'✓ 通过' if max_diff < 1e-5 else '✗ 失败'}")


if __name__ == "__main__":
    main()
