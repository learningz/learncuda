# ============================================================
# 12: LayerNorm PyTorch 扩展 — 编译配置
#
# 使用方法:
#   cd 12_layernorm_project
#   pip install -e .
#
# 这会调用 nvcc 编译 layernorm_cuda.cu，生成 Python 可导入的 .so 文件。
# 编译完成后，Python 中可以 import custom_layernorm。
#
# 编译链路详解见: 04_pytorch_extension/pytorch_extension.md
# ============================================================

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="custom_layernorm",
    ext_modules=[
        CUDAExtension(
            # 模块名 — import custom_layernorm 时用这个名字
            # 必须和 layernorm_cuda.cu 中 PYBIND11_MODULE 的第一个参数一致
            "custom_layernorm",
            # 源文件列表 — 这里只有一个 .cu 文件（同时包含 kernel 和 C++ binding）
            ["layernorm_cuda.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
