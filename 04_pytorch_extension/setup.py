from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="custom_gelu",
    ext_modules=[
        CUDAExtension(
            "custom_gelu",
            ["gelu_cuda.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
