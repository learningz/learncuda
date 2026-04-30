from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="custom_leaky_relu",
    ext_modules=[
        CUDAExtension(
            "custom_leaky_relu",
            ["leaky_relu_cuda.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
