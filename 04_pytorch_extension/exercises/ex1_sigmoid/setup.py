from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="custom_sigmoid",
    ext_modules=[
        CUDAExtension(
            "custom_sigmoid",
            ["sigmoid_cuda.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
