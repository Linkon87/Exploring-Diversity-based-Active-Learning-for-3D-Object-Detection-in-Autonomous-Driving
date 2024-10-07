# Copyright (c) Gorilla-Lab. All rights reserved.

import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_extension(name, module):
    if not torch.cuda.is_available(): return
    extersion = CUDAExtension
    return extersion(name=".".join([module, name]),
                     sources=[
                        "src/cdist.cpp",
                        "src/cdist.cu",
                     ],
                     include_dirs="./include",
                     extra_compile_args={
                         "cxx": ["-g"],
                         "nvcc": [
                             "-D__CUDA_NO_HALF_OPERATORS__",
                             "-D__CUDA_NO_HALF_CONVERSIONS__",
                             "-D__CUDA_NO_HALF2_OPERATORS__",
                         ],
                     },
                     define_macros=[("WITH_CUDA", None)])

setup(
    name="cdist",
    ext_modules=CUDAExtension(
        name="cdist_ext",
    ),
    packages=find_packages(),
    cmdclass={"build_ext": BuildExtension}
)