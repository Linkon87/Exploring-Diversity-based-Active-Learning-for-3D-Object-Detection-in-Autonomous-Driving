# Copyright (c) Gorilla-Lab. All rights reserved.
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="cdist",
    ext_modules=[
        CUDAExtension("cdist", [
            "src/cdist.cu",
            "src/cdist.cpp",
            # "src/bindings.cpp",
        ],
        extra_compile_args={"cxx": ["-g"],
                            "nvcc": ["-O2"]})
    ],
    cmdclass={"build_ext": BuildExtension}
)

