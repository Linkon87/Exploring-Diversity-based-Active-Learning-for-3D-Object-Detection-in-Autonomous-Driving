# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob

from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_ext_sources = glob.glob("src/*.cpp") + glob.glob("src/*.cu")

setup(
    name="cdist",
    ext_modules=[
        CUDAExtension(
            name=".cdist_ext",
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": ["-O2"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
