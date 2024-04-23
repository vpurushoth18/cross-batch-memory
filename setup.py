import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CppExtension

requirements = ["torch", "torchvision"]

setup(
    name="ret_benchmark",
    version="2.0",
    author="Malong Technologies",
    url="https://github.com/MalongTech/research-xbm",
    description="xbm",
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)