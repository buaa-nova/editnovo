# setup.py（放在 attennovo/ 根目录下）
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension

# 1. 定义要编译的扩展列表
ext_modules = [
    # CPU 版本
    CppExtension(
        name="editnovo.libnat",                        # 安装后通过 `import editnovo.libnat` 使用
        sources=[
            "editnovo/clib/libnat/edit_dist.cpp",     # 相对路径指向你的 C++ 源文件
        ],
        include_dirs=[],                               # 如果有额外头文件目录，可以在这里添加
    )
]
print(os.environ.get("CUDA_HOME", None))
# 2. 如果检测到 CUDA 环境，则再加上 GPU 加速版本
if os.environ.get("CUDA_HOME", None):
    ext_modules.append(
        CppExtension(
            name="editnovo.libnat_cuda",                          # 安装后通过 `import editnovo.libnat_cuda` 使用
            sources=[
                "editnovo/clib/libnat_cuda/edit_dist.cu",        # CUDA 核函数
                "editnovo/clib/libnat_cuda/binding.cpp",         # Python 绑定入口
            ],
            include_dirs=[],                                     # CUDA 头文件路径，若需要可添加
            extra_compile_args={                                 # 给 nvcc 的额外编译参数
                "nvcc": ["-O2", "--use_fast_math"]
            },
        )
    )

# 3. 调用 setuptools.setup
setup(
    name="attennovo",                # 你的项目名字
    version="0.1.0",
    packages=find_packages(),        # 自动发现 editnovo/ 及子包
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
