import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext

# setup(
#     name='graph utils',
#     cmdclass={'build_ext': BuildExtension},
#     ext_modules=[make_cuda_ext(
#                 name='index',
#                 module='',
#                 sources=[
#                     'src/index.cu'
#                 ]),
#         ]

# )

setup(
    name='graph_utils',
    cmdclass={'build_ext': BuildExtension},
    ext_modules=[
        make_cuda_ext(
        name='graph_utils_cuda',
        module='opencood.graph_utils',
        sources=[
            'src/find_relation.cu',
            'src/mapping_edgeidx.cu',
            'src/cal_overlap.cu',
            'src/graph_utils_api.cpp',
            'src/cal_overlap_dis.cu',
            'src/cal_overlap_xy.cu',
        ])
        ]
)