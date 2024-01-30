#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "graph_utils.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("cal_overlap_xy", &cal_overlap_xy, "Construct adjacency matrix using CUDA");
	m.def("cal_overlap_dis", &cal_overlap_dis, "Construct adjacency matrix using CUDA");
	m.def("cal_overlap", &cal_overlap, "Construct adjacency matrix using CUDA");
	m.def("mapping_edgeidx", &mapping_edgeidx, "Mapping idxs in edge set using CUDA");
	m.def("find_relation", &find_relation, "Construct adjacency matrix using CUDA");
	
}
