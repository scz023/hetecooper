// Assuming the necessary includes and definitions for the CUDA environment
#include <torch/torch.h>

__global__ void kernel(int num_edge, int num_node, int* edgeset, int* nodeset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / num_node;
    int j = idx % num_node;
    if (i<0 || i >= num_edge || j >=num_node)
        return;
    // printf("%d %d %d\n", edgeset[i], edgeset[i + num_edge], nodeset[j]);
    // edgeset[i + num_edge]索引计算错误
    if(edgeset[i] == nodeset[j]){
        edgeset[i] = j;
        // printf("%d %d %d\n", edgeset[i], nodeset[j], j);
    }
    if(edgeset[i + num_edge] == nodeset[j]){
        edgeset[i + num_edge] = j;
        // printf("%d %d %d\n", edgeset[i + num_edge], nodeset[j], j);
    }

}

// Host function to call the CUDA kernel
void mapping_edgeidx(
    torch::Tensor edgeset, torch::Tensor nodeset)
{

    int num_edge = edgeset.size(1);
    int num_node = nodeset.size(0);
    int block_size = 256;
    // printf("%d %d\n", num_edge, num_node);
    int grid_size = (num_edge*num_node + block_size - 1) / block_size;
    kernel<<<grid_size, block_size>>>(num_edge, num_node, edgeset.data<int>(), nodeset.data<int>());

    cudaDeviceSynchronize();
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaError));
        // 可以尝试打印相关的变量信息，帮助定位问题
    }
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("mapping_edgeidx", &mapping_edgeidx, "Mapping idxs in edge set using CUDA");
// }