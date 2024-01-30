// Assuming the necessary includes and definitions for the CUDA environment
#include <torch/torch.h>

__global__ void build_edge_kernel(int neb_num,
    int* neb_nonzero_indices,
    float* sc_neb, float* sc_ego, int H_ego, int W_ego, int H_neb, int W_neb,
    int* indices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= neb_num)
        return;

    // H row -- x , W col -- y
    int j = neb_nonzero_indices[idx];
    int neb_x = j / W_neb;
    int neb_y = j % W_neb;
    
    // ego和neb同一位置顶点到左上角物理距离相等，按这一原理寻找转换参数
    int x_lower = static_cast<int>(neb_x * sc_neb[0] / sc_ego[0] - 1.6);
    int x_upper = static_cast<int>(neb_x * sc_neb[0] / sc_ego[0] + 1.6);
    int y_lower = static_cast<int>(neb_y * sc_neb[0] / sc_ego[0] - 1.6);
    int y_upper = static_cast<int>(neb_y * sc_neb[0] / sc_ego[0] + 1.6);
    // int x_lower = static_cast<int>((sc_neb[0] * (neb_x - static_cast<float>(H_neb) / 2)) / sc_ego[0] + static_cast<float>(H_ego) / 2 - 1.5);
    // int x_upper = static_cast<int>((sc_neb[0] * (neb_x - static_cast<float>(H_neb) / 2)) / sc_ego[0] + static_cast<float>(H_ego) / 2 + 1.5 + 0.5);
    // int y_lower = static_cast<int>((sc_neb[1] * (neb_y - static_cast<float>(W_neb) / 2)) / sc_ego[1] + static_cast<float>(W_ego) / 2 - 1.5);
    // int y_upper = static_cast<int>((sc_neb[1] * (neb_y - static_cast<float>(W_neb) / 2)) / sc_ego[1] + static_cast<float>(W_ego) / 2 + 1.5 + 0.5);
    // printf("Self: %d\n", j);
    // printf("Nebs: %d %d %d %d\n", x_lower, x_upper, y_lower, y_upper);
    // float nc = neb_conf[neb_x * W_neb + neb_y];
    // float np = neb_proj[neb_x * W_neb + neb_y];
    int cc = 0;
    for (int ego_x = max(0, x_lower); ego_x < min(H_ego - 1, x_upper); ego_x++)
    {
        for (int ego_y = max(0, y_lower); ego_y < min(W_ego - 1, y_upper); ego_y++)
        {
            // int index_val = idx * 15 + cc;
            // values[index_val] = nc * (ego_proj[ego_x * W_ego + ego_y] + np);
            int index = (idx * 15 + cc) * 2;
            indices[index] = ego_x * W_ego + ego_y;
            indices[index + 1] = j;
            cc = cc + 1; // Assuming 2 entries per iteration
            if(cc>15) {break;}
        }
        if(cc>15) {break;}
    }
}

// Host function to call the CUDA kernel
void find_relation(
    const torch::Tensor neb_nonzero_indices, 
    const torch::Tensor sc_neb, const torch::Tensor sc_ego, int H_ego, int W_ego, int H_neb, int W_neb,
    torch::Tensor indices)
{

    int current_index = neb_nonzero_indices.device().index();
    cudaSetDevice(current_index);

    int neb_num = neb_nonzero_indices.size(0);
    int block_size = 256;
    int grid_size = (neb_num + block_size - 1) / block_size;
    build_edge_kernel<<<grid_size, block_size>>>(neb_num, neb_nonzero_indices.data<int>(),
        sc_neb.data<float>(), sc_ego.data<float>(), H_ego, W_ego, H_neb, W_neb,
        indices.data<int>());

    cudaDeviceSynchronize();
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaError));
        // 可以尝试打印相关的变量信息，帮助定位问题
    }
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("find_relation", &find_relation, "Construct adjacency matrix using CUDA");
// }