// Assuming the necessary includes and definitions for the CUDA environment
#include <torch/torch.h>

__device__ float overlap_ratio(float pos_ego_x, float pos_ego_y, float len_ego_x, float len_ego_y, float pos_neb_x, float pos_neb_y, float len_neb_x, float len_neb_y) {
    // 计算矩形ego的左上角和右下角坐标
    float ego_left_x = pos_ego_x - len_ego_x / 2;
    float ego_right_x = pos_ego_x + len_ego_x / 2;
    float ego_top_y = pos_ego_y + len_ego_y / 2;
    float ego_bottom_y = pos_ego_y - len_ego_y / 2;

    // 计算矩形neb的左上角和右下角坐标
    float neb_left_x = pos_neb_x - len_neb_x / 2;
    float neb_right_x = pos_neb_x + len_neb_x / 2;
    float neb_top_y = pos_neb_y + len_neb_y / 2;
    float neb_bottom_y = pos_neb_y - len_neb_y / 2;

    // 计算交叉矩形的左上角和右下角坐标
    float overlap_left_x = (ego_left_x > neb_left_x) ? ego_left_x : neb_left_x;
    float overlap_right_x = (ego_right_x < neb_right_x) ? ego_right_x : neb_right_x;
    float overlap_top_y = (ego_top_y < neb_top_y) ? ego_top_y : neb_top_y;
    float overlap_bottom_y = (ego_bottom_y > neb_bottom_y) ? ego_bottom_y : neb_bottom_y;

    // 计算交叉矩形的宽度和高度
    float overlap_width = overlap_right_x - overlap_left_x;
    float overlap_height = overlap_top_y - overlap_bottom_y;

    // 计算交叉矩形的面积
    float overlap_area = (overlap_width > 0 && overlap_height > 0) ? (overlap_width * overlap_height) : 0;

    // 计算矩形neb的面积
    float neb_area = len_neb_x * len_neb_y;

    // 计算交叉面积与neb面积的比值, 即neb中可用信息的比例
    float ratio = (overlap_area > 0) ? (overlap_area / neb_area) : 0;

    return ratio;
}



__global__ void build_edge_kernel(int neb_num, int max_related, 
    int* neb_nonzero_indices,
    float* sc_neb, float* sc_ego, int H_ego, int W_ego, int H_neb, int W_neb,
    int* indices, float* values)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= neb_num || idx < 0 )
        return;
    int L = neb_nonzero_indices[idx];

    int neb_i = L / W_neb;
    int neb_j = L % W_neb;
    // if(neb_i < 0 || neb_j < 0 || neb_i >= H_neb || neb_j >=W_ego )
        // return;
    
    
    float pos_neb_x = (neb_j - W_neb / 2.0 +0.5) * sc_neb[1];
    float pos_neb_y = (H_neb / 2.0 - neb_i -0.5) * sc_neb[0];
    
    // 可能与ego相邻的neb节点的范围
    int i_lower = static_cast<int>((neb_i-1) * H_ego / H_neb) - 1;
    int i_upper = static_cast<int>((neb_i+1) * H_ego / H_neb) + 1;
    int j_lower = static_cast<int>((neb_j-1) * W_ego / W_neb) - 1;
    int j_upper = static_cast<int>((neb_j+1) * W_ego / W_neb) + 1;
    
    int i_start = (i_lower>0) ? i_lower : 0; 
    int j_start = (j_lower>0) ? j_lower : 0; 
    int i_end = (i_upper > H_ego-1) ? H_ego: i_upper+1;
    int j_end = (j_upper > W_ego-1) ? W_ego: j_upper+1;
    
    int cc = 0;
    for (int ego_i=i_start;  ego_i<i_end; ego_i=ego_i+1)
    {
        for (int ego_j=j_start; ego_j<j_end; ego_j=ego_j+1)
        {
            float pos_ego_x = (ego_j - W_ego / 2.0 +0.5) * sc_ego[1];
            float pos_ego_y = (H_ego / 2.0 - ego_i-0.5) * sc_ego[0];
            float ratio = overlap_ratio(pos_ego_x, pos_ego_y, sc_ego[1], sc_ego[0], pos_neb_x, pos_neb_y, sc_neb[1], sc_neb[0] );
            if (ratio > 1e-4){ // if neb and ego overlap space is not none, set upper bonud max than zero condsier the compute exists precision error
                int val_index = idx * max_related + cc;
                values[val_index] = ratio;
    
                int index = (idx * max_related + cc) * 2;
                indices[index] = L;
                indices[index + 1] = static_cast<int>(ego_i * W_ego + ego_j);
                cc = cc + 1; // Assuming 2 entries per iteration
                if(cc>=max_related) {return;}
            }
        }
        if(cc>=max_related) {break;}
    }
}


void cal_overlap(
    int max_related,
    torch::Tensor neb_nonzero_indices, torch::Tensor sc_neb, torch::Tensor sc_ego, 
    int H_ego, int W_ego, int H_neb, int W_neb,
    torch::Tensor indices, torch::Tensor values
    )
{
    int current_index = neb_nonzero_indices.device().index();
    cudaSetDevice(current_index);

    int neb_num = neb_nonzero_indices.size(0);
    int len = neb_num * max_related;

    int block_size = 256;
    int grid_size = (neb_num + block_size - 1) / block_size;

    build_edge_kernel<<<grid_size, block_size>>>(neb_num, max_related, neb_nonzero_indices.data<int>(),
        sc_neb.data_ptr<float>(), sc_ego.data_ptr<float>(), H_ego, W_ego, H_neb, W_neb,
        indices.data_ptr<int>(), values.data_ptr<float>());

    cudaDeviceSynchronize();
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaError));
        // help to locate error state
    }
    return;
}
