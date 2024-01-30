// Assuming the necessary includes and definitions for the CUDA environment
#include <torch/torch.h>
#include <thrust/extrema.h>


__device__ float overlap_ratioxy(float pos_ego_x, float pos_ego_y, float len_ego_x, float len_ego_y, float pos_neb_x, float pos_neb_y, float len_neb_x, float len_neb_y) {
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

__device__ float disxy(float pos_ego_x, float pos_ego_y, float pos_neb_x, float pos_neb_y) {
    float delta_x = pos_ego_x - pos_neb_x;
    float delta_y = pos_ego_y - pos_neb_y;
    return sqrt(delta_x*delta_x + delta_y*delta_y);
}


__global__ void build_edge_kernel_xy(int neb_num, int max_related, float search_range, 
    int* neb_nonzero_indices,
    float* sc_neb, float* sc_ego, int H_ego, int W_ego, int H_neb, int W_neb,
    int* indices, float* values,
    int * trace, int value_num, int max_trace)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= neb_num || idx < 0 )
        return;
    int L = neb_nonzero_indices[idx];

    int neb_i = L / W_neb;
    int neb_j = L % W_neb;
    // if(neb_i < 0 || neb_j < 0 || neb_i >= H_neb || neb_j >=W_ego )
        // return;
    // printf("%d", L);
    float pos_neb_x = (neb_j - W_neb / 2.0 +0.5) * sc_neb[1];
    float pos_neb_y = (H_neb / 2.0 - neb_i -0.5) * sc_neb[0];
    
    float range_i = search_range / sc_neb[0];
    float range_j = search_range / sc_neb[1];

    int i_lower = (neb_i - range_i) * H_ego / H_neb;
    int i_upper = (neb_i + range_i) * H_ego / H_neb;
    int j_lower = (neb_j - range_j) * W_ego / W_neb;
    int j_upper = (neb_j + range_j) * W_ego / W_neb;

    i_lower = (i_lower > 0) ? i_lower : 0;
    j_lower = (j_lower > 0) ? j_lower : 0;
    i_upper = (i_upper > H_ego - 1) ? H_ego - 1 : i_upper;
    j_upper = (j_upper > W_ego - 1) ? W_ego - 1 : j_upper;

    int center_i = static_cast<int>(1.0 * neb_i * H_ego / H_neb);
    int center_j = static_cast<int>(1.0 * neb_j * W_ego / W_neb);
    float mid_len = max (search_range / sc_ego[0], search_range / sc_ego[1]);
    // printf("1");
    // trace为共享内存, 容易导致访问冲突
    trace[idx*max_trace + 0] = center_i;
    trace[idx*max_trace + 1] = center_j;
    int pos = 1;
    int sample_range = 1;
    // printf("2");
    // 生成采样序列
    while (sample_range) {
        // 左上到右下的采样
        int i_start = center_i - sample_range;
        int i_end = center_i + sample_range;
        int j_start = center_j - sample_range;
        int j_end = center_j + sample_range;

        // 更新indices和values数组，注意需要使用原子操作

        // 从上一个矩形递补
        for (int j = j_start + 1; j < j_end; j++) {
            if (i_start >= i_lower && j >= j_lower && j <= j_upper) {
                trace[idx*max_trace + pos*2] = i_start;
                trace[idx*max_trace + pos*2 + 1] = j;
                pos = pos + 1;
            }
        }
        for (int i = i_start + 1; i < i_end; i++) {
            if (i >= i_lower && i <= i_upper && j_end <= j_upper) {
                trace[idx*max_trace + pos*2] = i;
                trace[idx*max_trace + pos*2 + 1] = j_end;
                pos = pos + 1;
            }
        }
        for (int j = j_end - 1; j > j_start; j--) {
            if (i_end <= i_upper && j >= j_lower && j <= j_upper) {
                trace[idx*max_trace + pos*2] = i_end;
                trace[idx*max_trace + pos*2 + 1] = j;
                pos = pos + 1;
            }
        }
        for (int i = i_end - 1; i > i_start; i--) {
            if (i <= i_upper && i >= i_lower && j_start >= j_lower) {
                trace[idx*max_trace + pos*2] = i;
                trace[idx*max_trace + pos*2 + 1] = j_start;
                pos = pos + 1;
            }
        }

        // 采样矩形的四个角
        if (i_start >= i_lower && j_start >= j_lower) {
            trace[idx*max_trace + pos*2] = i_start;
            trace[idx*max_trace + pos*2 + 1] = j_start;
            pos = pos + 1;
        }
        if (i_start >= i_lower && j_end <= j_upper) {
            trace[idx*max_trace + pos*2] = i_start;
            trace[idx*max_trace + pos*2 + 1] = j_end;
            pos = pos + 1;
        }
        if (i_end <= i_upper && j_start >= j_lower) {
            trace[idx*max_trace + pos*2] = i_end;
            trace[idx*max_trace + pos*2 + 1] = j_start;
            pos = pos + 1;
        }
        if (i_end <= i_upper && j_end <= j_upper) {
            trace[idx*max_trace + pos*2] = i_end;
            trace[idx*max_trace + pos*2 + 1] = j_end;
            pos = pos + 1;
        }

        sample_range = sample_range + 1;
        if (sample_range > mid_len || pos > max_trace/2) {
            break;
        }
    }
    // printf("3");
    // 顺次读取采样序列， 赋予边权值
    int pos_max = pos;
    pos = 0;
    int cc=0;
    while(pos < pos_max)
    {
        int ego_i = trace[idx*max_trace + pos*2];
        int ego_j = trace[idx*max_trace + pos*2+1];
        float pos_ego_x = (ego_j + 0.5 - W_ego / 2.0 ) * sc_ego[1];
        float pos_ego_y = (H_ego / 2.0 - ego_i-0.5) * sc_ego[0];
        float ratio = overlap_ratioxy(pos_ego_x, pos_ego_y, sc_ego[1], sc_ego[0], pos_neb_x, pos_neb_y, sc_neb[1], sc_neb[0] );
        float dist = disxy(pos_ego_x, pos_ego_y, pos_neb_x, pos_neb_y);
        float dist_x = pos_neb_x - pos_ego_x;
        float dist_y = pos_neb_y - pos_ego_y;
        if (dist < search_range || ratio > 1e-4){ // if neb and ego overlap space is not none, set upper bonud max than zero condsier the compute exists precision error
            // int val_index = idx * max_related + cc;
            // values[val_index] = ratio;

            int index = (idx * max_related + cc) * 2;
            indices[index] = L;
            indices[index + 1] = static_cast<int>(ego_i * W_ego + ego_j);


            // 所乘倍数根据value的维度改变
            int index_val = (idx * max_related + cc) * value_num;
            values[index_val] = dist;
            values[index_val+1] = ratio; 
            values[index_val+2] = dist_x;
            values[index_val+3] = dist_y;

            cc = cc + 1;
            if(cc>=max_related) {break;}
        }
        pos = pos + 1; // Assuming 2 entries per iteration
    }
    // printf("4");
    // return;
}


void cal_overlap_xy(
    int max_related, float search_range,
    torch::Tensor neb_nonzero_indices, torch::Tensor sc_neb, torch::Tensor sc_ego, 
    int H_ego, int W_ego, int H_neb, int W_neb,
    torch::Tensor indices, torch::Tensor values
    )
{
    int current_index = neb_nonzero_indices.device().index();
    cudaSetDevice(current_index);

    int neb_num = neb_nonzero_indices.size(0);
    int value_num = values.size(1);
    // int len = neb_num * max_related;

    int block_size = 256;
    int grid_size = (neb_num + block_size - 1) / block_size;
    int* trace;

    int max_trace = 20; // 最多观测20个点
    // printf("%d\n", max_trace);
    cudaMalloc((void**)&trace, neb_num * max_trace * 2 * sizeof(int));
    // trace[idx][pos][0] 为每个neb分别分配一个trace空间, 防止访问冲突
    // printf("%d\n", max_trace);
    build_edge_kernel_xy<<<grid_size, block_size>>>(
        neb_num, max_related, search_range, 
        neb_nonzero_indices.data<int>(),
        sc_neb.data_ptr<float>(), sc_ego.data_ptr<float>(), H_ego, W_ego, H_neb, W_neb,
        indices.data_ptr<int>(), values.data_ptr<float>(),
        trace, value_num, max_trace*2);
    //  printf("5");
    cudaDeviceSynchronize();
    cudaFree(trace);
    // printf("6");
    //  printf("7");
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaError));
        // help to locate error state
    }
    
    return;
}





// __global__ void build_edge_kernel(int neb_num, int max_related, float search_range, 
//     int* neb_nonzero_indices,
//     float* sc_neb, float* sc_ego, int H_ego, int W_ego, int H_neb, int W_neb,
//     int* indices, float* values,
//     float * trace)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (idx >= neb_num || idx < 0 )
//         return;
//     int L = neb_nonzero_indices[idx];

//     int neb_i = L / W_neb;
//     int neb_j = L % W_neb;
//     // if(neb_i < 0 || neb_j < 0 || neb_i >= H_neb || neb_j >=W_ego )
//         // return;
    
    
//     float pos_neb_x = (neb_j - W_neb / 2.0 +0.5) * sc_neb[1];
//     float pos_neb_y = (H_neb / 2.0 - neb_i -0.5) * sc_neb[0];
    
//     // 可能与ego相邻的neb节点的范围
//     int i_lower = static_cast<int>((neb_i-1) * H_ego / H_neb) - 1;
//     int i_upper = static_cast<int>((neb_i+1) * H_ego / H_neb) + 1;
//     int j_lower = static_cast<int>((neb_j-1) * W_ego / W_neb) - 1;
//     int j_upper = static_cast<int>((neb_j+1) * W_ego / W_neb) + 1;
    
//     int i_start = (i_lower>0) ? i_lower : 0; 
//     int j_start = (j_lower>0) ? j_lower : 0; 
//     int i_end = (i_upper > H_ego-1) ? H_ego: i_upper+1;
//     int j_end = (j_upper > W_ego-1) ? W_ego: j_upper+1;
    
//     int cc = 0;
//     for (int ego_i=i_start;  ego_i<i_end; ego_i=ego_i+1)
//     {
//         for (int ego_j=j_start; ego_j<j_end; ego_j=ego_j+1)
//         {
//             float pos_ego_x = (ego_j - W_ego / 2.0 +0.5) * sc_ego[1];
//             float pos_ego_y = (H_ego / 2.0 - ego_i-0.5) * sc_ego[0];
//             float ratio = overlap_ratiod(pos_ego_x, pos_ego_y, sc_ego[1], sc_ego[0], pos_neb_x, pos_neb_y, sc_neb[1], sc_neb[0] );
//             float dist = dis(pos_ego_x, pos_ego_y, pos_neb_x, pos_neb_y);
//             if (dist < search_range || ratio > 1e-4){ // if neb and ego overlap space is not none, set upper bonud max than zero condsier the compute exists precision error
//                 // int val_index = idx * max_related + cc;
//                 // values[val_index] = ratio;
    
//                 int index = (idx * max_related + cc) * 2;
//                 indices[index] = L;
//                 indices[index + 1] = static_cast<int>(ego_i * W_ego + ego_j);

//                 int index_val = (idx * max_related + cc) * 4;
//                 values[index_val] = dist;
//                 values[index_val+1] = ratio; 
//                 values[index_val+2] = sin(pos_ego_x, pos_ego_y, pos_neb_x, pos_neb_y);
//                 values[index_val+3] = cos(pos_ego_x, pos_ego_y, pos_neb_x, pos_neb_y);

//                 cc = cc + 1; // Assuming 2 entries per iteration
//                 if(cc>=max_related) {return;}
//             }
//         }
//         if(cc>=max_related) {break;}
//     }
// }

// __global__ void build_edge_kernel(int neb_num, int max_related, float search_range, 
//     int* neb_nonzero_indices,
//     float* sc_neb, float* sc_ego, int H_ego, int W_ego, int H_neb, int W_neb,
//     int* indices, float* values,
//     float * trace)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (idx >= neb_num || idx < 0 )
//         return;
//     int L = neb_nonzero_indices[idx];

//     int neb_i = L / W_neb;
//     int neb_j = L % W_neb;
//     // if(neb_i < 0 || neb_j < 0 || neb_i >= H_neb || neb_j >=W_ego )
//         // return;
    
//     float pos_neb_x = (neb_j - W_neb / 2.0 +0.5) * sc_neb[1];
//     float pos_neb_y = (H_neb / 2.0 - neb_i -0.5) * sc_neb[0];
    
//     // 可能与ego相邻的neb节点的范围
//     // int idx_range = search_range / 
//     int i_lower = static_cast<int>((neb_i-1) * H_ego / H_neb) - 1;
//     int i_upper = static_cast<int>((neb_i+1) * H_ego / H_neb) + 1;
//     int j_lower = static_cast<int>((neb_j-1) * W_ego / W_neb) - 1;
//     int j_upper = static_cast<int>((neb_j+1) * W_ego / W_neb) + 1;
    
//     int i_start = (i_lower>0) ? i_lower : 0; 
//     int j_start = (j_lower>0) ? j_lower : 0; 
//     int i_end = (i_upper > H_ego-1) ? H_ego-1: i_upper;
//     int j_end = (j_upper > W_ego-1) ? W_ego-1: j_upper;

//     // save outside to inside traversal sequence

//     int len_i = i_end - i_start + 1;
//     int len_j = j_end - j_start + 1;
//     int len_trace = len_i * len_j;
//     // cudaMalloc((void**)&trace, len_trace * 2 * sizeof(float));
//     int center_i = i_end - i_start + 1;
//     int center_j = j_end - j_start + 1;

//     // int range = 0;
//     int pos = 0;
//     while (true) {
//         // Move right
//         for (int j = j_start; j <= j_end; j++) {
//             trace[idx*max_trace + pos*2] = i_start;
//             trace[idx*max_trace + pos*2 + 1] = j;
//             pos++;
//         }
//         i_start++;
//         if (i_start > i_end) {break;}

//         // Move down
//         for (int i = i_start; i <= i_end; i++) {
//             trace[idx*max_trace + pos*2] = i;
//             trace[idx*max_trace + pos*2+1] = j_end;
//             pos++;
//         }
//         j_end--;
//         if (j_end < j_start) {break;}

//         // Move left
//         for (int j = j_end; j >= j_start; j--) {
//             trace[idx*max_trace + pos*2] = i_end;
//             trace[idx*max_trace + pos*2+1] = j;
//             pos++;
//         }
//         i_end--;
//         if (i_start > i_end) {break;}

//         // Move up
//         for (int i = i_end; i >= i_start; i--) {
//             trace[idx*max_trace + pos*2] = i;
//             trace[idx*max_trace + pos*2+1] = j_start;
//             pos++;
//         }
//         j_start++;
//         if (j_end < j_start) {break;}
//     }

//     // 从后向前遍历trace, 得到由内向外的序列
//     int cc = 0;
//     for( int pos = len_trace-1; pos>=0; pos--)
//     {
//         int ego_i = trace[idx*max_trace + pos*2];
//         int ego_j = trace[idx*max_trace + pos*2+1];
//         float pos_ego_x = (ego_j - W_ego / 2.0 +0.5) * sc_ego[1];
//         float pos_ego_y = (H_ego / 2.0 - ego_i-0.5) * sc_ego[0];
//         float ratio = overlap_ratiod(pos_ego_x, pos_ego_y, sc_ego[1], sc_ego[0], pos_neb_x, pos_neb_y, sc_neb[1], sc_neb[0] );
//         float dist = dis(pos_ego_x, pos_ego_y, pos_neb_x, pos_neb_y);
//         if (dist < search_range || ratio > 1e-4){ // if neb and ego overlap space is not none, set upper bonud max than zero condsier the compute exists precision error
//             // int val_index = idx * max_related + cc;
//             // values[val_index] = ratio;

//             int index = (idx * max_related + cc) * 2;
//             indices[index] = L;
//             indices[index + 1] = static_cast<int>(ego_i * W_ego + ego_j);

//             int index_val = (idx * max_related + cc) * 4;
//             values[index_val] = dist;
//             values[index_val+1] = ratio; 
//             values[index_val+2] = sin(pos_ego_x, pos_ego_y, pos_neb_x, pos_neb_y);
//             values[index_val+3] = cos(pos_ego_x, pos_ego_y, pos_neb_x, pos_neb_y);

//             cc = cc + 1; // Assuming 2 entries per iteration
//             if(cc>=max_related) {return;}
//         }
//     }
//     return;
// }