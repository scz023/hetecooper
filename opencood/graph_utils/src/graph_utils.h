#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

void find_relation(
    const torch::Tensor neb_nonzero_indices, 
    const torch::Tensor sc_neb, const torch::Tensor sc_ego, int H_ego, int W_ego, int H_neb, int W_neb,
    torch::Tensor indices);

void mapping_edgeidx(
    torch::Tensor edgeset, torch::Tensor nodeset);

void cal_overlap(
    int max_related,
    torch::Tensor neb_nonzero_indices, torch::Tensor sc_neb, torch::Tensor sc_ego, 
    int H_ego, int W_ego, int H_neb, int W_neb,
    torch::Tensor indices, torch::Tensor values
);
void cal_overlap_dis(
    int max_related, float search_range,
    torch::Tensor neb_nonzero_indices, torch::Tensor sc_neb, torch::Tensor sc_ego, 
    int H_ego, int W_ego, int H_neb, int W_neb,
    torch::Tensor indices, torch::Tensor values
    );

void cal_overlap_xy(
    int max_related, float search_range,
    torch::Tensor neb_nonzero_indices, torch::Tensor sc_neb, torch::Tensor sc_ego, 
    int H_ego, int W_ego, int H_neb, int W_neb,
    torch::Tensor indices, torch::Tensor values
    );