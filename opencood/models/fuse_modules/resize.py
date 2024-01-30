import torch

def align_feature(x, target_size):
    # 获取A和B的维度信息
    _, c_a, h_a, w_a = A.size()
    _, c_b, h_b, w_b = B.size()

    # # 情况1：如果B的尺寸比A大，使用下采样将B的尺寸对齐到A
    # if h_b > h_a or w_b > w_a:
    #     scale_factor_h = h_a / h_b
    #     scale_factor_w = w_a / w_b
    #     B = torch.nn.functional.interpolate(B, scale_factor=(scale_factor_h, scale_factor_w), mode='bilinear', align_corners=False)

    # # 情况2：如果B的尺寸比A小，使用上采样将B的尺寸对齐到A
    # elif h_b < h_a or w_b < w_a:
    scale_factor_h = h_a / h_b
    scale_factor_w = w_a / w_b
    print(h_a, w_a)
    B = torch.nn.functional.interpolate(B, size=(h_a, w_a), mode='bilinear', align_corners=False)

    return A, B

# 示例用法
A = torch.randn(1, 3, 64, 64)  # 假设A的尺寸为[1, 3, 64, 64]
B = torch.randn(1, 3, 32, 128)  # 假设B的尺寸为[1, 3, 128, 128]

print(A[:1].size())
print(A[:0])

# 对齐B到A的尺寸
# aligned_A, aligned_B = align_feature(A, B)
# print(aligned_B.size())

# 现在aligned_A和aligned_B的尺寸与A相同，并保持原有信息不变
batch_node_features = torch.randn(2, 3, 1,2,2)
x_fuse = batch_node_features[:, 1:,:,:,:]
batch_node_features[:, 1:,:,:,:] = x_fuse
# print(x_fuse.size())
x = [batch_node_features]
x = torch.stack(x)
print(x.size())
