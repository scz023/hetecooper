import torch
import math

def overlap_ratio(pos_ego_x, pos_ego_y, len_ego_x, len_ego_y, pos_neb_x, pos_neb_y, len_neb_x, len_neb_y):
    # 计算矩形ego的左上角和右下角坐标
    ego_left_x = pos_ego_x - len_ego_x / 2
    ego_right_x = pos_ego_x + len_ego_x / 2
    ego_top_y = pos_ego_y + len_ego_y / 2
    ego_bottom_y = pos_ego_y - len_ego_y / 2

    # 计算矩形neb的左上角和右下角坐标
    neb_left_x = pos_neb_x - len_neb_x / 2
    neb_right_x = pos_neb_x + len_neb_x / 2
    neb_top_y = pos_neb_y + len_neb_y / 2
    neb_bottom_y = pos_neb_y - len_neb_y / 2

    # 计算交叉矩形的左上角和右下角坐标
    overlap_left_x = max(ego_left_x, neb_left_x)
    overlap_right_x = min(ego_right_x, neb_right_x)
    overlap_top_y = min(ego_top_y, neb_top_y)
    overlap_bottom_y = max(ego_bottom_y, neb_bottom_y)

    # 计算交叉矩形的宽度和高度
    overlap_width = overlap_right_x - overlap_left_x
    overlap_height = overlap_top_y - overlap_bottom_y

    # 计算交叉矩形的面积
    overlap_area = max(0, overlap_width) * max(0, overlap_height)

    # 计算矩形neb的面积
    neb_area = len_neb_x * len_neb_y

    # 计算交叉面积与neb面积的比值, 即neb中可用信息的比例
    ratio = overlap_area / neb_area if overlap_area > 0 else 0

    return ratio

def dis(pos_ego_x, pos_ego_y, pos_neb_x, pos_neb_y):
    delta_x = pos_ego_x - pos_neb_x
    delta_y = pos_ego_y - pos_neb_y
    return math.sqrt(delta_x*delta_x + delta_y*delta_y)


def build_edge_kernel(max_related, search_range, neb_nonzero_indices, sc_neb, sc_ego, H_ego, W_ego, H_neb, W_neb, indices, values):
    
    neb_num = neb_nonzero_indices.size()[0]

    # 建立邻接关系，即所有边权值相等的邻接矩阵
    indices = -1 * torch.ones((neb_num * max_related, 2), device=torch.device('cuda:0')).to(torch.int) # 初始化indices值为-1，后续若没有赋值，则排除
    values = torch.zeros((neb_num * max_related, 4), device=torch.device('cuda:0')).to(torch.float) # 初始化indices值为-1，后续若没有赋值，则排除
    trace = -torch.ones((100, 2),device=torch.device('cuda:0')).to(torch.int)
    for idx in range(neb_num):
        L = neb_nonzero_indices[idx]
        neb_i = L // W_neb
        neb_j = L % W_neb

        pos_neb_x = (neb_j - W_neb / 2 + 0.5) * sc_neb[1]
        pos_neb_y = (H_neb / 2 - neb_i - 0.5) * sc_neb[0]

        range_i = search_range / sc_neb[0]
        range_j = search_range / sc_neb[1]

        i_lower = (neb_i - range_i) * H_ego / H_neb
        i_upper = (neb_i + range_i) * H_ego / H_neb
        j_lower = (neb_j - range_j) * W_ego / W_neb
        j_upper = (neb_j + range_j) * W_ego / W_neb


        i_lower = i_lower if i_lower > 0 else 0
        j_lower = j_lower if j_lower > 0 else 0
        i_upper = H_ego-1 if i_upper > H_ego - 1 else i_upper
        j_upper = W_ego-1 if j_upper > W_ego - 1 else j_upper


        
        # pos = 0
        center_i = int(neb_i * H_ego / H_neb)
        center_j = int(neb_j * W_ego / W_neb)
        mid_len = max( (j_upper - j_lower +1)/2,  (i_upper - i_lower +1)/2)

        # 第一个点为特征图中心点
        trace[0][0] = center_i
        trace[0][1] = center_j
        pos = 1
        sample_range = 1
        while(sample_range):
            # left top: center_i-side, center_i+side, center_j-side, center_j+side, 
            # left to right
            i_start = center_i - sample_range
            i_end = center_i + sample_range
            j_start = center_j - sample_range
            j_end = center_j + sample_range

            # 从上一个矩形递补
            for j in range(j_start+1, j_end):
                if i_start >= i_lower and j>=j_lower and j<=j_upper:
                    trace[pos][0] = i_start
                    trace[pos][1] = j
                    pos = pos + 1
            for i in range(i_start+1, i_end):
                if i >= i_lower and i<=i_upper and j_end<=j_upper:               
                    trace[pos][0] = i
                    trace[pos][1] = j_end
                    pos = pos + 1
            for j in range(j_end-1, j_start, -1):
                if i_end <=i_upper and j>=j_lower and j<=j_upper:
                    trace[pos][0] = i_end
                    trace[pos][1] = j
                    pos = pos + 1
            for i in range(i_end-1, i_start, -1):
                if i <= i_upper and i>=i_lower and j_start>=j_lower:
                    trace[pos][0] = i
                    trace[pos][1] = j_start
                    pos = pos + 1
            
            # 采样矩形的四个角
            if(i_start>=i_lower and j_start>=j_lower):
                trace[pos][0]=i_start
                trace[pos][1]=j_start
                pos = pos + 1
            if(i_start>=i_lower and j_end<=j_upper):
                trace[pos][0]=i_start
                trace[pos][1]=j_end
                pos = pos + 1
            if(i_end<=i_upper and j_start>=j_lower):
                trace[pos][0]=i_end
                trace[pos][1]=j_start
                pos = pos + 1
            if(i_end<=i_upper and j_end<=j_upper):
                trace[pos][0]=i_end
                trace[pos][1]=j_end
                pos = pos + 1

            sample_range = sample_range + 1
            if sample_range > mid_len:
                break

        # print(trace)
        pos_max = pos
        pos = 0
        cc = 0
        while pos < pos_max:
            ego_i = trace[pos][0]
            ego_j = trace[pos][1]
            # print(ego_i, ego_j)
            pos_ego_x = (ego_j - W_ego / 2.0 + 0.5) * sc_ego[1]
            pos_ego_y = (H_ego / 2.0 - ego_i - 0.5) * sc_ego[0]
            ratio = overlap_ratio(pos_ego_x, pos_ego_y, sc_ego[1], sc_ego[0], pos_neb_x, pos_neb_y, sc_neb[1], sc_neb[0])
            dist = dis(pos_ego_x, pos_ego_y, pos_neb_x, pos_neb_y)
            if dist < search_range or ratio > 1e-4:
                index = (idx * max_related + cc)
                indices[index][0] = L  # You'll need to define L
                indices[index][1] = int(ego_i * W_ego + ego_j)

                values[index][0] = dist
                values[index][1] = ratio

                cc = cc + 1
                if cc >= max_related:
                    break
            pos += 1

    return indices, values
