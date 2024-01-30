# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, DistributedSampler
import open3d as o3d
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.tools import multi_gpu_utils
from opencood.models.main_module import MainModule
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils
from opencood.visualization import simple_vis
import matplotlib.pyplot as plt


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--model_dir_cooper', default='',
                        help='Continued training cooper model path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--save_vis_n', type=int, default=10,
                        help='save how many numbers of visualization result?')
    parser.add_argument('--eval_epoch', type=str, default=None,
                        help='Set the checkpoint')
    parser.add_argument('--eval_epoch_proj', type=str, default=None,
                        help='Set the checkpoint')
    parser.add_argument('--eval_epoch_cooper', type=str, default=None,
                        help='Set the checkpoint')
    parser.add_argument('--comm_thre', type=float, default=None,
                        help='Communication confidence threshold')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy_test file')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='whether to globally sort detections by confidence score.'
                             'If set to True, it is the mainstream AP computing method,'
                             'but would increase the tolerance for FP (False Positives).')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'intermediate_with_comm']
    assert not (opt.show_vis and opt.show_sequence), 'you can only visualize ' \
                                                    'the results in single ' \
                                                    'image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt)
    hypes_cooper = yaml_utils.load_yaml(None, opt, True)
    hypes_cooper['train_params'] = hypes['train_params']
    multi_gpu_utils.init_distributed_mode(opt)
    left_hand = True if "OPV2V" in hypes['root_dir'] else False

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False, dataset_cfg_neb=hypes_cooper)
    print(f"{len(opencood_dataset)} samples found.")
    if opt.distributed:
        sampler = DistributedSampler(opencood_dataset,
                                         shuffle=False)
        data_loader = DataLoader(opencood_dataset,
                                batch_size=1,
                                sampler=sampler,
                                num_workers=8,
                                collate_fn=opencood_dataset.collate_batch_test,
                                drop_last=False)
    
    else:
        data_loader = DataLoader(opencood_dataset,
                                batch_size=1,
                                num_workers=8,
                                collate_fn=opencood_dataset.collate_batch_test,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=False)

    print('Creating Model')
    model = MainModule(hypes['model'])
    model_cooper = MainModule(hypes_cooper['model'])
    # print(model.state_dict())
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
        model_cooper.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    
    # load ego parameters
    saved_path = opt.model_dir
    if opt.eval_epoch is not None:
        epoch_id = opt.eval_epoch
        epoch_id, model = train_utils.load_saved_model(saved_path, model, epoch_id)

    else:
        epoch_id, model = train_utils.load_saved_model(saved_path, model)

    # load ego proj parameters
    if opt.eval_epoch_proj is not None:
        epoch_id_proj = opt.eval_epoch_proj
        epoch_id_proj, model = train_utils.load_saved_model_proj(saved_path,
                                                         model, epoch_id_proj)
    else:
        epoch_id_proj, model = train_utils.load_saved_model_proj(saved_path,
                                                         model)
    
    # load cooper parameters
    if opt.eval_epoch_cooper is not None:
        epoch_id_cooper = opt.eval_epoch_cooper
        epoch_id_cooper, model_cooper = train_utils.load_saved_model(opt.model_dir_cooper, model_cooper, epoch_id_cooper)
    else:
        epoch_id_cooper, model_cooper = train_utils.load_saved_model(opt.model_dir_cooper, model_cooper)

    model_without_ddp = model
    model_without_ddp_cooper = model_cooper

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)
        # 返回未经并行处理的单卡模型参数，方便保存模型参数等操作

        model_cooper = \
            torch.nn.parallel.DistributedDataParallel(model_cooper,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)
        model_without_ddp = model.module
        model_without_ddp_cooper = model_cooper.module

    model.eval()
    # model_without_ddp.state["ITHP"] = "inference_hete"
    # model_without_ddp.state["ego"] = True

    model_cooper.eval()
    # model_without_ddp_cooper.state["ITHP"] = "inference_hete"
    # model_without_ddp_cooper.state["ego"] = False

    # Create the dictionary for evaluation.
    # also store the confidence score for each prediction
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    total_comm_rates = []
    if opt.show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().show_coordinate_frame = True

        # used to visualize lidar points
        vis_pcd = o3d.geometry.PointCloud()
        # used to visualize object bounding box, maximum 50
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(50):
            vis_aabbs_gt.append(o3d.geometry.LineSet())
            vis_aabbs_pred.append(o3d.geometry.LineSet())

    coop_name = None
    for i, batch_data in tqdm(enumerate(data_loader)):
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            
            # inference neb middle info
            neb_info = model_without_ddp_cooper.encoder(batch_data['neb'])
            coop_name = neb_info['model_name']
            neb_psm_single = model_without_ddp_cooper.head(neb_info["features_2d"])['psm']
            neb_info["psm_single"] = neb_psm_single
            
            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_late_fusion(batch_data,
                                                          model,
                                                          opencood_dataset)
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_early_fusion(batch_data,
                                                           model,
                                                           opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_intermediate_fusion(batch_data,
                                                                  model,
                                                                  opencood_dataset,
                                                                  neb_info)
            elif opt.fusion_method == 'intermediate_with_comm':
                pred_box_tensor, pred_score, gt_box_tensor, comm_rates = \
                    inference_utils.inference_intermediate_fusion_withcomm(batch_data,
                                                                  model,
                                                                  opencood_dataset,
                                                                  neb_info)
                total_comm_rates.append(comm_rates)
            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')

            if pred_box_tensor is None:
                continue

            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.7)
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                   gt_box_tensor,
                                                   batch_data['ego'][
                                                       'origin_lidar'][0],
                                                   i,
                                                   npy_save_path)

            if opt.show_vis or opt.save_vis:
                vis_save_path = ''
                if opt.save_vis:
                    vis_save_path = os.path.join(opt.model_dir, 'vis')
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)

                opencood_dataset.visualize_result(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'],
                                                  opt.show_vis,
                                                  vis_save_path,
                                                  dataset=opencood_dataset)

            if opt.show_sequence:
                pcd, pred_o3d_box, gt_o3d_box = \
                    vis_utils.visualize_inference_sample_dataloader(
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data['ego']['origin_lidar'],
                        vis_pcd,
                        mode='constant'
                        )
                if i == 0:
                    vis.add_geometry(pcd)
                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_pred,
                                                 pred_o3d_box,
                                                 update_mode='add')

                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_gt,
                                                 gt_o3d_box,
                                                 update_mode='add')

                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_pred,
                                             pred_o3d_box)
                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_gt,
                                             gt_o3d_box)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.001)
            
            if opt.save_vis_n and opt.save_vis_n >i:

                vis_save_path = os.path.join(opt.model_dir, 'vis_3d')
                if not os.path.exists(vis_save_path):
                    os.makedirs(vis_save_path)
                vis_save_path = os.path.join(opt.model_dir, 'vis_3d/3d_%05d.png' % i)
                simple_vis.visualize(pred_box_tensor,
                                    gt_box_tensor,
                                    batch_data['ego']['origin_lidar'][0],
                                    hypes['postprocess']['gt_range'],
                                    vis_save_path,
                                    method='3d',
                                    left_hand=left_hand,
                                    vis_pred_box=True)
                
                vis_save_path = os.path.join(opt.model_dir, 'vis_bev')
                if not os.path.exists(vis_save_path):
                    os.makedirs(vis_save_path)
                vis_save_path = os.path.join(opt.model_dir, 'vis_bev/bev_%05d.png' % i)
                simple_vis.visualize(pred_box_tensor,
                                    gt_box_tensor,
                                    batch_data['ego']['origin_lidar'][0],
                                    hypes['postprocess']['gt_range'],
                                    vis_save_path,
                                    method='bev',
                                    left_hand=left_hand,
                                    vis_pred_box=True)

    if len(total_comm_rates) > 0:
        comm_rates = (sum(total_comm_rates)/len(total_comm_rates))
    else:
        comm_rates = 0

    ap_30, ap_50, ap_70 = eval_utils.eval_final_results(result_stat,
                                                        opt.model_dir,
                                                        opt.global_sort_detections)
    if opt.show_sequence:
        vis.destroy_window()
    
    with open(os.path.join(saved_path, 'result.txt'), 'a+') as f:
        msg = 'Cooope with: ' + coop_name + \
               '\n  Epoch: {} | AP @0.3: {:.04f} | AP @0.5: {:.04f} | AP @0.7: {:.04f} | comm_rate: {:.06f}\n'.format(epoch_id, ap_30, ap_50, ap_70, comm_rates)
       
        if opt.comm_thre is not None:
            msg = 'Cooope with: ' + coop_name + \
             '\n  Epoch: {} | AP @0.3: {:.04f} | AP @0.5: {:.04f} | AP @0.7: {:.04f} | comm_rate: {:.06f} | comm_thre: {:.04f}\n'.format(epoch_id, ap_30, ap_50, ap_70, comm_rates, opt.comm_thre)
        f.write(msg)
        print(msg)
    

if __name__ == '__main__':
    main()
