# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import statistics

import torch
import tqdm
import wandb
import time
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

from opencood.models.main_module import MainModule
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.tools import multi_gpu_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils import eval_utils
from opencood.visualization import simple_vis
# os.environ["WANDB_API_KEY"] = 'KEY'
# os.environ["WANDB_MODE"] = "offline"


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision.")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='whether to globally sort detections by confidence score.'
                             'If set to True, it is the mainstream AP computing method,'
                             'but would increase the tolerance for FP (False Positives).')
    parser.add_argument('--save_vis_n', type=int, default=10,
                        help='save how many numbers of visualization result?')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    multi_gpu_utils.init_distributed_mode(opt)

    print('-----------------Dataset Building------------------')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)
    opencood_test_dataset = build_dataset(hypes, visualize=True, train=False)

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset,
                                         shuffle=False)
        sampler_test = DistributedSampler(opencood_test_dataset,
                                         shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                sampler=sampler_val,
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                drop_last=False)
        test_loader = DataLoader(opencood_test_dataset,
                             batch_size=1,
                             sampler = sampler_test,
                             num_workers=8,
                             collate_fn=opencood_test_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params']['batch_size'],
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=False,
                                  drop_last=True)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=True)
        test_loader = DataLoader(opencood_test_dataset,
                             batch_size=1,
                             num_workers=8,
                             collate_fn=opencood_test_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('---------------Creating Model------------------')
    model = MainModule(hypes['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path,
                                                         model)
        init_epoch = init_epoch + 1
    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)
        model_without_ddp = model.module

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    # lr scheduler setup
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    # record training
    writer = SummaryWriter(saved_path)

    torch.autograd.set_detect_anomaly(True)
    max_norm = 1.0  # 设置梯度的最大范数

    wandb.init(
        # set the wandb project where this run will be logged
        project=hypes["name"]+time.strftime('%m_%d_%H_%M_%S'),
        # track hyperparameters and run metadata
        config={
        "learning_rate": hypes["optimizer"]["lr"],
        "architecture": hypes["name"],
        "dataset": hypes["data_dir"],
        "epochs": hypes["train_params"]["epoches"],
        }
    )


    print('Training start')
    epoches = hypes['train_params']['epoches']
    pre_ap = 0
    valid_ave_loss = 100
    # used to help schedule learning rate

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)
        for param_group in optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])

        if opt.distributed:
            sampler_train.set_epoch(epoch)

        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)

        train_ave_loss = []
        for i, batch_data in enumerate(train_loader):
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)

            # case1 : late fusion train --> only ego needed,
            # and ego is random selected
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well

            if not opt.half:
                ouput_dict = model(batch_data['ego'])
                final_loss = criterion(ouput_dict,
                                       batch_data['ego']['label_dict'])
                if final_loss.isnan(): final_loss = 1e-6
                train_ave_loss.append(final_loss.item())
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    final_loss = criterion(ouput_dict,
                                        batch_data['ego']['label_dict'])
                    if final_loss.isnan(): final_loss = 1e-6
                    train_ave_loss.append(final_loss.item())       

            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)
            pbar2.update(1)

            if not opt.half:
                final_loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model_without_ddp.parameters(), max_norm=10, norm_type=2)
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                torch.nn.utils.clip_grad_norm_(parameters=model_without_ddp.parameters(), max_norm=10, norm_type=2)
                scaler.step(optimizer)
                scaler.update()

            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)
            
            torch.cuda.empty_cache()
            
        train_ave_loss = statistics.mean(train_ave_loss)
        wandb.log({"train_loss": train_ave_loss})

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model_without_ddp.state_dict(),
                os.path.join(saved_path, 'net_epoch%d.pth' % (epoch)))
            
        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []
            reg_losses = []
            conf_losses = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    # inf_start = time.time()
                    ouput_dict = model(batch_data['ego'])
                    # inf_end = time.time()
                    # print("inf time{}".format(inf_end - inf_start))

                    # loss_start = time.time()
                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    # loss_end = time.time()
                    # print("loss time{}".format(loss_end - loss_start))
                    valid_ave_loss.append(final_loss.item())
                    reg_losses.append(criterion.loss_dict['reg_loss'].item())
                    conf_losses.append(criterion.loss_dict['conf_loss'].item())
            valid_ave_loss = statistics.mean(valid_ave_loss)
            reg_loss = statistics.mean(reg_losses)
            conf_loss = statistics.mean(conf_losses)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)
            wandb.log({"val_loss": valid_ave_loss})
            wandb.log({"reg_loss": reg_loss})
            wandb.log({"conf_loss": conf_loss})
            time.sleep(0.1)
        torch.cuda.empty_cache()
        if valid_ave_loss < 1.5 and epoch % hypes['train_params']['test_freq'] == 0:
            ap_30, ap_50, ap_70, comm_rates = test(opt, hypes, test_loader, opencood_test_dataset, model_without_ddp, device)
            wandb.log({'ap_50': ap_50})
            wandb.log({'ap_70': ap_70})
            with open(os.path.join(saved_path, 'result.txt'), 'a+') as f:
                msg = 'Epoch: {} | AP @0.3: {:.04f} | AP @0.5: {:.04f} | AP @0.7: {:.04f} | comm_rate: {:.06f}\n'.format(epoch, ap_30, ap_50, ap_70, comm_rates)
                # if opt.comm_thre is not None:
                    # msg = 'Epoch: {} | AP @0.3: {:.04f} | AP @0.5: {:.04f} | AP @0.7: {:.04f} | comm_rate: {:.06f} | comm_thre: {:.04f}\n'.format(epoch, ap_30, ap_50, ap_70, comm_rates, opt.comm_thre)
                f.write(msg)
                print(msg)
            time.sleep(0.1)
            # 若模型出现过拟合, 中止训练
            if pre_ap > ap_50 + 0.3:
                break
            pre_ap = ap_50


    print('Training Finished, checkpoints saved to %s' % saved_path)
    torch.cuda.empty_cache()
    # wandb.finish()

def test(opt, hypes, data_loader, opencood_dataset, model, device, neb_info = False, model_cooper = None):
    left_hand = True if "OPV2V" in hypes['root_dir'] else False
    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    total_comm_rates = []
    # total_box = []
    for i, batch_data in tqdm.tqdm(enumerate(data_loader)):
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            
            if neb_info:
                # inference neb middle info
                neb_info = model_cooper.encoder(batch_data['neb'])
                coop_name = neb_info['model_name']
                neb_psm_single = model_cooper.head(neb_info["features_2d"])['psm']
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
                if not neb_info:
                    pred_box_tensor, pred_score, gt_box_tensor = \
                        inference_utils.inference_intermediate_fusion(batch_data,
                                                                    model,
                                                                    opencood_dataset)
                else:
                    pred_box_tensor, pred_score, gt_box_tensor = \
                        inference_utils.inference_intermediate_fusion(batch_data,
                                                                  model,
                                                                  opencood_dataset,
                                                                  neb_info)
                
            elif opt.fusion_method == 'intermediate_with_comm':
                if not neb_info:
                    pred_box_tensor, pred_score, gt_box_tensor, comm_rates = \
                        inference_utils.inference_intermediate_fusion_withcomm(batch_data,
                                                                    model,
                                                                    opencood_dataset)
                else:
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

    # print('total_box: ', sum(total_box)/len(total_box))

    if len(total_comm_rates) > 0:
        comm_rates = (sum(total_comm_rates)/len(total_comm_rates))
    else:
        comm_rates = 0
    ap_30, ap_50, ap_70 = eval_utils.eval_final_results(result_stat, 
                                                        opt.model_dir,
                                                        opt.global_sort_detections)
    return ap_30, ap_50, ap_70, comm_rates


if __name__ == '__main__':
    main()
