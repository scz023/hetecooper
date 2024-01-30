# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import statistics
import time

import torch
import tqdm
import wandb
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.tools.train_homo import test
from opencood.tools import multi_gpu_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.models.main_module import MainModule

os.environ["WANDB_API_KEY"] = 'KEY'
os.environ["WANDB_MODE"] = "offline"


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=False,
                        help='data generation yaml file needed ')
    parser.add_argument("--hypes_yaml_cooper", type=str, required=False,
                        help='cooper data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--model_dir_cooper', default='',
                        help='Continued training cooper model path')
    parser.add_argument('--eval_epoch', type=str, default=None,
                        help='Set the checkpoint')
    parser.add_argument('--eval_epoch_cooper', type=str, default=None,
                        help='Set the checkpoint')
    parser.add_argument('--eval_epoch_proj', type=str, default=None,
                        help='Set the checkpoint')
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
    hypes_cooper = yaml_utils.load_yaml(opt.hypes_yaml_cooper, opt, cooper=True)
    # hypes_cooper['fuse_train']['train_params'] = hypes['fuse_train']['train_params']

    multi_gpu_utils.init_distributed_mode(opt)

    print('-----------------Dataset Building------------------')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True, dataset_cfg_neb=hypes_cooper)
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False, dataset_cfg_neb=hypes_cooper)
    opencood_test_dataset = build_dataset(hypes, visualize=True, train=False, dataset_cfg_neb=hypes_cooper)

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset,
                                         shuffle=False)       
        sampler_test = DistributedSampler(opencood_test_dataset,
                                         shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['fuse_train']['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train)
        val_loader = DataLoader(opencood_validate_dataset,
                        batch_size=hypes['fuse_train']['train_params']['batch_size'],
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
                                  batch_size=hypes['fuse_train']['train_params']['batch_size'],
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=False,
                                  drop_last=True)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['fuse_train']['train_params']['batch_size'],
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
    model_cooper = MainModule(hypes_cooper['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epoch_id, epoch_id_proj, epoch_id_cooper = None, None, None
    if opt.eval_epoch is not None:
        epoch_id = opt.eval_epoch
    # if opt.eval_epoch_proj is not None:
    #     epoch_id_proj = opt.eval_epoch
    if opt.eval_epoch_cooper is not None:
        epoch_id_cooper = opt.eval_epoch_cooper
    
    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path,
                                                         model, epoch_id)
        init_epoch = init_epoch + 1        
        # init_epoch_proj, model = train_utils.load_saved_model_proj(saved_path,
        #                                                  model, epoch_id_proj)
    else:
        init_epoch = 0
        # init_epoch_proj = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)
        saved_path_cooper = train_utils.setup_train(hypes_cooper)

    # load ego proj parameters
    if opt.eval_epoch_proj is not None:
        epoch_id_proj = opt.eval_epoch_proj
        epoch_id_proj, model = train_utils.load_saved_model_proj(saved_path,
                                                         model, epoch_id_proj)
    else:
        epoch_id_proj, model = train_utils.load_saved_model_proj(saved_path,
                                                         model)
        # init_epoch_proj = init_epoch_proj + 1

    if opt.model_dir_cooper:
        saved_path_cooper = opt.model_dir_cooper
        _, model_cooper = train_utils.load_saved_model(saved_path_cooper,
                                                         model_cooper, epoch_id_cooper)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
        model_cooper.to(device)
    model_without_ddp = model
    model_without_ddp_cooper = model_cooper

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)
        model_without_ddp = model.module

        model_cooper = \
            torch.nn.parallel.DistributedDataParallel(model_cooper,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)
        model_without_ddp_cooper = model_cooper.module

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
        project='fuse'+hypes["name"]+time.strftime('%m_%d_%H_%M_%S'),
        # track hyperparameters and run metadata
        config={
        "learning_rate": hypes['fuse_train']["optimizer"]["lr"],
        "architecture": hypes["name"],
        "dataset": hypes["data_dir"],
        "epochs": hypes['fuse_train']["train_params"]["epoches"],
        }
    )


    for name, param in model_without_ddp.named_parameters():
        # only train fuse and head module
        if ('fuse' not in name) and ('head' not in name):
            param.requires_grad = False

    model_cooper.eval()

    # optimizer setup, already only transfer requires_grad parameters
    optimizer = train_utils.setup_optimizer_fuse(hypes, model_without_ddp)
    # lr scheduler setup
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular_fuse(hypes, optimizer, num_steps)


    print('Training start')
    epoches = hypes['fuse_train']['train_params']['epoches']
    # used to help schedule learning rate

    epoches = epoches + init_epoch  # 训练轮数在已有模型模型基础之上增加
    pre_ap = 0.0
    coop_name = None
    valid_ave_loss = 100.0
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes['fuse_train']['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        if hypes['fuse_train']['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)
        for param_group in optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])

        if opt.distributed:
            sampler_train.set_epoch(epoch)

        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)

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
                # inference neb feature
                neb_middle_info = model_without_ddp_cooper.encoder(batch_data['neb'])
                coop_name  = neb_middle_info['model_name']
                psm_single = model_without_ddp_cooper.head(neb_middle_info["features_2d"])['psm']
                neb_middle_info["psm_single"] = psm_single

                # compute loss of ego
                ouput_dict = model(batch_data['ego'], neb_middle_info)
                final_loss = criterion(ouput_dict,
                                       batch_data['ego']['label_dict'])
                if final_loss.isnan(): final_loss = 1e-6
            else:
                with torch.cuda.amp.autocast():
                    # inference neb feature
                    neb_middle_info = model_without_ddp_cooper.encoder(batch_data['neb'])
                    coop_name  = neb_middle_info['model_name']
                    psm_single = model_without_ddp_cooper.head(neb_middle_info["features_2d"])['psm']
                    neb_middle_info["psm_single"] = psm_single

                    # compute loss of ego
                    ouput_dict = model(batch_data['ego'], neb_middle_info)
                    final_loss = criterion(ouput_dict,
                                        batch_data['ego']['label_dict'])
                    if final_loss.isnan(): final_loss = 1e-6


            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)
            pbar2.update(1)

            if not opt.half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if hypes['fuse_train']['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)
            
            torch.cuda.empty_cache()

        if epoch % hypes['fuse_train']['train_params']['save_freq'] == 0:
            torch.save(model_without_ddp.state_dict(),
                os.path.join(saved_path, 'net_epoch%d.pth' % (epoch)))

        if epoch % hypes['fuse_train']['train_params']['eval_freq'] == 0:
            valid_ave_loss = []
            reg_losses = []
            conf_losses = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    model.eval()
                    batch_data = train_utils.to_device(batch_data, device)

                    # inference neb feature
                    neb_middle_info = model_without_ddp_cooper.encoder(batch_data['neb'])
                    # coop_name  = neb_middle_info['model_name']
                    psm_single = model_without_ddp_cooper.head(neb_middle_info["features_2d"])['psm']
                    neb_middle_info["psm_single"] = psm_single

                    # compute loss of ego
                    ouput_dict = model(batch_data['ego'], neb_middle_info)
                    final_loss = criterion(ouput_dict,
                                        batch_data['ego']['label_dict'])
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

        if epoch % hypes['fuse_train']['train_params']['test_freq'] == 0 and valid_ave_loss < 1.5 and epoch > 6:
            model.eval()
            ap_30, ap_50, ap_70, comm_rates = test(opt, hypes, test_loader, opencood_test_dataset, model_without_ddp, device, True, model_without_ddp_cooper)
            wandb.log({'ap_50': ap_50})
            wandb.log({'ap_70': ap_70})
            with open(os.path.join(saved_path, 'result.txt'), 'a+') as f:
                msg = 'Cooope with: ' + coop_name + \
                    '\n Epoch: {} | AP @0.3: {:.04f} | AP @0.5: {:.04f} | AP @0.7: {:.04f} | comm_rate: {:.06f}\n'.format(epoch, ap_30, ap_50, ap_70, comm_rates)
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

    time.sleep(2)
    # wandb.finish()


if __name__ == '__main__':
    main()
