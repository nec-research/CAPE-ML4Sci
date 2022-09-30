# -*- coding: utf-8 -*-
"""
*
*     SOFTWARE NAME
*
*        File:  train.py
*
*     Authors: Deleted for purposes of anonymity
*
*     Proprietor: Deleted for purposes of anonymity --- PROPRIETARY INFORMATION
*
* The software and its source code contain valuable trade secrets and shall be maintained in
* confidence and treated as confidential information. The software may only be used for
* evaluation and/or testing purposes, unless otherwise explicitly stated in the terms of a
* license agreement or nondisclosure agreement with the proprietor of the software.
* Any unauthorized publication, transfer to third parties, or duplication of the object or
* source code---either totally or in part---is strictly prohibited.
*
*     Copyright (c) 2022 Proprietor: Deleted for purposes of anonymity
*     All Rights Reserved.
*
* THE PROPRIETOR DISCLAIMS ALL WARRANTIES, EITHER EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO IMPLIED WARRANTIES OF MERCHANTABILITY
* AND FITNESS FOR A PARTICULAR PURPOSE AND THE WARRANTY AGAINST LATENT
* DEFECTS, WITH RESPECT TO THE PROGRAM AND ANY ACCOMPANYING DOCUMENTATION.
*
* NO LIABILITY FOR CONSEQUENTIAL DAMAGES:
* IN NO EVENT SHALL THE PROPRIETOR OR ANY OF ITS SUBSIDIARIES BE
* LIABLE FOR ANY DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES
* FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF INFORMATION, OR
* OTHER PECUNIARY LOSS AND INDIRECT, CONSEQUENTIAL, INCIDENTAL,
* ECONOMIC OR PUNITIVE DAMAGES) ARISING OUT OF THE USE OF OR INABILITY
* TO USE THIS PROGRAM, EVEN IF the proprietor HAS BEEN ADVISED OF
* THE POSSIBILITY OF SUCH DAMAGES.
*
* For purposes of anonymity, the identity of the proprietor is not given herewith.
* The identity of the proprietor will be given once the review of the
* conference submission is completed.
*
* THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
*
"""

import sys
import numpy as np
import math as mt
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import operator
from functools import reduce
from functools import partial

from timeit import default_timer

# torch.manual_seed(0)
# np.random.seed(0)

sys.path.append('.')
from .unet import UNet1d, UNet2d, UNet3d
from .utils import UNetDatasetSingle, UNetDatasetMult
from metrics import metrics
from PrmEmbd.PrmEmbd import PrmEmb_Block_1d, _wrap_model, L1, gradient_penalty
from loss.losses import PINO_loss, LpLoss

#@record
def run_training(if_training,
                 continue_training,
                 num_workers,
                 initial_step,
                 t_train,
                 in_channels,
                 out_channels,
                 init_features,
                 batch_size,
                 unroll_step,
                 ar_mode,
                 pushforward,
                 epochs,
                 learning_rate,
                 scheduler_step,
                 scheduler_gamma,
                 model_update,
                 flnm,
                 single_file,
                 reduced_resolution,
                 reduced_resolution_t,
                 reduced_batch,
                 plot,
                 channel_plot,
                 x_min,
                 x_max,
                 y_min,
                 y_max,
                 t_min,
                 t_max,
                 if_param_embed,
                 widening_factor,
                 kernel_size,
                 num_params,
                 num_PrmEmb_Pre,
                 if_L1loss,
                 num_channels_PrmEmb=1,
                 PrmEmb_coeff=1.e-1,
                 warmup_steps=5,
                 train_data=None,
                 val_data=None,
                 if_save=True,
                 if_return_data=False,
                 if_11cnv=False,
                 if_save_data=False,
                 if_load_data=False,
                 gp_coef=0,
                 gp_kk=1.,
                 pino_coef=0,
                 if_crc=False,
                 ):

    ### for distributed training
    ngpus_per_node = torch.cuda.device_count()

    if ngpus_per_node > 1:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = rank % ngpus_per_node
        print(f"Start running basic DDP example on rank {rank}.")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Epochs = {epochs}, learning rate = {learning_rate}, scheduler step = {scheduler_step}, scheduler gamma = {scheduler_gamma}')
    
    ################################################################
    # load data
    ################################################################
    
    if if_load_data:
        print('load Train/Val data...')
        with open('../data/'+flnm[0][:-5] + '_Unet_Train.pickle', 'rb') as f:
            train_data = pickle.load(f)
        with open('../data/'+flnm[0][:-5] + '_Unet_Val.pickle', 'rb') as f:
            val_data = pickle.load(f)

    if train_data is None:
        if single_file:
            # filename
            if len(flnm)==1:
                model_name = flnm[0][:-5] + '_Unet'
            else:
                model_name = flnm[0][:-5] + 'multi_Unet'

            # Initialize the dataset and dataloader
            train_data = UNetDatasetSingle(flnm,
                                           reduced_resolution=reduced_resolution,
                                           reduced_resolution_t=reduced_resolution_t,
                                           reduced_batch=reduced_batch,
                                           initial_step=initial_step)
            val_data = UNetDatasetSingle(flnm,
                                         reduced_resolution=reduced_resolution,
                                         reduced_resolution_t=reduced_resolution_t,
                                         reduced_batch=reduced_batch,
                                         initial_step=initial_step,
                                         if_test=True)
        
        else:
            # filename
            model_name = flnm + '_Unet'
    
            train_data = UNetDatasetMult(flnm,
                                         reduced_resolution=reduced_resolution,
                                         reduced_resolution_t=reduced_resolution_t,
                                         reduced_batch=reduced_batch)
            val_data = UNetDatasetMult(flnm,
                                       reduced_resolution=reduced_resolution,
                                       reduced_resolution_t=reduced_resolution_t,
                                       reduced_batch=reduced_batch,
                                       if_test=True,
                                       indexes=train_data.indexes)
        if if_return_data:
            return train_data, val_data

        if if_save_data:
            print('save Train/Val data...')
            pickle.dump(train_data, open('../data/' + flnm[0][:-5] + '_Unet_Train.pickle', "wb"))
            pickle.dump(val_data, open('../data/' + flnm[0][:-5] + '_Unet_Val.pickle', "wb"))
    else:
        if single_file:
            # filename
            if len(flnm)==1:
                model_name = flnm[0][:-5] + '_Unet'
            else:
                model_name = flnm[0][:-5] + 'multi_Unet'
        else:
            # filename
            model_name = flnm + '_Unet'

    gen_device='cuda'

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               num_workers=num_workers, shuffle=True,#)
                                               generator=torch.Generator(device=gen_device))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                             num_workers=num_workers, shuffle=False,#)
                                             generator=torch.Generator(device=gen_device))

    ################################################################
    # training and evaluation
    ################################################################
    _, _data, _ = next(iter(val_loader))
    dimensions = len(_data.shape)
    print('Spatial Dimension', dimensions - 3)
    if if_param_embed and num_PrmEmb_Pre > 0:
        _num_channels = in_channels * initial_step * (num_channels_PrmEmb + 1)
        _out_channels = out_channels * (num_channels_PrmEmb + 1)
    else:
        _num_channels = in_channels * initial_step
        _out_channels = out_channels

    if dimensions == 4:
        model = UNet1d(_num_channels, _out_channels, init_features).to(device)
        normed_shape = [_data.shape[1]]
    elif dimensions == 5:
        model = UNet2d(_num_channels, _out_channels, init_features).to(device)
        normed_shape = _data.shape[1:3]
    elif dimensions == 6:
        model = UNet3d(_num_channels, _out_channels, init_features).to(device)
        normed_shape = _data.shape[1:4]

    if ngpus_per_node > 1:
        process_group = dist.new_group()
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)

    if if_param_embed:
        model = _wrap_model(model,
                            widening_factor=widening_factor,
                            kernel_size=kernel_size,
                            num_params=num_params,
                            num_PrmEmb_Pre=num_PrmEmb_Pre,
                            num_channels=in_channels*initial_step,
                            out_channels=out_channels,
                            num_channels_PrmEmb=num_channels_PrmEmb,
                            if_11cnv=if_11cnv,
                            if_Unet=True,
                            n_dim=dimensions - 3,
                            normed_dim=normed_shape
                            ).to(device)
    else:
        model = _wrap_model(model,
                            widening_factor = widening_factor,
                            kernel_size = kernel_size,
                            num_params = num_params,
                            num_PrmEmb_Pre = 0,
                            if_Unet = True).to(device)

    if ngpus_per_node > 1:
        model = DDP(model, device_ids=[device])  # wrap model by DDP for distributed training

    if if_L1loss:
        reg = L1(weight=1.)

    # Set maximum time step of the data to train
    if t_train > _data.shape[-2]:
        t_train = _data.shape[-2]
    # Set maximum of unrolled time step for the pushforward trick
    if t_train - unroll_step < 1:
        unroll_step = t_train - 1

    if ar_mode:
        if pushforward:
            model_name = model_name + '-PF-' + str(unroll_step)
        if not pushforward:
            unroll_step = _data.shape[-2]
            model_name = model_name + '-AR'
    else:
        model_name = model_name + '-1-step'
        
    model_path = model_name + ".pt"
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters = {total_params}')

    if if_param_embed:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        optimizer_PrmEmb = torch.optim.Adam(params=model.PrmEmb_Pre.parameters(),
                                            lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    #loss_fn = nn.MSELoss(reduction="mean")
    loss_fn = LpLoss()
    loss_val_min = np.infty
    
    start_epoch = 0

    if not if_training:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        Lx, Ly, Lz = 1., 1., 1.
        errs = metrics(val_loader, model, Lx, Ly, Lz, plot, channel_plot,
                       model_name, x_min, x_max, y_min, y_max,
                       t_min, t_max, mode='Unet', initial_step=initial_step, t_train=t_train)
        pickle.dump(errs, open(model_name+'.pickle', "wb"))
            
        return

    # If desired, restore the network by loading the weights saved in the .pt
    # file
    if continue_training:
        print('Restoring model (that is the network\'s weights) from file...')
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.train()
        
        # Load optimizer state dict
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
                    
        start_epoch = checkpoint['epoch']
        loss_val_min = checkpoint['loss']

    print('start training...')
    
    if ar_mode:
    
        for ep in range(start_epoch, epochs):
            model.train()
            t1 = default_timer()
            train_l2_step = 0
            train_l2_full = 0
            train_l2_PrmEmb = 0
            cnt = 0

            for xx, yy, param in train_loader:
                loss = 0
                loss_PrmEmb = 0

                # xx: input tensor (first few time steps) [b, x1, ..., xd, t_init, v]
                # yy: target tensor [b, x1, ..., xd, t, v]
                # grid: meshgrid [b, x1, ..., xd, dims]
                xx = xx.to(device)
                yy = yy.to(device)
                param = param.to(device)
                
                # Initialize the prediction tensor
                pred = yy[..., :initial_step, :]
                
                # Extract shape of the input tensor for reshaping (i.e. stacking the
                # time and channels dimension together)
                inp_shape = list(xx.shape)
                inp_shape = inp_shape[:-2]
                inp_shape.append(-1)  # (nb, nx,.., -1)
        
                # Autoregressive loop
                for t in range(initial_step, t_train):
                    
                    if t < t_train-unroll_step and ep > warmup_steps:
                        with torch.no_grad():
                            # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                            inp = xx.reshape(inp_shape)
                            temp_shape = [0, -1]
                            temp_shape.extend([i for i in range(1,len(inp.shape)-1)])
                            inp = inp.permute(temp_shape)
                            
                            # Extract target at current time step
                            y = yy[..., t:t+1, :]
                    
                            # Model run
                            temp_shape = [0]
                            temp_shape.extend([i for i in range(2,len(inp.shape))])
                            temp_shape.append(1)
                            im, im_PrmEmb = model(inp, param)
                            im = im.permute(temp_shape).unsqueeze(-2)
                            # Concatenate the prediction at current time step into the
                            # prediction tensor
                            pred = torch.cat((pred, im), -2)
                
                            # Concatenate the prediction at the current time step to be used
                            # as input for the next time step
                            xx = torch.cat((xx[..., 1:, :], im), dim=-2)
                    
                    else:
                        # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                        inp = xx.reshape(inp_shape)
                        temp_shape = [0, -1]
                        temp_shape.extend([i for i in range(1,len(inp.shape)-1)])
                        inp = inp.permute(temp_shape)
                        
                        # Extract target at current time step
                        y = yy[..., t:t+1, :]
                    
                        # Model run
                        temp_shape = [0]
                        temp_shape.extend([i for i in range(2,len(inp.shape))])
                        temp_shape.append(1)
                        im, im_PrmEmb = model(inp, param)
                        im = im.permute(temp_shape).unsqueeze(-2)

                        # Loss calculation
                        _batch = im.size(0)
                        loss += loss_fn(im.reshape(_batch, -1), y.reshape(_batch, -1))
                        if im_PrmEmb is not None:
                            if (t < initial_step + 1 and ep < warmup_steps) or \
                                    (ep >= warmup_steps and t < t_train - num_channels_PrmEmb):
                                loss_PrmEmb += loss_fn(im_PrmEmb.reshape(_batch, -1), \
                                                       yy[..., t:t + num_channels_PrmEmb, :].reshape(_batch, -1))
                                loss += PrmEmb_coeff * loss_PrmEmb
                                train_l2_PrmEmb += loss_PrmEmb.item()

                        # Concatenate the prediction at current time step into the
                        # prediction tensor
                        pred = torch.cat((pred, im), -2)
            
                        # Concatenate the prediction at the current time step to be used
                        # as input for the next time step
                        xx = torch.cat((xx[..., 1:, :], im), dim=-2)
                            
                train_l2_step += loss.item()
                _batch = yy.size(0)
                _yy = yy[..., :t_train, :]  # if t_train is not -1
                l2_full = loss_fn(pred.reshape(_batch, -1), _yy.reshape(_batch, -1))
                train_l2_full += l2_full.item()

                # L1 norm on CNN Kernels
                if if_L1loss:
                    l1_loss = 0
                    for name, prm in model.named_parameters():
                        if name.split('.')[0][:3] == 'Prm' and name.split('.')[-2][:5] == 'CNN_1':
                            l1_loss += reg(prm)
                    loss += if_L1loss * l1_loss

                cnt += 1

                if if_param_embed:
                    if ep < warmup_steps:
                        optimizer_PrmEmb.zero_grad()
                        loss_PrmEmb.backward()
                        optimizer_PrmEmb.step()
                    else:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            train_l2_PrmEmb /= cnt * (unroll_step - num_channels_PrmEmb)
            train_l2_full /= cnt

            if ep % model_update == 0:
                val_l2_step = 0
                val_l2_full = 0
                val_l2_PrmEmb = 0
                cnt = 0
                with torch.no_grad():
                    for xx, yy, param in val_loader:
                        loss = 0
                        xx = xx.to(device)
                        yy = yy.to(device)
                        param = param.to(device)

                        pred = yy[..., :initial_step, :]
                        inp_shape = list(xx.shape)
                        inp_shape = inp_shape[:-2]
                        inp_shape.append(-1)
                
                        for t in range(initial_step, t_train):
                            inp = xx.reshape(inp_shape)
                            temp_shape = [0, -1]
                            temp_shape.extend([i for i in range(1,len(inp.shape)-1)])
                            inp = inp.permute(temp_shape)
                            y = yy[..., t:t+1, :]
                            temp_shape = [0]
                            temp_shape.extend([i for i in range(2,len(inp.shape))])
                            temp_shape.append(1)
                            im, im_PrmEmb = model(inp, param)
                            im = im.permute(temp_shape).unsqueeze(-2)
                            _batch = im.size(0)
                            loss += loss_fn(im.reshape(_batch, -1), y.reshape(_batch, -1))
                            if im_PrmEmb is not None and t < t_train - num_channels_PrmEmb:
                                val_l2_PrmEmb += loss_fn(im_PrmEmb.reshape(_batch, -1), \
                                                         yy[..., t:t + num_channels_PrmEmb, \
                                                         :].reshape(_batch, -1)).item()

                            pred = torch.cat((pred, im), -2)
                
                            xx = torch.cat((xx[..., 1:, :], im), dim=-2)
            
                        val_l2_step += loss.item()
                        _batch = yy.size(0)
                        _yy = yy[..., :t_train, :]
                        val_l2_full += loss_fn(pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()

                        cnt += 1

                    val_l2_full /= cnt
                    val_l2_PrmEmb /= cnt * (t_train - num_channels_PrmEmb - initial_step)

                    if val_l2_full < loss_val_min:
                        loss_val_min = val_l2_full
                        if if_save:
                            torch.save({
                                'epoch': ep,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss_val_min
                                }, model_path)

            t2 = default_timer()
            if ep > warmup_steps or not if_param_embed:
                scheduler.step()
            print('epoch: {0}, loss: {1:.5f}, t2-t1: {2:.5f}, trainL2: {3:.5f}, testL2: {4:.5f}, trainLP: {5:.5f}, testLP: {6:.5f}'\
                    .format(ep, loss.item(), t2 - t1, train_l2_step, val_l2_step, train_l2_PrmEmb, val_l2_PrmEmb))

    else:  # Teacher forcing
        for ep in range(start_epoch, epochs):
            model.train()
            t1 = default_timer()
            train_l2_step = 0
            train_l2_full = 0
            train_l2_PrmEmb = 0
            train_l2_PINO = 0
            cnt = 0

            for xx, yy, param in train_loader:
                loss = 0
                loss_PrmEmb = 0
                loss_PINO = 0

                # xx: input tensor (first few time steps) [b, x1, ..., xd, t_init, v]
                # yy: target tensor [b, x1, ..., xd, t, v]
                xx = xx.to(device)
                yy = yy.to(device)
                param = param.to(device)

                # Initialize the prediction tensor
                pred = yy[..., :initial_step, :]
                
                # Extract shape of the input tensor for reshaping (i.e. stacking the
                # time and channels dimension together)
                inp_shape = list(xx.shape)
                inp_shape = inp_shape[:-2]
                inp_shape.append(-1)

                # Autoregressive loop
                for t in range(initial_step, t_train):
                                        
                    # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                    inp = xx.reshape(inp_shape)
                    temp_shape = [0, -1]
                    temp_shape.extend([i for i in range(1,len(inp.shape)-1)])
                    inp = inp.permute(temp_shape)
                    inp = torch.normal(inp, 0.001)  # add random noise
                    
                    # Extract target at current time step
                    y = yy[..., t:t+1, :]
                
                    # Model run
                    temp_shape = [0]
                    temp_shape.extend([i for i in range(2,len(inp.shape))])
                    temp_shape.append(1)
                    im, im_PrmEmb = model(inp, param)
                    im = im.permute(temp_shape).unsqueeze(-2)

                    # Loss calculation
                    _batch = im.size(0)
                    loss += loss_fn(im.reshape(_batch, -1), y.reshape(_batch, -1))
                    if im_PrmEmb is not None and t < t_train - num_channels_PrmEmb:
                        loss_PrmEmb += loss_fn(im_PrmEmb.reshape(_batch, -1), \
                                               yy[..., t :t + num_channels_PrmEmb, :].reshape(
                                                   _batch, -1))
                        loss += PrmEmb_coeff * loss_PrmEmb
                        train_l2_PrmEmb += loss_PrmEmb.item()

                    # Concatenate the prediction at current time step into the
                    # prediction tensor
                    pred = torch.cat((pred, im), -2)

                    # gradient penalty
                    if if_param_embed and gp_coef and t == initial_step:
                        lossGP = gradient_penalty(model, inp, param,
                                                  kk=gp_kk, gp_weight=gp_coef)
                        loss += lossGP

                    # Concatenate the prediction at the current time step to be used
                    # as input for the next time step
                    # xx = torch.cat((xx[..., 1:, :], im), dim=-2)
                    if if_crc:
                        ep_norm = (ep - warmup_steps) / (epochs - warmup_steps)
                        #t_norm = ep_norm # linear
                        t_norm = 0.5 * (1. + mt.tanh( (ep_norm - 0.5)/0.2))  # tanh
                        t_ar = int(t_train * t_norm)
                        t_ar = min(t_train, max(1, t_ar)) # linear
                        if t < t_ar:
                            xx = torch.cat((xx[..., 1:, :], im), dim=-2)  # auto-regressible
                        else:
                            xx = torch.cat((xx[..., 1:, :], y), dim=-2)  # teacher forcing
                    else:
                        xx = torch.cat((xx[..., 1:, :], y), dim=-2)   # teacher forcing

        
                train_l2_step += loss.item()
                _batch = yy.size(0)
                _yy = yy[..., :t_train, :]
                l2_full = loss_fn(pred.reshape(_batch, -1), _yy.reshape(_batch, -1))
                train_l2_full += l2_full.item()

                # L1 norm on CNN Kernels
                if if_L1loss:
                    l1_loss = 0
                    for name, prm in model.named_parameters():
                        if name.split('.')[0][:3] == 'Prm' and name.split('.')[-2][:5] == 'CNN_1':
                            # print(name)
                            l1_loss += reg(prm)
                    loss += if_L1loss * l1_loss

                # PINO loss
                if pino_coef:
                    if model_name.split('_')[1] == 'Burgers':
                        mode_name = 'burgers'
                    elif model_name.split('_')[1] == 'Advection':
                        mode_name = 'advection'
                    else:
                        NotImplementedError
                    loss_PINO += PINO_loss(torch.permute(pred, (0, 2, 1, 3)),
                                           Lt=2., Lx=1.,
                                           param=param, mode=mode_name)
                    loss += loss_PINO * pino_coef
                    train_l2_PINO += loss_PINO.item()

                cnt += 1

                if if_param_embed:
                    if ep < warmup_steps:
                        optimizer_PrmEmb.zero_grad()
                        loss_PrmEmb.backward()
                        optimizer_PrmEmb.step()
                    else:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                else:
                    optimizer.zero_grad()
                    with torch.autograd.set_detect_anomaly(True):
                        loss.backward()
                    optimizer.step()

            train_l2_PrmEmb /= cnt * (t_train - num_channels_PrmEmb - initial_step)
            train_l2_PINO /= cnt
            train_l2_full /= cnt

            if ep % model_update == 0 or ep == epochs:
                val_l2_step = 0
                val_l2_full = 0
                val_l2_PrmEmb = 0
                val_l2_PINO = 0
                cnt = 0
                with torch.no_grad():
                    for xx, yy, param in val_loader:
                        loss = 0

                        xx = xx.to(device)
                        yy = yy.to(device)
                        param = param.to(device)

                        pred = yy[..., :initial_step, :]
                        inp_shape = list(xx.shape)
                        inp_shape = inp_shape[:-2]
                        inp_shape.append(-1)
                
                        for t in range(initial_step, t_train):
                            inp = xx.reshape(inp_shape)  # auto regressive
                            temp_shape = [0, -1]
                            temp_shape.extend([i for i in range(1,len(inp.shape)-1)])
                            inp = inp.permute(temp_shape)
                            y = yy[..., t:t+1, :]
                            temp_shape = [0]
                            temp_shape.extend([i for i in range(2,len(inp.shape))])
                            temp_shape.append(1)
                            im, im_PrmEmb = model(inp, param)
                            im = im.permute(temp_shape).unsqueeze(-2)
                            _batch = im.size(0)
                            loss += loss_fn(im.reshape(_batch, -1), y.reshape(_batch, -1))
                            if im_PrmEmb is not None and t < t_train - num_channels_PrmEmb:
                                val_l2_PrmEmb += loss_fn(im_PrmEmb.reshape(_batch, -1),\
                                                      yy[..., t : t + num_channels_PrmEmb,\
                                                      :].reshape(_batch, -1)).item()

                            pred = torch.cat((pred, im), -2)
                            xx = torch.cat((xx[..., 1:, :], im), dim=-2)  # auto-regressive
            
                        val_l2_step += loss.item()
                        _batch = yy.size(0)
                        _yy = yy[..., :t_train, :]
                        val_l2_full += loss_fn(pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()

                        # PINO loss
                        if model_name.split('_')[1] == 'Burgers':
                            mode_name = 'burgers'
                        elif model_name.split('_')[1] == 'Advection':
                            mode_name = 'advection'
                        else:
                            mode_name = 'nan'
                        if mode_name != 'nan':
                            loss_PINO += PINO_loss(torch.permute(pred, (0, 2, 1, 3)),
                                                   Lt=2., Lx=1.,
                                                   param=param, mode=mode_name)
                            val_l2_PINO += loss_PINO.item()

                        cnt += 1

                    val_l2_full /= cnt
                    val_l2_PINO /= cnt
                    val_l2_PrmEmb /= cnt * (t_train - num_channels_PrmEmb - initial_step)

                    if  val_l2_full < loss_val_min:
                        loss_val_min = val_l2_full
                        if if_save:
                            torch.save({
                                'epoch': ep,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss_val_min
                                }, model_path)
                    
                
            t2 = default_timer()
            if ep > warmup_steps or not if_param_embed:
                scheduler.step()
            if ngpus_per_node > 1:
                if rank == 0:
                    print('ep: {0}, loss: {1:.2f}, t2-t1: {2:.1f}, tr-L2: {3:.3f}, te-L2: {4:.3f}, tr-LP: {5:.3f}, '
                          'te-LP: {6:.3f}, tr-PINO: {7:.3f}, te-PINO: {8:.3f}' \
                          .format(ep, loss.item(), t2 - t1, train_l2_full, val_l2_full, train_l2_PrmEmb, val_l2_PrmEmb,
                                  train_l2_PINO, val_l2_PINO))
            else:
                print('ep: {0}, loss: {1:.2f}, t2-t1: {2:.1f}, tr-L2: {3:.3f}, te-L2: {4:.3f}, tr-LP: {5:.3f}, '
                      'te-LP: {6:.3f}, tr-PINO: {7:.3f}, te-PINO: {8:.3f}' \
                      .format(ep, loss.item(), t2 - t1, train_l2_full, val_l2_full, train_l2_PrmEmb, val_l2_PrmEmb,
                              train_l2_PINO, val_l2_PINO))

    # evaluation
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    Lx, Ly, Lz = 1., 1., 1.
    errs = metrics(val_loader, model, Lx, Ly, Lz, plot, channel_plot,
                   model_name, x_min, x_max, y_min, y_max,
                   t_min, t_max, mode='Unet', initial_step=initial_step, t_train=t_train)
    pickle.dump(errs, open(model_name+'.pickle', "wb"))


if __name__ == "__main__":
    run_training()
    print("Done.")