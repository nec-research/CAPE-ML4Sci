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
import glob
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

import optuna

# torch.manual_seed(0)
# np.random.seed(0)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sys.path.append('.')
from .fno import FNO1d, FNO2d, FNO3d
from .fno_test import FNO1d_test
from .utils import FNODatasetSingle, FNODatasetMult
from metrics import metrics
from PrmEmbd.PrmEmbd import PrmEmb_Block_1d, _wrap_model, L1, gradient_penalty
from loss.losses import PINO_loss, LpLoss

def run_training(if_training,
                 continue_training,
                 num_workers,
                 modes,
                 width,
                 initial_step,
                 t_train,
                 num_channels,
                 batch_size,
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
                 if_optuna=False,
                 if_return_data=False,
                 if_11cnv=False,
                 if_save_data=False,
                 if_load_data=False,
                 gp_coef=0,
                 gp_kk=1.,
                 pino_coef=0,
                 if_crc=False,
                 if_plot=False,
                 test_ratio=0.1,
                 main_loss_coef=1.,
                 if_TF=True,
                 if_CAPE_score=False,
                 if_conditional=False
                 ):

    ### for distributed training
    ngpus_per_node = torch.cuda.device_count()

    #if ngpus_per_node > 1:
    #    dist.init_process_group("gloo")
    #    rank = dist.get_rank()
    #    device = rank % ngpus_per_node
    #    print(f"Start running basic DDP example on rank {rank}.")
    #else:
    #    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Epochs = {epochs}, learning rate = {learning_rate}, scheduler step = {scheduler_step}, scheduler gamma = {scheduler_gamma}')

    ################################################################
    # load data
    ################################################################

    if if_load_data:
        print('load Train/Val data...')
        with open('../data/'+flnm[0][:-5] + '_Train.pickle', 'rb') as f:
            train_data = pickle.load(f)
        with open('../data/'+flnm[0][:-5] + '_Val.pickle', 'rb') as f:
            val_data = pickle.load(f)

    if train_data is None:
        if single_file:
            # filename
            if len(flnm)==1:
                model_name = flnm[0][:-5] + '_FNO'
            else:
                model_name = flnm[0][:-5] + 'multi_FNO'

            # Initialize the dataset and dataloader
            train_data = FNODatasetSingle(flnm,
                                          reduced_resolution=reduced_resolution,
                                          reduced_resolution_t=reduced_resolution_t,
                                          reduced_batch=reduced_batch,
                                          initial_step=initial_step,
                                          test_ratio=test_ratio)
            val_data = FNODatasetSingle(flnm,
                                        reduced_resolution=reduced_resolution,
                                        reduced_resolution_t=reduced_resolution_t,
                                        reduced_batch=reduced_batch,
                                        initial_step=initial_step,
                                        if_test=True,
                                        indexes=train_data.indexes,
                                        test_ratio=test_ratio)

        else:
            # filename
            model_name = flnm + '_FNO'

            train_data = FNODatasetMult(flnm,
                                    reduced_resolution=reduced_resolution,
                                    reduced_resolution_t=reduced_resolution_t,
                                    reduced_batch=reduced_batch,
                                    test_ratio=test_ratio)
            val_data = FNODatasetMult(flnm,
                                  reduced_resolution=reduced_resolution,
                                  reduced_resolution_t=reduced_resolution_t,
                                  reduced_batch=reduced_batch,
                                  if_test=True,
                                  test_ratio=test_ratio)
        if if_return_data:
            return train_data, val_data

        if if_save_data:
            print('save Train/Val data...')
            pickle.dump(train_data, open('../data/'+flnm[0][:-5] + '_Train.pickle', "wb"))
            pickle.dump(val_data, open('../data/'+flnm[0][:-5] + '_Val.pickle', "wb"))
            #np.save('../data/' + flnm[0][:-5] + '_TrainData', train_data.data)
            #np.save('../data/' + flnm[0][:-5] + '_TrainPrm', train_data.params)
            #np.save('../data/' + flnm[0][:-5] + '_ValData', val_data.data)
            #np.save('../data/' + flnm[0][:-5] + '_ValPrm', val_data.params)
    else:
        if single_file:
            # filename
            if len(flnm)==1:
                model_name = flnm[0][:-5] + '_FNO'
            else:
                model_name = flnm[0][:-5] + 'multi_FNO'
        else:
            # filename
            model_name = flnm + '_FNO'

    if if_optuna:
        gen_device='cpu'
    else:
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

    _, _data, _, _param = next(iter(val_loader))
    dimensions = len(_data.shape)
    print('Spatial Dimension', dimensions - 3)
    if if_param_embed and num_PrmEmb_Pre > 0:
        _num_channels = num_channels * (num_channels_PrmEmb + 1)
    elif if_conditional:
        _num_channels = num_channels + _param.size(-1)
    else:
        _num_channels = num_channels
    if dimensions == 4:
        #model = FNO1d_test(num_channels=_num_channels,
        model = FNO1d(num_channels=_num_channels,
                      width=width,
                      modes=modes,
                      initial_step=initial_step).to(device)
        normed_shape = [_data.shape[1]]
    elif dimensions == 5:
        model = FNO2d(num_channels=_num_channels,
                      width=width,
                      modes1=modes,
                      modes2=modes,
                      initial_step=initial_step).to(device)
        normed_shape = _data.shape[1:3]
    elif dimensions == 6:
        model = FNO3d(num_channels=_num_channels,
                      width=width,
                      modes1=modes,
                      modes2=modes,
                      modes3=modes,
                      initial_step=initial_step).to(device)
        normed_shape = _data.shape[1:4]

    if if_param_embed:
        model = _wrap_model(model,
                            widening_factor=widening_factor,
                            kernel_size=kernel_size,
                            num_params=num_params,
                            num_PrmEmb_Pre=num_PrmEmb_Pre,
                            num_channels=num_channels,
                            num_channels_PrmEmb=num_channels_PrmEmb,
                            if_11cnv=if_11cnv,
                            n_dim=dimensions-3,
                            normed_dim=normed_shape).to(device)
    else:
        model = _wrap_model(model,
                            widening_factor = widening_factor,
                            kernel_size = kernel_size,
                            num_params = num_params,
                            num_PrmEmb_Pre = 0).to(device)

    if ngpus_per_node > 1:
    #    model = DDP(model, device_ids=[device])  # wrap model by DDP for distributed training
        model = nn.DataParallel(model)  # wrap model by nn.DataParallel for distributed training
        model.to(device)

    if if_L1loss:
        reg = L1(weight=1.)

    # Set maximum time step of the data to train
    if t_train > _data.shape[-2]:
        t_train = _data.shape[-2]

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
                       t_min, t_max, initial_step=initial_step, t_train=t_train,
                       if_CAPE_score=if_CAPE_score, if_conditional=if_conditional)
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

    for ep in range(start_epoch, epochs):
        model.train()
        t1 = default_timer()
        train_l2_step = 0
        train_l2_full = 0
        train_l2_PrmEmb = 0
        train_l2_PINO = 0
        cnt = 0

        #if if_param_embed and ep >= warmup_steps:
        #    for param in model.PrmEmb_Pre.parameters():
        #        param.requires_grad = False

        for xx, yy, grid, param in train_loader:
            loss = 0
            loss_PrmEmb = 0
            loss_PINO = 0

            # xx: input tensor (first few time steps) [b, x1, ..., xd, t_init, v]
            # yy: target tensor [b, x1, ..., xd, t, v]
            # grid: meshgrid [b, x1, ..., xd, dims]
            if if_conditional:
                if dimensions == 4:
                    xx = torch.cat((xx, param[:,None,None,:].repeat(1, xx.size(1), xx.size(-2), 1)), dim=-1)
                elif dimensions == 5:
                    xx = torch.cat((xx, param[:,None,None,None,:].repeat(1, xx.size(1),xx.size(2), xx.size(-2), 1)), dim=-1)
                elif dimensions == 6:
                    xx = torch.cat((xx, param[:,None,None,None,None,:].repeat(1, xx.size(1), xx.size(2), xx.size(3), xx.size(-2), 1)), dim=-1)
            xx = xx.to(device)
            yy = yy.to(device)
            grid = grid.to(device)
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
                # for teacher forcing
                inp = torch.normal(inp, 0.001)  # add random noise

                # Extract target at current time step
                y = yy[..., t:t+1, :]

                # Model run
                im, im_PrmEmb = model(inp, param, grid)
                if if_conditional:
                    im = im[..., :yy.size(-1)]

                # Loss calculation
                _batch = im.size(0)
                loss += loss_fn(im.reshape(_batch, -1), y.reshape(_batch, -1)) * main_loss_coef
                if if_conditional:
                    if dimensions == 4:
                        y = torch.cat((y, param[:, None, None, :].repeat(1, xx.size(1), xx.size(-2), 1)), dim=-1)
                    elif dimensions == 5:
                        y = torch.cat(
                            (y, param[:, None, None, None, :].repeat(1, xx.size(1), xx.size(2), xx.size(-2), 1)),
                            dim=-1)
                    elif dimensions == 6:
                        y = torch.cat((y, param[:, None, None, None, None, :].repeat(1, xx.size(1), xx.size(2),
                                                                                       xx.size(3), xx.size(-2), 1)),
                                       dim=-1)
                #if im_PrmEmb is not None and t < initial_step + PrmEmb_unroll_step:
                if im_PrmEmb is not None and t < t_train - num_channels_PrmEmb:
                    loss_PrmEmb += loss_fn(im_PrmEmb.reshape(_batch, -1),\
                                           yy[..., t:t+num_channels_PrmEmb, :].reshape(_batch, -1))
                    loss += PrmEmb_coeff * loss_PrmEmb
                    train_l2_PrmEmb += loss_PrmEmb.item()

                # Concatenate the prediction at current time step into the
                # prediction tensor
                pred = torch.cat((pred, im), -2)

                # gradient penalty
                if if_param_embed and gp_coef and t == initial_step:
                    lossGP = gradient_penalty(model, inp, param, grid,
                                              kk=gp_kk, gp_weight=gp_coef)
                    loss += lossGP

                if if_conditional:
                    if dimensions == 4:
                        im = torch.cat((im, param[:, None, None, :].repeat(1, xx.size(1), xx.size(-2), 1)),
                                       dim=-1)
                    elif dimensions == 5:
                        im = torch.cat(
                            (im,
                             param[:, None, None, None, :].repeat(1, xx.size(1), xx.size(2), xx.size(-2), 1)),
                            dim=-1)
                    elif dimensions == 6:
                        im = torch.cat(
                            (im, param[:, None, None, None, None, :].repeat(1, xx.size(1), xx.size(2),
                                                                            xx.size(3), xx.size(-2), 1)),
                            dim=-1)
                # Concatenate the prediction at the current time step to be used
                if if_crc:
                    ep_norm = (ep - warmup_steps) / (epochs - warmup_steps)
                    #ep_norm = (ep - warmup_steps) / (epochs - 10 - warmup_steps)
                    #t_norm = ep_norm # linear
                    t_norm = 0.5 * (1. + mt.tanh( (ep_norm - 0.5)/0.2))  # tanh
                    t_ar = int(t_train * t_norm)
                    t_ar = min(t_train, max(1, t_ar))
                    if t < t_ar:
                    #if t < t_ar and t % 2 == 0:  # push-forward trick
                    #t_uplim = t_train//2
                    #if t < t_ar and t < t_uplim:
                        if main_loss_coef < 1.e-8:
                            PrmEmb_size = list(xx.shape)
                            PrmEmb_size[-2] = -1
                            xx = torch.cat((xx[..., 1:, :], im_PrmEmb.view(PrmEmb_size)[..., 0:1, :]), dim=-2)  # auto-regressible
                        else:
                            xx = torch.cat((xx[..., 1:, :], im), dim=-2)  # auto-regressible
                    else:
                        xx = torch.cat((xx[..., 1:, :], y), dim=-2)   # teacher forcing
                else:
                    if if_TF:
                        xx = torch.cat((xx[..., 1:, :], y), dim=-2)  # teacher forcing
                    else:
                        if main_loss_coef < 1.e-8:
                            PrmEmb_size = list(xx.shape)
                            PrmEmb_size[-2] = -1
                            xx = torch.cat((xx[..., 1:, :], im_PrmEmb.view(PrmEmb_size)[..., 0:1, :]), dim=-2)  # auto-regressible
                        else:
                            xx = torch.cat((xx[..., 1:, :], im), dim=-2)  # auto-regressible

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

            # PINO loss
            if pino_coef:
                if model_name.split('_')[1] == 'Burgers':
                    mode_name = 'burgers'
                elif model_name.split('_')[1] == 'Advection':
                    mode_name = 'advection'
                else:
                    NotImplementedError
                loss_PINO += PINO_loss(torch.permute(pred, (0, 2, 1, 3)),
                                       Lt = 2., Lx = 1.,
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
                loss.backward()
                optimizer.step()

        train_l2_PrmEmb /= cnt * (t_train - num_channels_PrmEmb - initial_step)
        train_l2_PINO /= cnt
        train_l2_full /= cnt

        if ep % model_update == 0:
            val_l2_step = 0
            val_l2_full = 0
            val_l2_PrmEmb = 0
            val_l2_PINO = 0
            cnt = 0
            with torch.no_grad():
                for xx, yy, grid, param in val_loader:
                    loss = 0

                    if if_conditional:
                        if dimensions == 4:
                            xx = torch.cat((xx, param[:, None, None, :].repeat(1, xx.size(1), xx.size(-2), 1)), dim=-1)
                        elif dimensions == 5:
                            xx = torch.cat(
                                (xx, param[:, None, None, None, :].repeat(1, xx.size(1), xx.size(2), xx.size(-2), 1)),
                                dim=-1)
                        elif dimensions == 6:
                            xx = torch.cat((xx, param[:, None, None, None, None, :].repeat(1, xx.size(1), xx.size(2),
                                                                                           xx.size(3), xx.size(-2), 1)),
                                           dim=-1)
                    xx = xx.to(device)
                    yy = yy.to(device)
                    grid = grid.to(device)
                    param = param.to(device)

                    pred = yy[..., :initial_step, :]
                    inp_shape = list(xx.shape)
                    inp_shape = inp_shape[:-2]
                    inp_shape.append(-1)

                    for t in range(initial_step, t_train):
                        inp = xx.reshape(inp_shape)
                        y = yy[..., t:t+1, :]

                        im, im_PrmEmb = model(inp, param, grid)
                        if if_conditional:
                            im = im[..., :yy.size(-1)]
                        _batch = im.size(0)

                        loss += loss_fn(im.reshape(_batch, -1), y.reshape(_batch, -1))
                        if im_PrmEmb is not None and t < t_train - num_channels_PrmEmb:
                            val_l2_PrmEmb += loss_fn(im_PrmEmb.reshape(_batch, -1),\
                                                     yy[..., t:t + num_channels_PrmEmb,\
                                                     :].reshape(_batch, -1)).item()

                        pred = torch.cat((pred, im), -2)
                        if if_conditional:
                            if dimensions == 4:
                                im = torch.cat((im, param[:, None, None, :].repeat(1, xx.size(1), xx.size(-2), 1)),
                                               dim=-1)
                            elif dimensions == 5:
                                im = torch.cat(
                                    (im,
                                     param[:, None, None, None, :].repeat(1, xx.size(1), xx.size(2), xx.size(-2), 1)),
                                    dim=-1)
                            elif dimensions == 6:
                                im = torch.cat(
                                    (im, param[:, None, None, None, None, :].repeat(1, xx.size(1), xx.size(2),
                                                                                    xx.size(3), xx.size(-2), 1)),
                                    dim=-1)
                        xx = torch.cat((xx[..., 1:, :], im), dim=-2)

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
                                               Lt = 2., Lx = 1.,
                                               param=param, mode=mode_name)
                        val_l2_PINO += loss_PINO.item()

                    cnt += 1

                val_l2_full /= cnt
                val_l2_PINO /= cnt
                val_l2_PrmEmb /= cnt * (t_train - num_channels_PrmEmb - initial_step)

                if val_l2_full < loss_val_min:
                    loss_val_min = val_l2_full
                    if if_save:
                        if ngpus_per_node > 1:  # assuming with nn.DataParallel
                            torch.save({
                                'epoch': ep,
                                'model_state_dict': model.module.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss_val_min
                            }, model_path)
                        else:
                            torch.save({
                                'epoch': ep,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss_val_min
                                }, model_path)

        t2 = default_timer()
        if ep > warmup_steps or not if_param_embed:
            scheduler.step()
        print('ep: {0}, loss: {1:.2f}, t2-t1: {2:.1f}, tr-L2: {3:.3f}, te-L2: {4:.3f}, tr-LP: {5:.3f}, '
              'te-LP: {6:.3f}, tr-PINO: {7:.3f}, te-PINO: {8:.3f}'\
                .format(ep, loss.item(), t2 - t1, train_l2_full, val_l2_full, train_l2_PrmEmb, val_l2_PrmEmb,
                        train_l2_PINO, val_l2_PINO))

    if if_optuna:
        return loss_val_min

    if if_plot:
        # plot results
        import matplotlib.pyplot as plt
        xx, yy, grid, param = next(iter(val_loader))

        xx = xx[0].view(1, xx.size(1), xx.size(2), xx.size(3)).repeat(10, 1, 1, 1)
        _param = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2., 4.], dtype=np.float32)
        param = torch.from_numpy(_param[:, None])

        xx = xx.to(device)
        grid = grid[:10].to(device)
        param = param.to(device)

        pred = yy[:10, ..., :initial_step, :]
        inp_shape = list(xx.shape)
        inp_shape = inp_shape[:-2]
        inp_shape.append(-1)
        for t in range(initial_step, t_train):
            inp = xx.reshape(inp_shape)
            im, _ = model(inp, param, grid)
            pred = torch.cat((pred, im), -2)
            xx = torch.cat((xx[..., 1:, :], im), dim=-2)

        pred = np.array(pred.data.cpu())
        fig, ax = plt.subplots(1,1, figsize=(6,6))
        for i in range(10):
            ax.plot(pred[i,:,5], label=str(_param[i])[:5])
        ax.legend()
        plt.savefig('evolve.pdf')
        plt.close()

    # evaluation
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    Lx, Ly, Lz = 1., 1., 1.
    errs = metrics(val_loader, model, Lx, Ly, Lz, plot, channel_plot,
                   model_name, x_min, x_max, y_min, y_max,
                   t_min, t_max, initial_step=initial_step, t_train=t_train,
                   if_CAPE_score=if_CAPE_score, if_conditional=if_conditional)
    pickle.dump(errs, open(model_name+'.pickle', "wb"))


def objective(trial, train_data, val_data):
    #learning_rate = trial.suggest_loguniform('lr', 1.e-5, 1.e-1)
    #learning_rate = trial.suggest_loguniform('lr', 1.e-4, 1.e-2)
    learning_rate = 3.e-3
    widening_factor = trial.suggest_categorical('widening_factor', [8, 16, 24, 32, 48, 64])
    #widening_factor = trial.suggest_categorical('widening_factor', [8, 24, 32, 64, 128])
    #widening_factor = 64
    #kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 15, 31])
    #kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])
    kernel_size = 5
    #num_PrmEmb_Pre = trial.suggest_int('num_PrmEmb_Pre', 1, 5)
    #num_PrmEmb_Pre = trial.suggest_int('num_PrmEmb_Pre', 1, 3)
    num_PrmEmb_Pre = 3
    #if_L1loss = trial.suggest_loguniform('L1', 1.e-6, 1.e-2)
    if_L1loss = 0
    #if_11cnv = trial.suggest_categorical('if_11cnv', [True, False])
    if_11cnv = True
    #num_channels_PrmEmb = trial.suggest_categorical('num_channels_PrmEmb', [1, 2, 4])
    num_channels_PrmEmb = 1
    PrmEmb_coeff = trial.suggest_loguniform('PrmEmb_coef', 1.e-6, 1.e-1)
    #warmup = trial.suggest_int('warmup', 1, 10)
    warmup = 3
    gp_coef = trial.suggest_loguniform('gp_coef', 1.e-6, 1.e0)
    #gp_coef = 0
    gp_kk = trial.suggest_loguniform('gp_kk', 1.e-2, 1.e1)
    #gp_kk = 0.
    #pino_coef = trial.suggest_loguniform('pino_coef', 1.e-3, 1.e0)
    pino_coef = 0

    flnm = ['2D_CFD_Rand_M0.1_Eta0.001_Zeta0.001_periodic_128_Train.hdf5',
            '2D_CFD_Rand_M1.0_Eta0.001_Zeta0.001_periodic_128_Train.hdf5',
            '2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5',
            '2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5',
            '2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5',
            '2D_CFD_Rand_M1.0_Eta0.1_Zeta0.1_periodic_128_Train.hdf5']

    val_score = run_training(if_training=True,
                 continue_training=False,
                 num_workers=0,
                 modes=12,
                 width=20,
                 initial_step=1,
                 t_train=200,
                 num_channels=4,
                 batch_size=20,
                 epochs=30,
                 learning_rate=learning_rate,
                 scheduler_step=5,
                 scheduler_gamma=0.5,
                 model_update=1,
                 flnm = flnm,
                 single_file=True,
                 reduced_resolution=2,
                 reduced_resolution_t=1,
                 reduced_batch=4,
                 plot=False,
                 channel_plot=False,
                 x_min=0.,
                 x_max=1.,
                 y_min=0.,
                 y_max=1.,
                 t_min=0.,
                 t_max=1.,
                 if_param_embed=True,
                 widening_factor=widening_factor,
                 kernel_size=kernel_size,
                 num_params=3,
                 num_PrmEmb_Pre=num_PrmEmb_Pre,
                 num_channels_PrmEmb=num_channels_PrmEmb,
                 PrmEmb_coeff=PrmEmb_coeff,
                 warmup_steps=warmup,
                 if_L1loss=if_L1loss,
                 train_data=train_data,
                 val_data=val_data,
                 if_optuna=True,
                 if_save=False,
                 if_11cnv=if_11cnv,
                 gp_coef=gp_coef,
                 gp_kk=gp_kk,
                 pino_coef=pino_coef,
                 if_crc=True
                )
    return val_score

def perform_optuna(n_trial):
    flnm = ['2D_CFD_Rand_M0.1_Eta0.001_Zeta0.001_periodic_128_Train.hdf5',
            '2D_CFD_Rand_M1.0_Eta0.001_Zeta0.001_periodic_128_Train.hdf5',
            '2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5',
            '2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5',
            '2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5',
            '2D_CFD_Rand_M1.0_Eta0.1_Zeta0.1_periodic_128_Train.hdf5']
    train_data, val_data = run_training(if_training=True,
                 continue_training=False,
                 num_workers=0,
                 modes=12,
                 width=20,
                 initial_step=1,
                 t_train=200,
                 num_channels=4,
                 batch_size=20,
                 epochs=25,
                 learning_rate=1.e-3,
                 scheduler_step=5,
                 scheduler_gamma=0.5,
                 model_update=1,
                 flnm = flnm,
                 single_file=True,
                 reduced_resolution=2,
                 reduced_resolution_t=1,
                 reduced_batch=4,
                 plot=False,
                 channel_plot=False,
                 x_min=0.,
                 x_max=1.,
                 y_min=0.,
                 y_max=1.,
                 t_min=0.,
                 t_max=1.,
                 if_param_embed=True,
                 widening_factor=None,
                 kernel_size=None,
                 num_params=3,
                 num_PrmEmb_Pre=None,
                 if_L1loss=None,
                 train_data=None,
                 val_data=None,
                 if_optuna=True,
                 if_return_data=True,
                 if_save=False
                )
    print('dataloaders are returned...')

    _objective = lambda trial: objective(trial, train_data, val_data)
    study = optuna.create_study()
    study.optimize(_objective, n_trials=n_trial)

    print('Acuracy: {}'.format(study.best_value))
    print('Best hyperparameters: {}'.format(study.best_params))
    #fname = 'FNO_LKS_Adv_study.pickle'
    #fname = 'FNO_LKS_BGS_study.pickle'
    #fname = 'FNO_vanilla_Adv_Attn_study.pickle'
    #fname = 'FNO_vanilla_Adv_PosPrm_study.pickle'
    #fname = 'FNO_vanilla_Adv_PEnewWarmup_study.pickle'
    #fname = 'FNO_vanilla_Adv_PEnew2WarmupCRC_study.pickle'
    #fname = 'FNO_MKL_PrmCh_study.pickle'
    #fname = 'FNO_Base_Adv_study.pickle'
    fname = 'FNO_vanilla_2DCFD_PEnew2WarmupCRC2_study.pickle'
    pickle.dump(study, open(fname, 'wb'))

if __name__ == "__main__":
    perform_optuna(n_trial=25)
    print("Done.")
