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

import operator
from functools import reduce
from functools import partial

from timeit import default_timer

import optuna

# torch.manual_seed(0)
# np.random.seed(0)

sys.path.append('.')
from .models_gnn import MP_PDE_Solver
from .utils import MPNNDatasetSingle, MPNNDatasetMult
from .utils_graph import GraphCreator
from .train_helper import train, test
from metrics import metrics


def run_training(if_training,
                 num_workers,
                 initial_step,
                 t_train,
                 batch_size,
                 unroll_step,
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
                 train_data=None,
                 val_data=None,
                 if_save=True,
                 if_return_data=False,
                 if_save_data=False,
                 if_load_data=False,
                 print_interval=50,
                 nr_gt_steps=2,
                 lr_decay=1.e-1,
                 neighbors=3,
                 time_window=20,
                 if_param_embed=True
                 ):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Epochs = {epochs}, learning rate = {learning_rate}, scheduler step = {scheduler_step}, scheduler gamma = {scheduler_gamma}')
    
    ################################################################
    # load data
    ################################################################
    
    if if_load_data:
        print('load Train/Val data...')
        with open('../data/'+flnm[0][:-5] + '_MPNN_Train.pickle', 'rb') as f:
            train_data = pickle.load(f)
        with open('../data/'+flnm[0][:-5] + '_MPNN_Val.pickle', 'rb') as f:
            val_data = pickle.load(f)

    if train_data is None:
        if single_file:
            # filename
            if len(flnm)==1:
                model_name = flnm[0][:-5] + '_MPNN'
            else:
                model_name = flnm[0][:-5] + 'multi_MPNN'

            # Initialize the dataset and dataloader
            train_data = MPNNDatasetSingle(flnm,
                                           reduced_resolution=reduced_resolution,
                                           reduced_resolution_t=reduced_resolution_t,
                                           reduced_batch=reduced_batch,
                                           initial_step=initial_step,
                                           if_param_embed=if_param_embed)
            val_data = MPNNDatasetSingle(flnm,
                                         reduced_resolution=reduced_resolution,
                                         reduced_resolution_t=reduced_resolution_t,
                                         reduced_batch=reduced_batch,
                                         initial_step=initial_step,
                                         if_test=True,
                                         if_param_embed=if_param_embed
                                         )
        
        else:
            # filename
            model_name = flnm + '_MPNN'
    
            train_data = MPNNDatasetMult(flnm,
                                         reduced_resolution=reduced_resolution,
                                         reduced_resolution_t=reduced_resolution_t,
                                         reduced_batch=reduced_batch)
            val_data = MPNNDatasetMult(flnm,
                                       reduced_resolution=reduced_resolution,
                                       reduced_resolution_t=reduced_resolution_t,
                                       reduced_batch=reduced_batch,
                                       if_test=True,
                                       indexes=train_data.indexes)
        if if_return_data:
            return train_data, val_data

        if if_save_data:
            print('save Train/Val data...')
            pickle.dump(train_data, open('../data/' + flnm[0][:-5] + '_MPNN_Train.pickle', "wb"))
            pickle.dump(val_data, open('../data/' + flnm[0][:-5] + '_MPNN_Val.pickle', "wb"))
            # np.save('../data/' + flnm[0][:-5] + '_TrainData', train_data.data)
            # np.save('../data/' + flnm[0][:-5] + '_TrainPrm', train_data.params)
            # np.save('../data/' + flnm[0][:-5] + '_ValData', val_data.data)
            # np.save('../data/' + flnm[0][:-5] + '_ValPrm', val_data.params)
    else:
        if single_file:
            # filename
            if len(flnm)==1:
                model_name = flnm[0][:-5] + '_MPNN'
            else:
                model_name = flnm[0][:-5] + 'multi_MPNN'
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

    # Equation specific input variables
    # assuming 1D Advection or Burgers equation
    eq_variables = {}
    eq_variables['beta'] = 1.

    # set parameters into a dict:pde
    t_max = val_data.t[-1]
    t_min = val_data.t[0]
    x_max = val_data.x[-1]
    x_min = val_data.x[0]
    pde = {}
    pde['tmin'] = t_min
    pde['tmax'] = t_max
    pde['dt'] = val_data.t[1] - val_data.t[0]
    pde['t_res'] = val_data.t.shape[0]
    pde['x_res'] = val_data.x.shape[0]
    pde['L'] = x_max
    pde['edge'] = 'neighbor'

    graph_creator = GraphCreator(pde=pde,
                                 neighbors=neighbors,
                                 time_window=time_window,
                                 initial_steps=initial_step,
                                 t_resolution=pde['t_res'],
                                 x_resolution=pde['x_res']).to(device)

    model = MP_PDE_Solver(pde=pde,
                          time_window=time_window,
                          initial_steps=initial_step,
                          eq_variables=eq_variables).to(device)

    model_path = model_name + ".pt"

    # count model-weight parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Number of parameters: {params}')

    # Set maximum time step of the data to train
    if t_train > pde['t_res']:
        t_train = pde['t_res']

    ######  needing to be modified
    if not if_training:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        Lx, Ly, Lz = 1., 1., 1.
        errs = metrics(val_loader, model, Lx, Ly, Lz, plot, channel_plot,
                       model_name, x_min, x_max, y_min, y_max,
                       t_min, t_max, mode='MPNN', initial_step=initial_step, t_train=t_train,
                       t_res=pde['t_res'], graph_creator=graph_creator)
        pickle.dump(errs, open(model_name + '.pickle', "wb"))

        return
    ######  needing to modify

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[unroll_step, 5, 10, 15], gamma=lr_decay)

    # Training loop
    min_val_loss = 1.e30
    test_loss = 1.e30
    criterion = torch.nn.MSELoss(reduction="sum")
    for ep in range(epochs):
        print(f"Epoch {ep}")
        train(ep, model, optimizer, train_loader, graph_creator, criterion, device=device,
              unrolling=unroll_step, batch_size=batch_size, print_interval=print_interval)
        print("Evaluation on validation dataset:")
        val_loss = test(model, val_loader, graph_creator, criterion, device=device,
                        batch_size=batch_size, nr_gt_steps=nr_gt_steps, base_resolution=pde['x_res'])
        if(val_loss < min_val_loss):
            # Save model
            print('save model')
            if if_save:
                torch.save({
                    'epoch': ep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss
                }, model_path)

            min_val_loss = val_loss

        scheduler.step()

    print(f"Test loss: {test_loss}")

    ######  needing to modify
    # evaluation
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    Lx, Ly, Lz = 1., 1., 1.
    errs = metrics(val_loader, model, Lx, Ly, Lz, plot, channel_plot,
                   model_name, x_min, x_max, y_min, y_max,
                   t_min, t_max, mode='MPNN', initial_step=initial_step, t_train=t_train,
                   t_res = pde['t_res'], graph_creator=graph_creator)
    pickle.dump(errs, open(model_name+'.pickle', "wb"))
    ######  needing to modify

if __name__ == "__main__":
    run_training()
    print("Done.")
