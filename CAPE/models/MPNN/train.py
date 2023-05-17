# -*- coding: utf-8 -*-
"""
       <NAME OF THE PROGRAM THIS FILE BELONGS TO>

  File:     train.py
  Authors:  Makoto Takamoto (makoto.takamoto@neclab.eu)


NEC Laboratories Europe GmbH, Copyright (c) <year>, All rights reserved.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.

       PROPRIETARY INFORMATION ---

SOFTWARE LICENSE AGREEMENT

ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution
or non-profit organization or self (called "Licensee" or "You" in this
Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
Agreement).  All rights not specifically granted to you in this Agreement
are reserved for Licensor.

RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
ownership of any copy of the Software (as defined below) licensed under this
Agreement and hereby grants to Licensee a personal, non-exclusive,
non-transferable license to use the Software for noncommercial research
purposes, without the right to sublicense, pursuant to the terms and
conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
Agreement, the term "Software" means (i) the actual copy of all or any
portion of code for program routines made accessible to Licensee by Licensor
pursuant to this Agreement, inclusive of backups, updates, and/or merged
copies permitted hereunder or subsequently supplied by Licensor,  including
all or any file structures, programming instructions, user interfaces and
screen formats and sequences as well as any and all documentation and
instructions related to it, and (ii) all or any derivatives and/or
modifications created or made by You to any of the items specified in (i).

CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
proprietary to Licensor, and as such, Licensee agrees to receive all such
materials and to use the Software only in accordance with the terms of this
Agreement.  Licensee agrees to use reasonable effort to protect the Software
from unauthorized use, reproduction, distribution, or publication. All
publication materials mentioning features or use of this software must
explicitly include an acknowledgement the software was developed by NEC
Laboratories Europe GmbH.

COPYRIGHT: The Software is owned by Licensor.

PERMITTED USES:  The Software may be used for your own noncommercial
internal research purposes. You understand and agree that Licensor is not
obligated to implement any suggestions and/or feedback you might provide
regarding the Software, but to the extent Licensor does so, you are not
entitled to any compensation related thereto.

DERIVATIVES: You may create derivatives of or make modifications to the
Software, however, You agree that all and any such derivatives and
modifications will be owned by Licensor and become a part of the Software
licensed to You under this Agreement.  You may only use such derivatives and
modifications for your own noncommercial internal research purposes, and you
may not otherwise use, distribute or copy such derivatives and modifications
in violation of this Agreement.

BACKUPS:  If Licensee is an organization, it may make that number of copies
of the Software necessary for internal noncommercial use at a single site
within its organization provided that all information appearing in or on the
original labels, including the copyright and trademark notices are copied
onto the labels of the copies.

USES NOT PERMITTED:  You may not distribute, copy or use the Software except
as explicitly permitted herein. Licensee has not been granted any trademark
license as part of this Agreement.  Neither the name of NEC Laboratories
Europe GmbH nor the names of its contributors may be used to endorse or
promote products derived from this Software without specific prior written
permission.

You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
whole or in part, or provide third parties access to prior or present
versions (or any parts thereof) of the Software.

ASSIGNMENT: You may not assign this Agreement or your rights hereunder
without the prior written consent of Licensor. Any attempted assignment
without such consent shall be null and void.

TERM: The term of the license granted by this Agreement is from Licensee's
acceptance of this Agreement by downloading the Software or by using the
Software until terminated as provided below.

The Agreement automatically terminates without notice if you fail to comply
with any provision of this Agreement.  Licensee may terminate this Agreement
by ceasing using the Software.  Upon any termination of this Agreement,
Licensee will delete any and all copies of the Software. You agree that all
provisions which operate to protect the proprietary rights of Licensor shall
remain in force should breach occur and that the obligation of
confidentiality described in this Agreement is binding in perpetuity and, as
such, survives the term of the Agreement.

FEE: Provided Licensee abides completely by the terms and conditions of this
Agreement, there is no fee due to Licensor for Licensee's use of the
Software in accordance with this Agreement.

DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
RELATED MATERIALS.

SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
provided as part of this Agreement.

EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
permitted under applicable law, Licensor shall not be liable for direct,
indirect, special, incidental, or consequential damages or lost profits
related to Licensee's use of and/or inability to use the Software, even if
Licensor is advised of the possibility of such damage.

EXPORT REGULATION: Licensee agrees to comply with any and all applicable
export control laws, regulations, and/or other laws related to embargoes and
sanction programs administered by law.

SEVERABILITY: If any provision(s) of this Agreement shall be held to be
invalid, illegal, or unenforceable by a court or other tribunal of competent
jurisdiction, the validity, legality and enforceability of the remaining
provisions shall not in any way be affected or impaired thereby.

NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
or remedy under this Agreement shall be construed as a waiver of any future
or other exercise of such right or remedy by Licensor.

GOVERNING LAW: This Agreement shall be construed and enforced in accordance
with the laws of Germany without reference to conflict of laws principles.
You consent to the personal jurisdiction of the courts of this country and
waive their rights to venue outside of Germany.

ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
entire agreement between Licensee and Licensor as to the matter set forth
herein and supersedes any previous agreements, understandings, and
arrangements between the parties relating hereto.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.

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
