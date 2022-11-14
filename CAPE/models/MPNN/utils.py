# -*- coding: utf-8 -*-
"""
*
*     SOFTWARE NAME
*
*        File:  utils.py
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

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import glob
import h5py
import numpy as np
import math as mt
import random

class MPNNDatasetSingle(Dataset):
    def __init__(self, filename,
                 initial_step=10,
                 saved_folder='../data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 if_test=False, test_ratio=0.1,
                 indexes=None,
                 if_param_embed=True):
        """
        
        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY

        """
        self.if_param_embed=if_param_embed

        if len(filename)==1:
            print('number of files is 1')
            self.data, param = self.init_single(filename[0],
                                                saved_folder,
                                                reduced_resolution,
                                                reduced_resolution_t,
                                                reduced_batch,
                                               )
            nb = self.data.shape[0]
            self.params = param * np.ones([nb, 1], dtype=np.float32)

            indexes = [i for i in range(nb)]
            random.shuffle(indexes)
            self.indexes = np.array(indexes, dtype=np.int)
            self.data = self.data[self.indexes]
        else:
            print('number of files is more than 1')
            for i, flnm in enumerate(filename):
                data, param = self.init_single(flnm,
                                               saved_folder,
                                               reduced_resolution,
                                               reduced_resolution_t,
                                               reduced_batch,
                                              )
                if i == 0:
                    dims = len(data.shape) # nb, nt, nx, .., nc
                    if dims == 4:  # 1D
                        _nb, nx, nt, nc = data.shape
                        nb = _nb * len(filename)
                        self.data = np.zeros([nb, nx, nt, nc],
                                             dtype=np.float32)
                    elif dims == 5:  # 2D
                        _nb, nx, ny, nt, nc = data.shape
                        nb = _nb * len(filename)
                        self.data = np.zeros([nb, nx, ny, nt, nc],
                                             dtype=np.float32)
                    elif dims == 6:  # 3D
                        _nb, nx, ny, nz, nt, nc = data.shape
                        nb = _nb * len(filename)
                        self.data = np.zeros([nb, nx, ny, nz, nt, nc],
                                             dtype=np.float32)

                    self.params = np.zeros([nb, len(param)], dtype=np.float32)
                print('param', param)
                print(_nb, nb, i)
                _nb = data.shape[0]
                self.data[i*_nb:(i+1)*_nb] = data
                for j in range((len(param))):
                    self.params[i*_nb:(i+1)*_nb, j] = param[j]
            if indexes is None:
                indexes = [i for i in range(nb)]
                random.shuffle(indexes)
            self.indexes = np.array(indexes, dtype=np.int)
            self.data = self.data[self.indexes]
            self.params = self.params[self.indexes]

        test_idx = int(self.data.shape[0] * test_ratio)
        if if_test:
            self.data = self.data[:test_idx]
            self.params = self.params[:test_idx]
        else:
            self.data = self.data[test_idx:]
            self.params = self.params[test_idx:]

        # Time steps used as initial conditions
        self.initial_step = initial_step

        self.params = self.params.astype(np.float32)

    def init_single(self, filename,
                    saved_folder='../data/',
                    reduced_resolution=1,
                    reduced_resolution_t=1,
                    reduced_batch=1,
                    ):

        # Define path to files
        root_path = os.path.abspath(saved_folder + filename)
        assert filename[-2:] != 'h5', 'HDF5 data is assumed!!'
        
        with h5py.File(root_path, 'r') as f:
            keys = list(f.keys())
            keys.sort()
            if 'tensor' not in keys:
                _data = np.array(f['density'], dtype=np.float32)  # batch, time, x,...
                idx_cfd = _data.shape
                if len(idx_cfd)==3:  # 1D
                    data = np.zeros([idx_cfd[0]//reduced_batch,
                                     idx_cfd[2]//reduced_resolution,
                                     mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                     3],
                                     dtype=np.float32)
                    #density
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    data[...,0] = _data   # batch, x, t, ch
                    # pressure
                    _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    data[...,1] = _data   # batch, x, t, ch
                    # Vx
                    _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    data[...,2] = _data   # batch, x, t, ch

                if len(idx_cfd)==4:  # 2D
                    data = np.zeros([idx_cfd[0]//reduced_batch,
                                     idx_cfd[2]//reduced_resolution,
                                     idx_cfd[3]//reduced_resolution,
                                     mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                     4],
                                     dtype=np.float32)
                    # density
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    data[...,0] = _data   # batch, x, t, ch
                    # pressure
                    _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    data[...,1] = _data   # batch, x, t, ch
                    # Vx
                    _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    data[...,2] = _data   # batch, x, t, ch
                    # Vy
                    _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    data[...,3] = _data   # batch, x, t, ch

                if len(idx_cfd)==5:  # 3D
                    data = np.zeros([idx_cfd[0]//reduced_batch,
                                     idx_cfd[2]//reduced_resolution,
                                     idx_cfd[3]//reduced_resolution,
                                     idx_cfd[4]//reduced_resolution,
                                     mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                     5],
                                    dtype=np.float32)

                    # density
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    data[...,0] = _data   # batch, x, t, ch
                    # pressure
                    _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    data[...,1] = _data   # batch, x, t, ch
                    # Vx
                    _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    data[...,2] = _data   # batch, x, t, ch
                    # Vy
                    _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    data[...,3] = _data   # batch, x, t, ch
                    # Vz
                    _data = np.array(f['Vz'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    data[...,4] = _data   # batch, x, t, ch

            else:  # scalar equations
                ## data dim = [t, x1, ..., xd, v]
                _data = np.array(f['tensor'], dtype=np.float32)  # batch, time, x,...
                if len(_data.shape) == 3:  # 1D
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    data = _data[:, :, :, None]  # batch, x, t, ch

                if len(_data.shape) == 4:  # 2D Darcy flow
                    # u: label
                    _data = _data[::reduced_batch,:,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                    #if _data.shape[-1]==1:  # if nt==1
                    #    _data = np.tile(_data, (1, 1, 1, 2))
                    data = _data
                    # nu: input
                    _data = np.array(f['nu'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch, None,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                    data = np.concatenate([_data, data], axis=-1)
                    data = data[:, :, :, :, None]  # batch, x, y, t, ch

            attrs = list(f.attrs.keys())
            param = []
            for attr in attrs:
                param.append(f.attrs[attr])

            # for MPNN model
            self.x = torch.tensor(np.array(f["x-coordinate"], dtype=np.float32)[::reduced_resolution], dtype=torch.float)
            self.t = torch.tensor(np.array(f["t-coordinate"], dtype=np.float32)[::reduced_resolution_t], dtype=torch.float)

        return data, param

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        variables = {}
        if self.if_param_embed:
            variables['beta'] = self.params[idx].squeeze()
        else:
            variables['beta'] = np.ones_like(self.params[0].squeeze())
        return self.data[idx], self.x, variables

class MPNNDatasetMult(Dataset):
    def __init__(self, filename,
                 initial_step=10,
                 saved_folder='../data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 if_test=False, test_ratio=0.1
                 ):
        """
        
        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional

        """
        
        # Define path to files
        self.file_path = os.path.abspath(saved_folder + filename + ".h5")
        
        # Extract list of seeds
        with h5py.File(self.file_path, 'r') as h5_file:
            data_list = sorted(h5_file.keys())

        test_idx = int(len(data_list) * (1-test_ratio))
        if if_test:
            self.data_list = np.array(data_list[test_idx:])
        else:
            self.data_list = np.array(data_list[:test_idx])
        
        # Time steps used as initial conditions
        self.initial_step = initial_step

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        
        # Open file and read data
        with h5py.File(self.file_path, 'r') as h5_file:
            seed_group = h5_file[self.data_list[idx]]
        
            # data dim = [t, x1, ..., xd, v]
            data = np.array(seed_group["data"], dtype='f')
            data = torch.tensor(data, dtype=torch.float)
            
            # convert to [x1, ..., xd, t, v]
            permute_idx = list(range(1,len(data.shape)-1))
            permute_idx.extend(list([0, -1]))
            data = data.permute(permute_idx)
        
        return data[...,:self.initial_step,:], data
