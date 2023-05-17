"""
       <NAME OF THE PROGRAM THIS FILE BELONGS TO>

  File:     PrmEmbd.py
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

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY
"""

#!/usr/bin/env python3

"""
parameter embedding layer
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class PrmEmb_Block_1d(nn.Module):
    """A 2-layer parameter embedding module for 1D data."""

    def __init__(self,
                 widening_factor: int = 16,
                 kernel_size: int = 5,
                 num_params: int = 1,
                 if_11cnv = False,
                 num_channels: int = 1,
                 num_channels_PrmEmb: int = 1,
                 modes = 16,
                 normed_dim=[128]
                 ):
        super().__init__()
        self.num_params = num_params
        self.num_channels_PrmEmb = num_channels_PrmEmb

        self.CNN_0 = nn.Conv1d(num_channels, widening_factor//2,
                               kernel_size=kernel_size, bias=False, padding='same')
        self.avpool = nn.AvgPool1d(4, 4)
        self.CNN_SC = SpectralConv1d(widening_factor//2, widening_factor, modes1=modes)
        self.CNN_AD = nn.Conv1d(widening_factor//2, widening_factor,
                                kernel_size=kernel_size, bias=False, padding='same',
                                groups=widening_factor//2)
        self.CNN_1 = nn.Conv1d(widening_factor // 2, widening_factor, kernel_size=1)
        self.CNN_AC = nn.Conv1d(widening_factor//2, widening_factor, kernel_size=1)
        if if_11cnv:
            self.CNN_2 = nn.Conv1d(widening_factor, num_channels * num_channels_PrmEmb, kernel_size=1)
        else:
            self.CNN_2 = nn.Conv1d(widening_factor, num_channels * num_channels_PrmEmb,
                                   kernel_size=kernel_size, bias=False, padding='same')
        self.widening_factor = widening_factor

        self.fc0a = nn.Linear(self.num_params, self.widening_factor//2)
        self.fc0b = nn.Linear(self.widening_factor//2, self.widening_factor)
        self.fc1a = nn.Linear(self.num_params, self.widening_factor//2)
        self.fc1b = nn.Linear(self.widening_factor//2, self.widening_factor)
        self.fc2a = nn.Linear(self.num_params, self.widening_factor//2)
        self.fc2b = nn.Linear(self.widening_factor//2, self.widening_factor)

        self.LN = nn.LayerNorm(normed_dim)

    def forward(self, inputs):
        x, x_p = inputs
        # parameter embedding  (b, widening_factor)
        inp0 = self.fc0b(F.gelu(self.fc0a(x_p)))
        inp1 = self.fc1b(F.gelu(self.fc1a(x_p)))
        inp2 = self.fc2b(F.gelu(self.fc2a(x_p)))
        # CNN
        y = self.CNN_0(x)
        y = F.gelu(y)
        y0 = self.avpool(y)
        y0 = self.CNN_SC(y0) * inp0[:, :, None]
        y0 = F.interpolate(y0,  scale_factor=4)
        y1 = self.CNN_AC(y) * inp1[:, :, None]
        y2 = self.CNN_AD(y) * inp2[:, :, None]
        y = (y0 + y1 + y2) + self.CNN_1(y)
        y = F.gelu(y)
        y = self.CNN_2(y)

        return [x.repeat(1, self.num_channels_PrmEmb, 1) + self.LN(y),
                x_p]

class PrmEmb_Block_2d(nn.Module):
    """A 2-layer parameter embedding module for 2D data."""

    def __init__(self,
                 widening_factor: int = 16,
                 kernel_size: int = 5,
                 num_params: int = 1,
                 if_11cnv = False,
                 num_channels: int = 1,
                 num_channels_PrmEmb: int = 1,
                 modes = 9,
                 normed_dim=[64,64]
                 ):
        super().__init__()
        self.num_params = num_params
        self.num_channels_PrmEmb = num_channels_PrmEmb

        self.CNN_0 = nn.Conv2d(num_channels, widening_factor//2,
                               kernel_size=kernel_size, bias=False, padding='same')
        self.avpool = nn.AvgPool2d(4, 4)
        self.CNN_SC = SpectralConv2d_fast(widening_factor//2, widening_factor, modes1=modes, modes2=modes)
        self.CNN_AD = nn.Conv2d(widening_factor//2, widening_factor,
                                kernel_size=kernel_size, bias=False, padding='same',
                                groups=widening_factor//2)
        self.CNN_1 = nn.Conv2d(widening_factor // 2, widening_factor, kernel_size=1)
        self.CNN_AC = nn.Conv2d(widening_factor//2, widening_factor, kernel_size=1)
        if if_11cnv:
            self.CNN_2 = nn.Conv2d(widening_factor, num_channels * num_channels_PrmEmb, kernel_size=1)
        else:
            self.CNN_2 = nn.Conv2d(widening_factor, num_channels * num_channels_PrmEmb,
                                   kernel_size=kernel_size, bias=False, padding='same')
        self.widening_factor = widening_factor

        self.fc0a = nn.Linear(self.num_params, self.widening_factor//2)
        self.fc0b = nn.Linear(self.widening_factor//2, self.widening_factor)
        self.fc1a = nn.Linear(self.num_params, self.widening_factor//2)
        self.fc1b = nn.Linear(self.widening_factor//2, self.widening_factor)
        self.fc2a = nn.Linear(self.num_params, self.widening_factor//2)
        self.fc2b = nn.Linear(self.widening_factor//2, self.widening_factor)

        self.LN = nn.LayerNorm(normed_dim)

    def forward(self, inputs):
        x, x_p = inputs
        # parameter embedding  (b, widening_factor)
        inp0 = self.fc0b(F.gelu(self.fc0a(torch.sigmoid(torch.log(x_p)))))
        inp1 = self.fc1b(F.gelu(self.fc1a(torch.sigmoid(torch.log(x_p)))))
        inp2 = self.fc2b(F.gelu(self.fc2a(torch.sigmoid(torch.log(x_p)))))
        # CNN
        y = self.CNN_0(x)
        y = F.gelu(y)
        y0 = self.avpool(y)
        y0 = self.CNN_SC(y0) * inp0[:, :, None, None]
        y0 = F.interpolate(y0,  scale_factor=4)
        y1 = self.CNN_AC(y) * inp1[:, :, None, None]
        y2 = self.CNN_AD(y) * inp2[:, :, None, None]
        y = (y0 + y1 + y2) + self.CNN_1(y)
        y = F.gelu(y)
        y = self.CNN_2(y)

        return [x.repeat(1, self.num_channels_PrmEmb, 1, 1) * (1. + self.LN(y)),
                x_p]

class _wrap_model(nn.Module):
    def __init__(self,
                 model,
                 widening_factor: int = 16,
                 kernel_size: int = 5,
                 num_params: int = 1,
                 num_PrmEmb_Pre: int = 1,
                 num_channels=1,
                 num_channels_PrmEmb=1,
                 out_channels=None,
                 if_11cnv=False,
                 if_Unet=False,
                 n_dim=1,
                 normed_dim=[128]):
        super().__init__()

        self.model = model
        self.num_channels = num_channels
        self.num_PrmEmb_Pre = num_PrmEmb_Pre
        self.num_channels = num_channels
        self.num_channels_PrmEmb = num_channels_PrmEmb
        self.if_Unet = if_Unet

        if num_PrmEmb_Pre != 0:
            PrmEmb_Pre = []
            if n_dim == 1:
                block = PrmEmb_Block_1d
            elif n_dim == 2:
                block = PrmEmb_Block_2d
            else:
                NotImplementedError
            for num in range(num_PrmEmb_Pre):
                if num == 0:
                    PrmEmb_Pre.append(block(widening_factor, kernel_size, num_params,
                                            if_11cnv,
                                            num_channels=num_channels * num_channels_PrmEmb**num,
                                            num_channels_PrmEmb=num_channels_PrmEmb,
                                            normed_dim=normed_dim))
            self.PrmEmb_Pre = nn.Sequential(*PrmEmb_Pre)
            if n_dim == 1:
                block_11 = nn.Conv1d
            elif n_dim == 2:
                block_11 = nn.Conv2d
            else:
                NotImplementedError

            if out_channels is not None:  # Unet
                self.mix_emb = block_11((num_channels_PrmEmb + 1) * num_channels, out_channels, kernel_size=1)
            else:
                self.mix_emb = block_11((num_channels_PrmEmb + 1) * num_channels, num_channels, kernel_size=1)

    def forward(self, x, xp, grid=None):
        nd = len(x.size())
        if nd == 3:  # 1D
            in_dims = (0, 2, 1)  # batch, nc, nx
            out_dims = (0, 2, 1)  # batch, nx, nc
        elif nd == 4:  # 2D
            in_dims = (0, 3, 1, 2)  # batch, nc, nx, ny
            out_dims = (0, 2, 3, 1)  # batch, nx, ny, nc
        elif nd == 5:  # 3D
            in_dims = (0, 4, 1, 2, 3)  # batch, nc, nx, ny, nz
            out_dims = (0, 2, 3, 4, 1)  # batch, nx, ny, nz, nc
        else:
            NotImplementedError

        y = None
        if self.num_PrmEmb_Pre != 0:
            if not self.if_Unet:  # FNO
                y = torch.permute(x, in_dims)
                y, _ = self.PrmEmb_Pre([y, xp])
                y = torch.permute(y, out_dims)
                x = torch.cat((x, y), dim=-1)  # input parameter embedding as additional channel
            else:  # Unet (batch, nc, nx,...)
                y, _ = self.PrmEmb_Pre([x, xp])
                x = torch.cat((x, y), dim=1)  # input parameter embedding as additional channel
        if self.if_Unet:  # Unet
            x = self.model(x)
        else:  # FNO
            x = self.model(x, grid)

        if self.num_PrmEmb_Pre != 0:
            if not self.if_Unet:  # FNO  batch, nx, nc
                if nd == 3:
                    nb, nx, nt, nc = x.size()
                    x = x.view(nb, nx, -1)
                elif nd == 4:
                    nb, nx, ny, nt, nc = x.size()
                    x = x.view(nb, nx, ny, -1)
                elif nd == 5:
                    nb, nx, ny, nz, nt, nc = x.size()
                    x = x.view(nb, nx, ny, nz, -1)
                x = torch.permute(x, in_dims)
                x = self.mix_emb(x)
                x = torch.permute(x, out_dims)
                if nd == 3:
                    x = x.view(nb, nx, nt, self.num_channels)
                elif nd == 4:
                    x = x.view(nb, nx, ny, nt, self.num_channels)
                elif nd == 5:
                    x = x.view(nb, nx, ny, nz, nt, self.num_channels)
            else:  # Unet:  batch, nc, nx
                x = self.mix_emb(x)

        return x, y

class L1(nn.Module):
    def __init__(self, weight):
        super(L1, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(
                torch.abs(f)
            )
        return norm/factors[0].shape[0]

def gradient_penalty(model, inp, param, grid = None, kk = 1., gp_weight = 1.e-4):
    batch_size = param.size(0)

    _inp = inp.clone().detach().requires_grad_(True)
    #_param = param.clone().detach().requires_grad_(True)
    __param = param.clone().detach().requires_grad_(True)
    p_mx = __param.max() * 2.
    p_mn = F.relu(__param.min() * 0.5)
    _param = (torch.randn(param.size(), requires_grad=True).cuda() + 1.) * 0.5  # (-1,1) --> (0, 1)
    _param = p_mn + _param * (p_mx - p_mn)  # (0, 1) --> (pmin, pmax)

    if grid is None:
        output, _ = model(_inp, _param)
    else:
        _grid = grid.clone().detach().requires_grad_(True)
        output, _ = model(_inp, _param, _grid)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(outputs=output,
                                    inputs=_param,
                                    grad_outputs=torch.ones(output.size()).cuda(),
                                    create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, ...),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return gp_weight * ((gradients_norm - kk) ** 2).mean()
