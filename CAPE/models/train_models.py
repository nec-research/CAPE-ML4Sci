"""
       <NAME OF THE PROGRAM THIS FILE BELONGS TO>

  File:     train_models.py
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
import sys, os
import hydra
from omegaconf import DictConfig

import operator
from functools import reduce
from functools import partial

from timeit import default_timer

sys.path.append(".")
from fno.train import run_training as run_training_FNO
from pinn.train import run_training as run_training_PINN
from unet.train import run_training as run_training_Unet
from MPNN.train import run_training as run_training_MPNN

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):

    if cfg.args.model_name == "FNO":
        print("FNO")
        run_training_FNO(
            if_training=cfg.args.if_training,
            continue_training=cfg.args.continue_training,
            num_workers=cfg.args.num_workers,
            modes=cfg.args.modes,
            width=cfg.args.width,
            initial_step=cfg.args.initial_step,
            t_train=cfg.args.t_train,
            num_channels=cfg.args.num_channels,
            batch_size=cfg.args.batch_size,
            epochs=cfg.args.epochs,
            learning_rate=cfg.args.learning_rate,
            scheduler_step=cfg.args.scheduler_step,
            scheduler_gamma=cfg.args.scheduler_gamma,
            model_update=cfg.args.model_update,
            flnm=cfg.args.filename,
            single_file=cfg.args.single_file,
            reduced_resolution=cfg.args.reduced_resolution,
            reduced_resolution_t=cfg.args.reduced_resolution_t,
            reduced_batch=cfg.args.reduced_batch,
            plot=cfg.args.plot,
            channel_plot=cfg.args.channel_plot,
            x_min=cfg.args.x_min,
            x_max=cfg.args.x_max,
            y_min=cfg.args.y_min,
            y_max=cfg.args.y_max,
            t_min=cfg.args.t_min,
            t_max=cfg.args.t_max,
            if_param_embed=cfg.args.if_param_embed,
            widening_factor=cfg.args.widening_factor,
            kernel_size=cfg.args.kernel_size,
            num_params=cfg.args.num_params,
            num_PrmEmb_Pre=cfg.args.num_PrmEmb_Pre,
            if_L1loss=cfg.args.if_L1loss,
            num_channels_PrmEmb=cfg.args.num_channels_PrmEmb,
            PrmEmb_coeff=cfg.args.PrmEmb_coeff,
            warmup_steps=cfg.args.warmup_steps,
            if_11cnv=cfg.args.if_11cnv,
            if_save=cfg.args.if_save,
            if_save_data=cfg.args.if_save_data,
            if_load_data=cfg.args.if_load_data,
            gp_coef=cfg.args.gp_coef,
            gp_kk=cfg.args.gp_kk,
            if_plot=cfg.args.if_plot,
            pino_coef=cfg.args.pino_coef,
            if_crc=cfg.args.if_crc,
            test_ratio=cfg.args.test_ratio,
            main_loss_coef=cfg.args.main_loss_coef,
            if_TF=cfg.args.if_TF,
            if_CAPE_score=cfg.args.if_CAPE_score,
            if_conditional = cfg.args.if_conditional
        )
    elif cfg.args.model_name == "Unet":
        print("Unet")
        run_training_Unet(
            if_training=cfg.args.if_training,
            continue_training=cfg.args.continue_training,
            num_workers=cfg.args.num_workers,
            initial_step=cfg.args.initial_step,
            t_train=cfg.args.t_train,
            in_channels=cfg.args.in_channels,
            out_channels=cfg.args.out_channels,
            init_features=cfg.args.init_features,
            epochs=cfg.args.epochs,
            learning_rate=cfg.args.learning_rate,
            batch_size=cfg.args.batch_size,
            unroll_step=cfg.args.unroll_step,
            ar_mode=cfg.args.ar_mode,
            pushforward=cfg.args.pushforward,
            scheduler_step=cfg.args.scheduler_step,
            scheduler_gamma=cfg.args.scheduler_gamma,
            model_update=cfg.args.model_update,
            flnm=cfg.args.filename,
            single_file=cfg.args.single_file,
            reduced_resolution=cfg.args.reduced_resolution,
            reduced_resolution_t=cfg.args.reduced_resolution_t,
            reduced_batch=cfg.args.reduced_batch,
            plot=cfg.args.plot,
            channel_plot=cfg.args.channel_plot,
            x_min=cfg.args.x_min,
            x_max=cfg.args.x_max,
            y_min=cfg.args.y_min,
            y_max=cfg.args.y_max,
            t_min=cfg.args.t_min,
            t_max=cfg.args.t_max,
            if_param_embed=cfg.args.if_param_embed,
            widening_factor=cfg.args.widening_factor,
            kernel_size=cfg.args.kernel_size,
            num_params=cfg.args.num_params,
            num_PrmEmb_Pre=cfg.args.num_PrmEmb_Pre,
            if_L1loss=cfg.args.if_L1loss,
            num_channels_PrmEmb=cfg.args.num_channels_PrmEmb,
            PrmEmb_coeff=cfg.args.PrmEmb_coeff,
            warmup_steps=cfg.args.warmup_steps,
            if_11cnv=cfg.args.if_11cnv,
            if_save=cfg.args.if_save,
            if_save_data=cfg.args.if_save_data,
            if_load_data=cfg.args.if_load_data,
            gp_coef=cfg.args.gp_coef,
            gp_kk=cfg.args.gp_kk,
            pino_coef=cfg.args.pino_coef,
            if_crc=cfg.args.if_crc,
            if_conditional=cfg.args.if_conditional
        )
    elif cfg.args.model_name == "PINN":
        print("PINN")
        run_training_PINN(
            scenario=cfg.args.scenario,
            epochs=cfg.args.epochs,
            learning_rate=cfg.args.learning_rate,
            model_update=cfg.args.model_update,
            flnm=cfg.args.filename,
            input_ch=cfg.args.input_ch,
            output_ch=cfg.args.output_ch,
            root_path=cfg.args.root_path,
            val_num=cfg.args.val_num,
            if_periodic_bc=cfg.args.if_periodic_bc,
            aux_params=cfg.args.aux_params,
            seed=cfg.args.seed
        )
    elif cfg.args.model_name == "MPNN":
        print("MPNN")
        run_training_MPNN(
            if_training=cfg.args.if_training,
            num_workers=cfg.args.num_workers,
            initial_step=cfg.args.initial_step,
            t_train=cfg.args.t_train,
            batch_size=cfg.args.batch_size,
            unroll_step=cfg.args.unroll_step,
            epochs=cfg.args.epochs,
            learning_rate=cfg.args.learning_rate,
            scheduler_step=cfg.args.scheduler_step,
            scheduler_gamma=cfg.args.scheduler_gamma,
            model_update=cfg.args.model_update,
            flnm=cfg.args.filename,
            single_file=cfg.args.single_file,
            reduced_resolution=cfg.args.reduced_resolution,
            reduced_resolution_t=cfg.args.reduced_resolution_t,
            reduced_batch=cfg.args.reduced_batch,
            plot=cfg.args.plot,
            channel_plot=cfg.args.channel_plot,
            x_min=cfg.args.x_min,
            x_max=cfg.args.x_max,
            y_min=cfg.args.y_min,
            y_max=cfg.args.y_max,
            t_min=cfg.args.t_min,
            t_max=cfg.args.t_max,
            if_save=cfg.args.if_save,
            if_save_data=cfg.args.if_save_data,
            if_load_data=cfg.args.if_load_data,
            print_interval=cfg.args.print_interval,
            nr_gt_steps=cfg.args.nr_gt_steps,
            lr_decay=cfg.args.lr_decay,
            neighbors=cfg.args.neighbors,
            time_window=cfg.args.time_window,
            if_param_embed = cfg.args.if_param_embed
            )

if __name__ == "__main__":
    main()
    print("Done.")
