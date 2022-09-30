"""
*
*     SOFTWARE NAME
*
*        File:  train_models.py
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
                                    

import sys, os
import hydra
from omegaconf import DictConfig

import operator
from functools import reduce
from functools import partial

from timeit import default_timer

sys.path.append(".")
from fno.train import run_training as run_training_FNO
from unet.train import run_training as run_training_Unet

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
            if_CAPE_score=cfg.args.if_CAPE_score
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
        )


if __name__ == "__main__":
    main()
    print("Done.")
