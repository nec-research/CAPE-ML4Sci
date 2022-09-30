## Unet
# base
#CUDA_VISIBLE_DEVICES='1' python3 train_models.py +args=config_PrmEmbd_2D.yaml ++args.if_save_data=True ++args.model_name='Unet'
CUDA_VISIBLE_DEVICES='1' python3 train_models.py +args=config_PrmEmbd_2D.yaml ++args.if_load_data=True ++args.model_name='Unet'
mv 2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_128_Trainmulti_Unet-1-step.pickle 2D_CFD_Rand_periodic_multi_Unet_PEB_Base_0.pickle
mv 2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_128_Trainmulti_Unet-1-step.pt 2D_CFD_Rand_periodic_multi_Unet_PEB_Base_0.pt
CUDA_VISIBLE_DEVICES='1' python3 train_models.py +args=config_PrmEmbd_2D.yaml ++args.if_load_data=True ++args.model_name='Unet'
mv 2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_128_Trainmulti_Unet-1-step.pickle 2D_CFD_Rand_periodic_multi_Unet_PEB_Base_1.pickle
mv 2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_128_Trainmulti_Unet-1-step.pt 2D_CFD_Rand_periodic_multi_Unet_PEB_Base_1.pt
CUDA_VISIBLE_DEVICES='1' python3 train_models.py +args=config_PrmEmbd_2D.yaml ++args.if_load_data=True ++args.model_name='Unet'
mv 2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_128_Trainmulti_Unet-1-step.pickle 2D_CFD_Rand_periodic_multi_Unet_PEB_Base_2.pickle
mv 2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_128_Trainmulti_Unet-1-step.pt 2D_CFD_Rand_periodic_multi_Unet_PEB_Base_2.pt

# base + PINO
#CUDA_VISIBLE_DEVICES='3' python3 train_models.py +args=config_PrmEmbd_2D.yaml ++args.if_load_data=True ++args.model_name='Unet' ++args.pino_coef=1.
#mv 2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_128_Trainmulti_Unet-1-step.pickle 2D_CFD_Rand_periodic_multi_Unet_PEB_BasePINO_0.pickle
#mv 2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_128_Trainmulti_Unet-1-step.pt 2D_CFD_Rand_periodic_multi_Unet_PEB_BasePINO_0.pt
#CUDA_VISIBLE_DEVICES='3' python3 train_models.py +args=config_PrmEmbd_2D.yaml ++args.if_load_data=True ++args.model_name='Unet' ++args.pino_coef=1.
#mv 2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_128_Trainmulti_Unet-1-step.pickle 2D_CFD_Rand_periodic_multi_Unet_PEB_BasePINO_1.pickle
#mv 2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_128_Trainmulti_Unet-1-step.pt 2D_CFD_Rand_periodic_multi_Unet_PEB_BasePINO_1.pt
#CUDA_VISIBLE_DEVICES='3' python3 train_models.py +args=config_PrmEmbd_2D.yaml ++args.if_load_data=True ++args.model_name='Unet' ++args.pino_coef=1.
##pmv 2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_128_Trainmulti_Unet-1-step.pickle 2D_CFD_Rand_periodic_multi_Unet_PEB_BasePINO_2.pickle
#mv 2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_128_Trainmulti_Unet-1-step.pt 2D_CFD_Rand_periodic_multi_Unet_PEB_BasePINO_2.pt

# PrmEmb
#CUDA_VISIBLE_DEVICES='1' python3 train_models.py +args=config_PrmEmbd_2D.yaml ++args.if_param_embed=True ++args.if_load_data=True ++args.model_name='Unet' ++args.init_features=30
#mv 2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_128_Trainmulti_Unet-1-step.pickle 2D_CFD_Rand_periodic_multi_Unet_PEB_newK5W64_0.pickle
#mv 2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_128_Trainmulti_Unet-1-step.pt 2D_CFD_Rand_periodic_multi_Unet_PEB_newK5W64_0.pt
#CUDA_VISIBLE_DEVICES='3' python3 train_models.py +args=config_PrmEmbd_2D.yaml ++args.if_param_embed=True ++args.if_load_data=True ++args.model_name='Unet' ++args.init_features=30
#mv 2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_128_Trainmulti_Unet-1-step.pickle 2D_CFD_Rand_periodic_multi_Unet_PEB_newK5W64_1.pickle
#mv 2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_128_Trainmulti_Unet-1-step.pt 2D_CFD_Rand_periodic_multi_Unet_PEB_newK5W64_1.pt
#CUDA_VISIBLE_DEVICES='3' python3 train_models.py +args=config_PrmEmbd_2D.yaml ++args.if_param_embed=True ++args.if_load_data=True ++args.model_name='Unet' ++args.init_features=30
#mv 2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_128_Trainmulti_Unet-1-step.pickle 2D_CFD_Rand_periodic_multi_Unet_PEB_newK5W64_2.pickle
#mv 2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_128_Trainmulti_Unet-1-step.pt 2D_CFD_Rand_periodic_multi_Unet_PEB_newK5W64_2.pt

# base + init_step=2
#CUDA_VISIBLE_DEVICES='3' python3 train_models.py +args=config_PrmEmbd_2D.yaml ++args.initial_step=2 ++args.model_name='Unet'
#mv 2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_128_Trainmulti_Unet-1-step.pickle 2D_CFD_Rand_periodic_multi_Unet_PEB_Baseinitsteps2_0.pickle
#mv 2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_128_Trainmulti_Unet-1-step.pt 2D_CFD_Rand_periodic_multi_Unet_PEB_Baseinitsteps2_0.pt
#CUDA_VISIBLE_DEVICES='3' python3 train_models.py +args=config_PrmEmbd_2D.yaml ++args.initial_step=2 ++args.model_name='Unet'
#mv 2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_128_Trainmulti_Unet-1-step.pickle 2D_CFD_Rand_periodic_multi_Unet_PEB_Baseinitsteps2_1.pickle
#mv 2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_128_Trainmulti_Unet-1-step.pt 2D_CFD_Rand_periodic_multi_Unet_PEB_Baseinitsteps2_1.pt
#CUDA_VISIBLE_DEVICES='3' python3 train_models.py +args=config_PrmEmbd_2D.yaml ++args.initial_step=2 ++args.model_name='Unet'
#mv 2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_128_Trainmulti_Unet-1-step.pickle 2D_CFD_Rand_periodic_multi_Unet_PEB_Baseinitsteps2_2.pickle
#mv 2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_128_Trainmulti_Unet-1-step.pt 2D_CFD_Rand_periodic_multi_Unet_PEB_Baseinitsteps2_2.pt
