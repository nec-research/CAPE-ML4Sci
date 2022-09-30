## 'Unet'
# Burgers
# base
#CUDA_VISIBLE_DEVICES='3' python3 train_models.py +args=config_PrmEmbd_Bgs.yaml ++args.if_save_data=True ++args.width=36 ++args.model_name='Unet'
#mv 1D_Burgers_Sols_Nu0.002multi_Unet-1-step.pickle 1D_Burgers_Sols_Nu_multi_Unet_PEB_Base_0.pickle
#mv 1D_Burgers_Sols_Nu0.002multi_Unet-1-step.pt 1D_Burgers_Sols_Nu_multi_Unet_PEB_Base_0.pt
#CUDA_VISIBLE_DEVICES='3' python3 train_models.py +args=config_PrmEmbd_Bgs.yaml ++args.if_load_data=True ++args.width=36 ++args.model_name='Unet'
#mv 1D_Burgers_Sols_Nu0.002multi_Unet-1-step.pickle 1D_Burgers_Sols_Nu_multi_Unet_PEB_Base_1.pickle
#mv 1D_Burgers_Sols_Nu0.002multi_Unet-1-step.pt 1D_Burgers_Sols_Nu_multi_Unet_PEB_Base_1.pt
CUDA_VISIBLE_DEVICES='3' python3 train_models.py +args=config_PrmEmbd_Bgs.yaml ++args.if_load_data=True ++args.width=36 ++args.model_name='Unet'
mv 1D_Burgers_Sols_Nu0.002multi_Unet-1-step.pickle 1D_Burgers_Sols_Nu_multi_Unet_PEB_Base_2.pickle
mv 1D_Burgers_Sols_Nu0.002multi_Unet-1-step.pt 1D_Burgers_Sols_Nu_multi_Unet_PEB_Base_2.pt

# base + PINO
#CUDA_VISIBLE_DEVICES='3' python3 train_models.py +args=config_PrmEmbd_Bgs.yaml ++args.if_load_data=True ++args.width=36 ++args.model_name='Unet' ++args.pino_coef=1.
#mv 1D_Burgers_Sols_Nu0.002multi_Unet-1-step.pickle 1D_Burgers_Sols_Nu_multi_Unet_PEB_BasePINO_0.pickle
#mv 1D_Burgers_Sols_Nu0.002multi_Unet-1-step.pt 1D_Burgers_Sols_Nu_multi_Unet_PEB_BasePINO_0.pt
#CUDA_VISIBLE_DEVICES='3' python3 train_models.py +args=config_PrmEmbd_Bgs.yaml ++args.if_load_data=True ++args.width=36 ++args.model_name='Unet' ++args.pino_coef=1.
#mv 1D_Burgers_Sols_Nu0.002multi_Unet-1-step.pickle 1D_Burgers_Sols_Nu_multi_Unet_PEB_BasePINO_1.pickle
#mv 1D_Burgers_Sols_Nu0.002multi_Unet-1-step.pt 1D_Burgers_Sols_Nu_multi_Unet_PEB_BasePINO_1.pt
CUDA_VISIBLE_DEVICES='3' python3 train_models.py +args=config_PrmEmbd_Bgs.yaml ++args.if_load_data=True ++args.width=36 ++args.model_name='Unet'
mv 1D_Burgers_Sols_Nu0.002multi_Unet-1-step.pickle 1D_Burgers_Sols_Nu_multi_Unet_PEB_BasePINO_2.pickle
mv 1D_Burgers_Sols_Nu0.002multi_Unet-1-step.pt 1D_Burgers_Sols_Nu_multi_Unet_PEB_BasePINO_2.pt

# PrmEmb
#CUDA_VISIBLE_DEVICES='3' python3 train_models.py +args=config_PrmEmbd_Bgs.yaml ++args.if_param_embed=True ++args.if_load_data=True ++args.model_name='Unet'
#mv 1D_Burgers_Sols_Nu0.002multi_Unet-1-step.pickle 1D_Burgers_Sols_Nu_multi_Unet_PEB_newK5W64_0.pickle
#mv 1D_Burgers_Sols_Nu0.002multi_Unet-1-step.pt 1D_Burgers_Sols_Nu_multi_Unet_PEB_newK5W64_0.pt
#CUDA_VISIBLE_DEVICES='3' python3 train_models.py +args=config_PrmEmbd_Bgs.yaml ++args.if_param_embed=True ++args.if_load_data=True ++args.model_name='Unet'
#mv 1D_Burgers_Sols_Nu0.002multi_Unet-1-step.pickle 1D_Burgers_Sols_Nu_multi_Unet_PEB_newK5W64_1.pickle
#mv 1D_Burgers_Sols_Nu0.002multi_Unet-1-step.pt 1D_Burgers_Sols_Nu_multi_Unet_PEB_newK5W64_1.pt
CUDA_VISIBLE_DEVICES='3' python3 train_models.py +args=config_PrmEmbd_Bgs.yaml ++args.if_param_embed=True ++args.if_load_data=True ++args.model_name='Unet'
mv 1D_Burgers_Sols_Nu0.002multi_Unet-1-step.pickle 1D_Burgers_Sols_Nu_multi_Unet_PEB_newK5W64_2.pickle
mv 1D_Burgers_Sols_Nu0.002multi_Unet-1-step.pt 1D_Burgers_Sols_Nu_multi_Unet_PEB_newK5W64_2.pt

# base + initsteps = 2
#echo "initsteps = 2 for Burgers"
#CUDA_VISIBLE_DEVICES='3' python3 train_models.py +args=config_PrmEmbd_Bgs.yaml ++args.if_load_data=False ++args.initial_step=2 ++args.width=36 ++args.model_name='Unet'
#mv 1D_Burgers_Sols_Nu0.002multi_Unet-1-step.pickle 1D_Burgers_Sols_Nu_multi_Unet_PEB_Baseinitsteps2_0.pickle
#mv 1D_Burgers_Sols_Nu0.002multi_Unet-1-step.pt 1D_Burgers_Sols_Nu_multi_Unet_PEB_Baseinitsteps2_0.pt
#CUDA_VISIBLE_DEVICES='3' python3 train_models.py +args=config_PrmEmbd_Bgs.yaml ++args.if_load_data=False ++args.initial_step=2 ++args.width=36 ++args.model_name='Unet'
#mv 1D_Burgers_Sols_Nu0.002multi_Unet-1-step.pickle 1D_Burgers_Sols_Nu_multi_Unet_PEB_Baseinitsteps2_1.pickle
#mv 1D_Burgers_Sols_Nu0.002multi_Unet-1-step.pt 1D_Burgers_Sols_Nu_multi_Unet_PEB_Baseinitsteps2_1.pt
CUDA_VISIBLE_DEVICES='3' python3 train_models.py +args=config_PrmEmbd_Bgs.yaml ++args.if_load_data=False ++args.initial_step=2 ++args.width=36 ++args.model_name='Unet'
mv 1D_Burgers_Sols_Nu0.002multi_Unet-1-step.pickle 1D_Burgers_Sols_Nu_multi_Unet_PEB_Baseinitsteps2_2.pickle
mv 1D_Burgers_Sols_Nu0.002multi_Unet-1-step.pt 1D_Burgers_Sols_Nu_multi_Unet_PEB_Baseinitsteps2_2.pt
