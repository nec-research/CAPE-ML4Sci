## 'MPNN'
# base
CUDA_VISIBLE_DEVICES='0' python3 train_models.py +args=config_PrmEmbd_Bgs.yaml ++args.if_save_data=True ++args.model_name='MPNN' ++args.unroll_step=1 ++args.learning_rate=1.e-4 args.epochs=20
mv 1D_Burgers_Sols_Nu0.002multi_MPNN.pickle 1D_Burgers_Sols_Nu_multi_MPNN_PEB_Base_0.pickle
mv 1D_Burgers_Sols_Nu0.002multi_MPNN.pt 1D_Burgers_Sols_Nu_multi_MPNN_PEB_Base_0.pt

# PrmEmb
CUDA_VISIBLE_DEVICES='3' python3 train_models.py +args=config_PrmEmbd_Bgs.yaml ++args.if_param_embed=True ++args.if_load_data=True ++args.model_name='MPNN' ++args.unroll_step=1 ++args.learning_rate=1.e-4 args.epochs=20
mv 1D_Burgers_Sols_Nu0.002multi_MPNN.pickle 1D_Burgers_Sols_Nu_multi_MPNN_PEB_PrmEmb_0.pickle
mv 1D_Burgers_Sols_Nu0.002multi_MPNN.pt 1D_Burgers_Sols_Nu_multi_MPNN_PEB_PrmEmb_0.pt
