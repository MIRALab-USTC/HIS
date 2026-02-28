# A Hierarchical Circuit Symbolic Discovery Framework for Efficient Logic Optimization
This is our code for our manuscript A Hierarchical Circuit Symbolic Discovery Framework for Efficient Logic Optimization.

## Installation of Packages
ABC Installation
In this repository, we provide the executable file abc and the static library libabc.a. However, we have observed that the executable file may encounter errors on different platforms. To address this, we have included the source code for our modified versions of abc, namely abc_py and abc_ai4mfs2_CMO. For detailed installation instructions, please refer to the documentation in [A Circuit Domain Generalization Framework for Efficient Logic Synthesis in Chip Design](https://github.com/MIRALab-USTC/AI4LogicSynthesis-PruneX).

## Requirements Installation
The python environment requires:

- **Hardware**: A GPU equipped machine
- **Python**: 3.9
- **Pytorch**: 2.3.0
- **Dependencies**: See requirements.txt for all required packages This python environment can be installed easily via:
```bash
pip install -r requirements.txt
```

## Training
We provide scripts for both in-domain and out-of-domain training.

```bash
# Train a model for in-domain circuits (Hyp)
mkdir out
CUDA_VISIBLE_DEVICES=4 nohup python o3_train_gs4co.py --config-name train_gs4co train_kwargs.score_func_name=focal_loss train_kwargs.use_layer_learning=True instance_kwargs.instance_type=Hyp 'instance_kwargs.Best_of_N=[1, 2, 4, 8, 16]' rl_algo_kwargs.kwargs.entropy_coef=0.2 train_data_loader.kwargs.npy_data_path=./dataset/in-domain-test/Hyp/train train_data_loader.kwargs.normalize=True train_data_loader.kwargs.feature_selection=no_lut test_data_loader.kwargs.npy_data_path=./dataset/in-domain-test/Hyp/test test_data_loader.kwargs.normalize=True test_data_loader.kwargs.feature_selection=no_lut > ./out/Hyp_focal_loss_score_normalize_entropy_coef_0.2_no_lut_normalize_pred_y_BON_124816_layer.out 2>&1 &

# Train a model for out-of-domain
# epfl2iwls normalize entropy_coef=0.2 feature=no_lut normalize_pred_y BON=[1, 2, 4, 8, 16] use_layer_learning=True
CUDA_VISIBLE_DEVICES=6 nohup python o3_train_gs4co.py --config-name train_gs4co train_kwargs.score_func_name=focal_loss train_kwargs.use_layer_learning=True instance_kwargs.instance_type=epfl2iwls 'instance_kwargs.Best_of_N=[1, 2, 4, 8, 16]' rl_algo_kwargs.kwargs.entropy_coef=0.2 train_data_loader.kwargs.npy_data_path=./dataset/out-of-domain-test/epfl2iwls/train train_data_loader.kwargs.normalize=True train_data_loader.kwargs.feature_selection=no_lut test_data_loader.kwargs.npy_data_path=./dataset/out-of-domain-test/epfl2iwls/test test_data_loader.kwargs.normalize=True test_data_loader.kwargs.feature_selection=no_lut > ./out/epfl2iwls_focal_loss_score_normalize_entropy_coef_0.2_no_lut_normalize_pred_y_BON_124816_layer.out 2>&1 &

# iwls2epfl normalize entropy_coef=0.2 feature=no_lut normalize_pred_y BON=[1, 2, 4, 8, 16] use_layer_learning=True
CUDA_VISIBLE_DEVICES=7 nohup python o3_train_gs4co.py --config-name train_gs4co train_kwargs.score_func_name=focal_loss train_kwargs.use_layer_learning=True instance_kwargs.instance_type=iwls2epfl 'instance_kwargs.Best_of_N=[1, 2, 4, 8, 16]' rl_algo_kwargs.kwargs.entropy_coef=0.2 train_data_loader.kwargs.npy_data_path=./dataset/out-of-domain-test/iwls2epfl/train train_data_loader.kwargs.normalize=True train_data_loader.kwargs.feature_selection=no_lut test_data_loader.kwargs.npy_data_path=./dataset/out-of-domain-test/iwls2epfl/test test_data_loader.kwargs.normalize=True test_data_loader.kwargs.feature_selection=no_lut > ./out/iwls2epfl_focal_loss_score_normalize_entropy_coef_0.2_no_lut_normalize_pred_y_BON_124816_layer.out 2>&1 &

# The default depth of symbolic trees is 2. To construct deeper trees (e.g., layer=3):
# epfl2iwls normalize entropy_coef=0.2 feature=no_lut normalize_pred_y BON=[1, 2, 4, 8] use_layer_learning=False seed=0 layer=3
CUDA_VISIBLE_DEVICES=2 nohup python o3_train_gs4co.py --config-name train_gs4co train_kwargs.seed=0 train_kwargs.score_func_name=focal_loss train_kwargs.use_layer_learning=False train_kwargs.use_multi_layer=True 'train_kwargs.num_messages=[5, 5, 5]' instance_kwargs.instance_type=epfl2iwls 'instance_kwargs.Best_of_N=[1, 2, 4, 8]' rl_algo_kwargs.kwargs.entropy_coef=0.2 'dso_agent_kwargs.transformer_kwargs.layer_min_length=[4, 4, 4, 4]' 'dso_agent_kwargs.transformer_kwargs.layer_max_length=[72, 48, 24, 8]' 'dso_agent_kwargs.transformer_kwargs.layer_soft_length=[38, 26, 14, 6]' train_data_loader.kwargs.npy_data_path=./dataset/out-of-domain-test/epfl2iwls/train train_data_loader.kwargs.normalize=True train_data_loader.kwargs.feature_selection=no_lut test_data_loader.kwargs.npy_data_path=./dataset/out-of-domain-test/epfl2iwls/test test_data_loader.kwargs.normalize=True test_data_loader.kwargs.feature_selection=no_lut > ./out/epfl2iwls_focal_loss_score_normalize_entropy_coef_0.2_no_lut_normalize_pred_y_BON_1248_layer_layer3.out 2>&1 &
```
## Testing

Next, the quick start for testing:

```bash
# offline test
bash shells/offline_recall_test.sh
# online test
bash online_test.sh
```