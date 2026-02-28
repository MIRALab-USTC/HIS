#!/bin/bash
# sixteen circuit

# 把标准输出和错误都写入日志
# exec > offline_sixteen_test.out 2>&1

# echo "Running iwls2epfl BON_16 model"
# echo "2000 epoch"
# python o4_test_gs4co.py --config_name train_gs4co \
#   train_data_loader.kwargs.feature_selection=no_lut \
#   test_data_loader.kwargs.feature_selection=no_lut \
#   test_data_loader.kwargs.npy_data_path=./dataset/sixteen/test \
#   test_data_loader.kwargs.max_batch_size=10240 \
#   test_kwargs.model_path=./results/results/train_graph/iwls2epfl/default/0827112603_cb1a02e/0_0/state_dict/hyp/BON_16/train_iter_2000.pkl

# echo "1500 epoch"
# python o4_test_gs4co.py --config_name train_gs4co \
#   train_data_loader.kwargs.feature_selection=no_lut \
#   test_data_loader.kwargs.feature_selection=no_lut \
#   test_data_loader.kwargs.npy_data_path=./dataset/sixteen/test \
#   test_data_loader.kwargs.max_batch_size=10240 \
#   test_kwargs.model_path=./results/results/train_graph/iwls2epfl/default/0827112603_cb1a02e/0_0/state_dict/hyp/BON_16/train_iter_1500.pkl

# echo "1000 epoch"
# python o4_test_gs4co.py --config_name train_gs4co \
#   train_data_loader.kwargs.feature_selection=no_lut \
#   test_data_loader.kwargs.feature_selection=no_lut \
#   test_data_loader.kwargs.npy_data_path=./dataset/sixteen/test \
#   test_data_loader.kwargs.max_batch_size=10240 \
#   test_kwargs.model_path=./results/results/train_graph/iwls2epfl/default/0827112603_cb1a02e/0_0/state_dict/hyp/BON_16/train_iter_1000.pkl

# echo "500 epoch"
# python o4_test_gs4co.py --config_name train_gs4co \
#   train_data_loader.kwargs.feature_selection=no_lut \
#   test_data_loader.kwargs.feature_selection=no_lut \
#   test_data_loader.kwargs.npy_data_path=./dataset/sixteen/test \
#   test_data_loader.kwargs.max_batch_size=10240 \
#   test_kwargs.model_path=./results/results/train_graph/iwls2epfl/default/0827112603_cb1a02e/0_0/state_dict/hyp/BON_16/train_iter_500.pkl
# # echo "Running iwls2epfl multiplier BON_4 model"

# # python o4_test_gs4co.py --config_name train_gs4co \
# #   train_data_loader.kwargs.feature_selection=no_lut \
# #   test_data_loader.kwargs.feature_selection=no_lut \
# #   test_data_loader.kwargs.npy_data_path=./dataset/sixteen/test \
# #   test_data_loader.kwargs.max_batch_size=10240 \
# #   test_kwargs.model_path=./results/results/train_graph/iwls2epfl/default/0827112603_cb1a02e/0_0/state_dict/multiplier/BON_4/train_iter_2000.pkl

# # echo "Running iwls2epfl square BON_4 model"

# # python o4_test_gs4co.py --config_name train_gs4co \
# #   train_data_loader.kwargs.feature_selection=no_lut \
# #   test_data_loader.kwargs.feature_selection=no_lut \
# #   test_data_loader.kwargs.npy_data_path=./dataset/sixteen/test \
# #   test_data_loader.kwargs.max_batch_size=10240 \
# #   test_kwargs.model_path=./results/results/train_graph/iwls2epfl/default/0827112603_cb1a02e/0_0/state_dict/square/BON_4/train_iter_2000.pkl

# echo "Running epfl2iwls des_perf BON_16 model"
# echo "2000 epoch"
# python o4_test_gs4co.py --config_name train_gs4co \
#   train_data_loader.kwargs.feature_selection=no_lut \
#   test_data_loader.kwargs.feature_selection=no_lut \
#   test_data_loader.kwargs.npy_data_path=./dataset/sixteen/test \
#   test_data_loader.kwargs.max_batch_size=10240 \
#   test_kwargs.model_path=./results/results/train_graph/epfl2iwls/default/0827112122_cb1a02e/0_0/state_dict/des_perf/BON_16/train_iter_2000.pkl

# echo "1500 epoch"
# python o4_test_gs4co.py --config_name train_gs4co \
#   train_data_loader.kwargs.feature_selection=no_lut \
#   test_data_loader.kwargs.feature_selection=no_lut \
#   test_data_loader.kwargs.npy_data_path=./dataset/sixteen/test \
#   test_data_loader.kwargs.max_batch_size=10240 \
#   test_kwargs.model_path=./results/results/train_graph/epfl2iwls/default/0827112122_cb1a02e/0_0/state_dict/des_perf/BON_16/train_iter_1500.pkl

# echo "1000 epoch"
# python o4_test_gs4co.py --config_name train_gs4co \
#   train_data_loader.kwargs.feature_selection=no_lut \
#   test_data_loader.kwargs.feature_selection=no_lut \
#   test_data_loader.kwargs.npy_data_path=./dataset/sixteen/test \
#   test_data_loader.kwargs.max_batch_size=10240 \
#   test_kwargs.model_path=./results/results/train_graph/epfl2iwls/default/0827112122_cb1a02e/0_0/state_dict/des_perf/BON_16/train_iter_1000.pkl

# echo "500 epoch"
# python o4_test_gs4co.py --config_name train_gs4co \
#   train_data_loader.kwargs.feature_selection=no_lut \
#   test_data_loader.kwargs.feature_selection=no_lut \
#   test_data_loader.kwargs.npy_data_path=./dataset/sixteen/test \
#   test_data_loader.kwargs.max_batch_size=10240 \
#   test_kwargs.model_path=./results/results/train_graph/epfl2iwls/default/0827112122_cb1a02e/0_0/state_dict/des_perf/BON_16/train_iter_500.pkl

# echo "Running epfl2iwls ethernet BON_4 model"

# python o4_test_gs4co.py --config_name train_gs4co \
#   train_data_loader.kwargs.feature_selection=no_lut \
#   test_data_loader.kwargs.feature_selection=no_lut \
#   test_data_loader.kwargs.npy_data_path=./dataset/sixteen/test \
#   test_data_loader.kwargs.max_batch_size=10240 \
#   test_kwargs.model_path=./results/results/train_graph/epfl2iwls/default/0827112122_cb1a02e/0_0/state_dict/ethernet/BON_4/train_iter_2000.pkl

# echo "Running epfl2iwls wb_conmax BON_4 model"

# python o4_test_gs4co.py --config_name train_gs4co \
#   train_data_loader.kwargs.feature_selection=no_lut \
#   test_data_loader.kwargs.feature_selection=no_lut \
#   test_data_loader.kwargs.npy_data_path=./dataset/sixteen/test \
#   test_data_loader.kwargs.max_batch_size=10240 \
#   test_kwargs.model_path=./results/results/train_graph/epfl2iwls/default/0827112122_cb1a02e/0_0/state_dict/wb_conmax/BON_4/train_iter_2000.pkl

exec > offline_square_test.out 2>&1

echo "Running iwls2epfl BON_4 model"
echo "2000 epoch"
python o4_test_gs4co.py --config_name train_gs4co \
  train_data_loader.kwargs.feature_selection=no_lut \
  test_data_loader.kwargs.feature_selection=no_lut \
  test_data_loader.kwargs.npy_data_path=/home/yqbai/HIS/gs4-ls-updated/dataset/out-of-domain-test/iwls2epfl/test \
  test_data_loader.kwargs.max_batch_size=70000 \
  test_kwargs.model_path=/home/yqbai/HIS/gs4-ls-updated/models/OOD/HIS/square/train_iter_2000.pkl
