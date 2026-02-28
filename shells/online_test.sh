#### 1 times test
# HIS
# exec > online_HIS_test.out 2>&1

# python o7_online_test.py --sel_percents 0.5 0.6 0.7 --test_blif square --test_npy_path ../../../dataset/out-of-domain-test/iwls2epfl/test/save_data_total_square.blif_epfl_arithmetic_5000_flow_num_1.npy --test_blif_path ../../../blif_data/epfl_arithmetic/square.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/HIS/square/train_iter_2000.pkl --dch --normalize --num_lo_flow 2 --save_dir square_online_test --method HIS --seed_num 1

# python o7_online_test.py --sel_percents 0.5 0.6 0.7 --test_blif hyp --test_npy_path ../../../dataset/out-of-domain-test/iwls2epfl/test/save_data_total_hyp.blif_epfl_arithmetic_5000_flow_num_1.npy --test_blif_path ../../../blif_data/epfl_arithmetic/hyp.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/HIS/hyp/train_iter_2000.pkl --normalize --num_lo_flow 2 --save_dir hyp_online_test --method HIS --seed_num 1

# python o7_online_test.py --sel_percents 0.5 0.6 0.7 --test_blif multiplier --test_npy_path ../../../dataset/out-of-domain-test/iwls2epfl/test/save_data_total_multiplier.blif_epfl_arithmetic_5000_flow_num_1.npy --test_blif_path ../../../blif_data/epfl_arithmetic/multiplier.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/HIS/multiplier/train_iter_2000.pkl --dch --normalize --num_lo_flow 2 --save_dir multiplier_online_test --method HIS --seed_num 1

# python o7_online_test.py --sel_percents 0.5 0.6 0.7 --test_blif des_perf --test_npy_path ../../../dataset/out-of-domain-test/epfl2iwls/test/save_data_total_des_perf.blif_iwls2005_5000_flow_num_1.npy --test_blif_path ../../../blif_data/iwls2005/des_perf.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/HIS/des_perf/train_iter_2000.pkl --dch --normalize --num_lo_flow 2 --save_dir des_perf_online_test --method HIS --seed_num 1

# python o7_online_test.py --sel_percents 0.5 0.6 0.7 --test_blif ethernet --test_npy_path ../../../dataset/out-of-domain-test/epfl2iwls/test/save_data_total_ethernet.blif_iwls2005_5000_flow_num_1.npy --test_blif_path ../../../blif_data/iwls2005/ethernet.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/HIS/ethernet/train_iter_2000.pkl --dch --normalize --num_lo_flow 2 --save_dir ethernet_online_test --method HIS --seed_num 1

# python o7_online_test.py --sel_percents 0.5 0.6 0.7 --test_blif wb_conmax --test_npy_path ../../../dataset/out-of-domain-test/epfl2iwls/test/save_data_total_wb_conmax.blif_iwls2005_5000_flow_num_1.npy --test_blif_path ../../../blif_data/iwls2005/wb_conmax.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/HIS/wb_conmax/train_iter_2000.pkl --dch --normalize --num_lo_flow 2 --save_dir wb_conmax_online_test --method HIS --seed_num 1
# wait

# COG
# exec > online_COG_test.out 2>&1

# python o7_online_test.py --sel_percents 0.5 0.6 0.7 --test_blif square --test_npy_path ../../../dataset/out-of-domain-test/iwls2epfl/test/save_data_total_square.blif_epfl_arithmetic_5000_flow_num_1.npy --test_blif_path ../../../blif_data/epfl_arithmetic/square.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/mfs2_GNN/iwls2epfl/normalize_model_no_lut/itr_1999.pkl --dch --normalize --num_lo_flow 2 --save_dir square_online_test --method COG --seed_num 1

# python o7_online_test.py --sel_percents 0.5 0.6 0.7 --test_blif hyp --test_npy_path ../../../dataset/out-of-domain-test/iwls2epfl/test/save_data_total_hyp.blif_epfl_arithmetic_5000_flow_num_1.npy --test_blif_path ../../../blif_data/epfl_arithmetic/hyp.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/mfs2_GNN/iwls2epfl/normalize_model_no_lut/itr_1999.pkl --normalize --num_lo_flow 2 --save_dir hyp_online_test --method COG --seed_num 1

# python o7_online_test.py --sel_percents 0.5 0.6 0.7 --test_blif multiplier --test_npy_path ../../../dataset/out-of-domain-test/iwls2epfl/test/save_data_total_multiplier.blif_epfl_arithmetic_5000_flow_num_1.npy --test_blif_path ../../../blif_data/epfl_arithmetic/multiplier.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/mfs2_GNN/iwls2epfl/normalize_model_no_lut/itr_1999.pkl --dch --normalize --num_lo_flow 2 --save_dir multiplier_online_test --method COG --seed_num 1

# python o7_online_test.py --sel_percents 0.5 0.6 0.7 --test_blif des_perf --test_npy_path ../../../dataset/out-of-domain-test/epfl2iwls/test/save_data_total_des_perf.blif_iwls2005_5000_flow_num_1.npy --test_blif_path ../../../blif_data/iwls2005/des_perf.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/mfs2_GNN/epfl2iwls/normalize_model_no_lut/itr_1999.pkl --dch --normalize --num_lo_flow 2 --save_dir des_perf_online_test --method COG --seed_num 1

# python o7_online_test.py --sel_percents 0.5 0.6 0.7 --test_blif ethernet --test_npy_path ../../../dataset/out-of-domain-test/epfl2iwls/test/save_data_total_ethernet.blif_iwls2005_5000_flow_num_1.npy --test_blif_path ../../../blif_data/iwls2005/ethernet.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/mfs2_GNN/epfl2iwls/normalize_model_no_lut/itr_1999.pkl --dch --normalize --num_lo_flow 2 --save_dir ethernet_online_test --method COG --seed_num 1

# python o7_online_test.py --sel_percents 0.5 0.6 0.7 --test_blif wb_conmax --test_npy_path ../../../dataset/out-of-domain-test/epfl2iwls/test/save_data_total_wb_conmax.blif_iwls2005_5000_flow_num_1.npy --test_blif_path ../../../blif_data/iwls2005/wb_conmax.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/mfs2_GNN/epfl2iwls/normalize_model_no_lut/itr_1999.pkl --dch --normalize --num_lo_flow 2 --save_dir wb_conmax_online_test --method COG --seed_num 1

# Random
# exec > online_Random_test.out 2>&1

# python o7_online_test.py --sel_percents 0.5 0.6 0.7 0.8 0.9 1.0 --test_blif square --test_npy_path ../../../dataset/out-of-domain-test/iwls2epfl/test/save_data_total_square.blif_epfl_arithmetic_5000_flow_num_1.npy --test_blif_path ../../../blif_data/epfl_arithmetic/square.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/random/Random_model_square_loss_mse_seed1.pkl --dch --normalize --num_lo_flow 1 --save_dir square_online_test --method Random --seed_num 1

# python o7_online_test.py --sel_percents 0.5 0.6 0.7 0.8 0.9 1.0 --test_blif hyp --test_npy_path ../../../dataset/out-of-domain-test/iwls2epfl/test/save_data_total_hyp.blif_epfl_arithmetic_5000_flow_num_1.npy --test_blif_path ../../../blif_data/epfl_arithmetic/hyp.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/random/Random_model_hyp_loss_mse_seed1.pkl --normalize --num_lo_flow 1 --save_dir hyp_online_test --method Random --seed_num 1

# python o7_online_test.py --sel_percents 0.5 0.6 0.7 0.8 0.9 1.0 --test_blif multiplier --test_npy_path ../../../dataset/out-of-domain-test/iwls2epfl/test/save_data_total_multiplier.blif_epfl_arithmetic_5000_flow_num_1.npy --test_blif_path ../../../blif_data/epfl_arithmetic/multiplier.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/random/Random_model_multiplier_loss_mse_seed1.pkl --dch --normalize --num_lo_flow 1 --save_dir multiplier_online_test --method Random --seed_num 1

# python o7_online_test.py --sel_percents 0.5 0.6 0.7 0.8 0.9 1.0 --test_blif des_perf --test_npy_path ../../../dataset/out-of-domain-test/epfl2iwls/test/save_data_total_des_perf.blif_iwls2005_5000_flow_num_1.npy --test_blif_path ../../../blif_data/iwls2005/des_perf.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/random/Random_model_des_perf_loss_mse_seed1.pkl --dch --normalize --num_lo_flow 1 --save_dir des_perf_online_test --method Random --seed_num 1

# python o7_online_test.py --sel_percents 0.5 0.6 0.7 0.8 0.9 1.0 --test_blif ethernet --test_npy_path ../../../dataset/out-of-domain-test/epfl2iwls/test/save_data_total_ethernet.blif_iwls2005_5000_flow_num_1.npy --test_blif_path ../../../blif_data/iwls2005/ethernet.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/random/Random_model_ethernet_loss_mse_seed1.pkl --dch --normalize --num_lo_flow 1 --save_dir ethernet_online_test --method Random --seed_num 1

# python o7_online_test.py --sel_percents 0.5 0.6 0.7 0.8 0.9 1.0 --test_blif wb_conmax --test_npy_path ../../../dataset/out-of-domain-test/epfl2iwls/test/save_data_total_wb_conmax.blif_iwls2005_5000_flow_num_1.npy --test_blif_path ../../../blif_data/iwls2005/wb_conmax.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/random/Random_model_wb_conmax_loss_mse_seed1.pkl --dch --normalize --num_lo_flow 1 --save_dir wb_conmax_online_test --method Random --seed_num 1
# wait

# SVD
# exec > online_SVD_test.out 2>&1

# python o7_online_test.py --sel_percents 0.5 0.6 0.7 0.8 0.9 1.0 --test_blif square --test_npy_path ../../../dataset/out-of-domain-test/iwls2epfl/test/save_data_total_square.blif_epfl_arithmetic_5000_flow_num_1.npy --test_blif_path ../../../blif_data/epfl_arithmetic/square.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/SVD/SVD_model_iwls2epfl_square_loss_GLNN_seed1.pkl --dch --normalize --num_lo_flow 1 --save_dir square_online_test --method SVD --seed_num 1

# python o7_online_test.py --sel_percents 0.5 0.6 0.7 0.8 0.9 1.0 --test_blif hyp --test_npy_path ../../../dataset/out-of-domain-test/iwls2epfl/test/save_data_total_hyp.blif_epfl_arithmetic_5000_flow_num_1.npy --test_blif_path ../../../blif_data/epfl_arithmetic/hyp.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/SVD/SVD_model_iwls2epfl_hyp_loss_GLNN_seed1.pkl --normalize --num_lo_flow 1 --save_dir hyp_online_test --method SVD --seed_num 1

# python o7_online_test.py --sel_percents 0.5 0.6 0.7 0.8 0.9 1.0 --test_blif multiplier --test_npy_path ../../../dataset/out-of-domain-test/iwls2epfl/test/save_data_total_multiplier.blif_epfl_arithmetic_5000_flow_num_1.npy --test_blif_path ../../../blif_data/epfl_arithmetic/multiplier.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/SVD/SVD_model_iwls2epfl_multiplier_loss_GLNN_seed1.pkl --dch --normalize --num_lo_flow 1 --save_dir multiplier_online_test --method SVD --seed_num 1

# python o7_online_test.py --sel_percents 0.5 0.6 0.7 0.8 0.9 1.0 --test_blif des_perf --test_npy_path ../../../dataset/out-of-domain-test/epfl2iwls/test/save_data_total_des_perf.blif_iwls2005_5000_flow_num_1.npy --test_blif_path ../../../blif_data/iwls2005/des_perf.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/SVD/SVD_model_epfl2iwls_des_perf_loss_GLNN_seed1.pkl --dch --normalize --num_lo_flow 1 --save_dir des_perf_online_test --method SVD --seed_num 1

# python o7_online_test.py --sel_percents 0.5 0.6 0.7 0.8 0.9 1.0 --test_blif ethernet --test_npy_path ../../../dataset/out-of-domain-test/epfl2iwls/test/save_data_total_ethernet.blif_iwls2005_5000_flow_num_1.npy --test_blif_path ../../../blif_data/iwls2005/ethernet.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/SVD/SVD_model_epfl2iwls_ethernet_loss_GLNN_seed1.pkl --dch --normalize --num_lo_flow 1 --save_dir ethernet_online_test --method SVD --seed_num 1

# python o7_online_test.py --sel_percents 0.5 0.6 0.7 0.8 0.9 1.0 --test_blif wb_conmax --test_npy_path ../../../dataset/out-of-domain-test/epfl2iwls/test/save_data_total_wb_conmax.blif_iwls2005_5000_flow_num_1.npy --test_blif_path ../../../blif_data/iwls2005/wb_conmax.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/SVD/SVD_model_epfl2iwls_wb_conmax_loss_GLNN_seed1.pkl --dch --normalize --num_lo_flow 1 --save_dir wb_conmax_online_test --method SVD --seed_num 1
# wait

# CMO
# exec > online_CMO_test.out 2>&1

# python o7_online_test.py --sel_percents 0.5 0.6 0.7 1.0 --test_blif square --test_npy_path ../../../dataset/out-of-domain-test/iwls2epfl/test/save_data_total_square.blif_epfl_arithmetic_5000_flow_num_1.npy --test_blif_path ../../../blif_data/epfl_arithmetic/square.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/CMO/no_lut/MCTS_ours_square_no_lut.pkl --dch --normalize --num_lo_flow 1 --save_dir square_online_test --method CMO --seed_num 1

# python o7_online_test.py --sel_percents 0.5 0.6 0.7 1.0 --test_blif hyp --test_npy_path ../../../dataset/out-of-domain-test/iwls2epfl/test/save_data_total_hyp.blif_epfl_arithmetic_5000_flow_num_1.npy --test_blif_path ../../../blif_data/epfl_arithmetic/hyp.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/CMO/no_lut/MCTS_ours_hyp_no_lut.pkl --normalize --num_lo_flow 1 --save_dir hyp_online_test --method CMO --seed_num 1

# python o7_online_test.py --sel_percents 0.5 0.6 0.7 1.0 --test_blif multiplier --test_npy_path ../../../dataset/out-of-domain-test/iwls2epfl/test/save_data_total_multiplier.blif_epfl_arithmetic_5000_flow_num_1.npy --test_blif_path ../../../blif_data/epfl_arithmetic/multiplier.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/CMO/no_lut/MCTS_ours_multiplier_no_lut.pkl --dch --normalize --num_lo_flow 1 --save_dir multiplier_online_test --method CMO --seed_num 1

# python o7_online_test.py --sel_percents 0.5 0.6 0.7 1.0 --test_blif des_perf --test_npy_path ../../../dataset/out-of-domain-test/epfl2iwls/test/save_data_total_des_perf.blif_iwls2005_5000_flow_num_1.npy --test_blif_path ../../../blif_data/iwls2005/des_perf.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/CMO/no_lut/MCTS_ours_des_perf_no_lut.pkl --dch --normalize --num_lo_flow 1 --save_dir des_perf_online_test --method CMO --seed_num 1

# python o7_online_test.py --sel_percents 0.5 0.6 0.7 1.0 --test_blif ethernet --test_npy_path ../../../dataset/out-of-domain-test/epfl2iwls/test/save_data_total_ethernet.blif_iwls2005_5000_flow_num_1.npy --test_blif_path ../../../blif_data/iwls2005/ethernet.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/CMO/no_lut/MCTS_ours_ethernet_no_lut.pkl --dch --normalize --num_lo_flow 1 --save_dir ethernet_online_test --method CMO --seed_num 1

# python o7_online_test.py --sel_percents 0.5 0.6 0.7 1.0 --test_blif wb_conmax --test_npy_path ../../../dataset/out-of-domain-test/epfl2iwls/test/save_data_total_wb_conmax.blif_iwls2005_5000_flow_num_1.npy --test_blif_path ../../../blif_data/iwls2005/wb_conmax.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/CMO/no_lut/MCTS_ours_wb_conmax_no_lut.pkl --dch --normalize --num_lo_flow 1 --save_dir wb_conmax_online_test --method CMO --seed_num 1
# wait

#### 2 times test
# COG
# exec > online_COG_test_2times.out 2>&1

# python o7_online_test.py --sel_percents 0.3 0.4 0.5 0.6 --test_blif square --test_npy_path ../../../dataset/out-of-domain-test/iwls2epfl/test/save_data_total_square.blif_epfl_arithmetic_5000_flow_num_1.npy --test_blif_path ../../../blif_data/epfl_arithmetic/square.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/mfs2_GNN/iwls2epfl/normalize_model_no_lut/itr_1999.pkl --dch --normalize --num_lo_flow 4 --save_dir square_online_test --method COG --seed_num 1

# python o7_online_test.py --sel_percents 0.3 0.4 0.5 0.6 --test_blif hyp --test_npy_path ../../../dataset/out-of-domain-test/iwls2epfl/test/save_data_total_hyp.blif_epfl_arithmetic_5000_flow_num_1.npy --test_blif_path ../../../blif_data/epfl_arithmetic/hyp.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/mfs2_GNN/iwls2epfl/normalize_model_no_lut/itr_1999.pkl --normalize --num_lo_flow 4 --save_dir hyp_online_test --method COG --seed_num 1

# python o7_online_test.py --sel_percents 0.3 0.4 0.5 0.6 --test_blif multiplier --test_npy_path ../../../dataset/out-of-domain-test/iwls2epfl/test/save_data_total_multiplier.blif_epfl_arithmetic_5000_flow_num_1.npy --test_blif_path ../../../blif_data/epfl_arithmetic/multiplier.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/mfs2_GNN/iwls2epfl/normalize_model_no_lut/itr_1999.pkl --dch --normalize --num_lo_flow 4 --save_dir multiplier_online_test --method COG --seed_num 1

# python o7_online_test.py --sel_percents 0.3 0.4 0.5 0.6 --test_blif des_perf --test_npy_path ../../../dataset/out-of-domain-test/epfl2iwls/test/save_data_total_des_perf.blif_iwls2005_5000_flow_num_1.npy --test_blif_path ../../../blif_data/iwls2005/des_perf.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/mfs2_GNN/epfl2iwls/normalize_model_no_lut/itr_1999.pkl --dch --normalize --num_lo_flow 4 --save_dir des_perf_online_test --method COG --seed_num 1

# python o7_online_test.py --sel_percents 0.3 0.4 0.5 0.6 --test_blif ethernet --test_npy_path ../../../dataset/out-of-domain-test/epfl2iwls/test/save_data_total_ethernet.blif_iwls2005_5000_flow_num_1.npy --test_blif_path ../../../blif_data/iwls2005/ethernet.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/mfs2_GNN/epfl2iwls/normalize_model_no_lut/itr_1999.pkl --dch --normalize --num_lo_flow 4 --save_dir ethernet_online_test --method COG --seed_num 1

# python o7_online_test.py --sel_percents 0.3 0.4 0.5 0.6 --test_blif wb_conmax --test_npy_path ../../../dataset/out-of-domain-test/epfl2iwls/test/save_data_total_wb_conmax.blif_iwls2005_5000_flow_num_1.npy --test_blif_path ../../../blif_data/iwls2005/wb_conmax.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/mfs2_GNN/epfl2iwls/normalize_model_no_lut/itr_1999.pkl --dch --normalize --num_lo_flow 4 --save_dir wb_conmax_online_test --method COG --seed_num 1

# HIS
# exec > online_HIS_test_2times.out 2>&1

# python o7_online_test.py --sel_percents 0.3 0.4 0.5 0.6 --test_blif hyp --test_npy_path ../../../dataset/out-of-domain-test/iwls2epfl/test/save_data_total_hyp.blif_epfl_arithmetic_5000_flow_num_1.npy --test_blif_path ../../../blif_data/epfl_arithmetic/hyp.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/HIS/hyp/train_iter_2000.pkl --normalize --num_lo_flow 4 --save_dir hyp_online_test --method HIS --seed_num 1

# python o7_online_test.py --sel_percents 0.3 0.4 0.5 0.6 --test_blif multiplier --test_npy_path ../../../dataset/out-of-domain-test/iwls2epfl/test/save_data_total_multiplier.blif_epfl_arithmetic_5000_flow_num_1.npy --test_blif_path ../../../blif_data/epfl_arithmetic/multiplier.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/HIS/multiplier/train_iter_2000.pkl --dch --normalize --num_lo_flow 4 --save_dir multiplier_online_test --method HIS --seed_num 1

# python o7_online_test.py --sel_percents 0.3 0.4 0.5 0.6 --test_blif des_perf --test_npy_path ../../../dataset/out-of-domain-test/epfl2iwls/test/save_data_total_des_perf.blif_iwls2005_5000_flow_num_1.npy --test_blif_path ../../../blif_data/iwls2005/des_perf.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/HIS/des_perf/train_iter_2000.pkl --dch --normalize --num_lo_flow 4 --save_dir des_perf_online_test --method HIS --seed_num 1

# python o7_online_test.py --sel_percents 0.3 0.4 0.5 0.6 --test_blif ethernet --test_npy_path ../../../dataset/out-of-domain-test/epfl2iwls/test/save_data_total_ethernet.blif_iwls2005_5000_flow_num_1.npy --test_blif_path ../../../blif_data/iwls2005/ethernet.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/HIS/ethernet/train_iter_2000.pkl --dch --normalize --num_lo_flow 4 --save_dir ethernet_online_test --method HIS --seed_num 1

# python o7_online_test.py --sel_percents 0.3 0.4 0.5 0.6 --test_blif wb_conmax --test_npy_path ../../../dataset/out-of-domain-test/epfl2iwls/test/save_data_total_wb_conmax.blif_iwls2005_5000_flow_num_1.npy --test_blif_path ../../../blif_data/iwls2005/wb_conmax.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/HIS/wb_conmax/train_iter_2000.pkl --dch --normalize --num_lo_flow 4 --save_dir wb_conmax_online_test --method HIS --seed_num 1

# python o7_online_test.py --sel_percents 0.3 0.4 0.5 0.6 --test_blif square --test_npy_path ../../../dataset/out-of-domain-test/iwls2epfl/test/save_data_total_square.blif_epfl_arithmetic_5000_flow_num_1.npy --test_blif_path ../../../blif_data/epfl_arithmetic/square.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/HIS/square/train_iter_2000.pkl --dch --normalize --num_lo_flow 4 --save_dir square_online_test --method HIS --seed_num 1
# wait

# CMO
exec > online_CMO_test_2times.out 2>&1

python o7_online_test.py --sel_percents 0.3 0.4 0.5 0.6 --test_blif hyp --test_npy_path ../../../dataset/out-of-domain-test/iwls2epfl/test/save_data_total_hyp.blif_epfl_arithmetic_5000_flow_num_1.npy --test_blif_path ../../../blif_data/epfl_arithmetic/hyp.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/CMO/no_lut/MCTS_ours_hyp_no_lut.pkl --normalize --num_lo_flow 3 --save_dir hyp_online_test --method CMO --seed_num 1

python o7_online_test.py --sel_percents 0.3 0.4 0.5 0.6 --test_blif multiplier --test_npy_path ../../../dataset/out-of-domain-test/iwls2epfl/test/save_data_total_multiplier.blif_epfl_arithmetic_5000_flow_num_1.npy --test_blif_path ../../../blif_data/epfl_arithmetic/multiplier.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/CMO/no_lut/MCTS_ours_multiplier_no_lut.pkl --dch --normalize --num_lo_flow 3 --save_dir multiplier_online_test --method CMO --seed_num 1

python o7_online_test.py --sel_percents 0.3 0.4 0.5 0.6 --test_blif des_perf --test_npy_path ../../../dataset/out-of-domain-test/epfl2iwls/test/save_data_total_des_perf.blif_iwls2005_5000_flow_num_1.npy --test_blif_path ../../../blif_data/iwls2005/des_perf.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/CMO/no_lut/MCTS_ours_des_perf_no_lut.pkl --dch --normalize --num_lo_flow 3 --save_dir des_perf_online_test --method CMO --seed_num 1

python o7_online_test.py --sel_percents 0.3 0.4 0.5 0.6 --test_blif ethernet --test_npy_path ../../../dataset/out-of-domain-test/epfl2iwls/test/save_data_total_ethernet.blif_iwls2005_5000_flow_num_1.npy --test_blif_path ../../../blif_data/iwls2005/ethernet.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/CMO/no_lut/MCTS_ours_ethernet_no_lut.pkl --dch --normalize --num_lo_flow 3 --save_dir ethernet_online_test --method CMO --seed_num 1

python o7_online_test.py --sel_percents 0.3 0.4 0.5 0.6 --test_blif wb_conmax --test_npy_path ../../../dataset/out-of-domain-test/epfl2iwls/test/save_data_total_wb_conmax.blif_iwls2005_5000_flow_num_1.npy --test_blif_path ../../../blif_data/iwls2005/wb_conmax.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/CMO/no_lut/MCTS_ours_wb_conmax_no_lut.pkl --dch --normalize --num_lo_flow 3 --save_dir wb_conmax_online_test --method CMO --seed_num 1

python o7_online_test.py --sel_percents 0.3 0.4 0.5 0.6 --test_blif square --test_npy_path ../../../dataset/out-of-domain-test/iwls2epfl/test/save_data_total_square.blif_epfl_arithmetic_5000_flow_num_1.npy --test_blif_path ../../../blif_data/epfl_arithmetic/square.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/CMO/no_lut/MCTS_ours_square_no_lut.pkl --dch --normalize --num_lo_flow 3 --save_dir square_online_test --method CMO --seed_num 1
wait

# for test
# python o7_online_test.py --sel_percents 0.5 --test_blif square --test_npy_path ../../../dataset/out-of-domain-test/iwls2epfl/test/save_data_total_square.blif_epfl_arithmetic_5000_flow_num_1.npy --test_blif_path ../../../blif_data/epfl_arithmetic/square.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/HIS/square/train_iter_2000.pkl --dch --normalize --num_lo_flow 2 --save_dir square_online_test --method HIS --seed_num 1
# python o7_online_test.py --sel_percents 0.5 --test_blif square --test_npy_path ../../../dataset/out-of-domain-test/iwls2epfl/test/save_data_total_square.blif_epfl_arithmetic_5000_flow_num_1.npy --test_blif_path ../../../blif_data/epfl_arithmetic/square.blif --model_path /home/yqbai/HIS/gs4-ls-updated/models/OOD/mfs2_GNN/iwls2epfl/normalize_model_no_lut/itr_1999.pkl --dch --normalize --num_lo_flow 2 --save_dir square_online_test --method COG --seed_num 1