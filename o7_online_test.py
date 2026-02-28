import argparse
import json
import os
import subprocess
import numpy as np
import re
import datetime
def process_output(output):
    stats = {
        'and': [],
        'edge': [],
        'lev': [],
        'run_time': [],
        'all_run_time': [],
        'time_process_csv': [],
        'time_process_children': [],
        'time_process_graph_samples': [],
        'time_load_graph_model': [],
        'time_graph_data_load': [],
        'time_policy_inference': [],
        'online_time_all': [],
        'offline_time_all': []
    }
    output = output.decode()
    output = re.sub(r'\x1b\[\d(;\d+)?m', '', output)
    output = output.split('\n')
    output = output[2:]
    print(f"output: {output}")
    offline_time_all = 0
    online_time_all = 0
    for o_str in output:
        if 'nd' in o_str and 'seconds' not in o_str:
            o_str_list = o_str.split()
            print(f"o str list: {o_str_list}")
            and_index = o_str_list.index('nd')
            if o_str_list[and_index+1] != '=':
                stats['and'].append(int(o_str_list[and_index+1][1:]))
            else:
                stats['and'].append(int(o_str_list[and_index+2]))
        if 'lev' in o_str:
            o_str_list = o_str.split()
            lev_index = o_str_list.index('lev')
            if o_str_list[lev_index+1] != '=':
                stats['lev'].append(int(o_str_list[lev_index+1][1:]))
            else:
                stats['lev'].append(int(o_str_list[lev_index+2]))
        if 'edge' in o_str:
            o_str_list = o_str.split()
            lev_index = o_str_list.index('edge')
            if o_str_list[lev_index+1] != '=':
                stats['edge'].append(int(o_str_list[lev_index+1][1:]))
            else:
                stats['edge'].append(int(o_str_list[lev_index+2]))
        if 'elapse:' in o_str:
            o_str_list = o_str.split()
            time_index = o_str_list.index('elapse:')
            run_time = float(o_str_list[time_index+1])
            stats['run_time'].append(run_time)
        if 'total:' in o_str:
            o_str_list = o_str.split()
            time_index = o_str_list.index('total:')
            all_run_time = float(o_str_list[time_index+1])
            stats['all_run_time'].append(all_run_time)
        if 'process_csv:' in o_str:
            o_str_list = o_str.split()
            time_index = o_str_list.index('process_csv:')
            time_process_csv = float(o_str_list[time_index+1])
            stats['time_process_csv'].append(time_process_csv)
            offline_time_all += time_process_csv
        if 'process_children:' in o_str:
            o_str_list = o_str.split()
            time_index = o_str_list.index('process_children:')
            time_process_children = float(o_str_list[time_index+1])
            stats['time_process_children'].append(time_process_children)
            offline_time_all += time_process_children
        if 'process_graph_samples:' in o_str:
            o_str_list = o_str.split()
            time_index = o_str_list.index('process_graph_samples:')
            time_process_graph_samples = float(o_str_list[time_index+1])
            stats['time_process_graph_samples'].append(time_process_graph_samples)
            offline_time_all += time_process_graph_samples
        if 'load_graph_mode:' in o_str:
            o_str_list = o_str.split()
            time_index = o_str_list.index('load_graph_model:')
            time_load_graph_model = float(o_str_list[time_index+1])
            stats['time_load_graph_model'].append(time_load_graph_model)
            offline_time_all += time_load_graph_model
        if 'graph_data_load:' in o_str:
            o_str_list = o_str.split()
            time_index = o_str_list.index('graph_data_load:')
            time_load_graph_model = float(o_str_list[time_index+1])
            stats['time_graph_data_load'].append(time_load_graph_model)
            offline_time_all += time_load_graph_model
        if 'inference:' in o_str:
            o_str_list = o_str.split()
            print(f"o str list: {o_str_list}")
            time_index = o_str_list.index('inference:')
            time_inference = float(o_str_list[time_index+1])
            stats['time_policy_inference'].append(time_inference)
            offline_time_all += time_inference
    online_time_all = offline_time_all + stats['run_time'][1]
    stats['offline_time_all'].append(offline_time_all)
    stats['online_time_all'].append(online_time_all)
    return stats

def save_data(save_data, save_dir, test_blif_name, seed, method, num_lo_flow):
    if save_dir == None:
        pass
    else:
        save_path = f'../../../online_test/mfs2/{save_dir}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if num_lo_flow == 3 or num_lo_flow == 4 or num_lo_flow == 5:
            save_npy = f'{save_path}/stats_{test_blif_name}_method_{method}_twice_seed{seed}.npy'
        else:
            save_npy = f'{save_path}/stats_{test_blif_name}_method_{method}_seed{seed}.npy'
        np.save(save_npy, save_data)


def get_command(test_blif_path, dch, num_lo_flow):
    # flow0: 1 times mfs2 default p=0
    # flow1: 1 times SR/Random/Trees/LR_mfs2/SVD p=1
    # flow2: 1 times GNN/HIS p=2
    # flow3: 2 times SR/Random/Trees/LR_mfs2/SVD p=4
    # flow4: 2 times GNN/HIS p=5
    if num_lo_flow == 0:
        if dch:
            command = f"./abc -c 'r {test_blif_path}; strash; dch; if -C 12; print_stats -t; mfs2 -W 4 -M 5000 -l -p 0; print_stats -t'"
        else:
            command = f"./abc -c 'r {test_blif_path}; strash; if -C 12; print_stats -t; mfs2 -W 4 -M 5000 -l -p 0; print_stats -t'"
    elif num_lo_flow == 1:
        if dch:
            command = f"./abc -c 'r {test_blif_path}; strash; dch; if -C 12; print_stats -t; mfs2 -W 4 -M 5000 -l -p 1; print_stats -t'"
        else:
            command = f"./abc -c 'r {test_blif_path}; strash; if -C 12; print_stats -t; mfs2 -W 4 -M 5000 -l -p 1; print_stats -t'"
    elif num_lo_flow == 2:
        if dch:
            command = f"./abc -c 'r {test_blif_path}; strash; dch; if -C 12; print_stats -t; mfs2 -W 4 -M 5000 -l -p 2; print_stats -t'"
        else:
            command = f"./abc -c 'r {test_blif_path}; strash; if -C 12; print_stats -t; mfs2 -W 4 -M 5000 -l -p 2; print_stats -t'"
    elif num_lo_flow == 3:
        if dch:
            command = f"./abc -c 'r {test_blif_path}; strash; dch; if -C 12; print_stats -t; mfs2 -W 4 -M 5000 -l -p 1; strash; if -C 12; mfs2 -W 4 -M 5000 -l -p 1; print_stats -t'"
        else:
            command = f"./abc -c 'r {test_blif_path}; strash; if -C 12; print_stats -t; mfs2 -W 4 -M 5000 -l -p 1; strash; if -C 12; mfs2 -W 4 -M 5000 -l -p 1; print_stats -t'"
    elif num_lo_flow == 4:
        if dch:
            command = f"./abc -c 'r {test_blif_path}; strash; dch; if -C 12; print_stats -t; mfs2 -W 4 -M 5000 -l -p 2; strash; if -C 12; mfs2 -W 4 -M 5000 -l -p 2; print_stats -t'"
        else:
            command = f"./abc -c 'r {test_blif_path}; strash; if -C 12; print_stats -t; mfs2 -W 4 -M 5000 -l -p 2; strash; if -C 12; mfs2 -W 4 -M 5000 -l -p 2; print_stats -t'"
    return command

def create_and_enter_working_directory():
    # 获取当前日期和时间
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    
    # 创建基础目录
    base_dir = 'outputs'
    date_dir = os.path.join(base_dir, date_str)
    
    # 创建日期目录（如果不存在的话）
    if not os.path.exists(date_dir):
        os.makedirs(date_dir)
    
    # 创建时间目录
    working_dir = os.path.join(date_dir, time_str)
    os.makedirs(working_dir, exist_ok=True)
    
    # 更改当前工作目录
    os.chdir(working_dir)
    import shutil
    source = '../../../abc'
    destination = './abc'
    shutil.copy(source, destination)
    source = '../../../abc.rc'
    destination = './abc.rc'
    shutil.copy(source, destination)
    # 输出创建的目录路径
    print(f"Working directory created and entered: {os.getcwd()}")

if __name__ == "__main__":
    create_and_enter_working_directory()
    parser = argparse.ArgumentParser(description='test scripts')
    parser.add_argument('--sel_percents', nargs="+", default=[0.5], type=float)
    parser.add_argument('--model_path', nargs="+", default=["/home/yqbai/HIS/gs4-ls-updated/models/OOD/HIS/square/train_iter_2000.pkl"], type=str)
    # parser.add_argument('--no_lut_model_path', type=str,
    #                     default="./model/epfl/hyp/hyp_no_smote_15gen_GNN_mae_non_normalize_all_no_lut.pkl")
    # parser.add_argument('--lut_model_path', type=str,
    #                     default="./model/epfl/hyp/hyp_no_smote_15gen_GNN_mae_non_normalize_all_lut.pkl")
    parser.add_argument('--test_blif_path', type=str,
                        default="../../../blif_data/epfl_arithmetic/square.blif")
    parser.add_argument('--test_npy_path', type=str,
                        default="../../../dataset/out-of-domain-test/iwls2epfl/test/save_data_total_square.blif_epfl_arithmetic_5000_flow_num_1.npy")
    parser.add_argument('--test_blif', type=str,
                        default="square")
    parser.add_argument('--feature_selection', type=str, default="no_lut")
    parser.add_argument('--dch', action='store_true')
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--num_lo_flow', type=int, default=2)
    parser.add_argument('--save_dir', type=str, default='square_online_test')
    # METHOD includes: HIS, CMO, COG, Random, SVD
    parser.add_argument('--method', type=str, default='HIS')
    parser.add_argument('--seed_num', type=int, default=1)
    args = parser.parse_args()
    print('args.model_path', args.model_path)
    print('dch', args.dch)
    print('file_name', args.test_blif)
    print('normalize', args.normalize)
    print('random', args.random)
    print('The testing method is', args.method)
    # GLOBAL VAR
    SEL_PERCENTS = args.sel_percents
    DEVICE = "cpu"

    # command
    command = get_command(args.test_blif_path, args.dch, args.num_lo_flow)
    print('the command is', command)   
    for seed in range(args.seed_num):
        print(f'the information of seed {seed}')
        stats_dict = {}
        for sel_percent in SEL_PERCENTS:
            json_kwargs = {
                "MODEL": args.model_path[seed],
                "npy_file_path": args.test_npy_path,
                "feature_selection": args.feature_selection,
                "DEVICE": DEVICE,
                "SEL_PERCENT": sel_percent,
                "RANDOM": args.random,
                "NORMALIZE": args.normalize,
                "METHOD": args.method
            }
            with open('./configs.json', 'w') as f:
                json.dump(json_kwargs, f)
            output = subprocess.check_output(command, shell=True)
            stats = process_output(output)
            # print(stats)
            k = f"sel_percent_{json_kwargs['SEL_PERCENT']}"
            stats_dict[k] = stats
        print(f"cur blif: {args.test_blif_path}, {stats_dict}")
        save_data(stats_dict, args.save_dir, args.test_blif, seed, args.method, args.num_lo_flow)
