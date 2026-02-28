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
        if 'and' in o_str and 'seconds' not in o_str:
            o_str_list = o_str.split()
            print(f"o str list: {o_str_list}")
            and_index = o_str_list.index('and')
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
        if 'elapse' in o_str:
            o_str_list = o_str.split()
            time_index = o_str_list.index('elapse:')
            run_time = float(o_str_list[time_index+1])
            stats['run_time'].append(run_time)
        if 'total' in o_str:
            o_str_list = o_str.split()
            time_index = o_str_list.index('total:')
            all_run_time = float(o_str_list[time_index+1])
            stats['all_run_time'].append(all_run_time)
        if 'process_csv' in o_str:
            o_str_list = o_str.split()
            time_index = o_str_list.index('process_csv:')
            time_process_csv = float(o_str_list[time_index+1])
            stats['time_process_csv'].append(time_process_csv)
            offline_time_all += time_process_csv
        if 'process_children' in o_str:
            o_str_list = o_str.split()
            time_index = o_str_list.index('process_children:')
            time_process_children = float(o_str_list[time_index+1])
            stats['time_process_children'].append(time_process_children)
            offline_time_all += time_process_children
        if 'process_graph_samples' in o_str:
            o_str_list = o_str.split()
            time_index = o_str_list.index('process_graph_samples:')
            time_process_graph_samples = float(o_str_list[time_index+1])
            stats['time_process_graph_samples'].append(time_process_graph_samples)
            offline_time_all += time_process_graph_samples
        if 'load_graph_model' in o_str:
            o_str_list = o_str.split()
            time_index = o_str_list.index('load_graph_model:')
            time_load_graph_model = float(o_str_list[time_index+1])
            stats['time_load_graph_model'].append(time_load_graph_model)
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
        save_path = f'../../../online_test/resub_sr/{save_dir}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if num_lo_flow == 4 or num_lo_flow == 5:
            save_npy = f'{save_path}/stats_{test_blif_name}_method_{method}_twice_seed{seed}.npy'
        else:
            save_npy = f'{save_path}/stats_{test_blif_name}_method_{method}_seed{seed}.npy'
        np.save(save_npy, save_data)


def get_command(test_blif_path, num_lo_flow):
    # flow0: 1 times resub default p=0
    # flow1: 1 times SR/Random/Trees/LR_mfs2/SVD p=1
    # flow2: 1 times GNN_mfs2 p=2
    # flow3: 1 times SVD decomposition p=3
    # flow4: 2 times SR/Random/Trees/LR_mfs2/SVD p=4
    # flow5: 2 times GNN_mfs2 p=5

    if num_lo_flow == 0:
        command = f"./abc -c 'r {test_blif_path}; strash; balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance; print_stats -t; resub -K 16 -N 3 -z; print_stats -t'"
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
    parser.add_argument('--model_path', nargs="+", default=["../../../models/ours/resub/MCTS_ours_square_resub.pkl"], type=str)
    # parser.add_argument('--no_lut_model_path', type=str,
    #                     default="./model/epfl/square/square_no_smote_15gen_GNN_mae_non_normalize_all_no_lut.pkl")
    # parser.add_argument('--lut_model_path', type=str,
    #                     default="./model/epfl/square/square_no_smote_15gen_GNN_mae_non_normalize_all_lut.pkl")
    parser.add_argument('--test_blif_path', type=str,
                        default="../../../../dataset/open_source/ethernet.blif")
    parser.add_argument('--test_blif', type=str,
                        default="ethernet")
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--num_lo_flow', type=int,
                        default=0)
    parser.add_argument('--save_dir', type=str, default='ethernet_online_test')
    # METHOD includes: MCTS, GNN, Random, SVD, Trees, LR
    parser.add_argument('--method', type=str, default='MCTS_ours')
    parser.add_argument('--seed_num', type=int, default=1)

    args = parser.parse_args()
    print('args.model_path', args.model_path)
    print('file_name', args.test_blif)
    print('normalize', args.normalize)
    print('random', args.random)
    print('The testing method is', args.method)
    # GLOBAL VAR
    SEL_PERCENTS = args.sel_percents
    DEVICE = "cpu"

    # command
    command = get_command(args.test_blif_path, args.num_lo_flow)
    print('the command is', command)
    # EPFL
    # num_run_2_eq_penalty
    # SR_agent_no_lut = ['x_4-x_2*x_4*sin(x_4)-exp(x_2)/sin(x_0)+x_4-x_3-x_2*exp(exp(x_0))/cos(cos(x_3))*exp(cos(x_0))',
    #                    'x_4-sin(x_1)-exp(x_2)/x_0*exp(x_2)+cos(exp(x_3))-x_2-x_2/x_4-sin(exp(exp(x_4)))/exp(x_3)',
    #                    'x_0-exp(cos(cos(x_4)))-x_2*exp(x_0)*exp(x_4)+x_4-x_3+x_4-x_2-x_2*exp(exp(exp(sin(x_4))))',
    #                    'exp(x_0)*exp(cos(exp(x_3)))/cos(x_4)*x_4+x_3-exp(x_2+x_3)/x_0*sin(exp(x_2))-x_2-sin(x_2)*exp(exp(x_4))*exp(exp(x_2))*exp(exp(x_0))',
    #                    'cos(x_3)*x_0*x_4-exp(x_2)-x_3-x_2-x_2*exp(exp(x_4))+sin(x_4)-cos(x_0+x_0)']
    #num_run_2_eq_penalty
    # SR_agent_no_lut = ['sin(x_2)+x_1+sin(x_1)+x_2-x_1-exp(x_3-x_0)-x_3-x_2*x_2*x_2*exp(sin(x_2))*exp(x_4)/cos(x_2)/cos(x_2)-cos(x_2)+x_0*sin(exp(x_2))',
    #                    'cos(x_4)*sin(x_0)-x_3+sin(x_1)+x_0-x_3-exp(x_2)/cos(x_2)/sin(exp(sin(x_2)))+sin(x_2)*sin(exp(x_4))*exp(sin(sin(exp(exp(x_0+exp(x_1))))))',
    #                    'sin(x_0)*x_4-x_4*x_2+sin(exp(x_2))*x_0*cos(sin(x_3))-x_3-exp(x_2*x_2)/cos(x_2)/cos(x_2)/cos(x_4)+x_2+x_2',
    #                    'x_4*x_1-x_3-cos(x_0)-x_2*x_2+x_0-x_4/cos(x_2)-sin(exp(exp(x_2)))-x_2',
    #                    'cos(x_4)*exp(sin(exp(x_0)))*exp(cos(x_2))*x_0*x_3*x_2+exp(x_2)*cos(x_3+exp(x_2))-x_3/cos(x_4)+x_0-cos(x_1)+x_1*x_4']   
    # SR_agent_lut = ['(1-(x_24*(x_21*x_14)))',
    #                 '(1-(x_43*(x_13*x_56)))',
    #                 '((1-((1-x_6)*x_55))*(1-(x_43*(x_56*x_21))))',
    #                 '((1-((1-x_38)*x_39))*(1-((x_16*x_5)*x_11)))',
    #                 '((1-(x_56*(x_59*x_5)))*(1-(x_23*(1-x_22))))'
    #                 ]
    # IWLS
    #num_run_2_eq_penalty
    # SR_agent_no_lut = ['cos(x_3)*exp(x_2)*exp(x_4)*exp(x_2)*exp(x_0)*x_3-cos(sin(x_4))/x_0-cos(exp(x_2))*x_0*exp(x_3)/sin(cos(x_3))-x_2*x_2*exp(x_4)',
    #                    'x_0*x_4-cos(x_4)/x_0-x_1+x_3+x_2*x_3*exp(cos(x_4))*exp(x_4)*exp(x_0)-x_1-x_3-x_1*exp(x_4+exp(x_0))*exp(exp(x_0))+x_3',
    #                    'exp(x_4)*exp(x_2)*exp(x_2)*exp(sin(x_2))*x_3-cos(x_1)/x_0+x_4+x_3-x_3*x_3*exp(x_0)-x_2*x_2/cos(x_0)-x_3*sin(x_3-x_2)/exp(x_1-exp(x_4))',
    #                    'x_0*exp(x_2)*cos(x_3)*exp(x_4)*exp(x_2)*exp(x_2)*x_3-cos(x_1)/x_0+sin(x_4)+x_3+x_2*x_3*exp(exp(x_0))-x_3-x_3*x_3*exp(x_0)*exp(x_4)'
    #                     ]
    # SR_agent_lut = ['(1-(((1-x_53)*x_14)*x_7))',
    #                 '(1-((1-x_55)*x_7))',
    #                 '(1-(x_5*(1-((1-((x_47*x_49)*(1-x_32)))*(1-((1-x_32)*x_32))))))',
    #                 '(1-((1-((1-((x_49*(1-x_52))*x_37))*(1-(x_32*(1-x_16)))))*x_51))',
    #                 ]

    # EPFL
    # num_run_2_eq_penalty_getscore_modify_without_memctrl
    # SR_agent_no_lut = {
    #                 'seed1': {
    #                         'square': '((((((x_1-cos(x_0))-exp(x_2))/exp(x_1))-(x_1*x_1))/sin(exp(x_1)))+(x_0*x_1))',
    #                         'square': '(((x_1-exp((exp(x_2)-x_0)))-sin(exp(exp(x_1))))*cos(cos(cos(x_2))))',
    #                         'multiplier': '((((((x_0-exp(x_1))-exp(x_2))+sin(x_1))+x_1)/sin(exp(x_4)))/exp(x_1))',
    #                         'sin': '((((x_3-x_2)-exp(cos(x_0)))/sin(exp(x_1)))+x_1)',
    #                         'square': '(x_2+(((x_0-cos(x_1))-exp(x_2))/cos((x_1-x_4))))'
    #                         },
    #                 'seed2': {
    #                         'square': '(((((sin(x_1)-exp((cos(x_0)+x_2)))/exp(x_1))/exp(x_1))+x_2)-(x_1*x_1))',
    #                         'square': '((x_0-exp(cos(sin(x_3))))-(x_2-cos((x_4-exp(sin((exp(exp(x_1))-x_4)))))))',
    #                         'multiplier': '(((x_1*x_1)-(exp(x_2)/sin(x_0)))*cos(cos(x_0)))',
    #                         'sin': '((((((x_0-(exp(x_2)/sin(x_0)))-x_0)+x_1)*cos(cos(x_0)))/cos(x_4))/exp(x_1))',
    #                         'square': '(((x_1-exp((exp(x_2)-x_0)))+x_2)/sin(exp(x_4)))'
    #                         },
    #                 'seed3': {
    #                         'square': '(((sin(x_1)+((x_1-exp(cos(x_0)))-x_2))/exp(x_1))-(x_1*x_1))',
    #                         'square': '(((((x_3*x_4)-(exp(x_2)/sin(x_0)))-(x_4*x_3))+x_1)*cos(sin(exp((x_0+sin(exp(x_1)))))))',
    #                         'multiplier': '((x_2+(((x_1-(exp(x_2)/x_0))*cos(cos(x_0)))/sin(cos(x_4))))/exp(x_1))',
    #                         'sin': '((((x_1-sin(x_4))-exp(sin(x_2)))/(x_4+x_0))/cos(x_4))',
    #                         'square': '(((x_0-exp(cos(sin(x_3))))/cos(x_4))/exp(x_1))'
    #                         }
    #                     }
    # SR_agent_lut = {
    #                 # 'seed1': ['(1-(x_53*x_14))',
    #                 #         '((1-(x_39*x_28))*(1-(x_45*x_14)))',
    #                 #         '((1-((1-x_47)*x_61))*(1-(x_12*x_39)))',
    #                 #         '((1-(x_21*x_30))*(1-(x_12*x_55)))',
    #                 #         '((1-(x_28*x_7))*(1-(x_30*x_45)))']
    #                 # 'seed1': ['(1-(x_24*(x_21*x_14)))',
    #                 # '(1-(x_43*(x_13*x_56)))',
    #                 # '((1-((1-x_6)*x_55))*(1-(x_43*(x_56*x_21))))',
    #                 # '((1-((1-x_38)*x_39))*(1-((x_16*x_5)*x_11)))',
    #                 # '((1-(x_56*(x_59*x_5)))*(1-(x_23*(1-x_22))))'
    #                 # ]
    #                 'seed1': {
    #                     'square': '(1-(x_24*(x_21*x_14)))',
    #                     'square': '(1-(x_43*(x_13*x_56)))',
    #                     'multiplier': '((1-((1-x_47)*x_61))*(1-(x_12*x_39)))',
    #                     'sin': '((1-((1-x_38)*x_39))*(1-((x_16*x_5)*x_11)))',
    #                     'square': '((1-(x_56*(x_59*x_5)))*(1-(x_23*(1-x_22))))'
    #                     }
    #                 }   

    # IWLS
    # num_run_2_eq_penalty_getscore_modify_without_memctrl
    # SR_agent_no_lut = {
    #                 # 'seed1': ['cos(x_3)*exp(x_2)*exp(x_4)*exp(x_2)*exp(x_0)*x_3-cos(sin(x_4))/x_0-cos(exp(x_2))*x_0*exp(x_3)/sin(cos(x_3))-x_2*x_2*exp(x_4)',
    #                 #    'x_0*x_4-cos(x_4)/x_0-x_1+x_3+x_2*x_3*exp(cos(x_4))*exp(x_4)*exp(x_0)-x_1-x_3-x_1*exp(x_4+exp(x_0))*exp(exp(x_0))+x_3',
    #                 #    'exp(x_4)*exp(x_2)*exp(x_2)*exp(sin(x_2))*x_3-cos(x_1)/x_0+x_4+x_3-x_3*x_3*exp(x_0)-x_2*x_2/cos(x_0)-x_3*sin(x_3-x_2)/exp(x_1-exp(x_4))',
    #                 #    'x_0*exp(x_2)*cos(x_3)*exp(x_4)*exp(x_2)*exp(x_2)*x_3-cos(x_1)/x_0+sin(x_4)+x_3+x_2*x_3*exp(exp(x_0))-x_3-x_3*x_3*exp(x_0)*exp(x_4)'
    #                 #     ],
    #                 # 'seed2': ['cos(x_3)*exp(x_2)*exp(x_4)*exp(x_2)*exp(x_0)*x_3-cos(sin(x_4))/x_0-cos(exp(x_2))*x_0*exp(x_3)/sin(cos(x_3))-x_2*x_2*exp(x_4)',
    #                 #    'x_0*x_4-cos(x_4)/x_0-x_1+x_3+x_2*x_3*exp(cos(x_4))*exp(x_4)*exp(x_0)-x_1-x_3-x_1*exp(x_4+exp(x_0))*exp(exp(x_0))+x_3',
    #                 #    'exp(x_4)*exp(x_2)*exp(x_2)*exp(sin(x_2))*x_3-cos(x_1)/x_0+x_4+x_3-x_3*x_3*exp(x_0)-x_2*x_2/cos(x_0)-x_3*sin(x_3-x_2)/exp(x_1-exp(x_4))',
    #                 #    'x_0*exp(x_2)*cos(x_3)*exp(x_4)*exp(x_2)*exp(x_2)*x_3-cos(x_1)/x_0+sin(x_4)+x_3+x_2*x_3*exp(exp(x_0))-x_3-x_3*x_3*exp(x_0)*exp(x_4)'
    #                 #     ],
    #                 # 'seed3': ['cos(x_3)*exp(x_2)*exp(x_4)*exp(x_2)*exp(x_0)*x_3-cos(sin(x_4))/x_0-cos(exp(x_2))*x_0*exp(x_3)/sin(cos(x_3))-x_2*x_2*exp(x_4)',
    #                 #    'x_0*x_4-cos(x_4)/x_0-x_1+x_3+x_2*x_3*exp(cos(x_4))*exp(x_4)*exp(x_0)-x_1-x_3-x_1*exp(x_4+exp(x_0))*exp(exp(x_0))+x_3',
    #                 #    'exp(x_4)*exp(x_2)*exp(x_2)*exp(sin(x_2))*x_3-cos(x_1)/x_0+x_4+x_3-x_3*x_3*exp(x_0)-x_2*x_2/cos(x_0)-x_3*sin(x_3-x_2)/exp(x_1-exp(x_4))',
    #                 #    'x_0*exp(x_2)*cos(x_3)*exp(x_4)*exp(x_2)*exp(x_2)*x_3-cos(x_1)/x_0+sin(x_4)+x_3+x_2*x_3*exp(exp(x_0))-x_3-x_3*x_3*exp(x_0)*exp(x_4)'
    #                 #     ]
    #                 'seed1': {
    #                         'des_perf': '((((((((((x_3*x_2)+(((((x_0-(exp(cos(x_4))-x_3))+sin(x_2))/sin(exp(x_3)))+x_3)*exp(x_4)))-x_1)/cos((x_3-x_4)))-(x_3*x_3))/exp(x_3))+(x_3*x_2))-x_1)-x_1)/cos(x_3))',
    #                         'ethernet': 'x_4+exp(x_0)-exp(exp(x_1))/cos(x_3-exp(x_1)+x_2*sin(x_3))',
    #                         'vga_lcd': '((((((((((((x_3-exp(x_1))/(cos((x_3-exp(x_1)))+x_3))-x_1)/x_0)+x_2)/cos(x_0))-x_1)+(x_4-x_3))/cos(x_4))-x_1)-x_1)+(cos((x_0-exp(x_3)))*x_3))',
    #                         'wb_conmax': '(x_4-(cos(x_0)-((((((((((((cos(exp((x_3+exp(x_3))))/(x_2+x_3))/exp(sin(x_3)))+(x_0-x_3))-x_1)-x_4)-x_1)-x_1)+(x_3*x_4))-x_1)+(x_3*x_2))/sin(exp(x_3)))-(x_1+x_1))))'
    #                         },
    #                 'seed2': {
    #                         'des_perf': '((((((x_4+(((((x_2+((x_0-exp((exp((x_3*x_1))-x_3)))/exp(x_3)))-x_1)-x_1)/x_0)/cos(sin(x_0))))-x_3)/cos(cos(exp(x_0))))+x_3)-(x_3*x_3))+(x_2*x_3))',
    #                         'ethernet': 'exp(cos(exp(x_3)))*exp(x_2)*x_3-exp(cos(x_0)-x_0)/cos(x_3)/sin(exp(x_3))+x_3*x_4-x_1*exp(x_0)*exp(exp(x_0))-cos(exp(x_1))+x_3*x_2+x_3',
    #                         'vga_lcd': '((((((((((((x_3-exp(x_1))/(cos((x_3-exp(x_1)))+x_3))-x_1)/x_0)+x_2)/cos(x_0))-x_1)+(x_4-x_3))/cos(x_4))-x_1)-x_1)+(cos((x_0-exp(x_3)))*x_3))',
    #                         'wb_conmax': '(((((((cos(x_2)*(((x_2+((x_2-x_3)+((x_3+((x_3-exp((x_3-x_0)))/x_0))/(x_3+x_3))))-x_1)*exp(x_3)))*cos(x_2))-x_1)-x_1)-(x_1+x_1))+(x_3*x_4))-x_1)'
    #                         },
    #                 'seed3': {
    #                          'des_perf': '(((((((((((x_3/cos(x_4))-exp(x_1))/(x_2+x_3))+sin(x_4))-x_1)-(x_3*x_3))/x_0)+(x_3*x_2))/(x_3+cos(x_0)))-sin(x_1))-cos((x_0+x_0)))',
    #                          'ethernet': 'cos(x_3)*exp(x_2)*x_3+sin(x_3)-exp(x_1)/x_0+cos(exp(x_3))*x_3-x_1*exp(x_0)*exp(exp(x_3))+x_2*x_3*exp(x_4)/sin(cos(x_4))*exp(x_2)-x_2*x_0',
    #                          'vga_lcd': '((((((exp(x_2)*(((((((x_3-exp((x_1+x_1)))/x_0)+(x_4*x_3))+(x_3*x_2))/(x_2+x_3))/exp(x_3))+sin(cos(exp(x_3)))))+sin(x_2))-(x_2*x_2))-x_1)+(x_3*x_2))*exp(x_3))',
    #                          'wb_conmax': '(((((((cos(x_2)*(((x_2+((x_2-x_3)+((x_3+((x_3-exp((x_3-x_0)))/x_0))/(x_3+x_3))))-x_1)*exp(x_3)))*cos(x_2))-x_1)-x_1)-(x_1+x_1))+(x_3*x_4))-x_1)'
    #                         }
    #                     }
    # SR_agent_lut = {
    #                 'seed1': {
    #                 'des_perf': '((1-((x_10*(x_29*x_12))*x_22))*(1-(((1-x_36)*x_43)*x_7)))',
    #                 'ethernet': '(1-((1-x_50)*x_34))',
    #                 'vga_lcd': '(1-(((1-x_26)*x_55)*x_59))',
    #                 'wb_conmax': '(1-((1-x_18)*x_34))'
    #                 }
    #                 # num_run_2_eq_penalty
    #                 # 'seed1': ['(1-(((1-x_53)*x_14)*x_7))',
    #                 # '(1-((1-x_55)*x_7))',
    #                 # '(1-(x_5*(1-((1-((x_47*x_49)*(1-x_32)))*(1-((1-x_32)*x_32))))))',
    #                 # '(1-((1-((1-((x_49*(1-x_52))*x_37))*(1-(x_32*(1-x_16)))))*x_51))',
    #                 # ]
    #                 } 
    # Open_source circuits 
    SR_agent = {
                'seed1': {
                        'hyp': '((cos(x_11)-exp((x_15+x_2)))+(x_8+(x_0*(x_15-exp(x_2)))))',
                        'multiplier': '(((cos(x_4)*(((sin(cos((x_6+cos(x_11))))-(x_10/x_17))+(x_7/x_3))-(x_12/x_8)))-(x_2/x_14))*(x_16/x_2))',
                        'square': '((x_8+(((((x_17-(x_14+x_12))*exp(x_0))+(x_12-x_6))-x_16)+(x_11-x_3)))-(x_17*(x_10+x_7)))',
                        'des_perf': '(((((((x_2-exp(x_0))+(x_11-x_15))-x_9)-(x_0-x_11))/exp(x_11))-x_9)-x_9)',
                        'ethernet': '(((x_8+x_16)-x_13)+(((exp(x_19)*x_8)-(x_3+x_15))+x_11))',
                        'wb_conmax': '(x_11+(x_6-x_13))'
                        },
                }
 
    for seed in range(args.seed_num):
        print(f'the information of seed {seed}')
        stats_dict = {}
        for sel_percent in SEL_PERCENTS:
            # json_kwargs = {
            #     "GCNMODEL": args.model_path,
            #     "DEVICE": DEVICE,
            #     "SEL_PERCENT": sel_percent,
            #     "RANDOM": args.random,
            #     "NORMALIZE": args.normalize
            # }
            if args.method == 'MCTS_ours':
                json_kwargs = {
                "MODEL": args.model_path,
                "SRMODEL": SR_agent[f'seed{seed+1}'][args.test_blif],
                "DEVICE": DEVICE,
                "SEL_PERCENT": sel_percent,
                "RANDOM": args.random,
                "NORMALIZE": args.normalize,
                "METHOD": args.method
                }
            else:
                json_kwargs = {
                    "MODEL": args.model_path[seed],
                    "SRMODEL": SR_agent[f'seed{seed+1}'][args.test_blif],
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
