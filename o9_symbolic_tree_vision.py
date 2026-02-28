import argparse
from collections import Counter
import csv
from doctest import testfile
from gettext import find
import os
from pathlib import Path
import re
from sympy import li
import torch
import settings.consts as consts

# 所有可能出现的“键”
KEYS = [
    'test_file_name', 'iteration', 'best of N', 'train_loss',
    'BON_test_loss', 'BON_test_top50_recall',
    'BON_exp_nlp', 'BON_exp_expression'
] + [f'variable_generated_{i}_{s}' for i in range(1, 6) for s in ('nlp', 'expression')] \
  + [f'constraint_generated_{i}_{s}' for i in range(1, 6) for s in ('nlp', 'expression')]

# 拼成前瞻：(?=key1:|key2:|...|$)
key_pat = '|'.join(map(re.escape, KEYS))
regex   = re.compile(fr'({key_pat}):\s*(.*?)(?=(?:{key_pat}):|$)')

def parse_line(line:str): # 正则匹配，返回字典
    data = {}
    # pattern = re.compile(r'(\S+?):\s*(.*?)(?=\s+\S+:|$)')
    for key, value in regex.findall(line):
    # for match in re.finditer(r'(\S+?):([^\s]+)(?=\s+|$)', line):
        # key, value = match.groups()
        value = value.strip()
        # if key in ('test_file_name', 'iteration', 'N', 'train_loss', 'BON_test_top50_recall'):
        if key == 'iteration':
            data[key] = int(float(value))
        elif key == 'best of N':
            data['best of N'] = int(float(value))
        elif key in ('train_loss', 'BON_test_top50_recall'):
            data[key] = float(value)
        else:
            data[key] = value.strip()
        # elif key in ('BON_exp_nlp', 'BON_exp_expression'):
        #     data[key] = value.strip()
        # # elif key in ([f'variable_generated_{i}_{s}' for i in range(1, 6) for s in ('nlp', 'expression')]):
        # #     data[key] = value.strip()
        # else:
        #     data[key] = value.strip()

    return data
def variable_statistics(log_path:str, thres:float, test_file:str, out_csv:str):
    """
    统计 BON_exp_nlp 中各个变量（来自 consts.GRAPH_NAMES）出现的频率，
    仅考虑 recall >= thres 的样本，结果写入 out_csv。
    """
    variable_names = ['scatter_sum', 'scatter_mean', 'scatter_max', 'scatter_min']
    freq = Counter({v: 0 for v in variable_names})   # 初始化 0

    with open(log_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('test_file_name'):
                continue
            info = parse_line(line)          # 复用你已有的函数
            if info.get('test_file_name', '')!=test_file or float(info.get('BON_test_top50_recall', -1)) < thres:
                continue

            expr = info.get('BON_exp_nlp', '')
            for v in variable_names:
                freq[v] += expr.count(v)     # 简单计数，可重叠

    # 写出 CSV
    with open(out_csv, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        writer.writerow(['variable', 'frequency'])
        for var, cnt in freq.items():
            writer.writerow([var, cnt])

def find_expression_iter(log_path:str, test_file:str, iteration=2000):
    # train_best_loss = float('-inf')
    # test_best_recall = [None] * 5
    # best_iteration = None
    BON_exp_nlp, BON_exp_expression = None, None
    variable_nlp, variable_expression = [], []
    constraint_nlp, constraint_expression = [], []

    itr_flag = False
    var_keys = [f'variable_generated_{i}_{s}' for i in range(1, 6) for s in ('nlp', 'expression')]
    con_keys = [f'constraint_generated_{i}_{s}' for i in range(1, 6) for s in ('nlp', 'expression')]
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('test_file_name'):
                if itr_flag: # 读完目标块，跳过其他
                    itr_flag = False
                    break
                info = parse_line(line)
                if (info.get('test_file_name') == test_file) and (info.get('iteration') == iteration) and (info.get('best of N') == 16):
                    itr_flag = True
                    BON_exp_nlp = info.get('BON_exp_nlp')
                    BON_exp_expression = info.get('BON_exp_expression')
            elif itr_flag:
                info = parse_line(line)
                for key, val in info.items():
                    if key in var_keys:
                        if key.endswith('_nlp'):
                            variable_nlp.append(val)
                        else:
                            variable_expression.append(val)
                    elif key in con_keys:
                        if key.endswith('_nlp'):
                            constraint_nlp.append(val)
                        else:
                            constraint_expression.append(val)

    print(f'test file:{test_file}\t iteration:{iteration}')
    print(f'BON_exp_nlp:{BON_exp_nlp}')
    print(f'BON_exp_alloc:{BON_exp_expression}')
    for i in range(5):
        print(f'variable_{i+1}_nlp:{variable_nlp[i]}\t alloc:{variable_expression[i]}')
    for i in range(5):
        print(f'constraint_{i+1}_nlp:{constraint_nlp[i]}\t alloc:{constraint_expression[i]}')

    return {
        'BON_exp_nlp': BON_exp_nlp,
        'BON_exp_expression': BON_exp_expression,
        'variable_nlp': variable_nlp,
        'variable_expression': variable_expression,
        'constraint_nlp': constraint_nlp,
        'constraint_expression': constraint_expression
    }

                # n = info.get('best of N')
                # idx = int(log(n, 2))
                # loss = info['train_loss']
                # recall = info['BON_test_top50_recall']
                # if loss is not None and recall is not None:
                #     if loss >= train_best_loss:
                #         train_best_loss = loss
                #         test_best_recall[idx] = recall
                #         best_iteration = info['iteration']

    # print(f"test file:{test_file}\t train_best_loss:{train_best_loss}\t iteration:{best_iteration}\n")
    # for i in range(5):
    #     print(f"best of N:{int(pow(2, i))}\t BON_test_top50_recall:{test_best_recall[int(i)]}")

def expressions_to_csv(log_path: str, test_files: list, iteration: int, csv_path: str):
    """把指定 test_files + iteration 的所有表达式写成 CSV"""
    fieldnames = ['test_file', 'iteration', 'type', 'id', 'nlp', 'expression']
    

    for tf in test_files:
        csv_path = os.path.join(csv_path, tf)
        csv_path += f"expression_iter2000.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        data = find_expression_iter(log_path, tf, iteration)

        # BON
        if data.get('BON_exp_nlp') is not None:
            writer.writerow({
                'test_file': tf,
                'iteration': iteration,
                'type': 'BON',
                'id': '',
                'nlp': data['BON_exp_nlp'],
                'expression': data['BON_exp_expression']
            })
            # variable
            # for i, nlp in enumerate(data['variable_nlp'], 1):
            #     writer.writerow({
            #         'test_file': tf,
            #         'iteration': iteration,
            #         'type': 'variable',
            #         'id': i,
            #         'nlp': nlp,
            #         'expression': data['variable_expression'][i-1]
            #     })
            # # constraint
            # for i, nlp in enumerate(data['constraint_nlp'], 1):
            #     writer.writerow({
            #         'test_file': tf,
            #         'iteration': iteration,
            #         'type': 'constraint',
            #         'id': i,
            #         'nlp': nlp,
            #         'expression': data['constraint_expression'][i-1]
            #     })
    print(f"CSV 已生成 → {Path(csv_path).resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='symbolic version')
    parser.add_argument('--file_name', default='iwls2epfl', type=str)
    parser.add_argument('--model_path', default='./models/OOD/HIS/HIS/des_perf/train_iter_2000.pkl', type=str)
    parser.add_argument('--out_path', default='/home/yqbai/HIS/gs4-ls-updated/out/epfl2iwls_focal_loss_score_normalize_entropy_coef_0.2_no_lut_normalize_pred_y_BON_1248_layer_layer1.out', type=str)
    # parser.add_argument('--out_path', default='./out/epfl2iwls_focal_loss_score_normalize_entropy_coef_0.2_no_lut_normalize_pred_y_BON_124816_layer.out', type=str)

    args = parser.parse_args()
    # state_obj = torch.load(args.model_path)
    # expr = torch.load(args.model_path)["best_expr"]
    # print(f"expression for file {args.file_name} is:{expr.expression}")
    with open(args.out_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(r'/[^\s]+', line)
            if match:
                result_path = match.group(0)
                print(f"result path:{result_path}")
                break
    best_path = result_path+'/best.txt'
    epfl2iwls_test_file = ['wb_conmax', 'ethernet', 'des_perf']
    iwls2epfl_test_file = ['hyp', 'multiplier', 'square']
    # if args.file_name == 'epfl2iwls':
    #     for test_file in epfl2iwls_test_file:
    #         find_expression_iter(best_path, test_file, 2000)
    # elif args.file_name == 'iwls2epfl':
    #     for test_file in iwls2epfl_test_file:
    #         find_expression_iter(best_path, test_file)
    # else:
    #     find_expression_iter(best_path, args.file_name)
    test_files = epfl2iwls_test_file if args.file_name == 'epfl2iwls' else iwls2epfl_test_file

    csv_out = f"./csv_results/{args.file_name}"
    expressions_to_csv(best_path, test_files, 2000, csv_out)
    # for file in test_files:
    #     csv_out = f"./csv_results/{args.file_name}/{file}_scatter_frequency_statistecs.csv"
    #     variable_statistics(best_path, 0.85, file, csv_out)
