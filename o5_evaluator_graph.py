import pandas as pd
import numpy as np
import torch
import time
import json
import re
from torch_geometric import loader
from o4_test_gs4co import GraphDataset, GraphDataLoader
import utils_GNN.gcn_policy as gcn_policy
import dso_utils_graph

# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)

FEATURE_CSV = './features.csv'
TRUTHTABLE_CSV = './truth_table.csv'
DEPTH = 2
FANIN_NODES_TYPE = 'remove'
GCNPOLICY_CLASS = 'GCNPolicy'
BATCH_SIZE = 70000
POLICY_KWARGS = {
    'mean_max': 'mean',
    'emd_size': 128,
    'out_size': 2,
    'num_pivot_node_features': 5,
    'num_children_node_features': 5
}
import os
import settings.consts as consts
current_directory = os.getcwd()
os.chdir(current_directory)
FEATURE_CSV = os.path.join(current_directory, './features.csv')
TRUTHTABLE_CSV = os.path.join(current_directory, './truth_table.csv')
LABLES_CSV = os.path.join(current_directory, './labels.csv')
with open('./configs.json', 'r') as f:
    json_kwargs = json.load(f)
MODEL = json_kwargs['MODEL']
DEVICE = json_kwargs['DEVICE']
SEL_PERCENT = json_kwargs['SEL_PERCENT']
RANDOM = json_kwargs['RANDOM']
NORMALIZE = json_kwargs['NORMALIZE']
METHOD = json_kwargs['METHOD']
FEATURE_SELECTION = json_kwargs['feature_selection']
NPY_FILE_PATH = json_kwargs['npy_file_path']
TEST_DATA_LOADER_KWARGS = {
    'npy_data_path': NPY_FILE_PATH,
    'save_dir': 'v2_processed_data',
    'processed_npy_path': None,
    'sample_bool': False,
    'train_type': 'test',
    'max_batch_size': BATCH_SIZE,
    'sample_type': 'upsampling',
    'load_type': 'default_order',
    'depth': 2,
    'fanin_nodes_type': 'remove',
    'normalize': NORMALIZE,
    'feature_selection': FEATURE_SELECTION
} 
def normalize_minmax(returns): # 使用最小值和最大值进行归一化
    for i in range(returns.shape[0]):
        min_return = returns[i].min()
        delta = returns[i].max() - min_return
        if delta < consts.SAFE_EPSILON:
            delta = 1.
        returns[i] = (returns[i] - min_return) / delta
    return returns

def fix_expression(expr: str) -> str:
    """
    在 expr 中找出所有 scatter_<op>(..., v_edge_index) 并补全 dim 参数
    """
    # 1. 预编译：匹配 scatter_<op>( 开头
    opener = re.compile(r'(scatter_(?:mean|sum|min|max))\s*\(')

    pos = 0
    pieces = []

    while True:
        m = opener.search(expr, pos)
        if not m:
            pieces.append(expr[pos:])
            break

        # 2. 记录函数名和起始位置
        func_start = m.start()
        pieces.append(expr[pos:func_start])  # 保留之前内容
        func_name = m.group(1)
        scan = m.end()                       # 从 '(' 后开始扫描

        # 3. 括号计数找匹配的 ')'
        level = 1
        comma_pos = -1
        right_pos = -1
        for i, ch in enumerate(expr[scan:], start=scan):
            if ch == '(':
                level += 1
            elif ch == ')':
                level -= 1
                if level == 0:
                    right_pos = i
                    break
            elif ch == ',' and level == 1:
                comma_pos = i  # 记录顶层逗号位置

        # 4. 检查第二个参数是不是 v_edge_index（允许前后空格）
        second_arg = expr[comma_pos + 1:right_pos].strip()
        if second_arg == 'v_edge_index':
            # 5. 替换：在右括号前插入 , dim=0, dim_size=variable.size(0)
            new_call = (
                expr[func_start:comma_pos] +
                ', v_edge_index, dim=0, dim_size=variable.size(0))'
            )
            pieces.append(new_call)
        else:
            # 不匹配，原样保留
            pieces.append(expr[func_start:right_pos + 1])
        pos = right_pos + 1

    return ''.join(pieces)

def _normalize(features):
    max_vals = np.max(features, axis=0)
    min_vals = np.min(features, axis=0)
    normalized_features = (features - min_vals) / \
        (max_vals - min_vals + 1e-3)

    # print(f"debug log normalized features: {normalized_features}")
    return normalized_features

def _process_csv():
    # process collected csv
    data_frame = pd.read_csv(FEATURE_CSV)
    data_frame_truth_table = pd.read_csv(TRUTHTABLE_CSV)
    numpy_data = data_frame.to_numpy()
    numpy_data_truth_table = data_frame_truth_table.to_numpy()

    # get features current node
    features = numpy_data[:, 1:6].astype(np.int32)
    # get features truth table
    features_truth_table = numpy_data_truth_table[:, 1:]
    # get features x and label y
    features = np.concatenate((features, features_truth_table), axis=1)

    # first traverse id
    first_traverse_id = features[:, 4][0]

    # normalize features
    if NORMALIZE:
        features = _normalize(features)
    children = numpy_data[:, -2]
    parents = numpy_data[:, -1]
    return features, children, parents, first_traverse_id

def process_children(children, first_traverse_id):
    processed_children = []
    for i in range(children.shape[0]):
        processed_chhildren_i = []
        for ind in children[i].split("/"):
            try:
                processed_chhildren_i.append(int(ind))
            except:
                continue
            if DEPTH == 2:
                try:
                    actual_index = int(ind) - first_traverse_id
                    # modified by yqbai 240728
                    # if actual_index < 0:
                    #     continue
                    # modified by yqbai 240728
                    for ind_2 in children[actual_index].split("/"):
                        try:
                            processed_chhildren_i.append(int(ind_2))
                        except:
                            continue
                except:
                    pass
                    #  print(f"found fanin nodes")
        processed_children.append(processed_chhildren_i)
        # print(f"{i} th children index: {processed_chhildren_i}")
    return processed_children


def get_children(children, i, first_traverse_id, features):
    children_indexes = children[i]
    children_features = []
    for id in children_indexes:
        index = id - first_traverse_id
        index = int(index)
        # print('id', id)
        # print('first_traverse_id', first_traverse_id)
        if index >= 0:
            children_feature = features[index:index+1, :]
            children_features.append(children_feature)
        else:
            if FANIN_NODES_TYPE == 'zero':
                children_feature = np.zeros((1, 69), dtype=np.int32)
            elif FANIN_NODES_TYPE == 'one':
                children_feature = np.ones((1, 69), dtype=np.int32)
            elif FANIN_NODES_TYPE == 'remove':
                continue
            children_features.append(children_feature)

    if len(children_features) == 0:
        children_features.append(np.ones((1, 69), dtype=np.int32))
    return np.vstack(children_features)

def get_edge_indexes(num_children):
    zero_row = np.zeros((1, num_children), dtype=np.int32)
    one_row = np.arange(num_children).astype(np.int32)
    one_row = np.expand_dims(one_row, axis=0)
    edge_indexes = np.vstack([zero_row, one_row])

    return edge_indexes

def process_data(features, children, first_traverse_id):
    samples = []
    for i in range(features.shape[0]):
        pivot_node_features = features[i:i+1, :]
        children_features = get_children(
            children, i, first_traverse_id, features)
        edge_indexes = get_edge_indexes(children_features.shape[0])
        label = np.array([0])
        if FEATURE_SELECTION == 'no_lut':
            pivot_node_features = pivot_node_features[:, 1:6]
            children_features = children_features[:, 1:6]
        sample = (pivot_node_features, edge_indexes, children_features, label)
        samples.append(sample)

    return samples

def evaluate_policy(data, graph_policy, first_traverse_id, time_dict):
    predict_scores_list = []
    for batch in data.graph_data_loader:
        # print("evaluating data loader ..............")
        batch = batch.to(DEVICE)
        batch_pivot_node_features = batch.x_variable.to(DEVICE)
        batch_children_node_features = batch.x_constraint.to(DEVICE)
        batch_edge_indexes = batch.x_cv_edge_index.to(DEVICE)
        batch_edge_indexes = batch_edge_indexes.flip(0)
        # print('first line of batch_edge_indexes', batch_edge_indexes[0,:])
        # print('second line of batch_edge_indexes', batch_edge_indexes[1,:])
        # print('batch_pivot_node_features', batch_pivot_node_features.shape)
        # print('batch_children_node_features', batch_children_node_features.shape)
        # print('batch_edge_indexes', batch_edge_indexes.shape)
        # print('batch_pivot_node_features', batch_pivot_node_features.numpy()[0:10, 0:5])
        # print('batch_children_node_features', batch_children_node_features.numpy()[0:10, 0:5])
        # print('batch_edge_indexes', batch_edge_indexes.numpy())
        # added for test
        # sin_data = np.load('/yqbai/GLENORE/ai4mfs2/npy_data/temp/sin_temp.npy', allow_pickle=True).item()
        # sin_batch_pivot_node_features = sin_data['node_features']
        # sin_batch_children_node_features = sin_data['children_node_features']
        # sin_batch_edge_indexes = sin_data['edge_indexes']
        # with torch.no_grad():
        #     sin_predict_scores = graph_policy(
        #         sin_batch_pivot_node_features,
        #         sin_batch_edge_indexes,
        #         sin_batch_children_node_features
        #     )
        #     # print(graph_policy)
        # sin_predict_scores = sin_predict_scores.cpu().detach().numpy()
        # sin_predict_scores = np.mean(sin_predict_scores, axis=1, keepdims=True)
        # added for test
        time_dict['graph_data_load'] = time.time()
        if METHOD == 'COG':
            with torch.no_grad():
                predict_scores = graph_policy(
                    batch_pivot_node_features,
                    batch_edge_indexes,
                    batch_children_node_features
                )
                predict_scores = predict_scores.mean(dim=1, keepdim=True)
                # predict_scores = predict_scores.cpu().detach().numpy()
                # predict_scores = np.mean(predict_scores, axis=1, keepdims=True)
        elif METHOD == 'HIS':
            predict_scores = graph_policy(batch, train_mode=False)
            # predict_scores = predict_scores.cpu().detach().numpy()
        # predict_scores = np.mean(predict_scores, axis=1, keepdims=True)
        predict_scores_list.append(predict_scores)
        torch.cuda.empty_cache()
    pred_y = torch.cat(predict_scores_list, dim=1)
    pred_y = normalize_minmax(pred_y)
    pred_y = pred_y.view(-1)
    total_num = pred_y.size(0)
    _, ascending_indexes = torch.sort(pred_y, descending=True)
    # print(f'The first 10 indexes of prediction indexes are {ascending_indexes[:10]}')
    ascending_indexes += first_traverse_id
    sel_num = int(total_num * SEL_PERCENT)
    # predict_scores = np.vstack(predict_scores_list)
    # predict_scores = predict_scores.squeeze()
    # print(predict_scores.shape)
    # ascending_indexes = np.argsort(
    #     predict_scores)  # default ascending order
    # # print('the predicted indexes are', ascending_indexes)
    # print(f'The first 10 indexes of prediction indexes are {ascending_indexes[:10]}')
    # ascending_indexes += first_traverse_id
    # total_num = predict_scores.shape[0]
    # # print(f'total_num is {total_num}')
    # sel_num = int(total_num * SEL_PERCENT)
    if METHOD == 'HIS':
        sel_indexes = np.array(
            ascending_indexes[:sel_num], dtype=np.int32)
    else:
        sel_indexes = np.array(
            ascending_indexes[total_num-sel_num:], dtype=np.int32)
    # sel_indexes = sel_indexes[::-1]
    sel_indexes.sort()
    # print(sel_indexes)
    # print(sel_indexes.shape[0])
    # print('len sel_indexes is', len(sel_indexes))
    return sel_indexes

def load_graph_model():
    if METHOD == 'COG':
        graph_policy = getattr(gcn_policy, GCNPOLICY_CLASS)(
        **POLICY_KWARGS).to(DEVICE)
        model_state_dict = torch.load(MODEL, map_location=DEVICE)
        graph_policy.load_state_dict(model_state_dict)
        graph_policy.eval()
    elif METHOD == 'HIS':
        expr = torch.load(MODEL)["best_expr"]
        # print(f"expression before fix:{expr.expression}")
        expr.expression = fix_expression(expr.expression)
        # print(f"expression after fix:{expr.expression}")
        graph_policy = dso_utils_graph.EnsemBleExpression([expr])
    return graph_policy

def get_random_indexes(num_samples, first_traverse_id):
    random_scores = [np.random.rand() for _ in range(num_samples)]
    ascending_indexes = np.argsort(random_scores)
    ascending_indexes += first_traverse_id
    sel_num = int(num_samples * SEL_PERCENT)
    sel_indexes = np.array(
        ascending_indexes[num_samples-sel_num:], dtype=np.int32)
    sel_indexes.sort()
    return sel_indexes

def online_inference():
    """
    # input: csv files and gnn models
    # output: the selected traverse ids
    """
    time_dict = {}
    time_dict["st"] = time.time()
    # First step: read csv process to numpy
    features, children, parents, first_traverse_id = _process_csv()
    # print('the features', features[0,:])
    time_dict["process_csv"] = time.time()
    # first_traverse_id = features[:, 4][0]
    num_samples = features.shape[0]
    processed_children = process_children(children, first_traverse_id)
    time_dict["process_children"] = time.time()
    # Second step: process numpy to graphs
    graph_samples = process_data(
        features, processed_children, first_traverse_id)
    time_dict["process_graph_samples"] = time.time()
    graph_data_loader = GraphDataLoader(**TEST_DATA_LOADER_KWARGS)
    # graph_dataset = GraphDataset(graph_samples)
    # graph_data_loader = loader.DataLoader(
    #     graph_dataset,
    #     BATCH_SIZE,
    #     shuffle=False
    # )
    # Third step: read models and inference
    graph_policy = load_graph_model()
    time_dict["load_graph_model"] = time.time()
    if RANDOM:
        sel_indexes = get_random_indexes(num_samples, first_traverse_id)
    else:
        sel_indexes = evaluate_policy(
            graph_data_loader, graph_policy, first_traverse_id, time_dict)
    time_dict["policy inference"] = time.time()

    last_time = 0
    for k in time_dict.keys():
        if k == 'st':
            last_time = time_dict[k]
            continue
        print(f"time {k}: {time_dict[k]-last_time}")
        last_time = time_dict[k]
    # Fourth step: return
    return sel_indexes


if __name__ == "__main__":
    online_inference()
