from fileinput import filename
import glob
import os, random, gzip, argparse
import re
import sys
from sklearn import ensemble
import os.path as osp, torch.nn as nn
from torch import as_tensor
from torch_scatter import scatter_sum, scatter_mean
import torch_scatter

import torch, pickle
import numpy as np
import torch_geometric
from torch_scatter import scatter_mean, scatter_max, scatter_sum
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
from torch_geometric import loader

from utils.utilities import normalize_features, scatter_max, scatter_min, scatter_max_raw

import settings.consts as consts
consts.DEVICE = torch.device("cpu")
torch.set_default_device(consts.DEVICE)

import hydra # 用于配置管理的库，支持从YAML文件中加载配置
from omegaconf import OmegaConf # 用于处理配置文件的库，支持嵌套配置和配置解析

import time
import dso_utils_graph

class Expression(nn.Module): # 定义一个可执行的表达式模型
    def __init__(self, expression):
        super().__init__()

        assert ";;" in expression
        self.alloc_expression, self.expression = expression.split(";;") # 接受一个表达式字符串，并将其分为两部分

    def forward(self, constraint, variable, cv_edge_index, edge_attr, cand_mask): # 定义了模型的前向传播过程，执行 alloc_expression 并评估 expression
        c_edge_index, v_edge_index = cv_edge_index
        exec(self.alloc_expression)
        result = eval(self.expression)
        return result[cand_mask]

class BipartiteNodeData(torch_geometric.data.Data): # 继承自 torch_geometric.data.Data，用于定义二分图数据结构
    # def __inc__(self, key, value, *args, **kwargs):
    #     if key == "x_cv_edge_index": # 当属性名为 "x_cv_edge_index" 时，该方法返回一个2x2的张量，指示了在批处理过程中需要为边索引增加的值
    #         return torch.tensor(
    #             [[self.x_constraint.size(0)], [self.x_variable.size(0)]]
    #         )
    #     if key == "y_cand_mask": # 当属性名为 "y_cand_mask" 时，该方法返回的是变量节点的数量(self.x_variable.size(0))
    #         return self.x_variable.size(0)
    #     return super().__inc__(key, value, *args, **kwargs)
    def __init__(self, x_constraint, x_variable, x_cv_edge_index, x_edge_attr, y_cand_mask, y_cand_score, y_cand_label, depth=None):
        super().__init__()
        self.x_constraint = x_constraint  # 约束节点特征
        self.x_variable = x_variable  # 变量节点特征
        self.x_cv_edge_index = x_cv_edge_index  # 约束节点和变量节点之间的边索引
        self.x_edge_attr = x_edge_attr  # 边的属性
        self.y_cand_mask = y_cand_mask  # 候选变量的掩码
        self.y_cand_score = y_cand_score  # 候选变量的评分
        self.y_cand_label = y_cand_label  # 候选变量的标签
        self.depth = depth  # 层数信息（可选）

    def __inc__(self, key, value, *args, **kwargs):
        if key == "x_cv_edge_index":  # 边索引的增量
            return torch.tensor([[self.x_constraint.size(0)], [self.x_variable.size(0)]])
        if key == "y_cand_mask":  # 候选掩码的增量
            return self.x_variable.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class GraphDataset(torch_geometric.data.Dataset): # 继承自 utilities.BranchDataset，用于加载和处理图数据
    # def __init__(self, root, data_num, raw_dir_name="train", processed_suffix="_processed"):
    #     super().__init__(root, data_num, raw_dir_name, processed_suffix)

    # def process_sample(self, sample): # 重写了 process_sample 方法，用于将原始样本转换为一个适合图神经网络处理的 BipartiteNodeData 对象
    #     obss = sample["obss"] # 从 sample["obss"] 中获取变量(vars_all)、约束特征(cons_feature)、边信息(edge)、 层数信息(depth)和评分(scores)

    #     vars_all, cons_feature, edge = obss[0][0], obss[0][1], obss[0][2]
    #     depth = obss[2]["depth"]

    #     scores = obss[2]["scores"]
    #     vars_feature, indices = vars_all[:,:19], vars_all[:,-1].astype(bool) # 前者包含每个变量的前19个特征，后者是一个布尔数组，表示哪些变量是有效的分支变量
    #     indices = np.where(indices)[0] # 使用 np.where(indices) 找出所有有效分支变量的索引
    #     scores = scores[indices]
    #     labels = scores >= scores.max() # 创建二值标签(labels)

    #     scores = utilities.normalize_features(scores) # 对评分进行归一化处理

    #     data = BipartiteNodeData(x_constraint=as_tensor(cons_feature, dtype=torch.float, device="cpu"), x_variable=as_tensor(vars_feature, dtype=torch.float, device="cpu"),
    #                             x_cv_edge_index=as_tensor(edge['indices'], dtype=torch.long, device="cpu"), x_edge_attr=as_tensor(edge['values'].squeeze(1), dtype=torch.float, device="cpu"),
    #                             y_cand_mask=as_tensor(indices, dtype=torch.long, device="cpu"), y_cand_score=as_tensor(scores, dtype=torch.float, device="cpu"), y_cand_label=as_tensor(labels, dtype=torch.bool, device="cpu"),
    #                             depth=depth,)
    #     return data
    def __init__(self, samples):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.samples = samples

    def len(self):
        return len(self.samples)

    def get(self, index):
        sample = self.samples[index]
        pivot_node_features, edge_indices, children_node_features, label = sample

        # 将 pivot_node_features 作为变量节点，children_node_features 作为约束节点
        x_variable = torch.FloatTensor(pivot_node_features)  # 变量节点特征
        x_constraint = torch.FloatTensor(children_node_features)  # 约束节点特征

        # 调整边索引：确保边连接的是变量节点 -> 约束节点
        # 假设 edge_indices 的第一行是约束节点索引，第二行是变量节点索引
        # 我们需要交换两行，使其变为变量节点 -> 约束节点
        edge_indices = np.flip(edge_indices, axis=0)  # 交换两行
        x_cv_edge_index = torch.LongTensor(edge_indices.astype(np.int32))  # 边索引

        # 边的属性（初始化为全 1）
        x_edge_attr = torch.ones(x_cv_edge_index.size(1))

        # 候选变量的掩码（假设所有变量节点都是候选）
        y_cand_mask = torch.arange(x_variable.size(0), dtype=torch.long)

        # 候选变量的评分（初始化为全 1）
        y_cand_score = torch.FloatTensor([1.0] * x_variable.size(0))

        # 候选变量的标签
        y_cand_label = torch.FloatTensor(label.astype(np.float64))
        y_cand_label = y_cand_label.squeeze()

        # 创建 BipartiteNodeData 对象
        graph = BipartiteNodeData(
            x_constraint=x_constraint,
            x_variable=x_variable,
            x_cv_edge_index=x_cv_edge_index,
            x_edge_attr=x_edge_attr,
            y_cand_mask=y_cand_mask,
            y_cand_score=y_cand_score,
            y_cand_label=y_cand_label,
        )
        graph.num_nodes = x_constraint.size(0) + x_variable.size(0)
        return graph


class GraphDataLoader(DataLoader):
    def __init__(
        self,
        npy_data_path=None,
        save_dir=None,
        processed_npy_path=None,
        sample_rate=2,
        sample_bool=True,
        train_type='train',
        sample_type='upsampling',
        load_type='default_order',
        batch_size=128,
        max_batch_size=10240,
        depth=2,
        # debug
        debug_log=False,
        # domain attribute
        label_dict=None,
        # fanin_nodes
        fanin_nodes_type='zero',
        normalize=False,
        feature_selection=None
    ):
        self.npy_data_path = npy_data_path
        self.sample_rate = sample_rate
        self.sample_type = sample_type
        self.load_type = load_type
        self.sample_bool = sample_bool
        self.train_type = train_type
        self.save_dir = save_dir
        self.depth = depth
        self.debug_log = debug_log
        self.label_dict = label_dict
        self.fanin_nodes_type = fanin_nodes_type
        self.max_batch_size = max_batch_size
        self.normalize = normalize
        self.feature_selection = feature_selection

        if self.sample_type == 'downsampling':
            process_data = self.process_data
        elif self.sample_type == 'upsampling':
            process_data = self.process_data_up_sample
        else:
            raise NotImplementedError
        self.dataset = {
            "samples": [],
            "default_order_features": None,
            "default_order_label": None,
            "default_order_children_features": None,
            "default_traverse_id": None,
            "first_traverse_id": None
        }

        self.end_to_end_stats = None
        self.features = None
        self.labels = None
        self.children = None
        # num samples
        self.num_samples = 0
        self.num_positive_samples = 0
        self.num_negative_samples = 0
        samples_list = []
        samples_node_feature_list = []
        if processed_npy_path is None:
            if npy_data_path.endswith('.npy'):
                npy_file_path = npy_data_path
                self.load_data(npy_file_path)
                samples_node_feature, samples, bool_con = process_data()
                samples_list.extend(samples)
                samples_node_feature_list.extend(samples_node_feature)
            else:
                for npy_file in os.listdir(npy_data_path):
                    npy_file_path = os.path.join(npy_data_path, npy_file)
                    print(f"cur npy: {npy_file}")
                    self.load_data(npy_file_path)
                    samples_node_feature, samples, bool_con = process_data()
                    if bool_con:
                        continue
                    samples_list.extend(samples)
                    samples_node_feature_list.extend(samples_node_feature)
            # save processed samples
            # np.save(
            #     f"../../../npy_data/v2/{save_dir}/graph_samples_train_type_{train_type}_sample_bool_{self.sample_bool}.npy", samples_list)
        else:
            samples_list = np.load(processed_npy_path, allow_pickle=True)
        num_samples = len(samples_node_feature_list)
        self.data_loader = loader.DataLoader(samples_node_feature_list, num_samples, shuffle=False)
        self.graph_dataset = GraphDataset(samples_list)
        if self.train_type == 'train':
            total_num = len(samples_list)
            if total_num <= self.max_batch_size:
                self.graph_data_loader = loader.DataLoader(
                    self.graph_dataset, total_num,shuffle=False)
                print()
            else:
                sampler = BatchSampler(
                    SubsetRandomSampler(range(total_num)),
                    self.max_batch_size,
                    drop_last=False)
                self.graph_data_loader = loader.DataLoader(
                    self.graph_dataset, batch_sampler=sampler)
        elif self.train_type == 'test':
            total_num = len(samples_list)
            if total_num <= self.max_batch_size:
                self.graph_data_loader = loader.DataLoader(
                    self.graph_dataset, self.num_samples, shuffle=False)
            else:
                self.graph_data_loader = loader.DataLoader(
                    self.graph_dataset, self.max_batch_size, shuffle=False)

    def _get_label(self, npy_file_path):
        for k in self.label_dict.keys():
            if k in npy_file_path:
                label_value = self.label_dict[k]
        return label_value

    def _normalize(self):
        max_vals = np.max(self.features, axis=0)
        min_vals = np.min(self.features, axis=0)
        normalized_features = (self.features - min_vals) / \
            (max_vals - min_vals + 1e-3)
        self.features = normalized_features
        
    def load_data(self, npy_file_path):
        npy_data = np.load(npy_file_path, allow_pickle=True).item()
        self.end_to_end_stats = npy_data['end_to_end_stats']
        self.dataset['default_order_label'] = npy_data['labels_list'][0]
        self.dataset['default_traverse_id'] = list(
            npy_data['features_list'][0][:, 4])
        self.dataset['default_order_children_features'] = self.process_children(
            [npy_data['children'][0]])
        self.dataset['first_traverse_id'] = self.dataset['default_traverse_id'][0]
        print('the first traverse id is', self.dataset['first_traverse_id'])
        self.features = npy_data['features_list'][0]
        if self.normalize:
            self._normalize()
        self.dataset['default_order_features'] = self.features

        if self.label_dict is None:
            self.labels = npy_data['labels_list'][0]
        else:
            label_value = self._get_label(npy_file_path)
            self.labels = np.ones(
                (self.features.shape[0], 1)) * label_value
            self.labels = self.labels.astype(np.int64)
        self.children = self.process_children([npy_data['children'][0]])
        self.num_samples = self.labels.shape[0]
        if self.debug_log:
            print(f"debug log features: {self.features}")
            print(f"debug log labels: {self.labels}")
            print(
                f"debug log default_traverse_id: {self.dataset['default_traverse_id']}")
            print(f"debug log children: {self.children}")
            print(f"debug log npy file: {npy_file_path}")
            print(
                f"debug log first_traverse_id: {self.dataset['first_traverse_id']}")

    def process_children(self, children):
        processed_children = []
        for child_list in children:
            for i in range(child_list.shape[0]):
                processed_chhildren_i = []
                for ind in child_list[i].split("/"):
                    try:
                        processed_chhildren_i.append(int(ind))
                    except:
                        continue
                    if self.depth == 2:
                        try:
                            # actual_index = self.dataset['default_traverse_id'].index(
                            #     int(ind))
                            actual_index = int(ind) - self.dataset['first_traverse_id']
                            # print(f"found actual index: {actual_index}")
                            for ind_2 in child_list[actual_index].split("/"):
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

    def get_children(self, i):
        children_indexes = self.children[i]
        children_features = []
        #added by yqbai 20240607
        if self.feature_selection == 'no_lut':
            for id in children_indexes:
                index = id - self.dataset['first_traverse_id']
                if index >= 0:
                    # index = self.dataset['default_traverse_id'].index(id)
                    children_feature = self.dataset['default_order_features'][index:index+1, :5]
                    children_features.append(children_feature)
                else:
                    if self.fanin_nodes_type == 'zero':
                        children_feature = np.zeros((1, 5), dtype=np.int32)
                    elif self.fanin_nodes_type == 'one':
                        children_feature = np.ones((1, 5), dtype=np.int32)
                    elif self.fanin_nodes_type == 'remove':
                        continue
                    children_features.append(children_feature)
            if len(children_features) == 0:
                children_features.append(np.ones((1, 5), dtype=np.int32))
        elif self.feature_selection == 'lut':
            for id in children_indexes:
                index = id - self.dataset['first_traverse_id']
                if index >= 0:
                    # index = self.dataset['default_traverse_id'].index(id)
                    children_feature = self.dataset['default_order_features'][index:index+1, 5:]
                    children_features.append(children_feature)
                else:
                    if self.fanin_nodes_type == 'zero':
                        children_feature = np.zeros((1, 64), dtype=np.int32)
                    elif self.fanin_nodes_type == 'one':
                        children_feature = np.ones((1, 64), dtype=np.int32)
                    elif self.fanin_nodes_type == 'remove':
                        continue
                    children_features.append(children_feature)
            if len(children_features) == 0:
                children_features.append(np.ones((1, 64), dtype=np.int32))
        elif self.feature_selection == 'all':
            for id in children_indexes:
                index = id - self.dataset['first_traverse_id']
                if index >= 0:
                    # index = self.dataset['default_traverse_id'].index(id)
                    children_feature = self.dataset['default_order_features'][index:index+1, :]
                    children_features.append(children_feature)
                else:
                    if self.fanin_nodes_type == 'zero':
                        children_feature = np.zeros((1, 69), dtype=np.int32)
                    elif self.fanin_nodes_type == 'one':
                        children_feature = np.ones((1, 69), dtype=np.int32)
                    elif self.fanin_nodes_type == 'remove':
                        continue
                    children_features.append(children_feature)
            if len(children_features) == 0:
                children_features.append(np.ones((1, 69), dtype=np.int32))
        #added by yqbai 20240607
            # try:
            #     index = self.dataset['default_traverse_id'].index(id)
            #     children_feature = self.dataset['default_order_features'][index:index+1,:]
            #     children_features.append(children_feature)
            # except:
            #     # print(f"current node ones fanin!!!!")
            #     if self.fanin_nodes_type == 'zero':
            #         children_feature = np.zeros((1,69), dtype=np.int32)
            #     elif self.fanin_nodes_type == 'one':
            #         children_feature = np.ones((1,69), dtype=np.int32)
            #     elif self.fanin_nodes_type == 'remove':
            #         continue
            #     children_features.append(children_feature)
        return np.vstack(children_features)

    def get_edge_indexes(self, num_children):
        zero_row = np.zeros((1, num_children), dtype=np.int32)
        one_row = np.arange(num_children).astype(np.int32)
        one_row = np.expand_dims(one_row, axis=0)
        edge_indexes = np.vstack([zero_row, one_row])

        return edge_indexes

    def process_data_up_sample(self):
        # get positive and negative samples
        positive_indexes = np.nonzero(self.labels)[0]
        negative_indexes = np.nonzero(self.labels == 0)[0]
        # upsampling positive samples
        num_positive_sample = len(positive_indexes)
        num_negative_sample = len(negative_indexes)
        self.num_positive_samples += num_positive_sample
        self.num_negative_samples += num_negative_sample
        if num_positive_sample <= 0:
            return [], True
        upsample_rate = int(num_negative_sample / num_positive_sample)

        print(f"children shape: {len(self.children)}")
        print(f"total samples: {num_positive_sample+num_negative_sample}")
        print(f"num_positive_samples: {num_positive_sample}")
        print(f"num_negative_samples: {num_negative_sample}")
        samples = []
        samples_node_feature = []
        for i in range(num_positive_sample+num_negative_sample):
            label = self.labels[i:i+1, :]
            children_features = self.get_children(i)
            edge_indexes = self.get_edge_indexes(children_features.shape[0])
            # added by yqbai 20240607
            if self.feature_selection == 'no_lut':
                pivot_node_features = self.features[i:i+1, :5]
                samples_node_feature.append((torch.FloatTensor(self.features[i:i+1, :5]), torch.LongTensor(self.labels[i:i+1, :])))
            elif self.feature_selection == 'lut':
                pivot_node_features = self.features[i:i+1, 5:]
                samples_node_feature.append((torch.FloatTensor(self.features[i:i+1, 5:]), torch.LongTensor(self.labels[i:i+1, :])))
            elif self.feature_selection == 'all':
                pivot_node_features = self.features[i:i+1, :]
                samples_node_feature.append((torch.FloatTensor(self.features[i:i+1, :]), torch.LongTensor(self.labels[i:i+1, :])))
            # added by yqbai 20240607
            sample = (pivot_node_features, edge_indexes,
                      children_features, label)
            samples.append(sample)
            if self.sample_bool:
                if i in positive_indexes:
                    # repeat positive samples
                    for _ in range(upsample_rate):
                        samples.append(sample)
            if self.debug_log:
                print(
                    f"debug log sample {i}, pivot_node_features: {pivot_node_features}")
                print(f"debug log sample {i}, label: {label}")
                print(
                    f"debug log sample {i}, children_features: {children_features}")
                print(f"debug log sample {i}, edge_indexes: {edge_indexes}")

        self.dataset['samples'] = samples
        print(f"debug log samples: {len(samples)}")
        return samples_node_feature, samples, False

def get_all_dataset(test_args={}): # 加载训练和验证数据集，返回对应的 DataLoader
    test_data_loader = []
    test_data_loader = GraphDataLoader(**test_args["kwargs"])
    return test_data_loader


def get_batch_score_precision(model, batch):
    # 模型预测
    data = batch.x_constraint, batch.x_variable, batch.x_cv_edge_index, batch.x_edge_attr, batch.y_cand_mask
    start = time.time()
    pred_y = model(*data)
    end1 = time.time()
    y_cand_label = batch.y_cand_label 
    # 统计 y_cand_label 中 1 的个数 k
    k = int(torch.sum(y_cand_label).item())  # 将 k 转换为整数

    m = int(len(y_cand_label)/2)


    # 对 pred_y 的每一行进行降序排序
    _, sorted_indices = torch.sort(pred_y, descending=True)  

    # 初始化 recall 分数
    recall_scores = 0.0

    # 遍历每一行，计算 recall
    # 获取当前行的排序索引
    top_k_indices = sorted_indices[:m]  # 前 k 个预测为正例的索引

    # 计算 TP 和 FN
    TP = torch.sum(y_cand_label[top_k_indices] == 1).item()  # 真正例
    # FN = torch.sum(y_cand_label == 1).item() - TP  # 假负例
    FN = k - TP  # 假负例

    # 计算 recall，避免除零错误
    recall_scores = TP / (TP + FN + 1e-3)

    print(f"耗时: {end1 - start:.4f}秒")
    return recall_scores  

def normalize_minmax(returns): # 使用最小值和最大值进行归一化
    for i in range(returns.shape[0]):
        min_return = returns[i].min()
        delta = returns[i].max() - min_return
        if delta < consts.SAFE_EPSILON:
            delta = 1.
        returns[i] = (returns[i] - min_return) / delta
    return returns

def get_score_recall(pred_y, label, percent=0.5):
    """
    Compute recall@top_k where top_k = percent * total samples

    Args:
        pred_y (Tensor): Predicted scores, shape (N,) or (N, 1)
        label (Tensor): Ground truth labels, shape (N,) or (N, 1), assumed binary (0 or 1)
        percent (float): Percentage of top-k samples to consider, default 0.5 (i.e., top 50%)

    Returns:
        recall (Tensor): Recall value (scalar tensor)
    """
    pred_y = pred_y.view(-1)
    label = label.view(-1)

    # Total number of samples
    total_num = pred_y.size(0)
    top_k = int(total_num * percent)

    # Sort predictions in descending order and get top-k indices
    _, sorted_indices = torch.sort(pred_y, descending=True)
    top_k_indices = sorted_indices[:top_k]
    print(f'The first 10 indexes of top_k_indices are {top_k_indices[:10]}')
    # Get indices of positive samples
    positive_indices = (label > 0).nonzero(as_tuple=True)[0]
    num_positives = positive_indices.numel()

    # Compute number of positives in top-k
    if num_positives == 0:
        return torch.tensor(0.0, dtype=torch.float32)  # Avoid division by zero

    # Use set intersection via tensor comparison
    is_positive_in_top_k = torch.isin(top_k_indices, positive_indices)
    true_positives_in_top_k = is_positive_in_top_k.sum()

    # Recall = TP / Total Positive
    recall = true_positives_in_top_k.float() / num_positives
    return recall

def get_score_focal_loss(pred_y, label):
    epsilon = 1e-3
    focal_gamma = 2

    num_positive_samples = torch.sum(label).item()
    num_negative_samples = label.numel() - num_positive_samples
    total_samples = num_positive_samples + num_negative_samples
    alpha_positive = (num_negative_samples + epsilon) / total_samples
    alpha_negative = (num_positive_samples + epsilon) / total_samples

    weight_pos = torch.pow(1. - pred_y + epsilon, focal_gamma)
    focal_pos = -alpha_positive * weight_pos * torch.log(pred_y + epsilon)

    weight_neg = torch.pow(pred_y, focal_gamma)
    focal_neg = -alpha_negative * weight_neg * torch.log(1. - pred_y + epsilon)

    loss = label * focal_pos + (1 - label) * focal_neg
    loss = loss.mean(dim=1)  # 如果 pred_y 是二维 (batch, num_classes)
    return 1 - loss

@torch.no_grad()
def get_score(model, data, score_func_name="recall"):
    score_func = globals()[f"get_score_{score_func_name}"] # 获取 get_score_recall 函数
    pred_y_list = []
    label_list = []
    inference_time = 0.0
    for batch in data.graph_data_loader:
        print(batch) # batch:{'x_constraint':(113837, 69);'x_variable':(10240, 69);'x_cv_edge_index':(2, 113837);'x_edge_attr':(113837, );'y_cand_mask':(n, );'y_cand_score':(n, );'y_cand_label':(n, )}
        batch = batch.to(consts.DEVICE)
        start_time = time.time()
        pred_y = model(batch, train_mode=False) # (512, 10240) (batch_size, data_batch_size)
        end_time = time.time()
        inference_time = inference_time + end_time - start_time
        label = batch.y_cand_label # (10240, )
        pred_y_list.append(pred_y)
        label_list.append(label)
    pred_y = torch.cat(pred_y_list, dim=1)
    pred_y = normalize_minmax(pred_y) # (512, 10240)
    label = torch.cat(label_list)
    results = score_func(pred_y, label)
    return pred_y, results, inference_time

@torch.no_grad()
def get_precision_iteratively(model, data, partial_sample=None, score_func_name="precision"):
    score_func = globals()[f"get_batch_score_{score_func_name}"] # 获取 get_batch_score_precision 函数
    scores_sum, data_sum, s_sum = 0, 0, 0
    if partial_sample is None: # 如果 partial_sample 未提供，则默认设置为其等于数据集的长度
        partial_sample = len(data.graph_data_loader) * 70000
        # print('partial_sample:', partial_sample)
    # 设置足够大的test batch，使其只需取一个batch，在开源数据集上可以，但是工业电路可能内存不够
    for batch in data.graph_data_loader:
        print(batch)
        batch = batch.to(consts.DEVICE)
        start_time = time.time()
        score_batch = score_func(model, batch)
        end_time = time.time()
        # print(f"Inference Time: {end1 - start:.4f}秒")
        scores_sum += score_batch
        data_sum += len(batch)
        s_sum += 1
        if data_sum >= partial_sample:
            break
    result = scores_sum / s_sum
    inference_time = end_time - start_time
    return result, inference_time


def get_expression(dataset_name): # 从指定文件中读取表达式
    with open(osp.join("./expressions", dataset_name), "r") as txt:
        expression = next(txt) # 从打开的文件对象 txt 中读取第一行内容，并赋值给变量 expression
    return expression

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

@hydra.main(config_path='settings', config_name='train_gs4co', version_base=None) # 装饰器，用于指定配置文件的路径和名称。config_path指定配置文件的目录，config_name指定配置文件的名称（不包含扩展名）
def main(conf):
    inference_time = 0.0  # 推理时间
    data_loading_time = 0.0  # 数据加载时间
    total_time = 0.0  # 总时间 
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     'problem',
    #     help='MILP instance type to process.',
    #     choices=['Conmax', 'Desperf', 'Ethernet', 'Hyp', 'Multiplier', 'Square'],
    # )
    # args = parser.parse_args()
    # dataset_name = args.problem
    new_conf = OmegaConf.to_container(conf, resolve=True)
    # new_conf['test_data_loader']['kwargs']['npy_data_path'] = test_file
    data_loading_time_begin = time.time()
    # test_data = GraphDataLoader(new_conf['test_data_loader'])
    test_data_list = []
    npy_dir_path = new_conf['test_data_loader']['kwargs']['npy_data_path']
    for npy_file in os.listdir(npy_dir_path):
        npy_data_path = os.path.join(npy_dir_path, npy_file)
        print(f"cur npy: {npy_file}")
        new_conf["test_data_loader"]["kwargs"]["npy_data_path"] = npy_data_path
        test_data_list.append(GraphDataLoader(**new_conf["test_data_loader"]["kwargs"]))
    data_loading_time_end = time.time()
    data_loading_time = data_loading_time_end - data_loading_time_begin
    print(f"data_loading_time: {data_loading_time:.2f}s")
    dataset_name = new_conf["instance_kwargs"]["instance_type"]
    # 替换npy_data_path中的占位符
    # new_conf["train_data_loader"]['kwargs']['npy_data_path'] = \
    #     new_conf["train_data_loader"]['kwargs']['npy_data_path'].replace('{{instance_type}}', dataset_name)
    # new_conf["test_data_loader"]['kwargs']['npy_data_path'] = \
    #     new_conf["test_data_loader"]['kwargs']['npy_data_path'].replace('{{instance_type}}', dataset_name)
    # start_data_loading_time = time.time()
    # dataloader = get_all_dataset(new_conf["test_data_loader"])
    # test_data = GraphDataLoader(new_conf["test_data_loader"])
    # end_data_loading_time = time.time()
    # model = Expression(get_expression(dataset_name))

    state_obj = torch.load(new_conf['test_kwargs']['model_path'])
    agent = state_obj['model_state']
    expr = state_obj["best_expr"]
    print(f"expression before fix:{expr.expression}")
    expr.expression = fix_expression(expr.expression)
    print(f"expression after fix:{expr.expression}")
    ensemble_expr = dso_utils_graph.EnsemBleExpression([expr])

    for test_data in test_data_list:
        # recall, inference_time = get_precision_iteratively(ensemble_expr, test_data)
        _, recall, inference_time = get_score(ensemble_expr, test_data, score_func_name="recall")
        file_name = re.search(r"save_data_total_(.*?)\.", test_data.npy_data_path).group(1)
        print(f"test_file_name: {file_name}")
        print(f"top-k recall: {recall:.4f}")
        print(f"Inference time: {inference_time:.2f}s")

    # precision, inference_time = get_precision_iteratively(model, dataloader)
    # print(f"the top 50% accuracy of {dataset_name} is: {precision:.4f}")
    # data_loading_time = end_data_loading_time - start_data_loading_time
    # total_time = inference_time + data_loading_time
    # print(f"Inference Time: {inference_time:.2f} seconds")
    # print(f"Data Loading Time: {data_loading_time:.2f}s")
    # print(f"Total Time (Inference + Data Loading): {total_time:.2f}s")

# @hydra.test(config_path='settings', config_name='train_gs4co', version_base=None)
# def test(conf, model_path, test_file):
#     inference_time = 0.0
#     data_loading_time = 0.0
#     total_time = 0.0

#     new_conf = OmegaConf.to_container(conf, resolve=True)
#     new_conf['test_data_loader']['kwargs']['npy_data_path'] = test_file
#     data_loading_time_begin = time.time()
#     test_data = GraphDataLoader(new_conf['test_data_loader'])
#     data_loading_time_end = time.time()
#     data_loading_time = data_loading_time_end - data_loading_time_begin

#     state_obj = torch.load(model_path)
#     agent = state_obj['model_state']
#     expr = state_obj["best_expr"]
#     ensemble_expr = dso_utils_graph.EnsemBleExpression(expr)

#     recall, inference_time = get_precision_iteratively(expr, test_data)
#     print(f"top-k recall: {recall}")
#     print(f"inference time: {inference_time}")


if __name__ == "__main__":
    # sys.argv.extend([
    # "model_path=/home/yqbai/ldpan/GS4LS/gs4-ls-updated/results/results/train_graph/Conmax/default/0827104457_cb1a02e/0_0/state_dict/wb_conmax/BON_16/train_iter_2000.pkl",
    # "test_file=/home/yqbai/ldpan/GS4LS/gs4-ls-updated/dataset/in-domain-test/Conmax/test"
    # ])
    main()