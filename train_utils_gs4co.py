import copy
from gzip import FNAME
import os
import re
from os import path as osp
# import pickle
# import gzip
# from cmo.code.SR.MCTS import score
import torch
from torch import as_tensor
import numpy as np
from time import time as time
import random
from copy import deepcopy

import settings.consts as consts
import utils.logger as logger
import utils.utilities as utilities

import torch
import torch_geometric
from torch_scatter import scatter_mean, scatter_max, scatter_sum
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
from torch_geometric import loader

import dso_utils_graph
from utils.rl_algos import PPOAlgo

from torch.utils.tensorboard import SummaryWriter
# 创建一个SummaryWriter实例，指定日志目录
writer = SummaryWriter('runs/experiment_focal_loss_iwls2epfl_exp/layer')

NUMFeatures = 20

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
        max_samples=None,
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
        if max_samples and len(samples_list) > max_samples:
            import random
            samples_list = random.sample(samples_list, max_samples)
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

class GraphResubDataLoader(GraphDataLoader):
    def __init__(
        self,
        npy_data_path=None,
        save_dir=None,
        processed_npy_path=None,
        train_type='train',
        batch_size=128,
        max_batch_size=10240,
        # debug
        debug_log=False,
        # domain attribute
        label_dict=None,
        # fanin_nodes
        fanin_nodes_type='zero',
        normalize=False
    ):
        self.npy_data_path = npy_data_path
        self.train_type = train_type
        self.save_dir = save_dir
        self.debug_log = debug_log
        self.label_dict = label_dict
        self.fanin_nodes_type = fanin_nodes_type
        self.max_batch_size = max_batch_size
        self.normalize = normalize

        # datasets
        self.end_to_end_stats = None
        self.features = None
        self.labels = None
        self.divisors = None
        self.first_traverse_id = None
        # num samples
        self.num_samples = 0
        self.num_positive_samples = 0
        self.num_negative_samples = 0
        samples_list = []
        if processed_npy_path is None:
            if npy_data_path.endswith('.npy'):
                npy_file_path = npy_data_path
                self.load_data(npy_file_path)
                samples, bool_con = self.process_data()
                if bool_con:
                    print(
                        f"warning! current npy data {npy_file_path} zero positive sample")
                samples_list.extend(samples)
            else:
                for npy_file in os.listdir(npy_data_path):
                    print(f"current npy file: {npy_file}")
                    if npy_file.endswith('.npy'):
                        npy_file_path = os.path.join(npy_data_path, npy_file)
                        self.load_data(npy_file_path)
                        samples, bool_con = self.process_data()
                        print(f"bool continuous: {bool_con}")
                        if bool_con:
                            continue
                        samples_list.extend(samples)
            # save processed samples
            # np.save(
            #     f"../../../npy_data/v2/{save_dir}/graph_samples_train_type_{train_type}_sample_bool_{self.sample_bool}.npy", samples_list)
        else:
            samples_list = np.load(processed_npy_path, allow_pickle=True)
        self.graph_dataset = GraphDataset(samples_list)
        if self.train_type == 'train':
            total_num = len(samples_list)
            if total_num <= self.max_batch_size:
                self.graph_data_loader = loader.DataLoader(
                    self.graph_dataset, self.num_samples, shuffle=False)
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

    def _normalize(self):
        max_vals = np.max(self.features, axis=0)
        min_vals = np.min(self.features, axis=0)
        normalized_features = (self.features - min_vals) / \
            (max_vals - min_vals + 1e-3)
        self.features = normalized_features

    def load_data(self, npy_file_path):
        npy_data = np.load(npy_file_path, allow_pickle=True).item()
        self.end_to_end_stats = npy_data['end_to_end_stats']
        self.features = npy_data['features_list'][0]
        self.first_traverse_id = self.features[0, 0]
        if self.normalize:
            self._normalize()
        if self.label_dict is None:
            self.labels = npy_data['labels_list'][0]
        else:
            label_value = self._get_label(npy_file_path)
            self.labels = np.ones(
                (self.features.shape[0], 1)) * label_value
            self.labels = self.labels.astype(np.int32)
        self.divisors = npy_data['divs_list'][0]
        self.num_samples = self.labels.shape[0]

        if self.debug_log:
            print(f"debug log features: {self.features}")
            print(f"debug log labels: {self.labels}")
            print(f"debug log npy file: {npy_file_path}")

    def process_data(self):
        # get positive and negative samples
        positive_indexes = np.nonzero(self.labels)[0]
        negative_indexes = np.nonzero(self.labels == 0)[0]
        # upsampling positive samples
        num_positive_sample = len(positive_indexes)
        num_negative_sample = len(negative_indexes)
        print(f"num_positive_sample: {num_positive_sample}")
        print(f"total_num_samples: {num_positive_sample+num_negative_sample}")
        self.num_positive_samples += num_positive_sample
        self.num_negative_samples += num_negative_sample
        bool_con = False

        samples = []
        for i in range(self.num_samples):
            if num_positive_sample <= 0:
                print(f"warning! num_positive_sample < 0")
                bool_con = True
                break
            pivot_node_features = self.features[i:i+1, :]
            label = self.labels[i:i+1, :]
            children_features = self.get_children(i)
            edge_indexes = self.get_edge_indexes(children_features.shape[0])
            sample = (pivot_node_features, edge_indexes,
                      children_features, label)
            samples.append(sample)
            if self.debug_log:
                print(
                    f"debug log sample {i}, pivot_node_features: {pivot_node_features}")
                print(f"debug log sample {i}, label: {label}")
                print(
                    f"debug log sample {i}, children_features: {children_features}")
                print(f"debug log sample {i}, edge_indexes: {edge_indexes}")

        return samples, bool_con

    def get_children(self, i):
        children_indexes = self.divisors[i]
        children_features = []
        for index in children_indexes:
            if index >= 0:
                children_feature = self.features[index:index+1, :]
                children_features.append(children_feature)
            else:
                if self.fanin_nodes_type == 'zero':
                    children_feature = np.zeros(
                        (1, NUMFeatures), dtype=np.int32)
                elif self.fanin_nodes_type == 'one':
                    children_feature = np.ones(
                        (1, NUMFeatures), dtype=np.int32)
                elif self.fanin_nodes_type == 'remove':
                    continue
                children_features.append(children_feature)

        if len(children_features) == 0:
            children_features.append(np.ones((1, NUMFeatures), dtype=np.int32))
        return np.vstack(children_features)

def get_all_dataset(instance_type, dataset_type=None, train_num=150000, valid_num=100000, batch_size_train=400, batch_size_valid=400, get_train=True, get_valid=True, train_args={}, test_args={},): # 加载训练和验证数据集，返回对应的 DataLoader
    # file_dir = osp.join(consts.SAMPLE_DIR, instance_type, consts.TRAIN_NAME_DICT[instance_type] if dataset_type is None else dataset_type)
    # if get_train:
    #     train_dataset = GraphDataset(file_dir, train_num)
    #     train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size_train, shuffle=True, follow_batch=["y_cand_mask"], generator=torch.Generator(device=consts.DEVICE))
    # else:
    #     train_loader = None

    # if get_valid:
    #     valid_dataset = GraphDataset(file_dir, valid_num, raw_dir_name="valid")
    #     valid_loader = torch_geometric.loader.DataLoader(valid_dataset, batch_size_valid, shuffle=False, follow_batch=["y_cand_mask"])
    # else:
    #     valid_loader = None
    # return train_loader, valid_loader
    train_data_loader = []
    # npy_data_path = train_args["kwargs"]["npy_data_path"]
    # for npy_file in os.listdir(npy_data_path):
    #     print(npy_file)
    #     cur_npy_file_path = os.path.join(npy_data_path, npy_file)
    #     train_args["kwargs"]["npy_data_path"] = cur_npy_file_path
    #     train_data_loader.append(
    #         GraphDataLoader(
    #             **train_args["kwargs"]).graph_data_loader
    #     )
    train_data_loader = GraphDataLoader(**train_args["kwargs"])
    
    test_data_loader = []
    # npy_data_path = test_args["kwargs"]["npy_data_path"]
    # for npy_file in os.listdir(npy_data_path):
    #     print(npy_file)
    #     cur_npy_file_path = os.path.join(npy_data_path, npy_file)
    #     test_args["kwargs"]["npy_data_path"] = cur_npy_file_path
    #     test_data_loader.append(
    #         GraphDataLoader(
    #             **test_args["kwargs"]).graph_data_loader
    #     )
    test_data_loader = GraphDataLoader(**test_args["kwargs"])
    return train_data_loader, test_data_loader

def focal_loss(self, predict_y, target_y):
        if self.single_domain_ensemble:
            target_y = target_y.repeat(1, predict_y.shape[1])
        epilson = 1e-3
        alpha_positive = (self.data_loader.num_negative_samples+1e-3) / (
            self.data_loader.num_positive_samples+self.data_loader.num_negative_samples)
        alpha_negative = (self.data_loader.num_positive_samples+1e-3) / (
            self.data_loader.num_positive_samples+self.data_loader.num_negative_samples)
        # positive
        weight_positive = torch.pow(1.-predict_y+epilson, self.focal_gamma)
        focal_positive = -alpha_positive * \
            weight_positive * torch.log(predict_y+epilson)
        loss_positive = target_y * focal_positive
        if self.debug_log:
            print(
                f"debug log focal loss weight positive: {weight_positive.shape}")
            print(
                f"debug log focal loss focal_positive: {focal_positive.shape}")
            print(f"debug log focal loss loss_positive: {loss_positive.shape}")
        # negative
        weight_negative = torch.pow(predict_y, self.focal_gamma)
        focal_negative = -alpha_negative * \
            weight_negative * torch.log(1.-predict_y+epilson)
        loss_negative = (1.-target_y) * focal_negative

        loss = torch.mean(loss_positive+loss_negative)
        print(f"debug log focal loss: {loss.shape}")
        return loss


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
    _, sorted_indices = torch.sort(pred_y, descending=True) # sort is from small to large
    top_k_indices = sorted_indices[:top_k]

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

# def get_score_focal_loss(pred_y, label):
#     epsilon=1e-3
#     num_positive_samples = torch.sum(label).item()
#     num_negative_samples = len(label) - num_positive_samples
#     total_samples = num_positive_samples + num_negative_samples
#     alpha_positive = (num_negative_samples + epsilon) / total_samples
#     alpha_negative = (num_positive_samples + epsilon) / total_samples

#     # 初始化 recall 分数
#     loss = torch.zeros(pred_y.size(0))  

#     focal_gamma=2
#     # 遍历每一行，计算 recall
#     for i in range(pred_y.size(0)):
#         weight_positive = torch.pow(1. - pred_y[i] + epsilon, focal_gamma)
#         focal_positive = -alpha_positive * weight_positive * torch.log(pred_y[i] + epsilon)
#         loss_positive = label * focal_positive
        
#         # negative部分
#         weight_negative = torch.pow(pred_y[i], focal_gamma)
#         focal_negative = -alpha_negative * weight_negative * torch.log(1. - pred_y[i] + epsilon)
#         loss_negative = (1. - label) * focal_negative

#         # 合并positive和negative的损失，并取平均值
#         loss[i] = - torch.mean(loss_positive + loss_negative) + 1

#     return loss


def get_score_mse(pred_y, label):
    # 初始化 recall 分数
    loss = torch.zeros(pred_y.size(0))  

    # 遍历每一行，计算 recall
    for i in range(pred_y.size(0)):
        loss[i] = - torch.mean((pred_y[i] - label) ** 2) + 1

    return loss


@torch.no_grad()
def get_score(model, data, score_func_name="recall"):
    score_func = globals()[f"get_score_{score_func_name}"] # 获取 get_score_recall 函数
    pred_y_list = []
    label_list = []
    for batch in data.graph_data_loader:
        print(batch) # batch:{'x_constraint':(113837, 69);'x_variable':(10240, 69);'x_cv_edge_index':(2, 113837);'x_edge_attr':(113837, );'y_cand_mask':(n, );'y_cand_score':(n, );'y_cand_label':(n, )}
        batch = batch.to(consts.DEVICE)
        pred_y = model(batch, train_mode=False) # (512, 10240) (batch_size, data_batch_size)
        pred_y = normalize_minmax(pred_y) # (512, 10240)
        label = batch.y_cand_label # (10240, )
        pred_y_list.append(pred_y)
        label_list.append(label)
    pred_y = torch.cat(pred_y_list, dim=1)
    label = torch.cat(label_list)
    results = score_func(pred_y, label)
    return pred_y, results

class TrainDSOAgent(object):
    def __init__(self, 

                seed=0,
                use_layer_learning = False,
                use_multi_layer = False,
                num_messages=None,
                batch_size=1024, # number of generated expressions 
                data_batch_size=2000, # number of data to evaluate fitness
                eval_expression_num=48, # number of active expressions
                score_func_name='recall',

                record_expression_num=16, # top k expressions from fitness evaluation to evaluate on valid dataset
                record_expression_freq=10, # evaluation frequency

                early_stop=1000,

                total_iter=None,
                
                # operator type (mfs2, resub)
                operator_type=None,
                
                # env args
                instance_kwargs={},

                # expression
                expression_kwargs={},

                # agent
                dso_agent_kwargs={},

                # rl_algo
                rl_algo_kwargs={},

                # train_data_loader
                train_data_loader={},

                # test_data_loader
                test_data_loader={},

                ):
        self.batch_size, self.data_batch_size, self.eval_expression_num, self.seed = batch_size, data_batch_size, eval_expression_num, seed
        # if random_seed:
        #     self.seed = random.randint(0, 2**32-1)
        self.score_func_name = score_func_name
        self.early_stop, self.current_early_stop = early_stop, 0

        self.record_expression_num, self.record_expression_freq = record_expression_num, record_expression_freq
        self.instance_type = instance_kwargs["instance_type"]
        self.best_of_N = instance_kwargs["Best_of_N"] # select the best N expressions for test
        self.total_iter = consts.ITER_DICT[self.instance_type] if total_iter is None else total_iter
        # self.train_data, self.valid_data = get_all_dataset(**instance_kwargs, train_args=train_data_loader, test_args=test_data_loader)
        self.operator_type = operator_type
        if self.operator_type == "mfs2":
            self.train_data = GraphDataLoader(**train_data_loader["kwargs"])
        elif self.operator_type == "resub":
            self.train_data = GraphResubDataLoader(**train_data_loader["kwargs"])
        # test_data
        self.test_data = []
        npy_dir_path = test_data_loader["kwargs"]["npy_data_path"]
        for npy_file in os.listdir(npy_dir_path):
            npy_file_path = os.path.join(npy_dir_path, npy_file)
            print(f"cur npy: {npy_file}")
            test_data_loader["kwargs"]["npy_data_path"] = npy_file_path
            if self.operator_type == "mfs2":
                self.test_data.append(GraphDataLoader(**test_data_loader["kwargs"]))
            elif self.operator_type == "resub":
                self.test_data.append(GraphResubDataLoader(**test_data_loader["kwargs"]))
        # expression
        self.operators = dso_utils_graph.Operators(**expression_kwargs, use_layer_learning=use_layer_learning, use_multi_layer=use_multi_layer, num_messages=num_messages)
        # if expression_kwargs["math_list"] == "all":
        #     self.operators_simple = dso_utils_graph.Operators(use_layer_learning=use_layer_learning, use_multi_layer=use_multi_layer, num_messages=num_messages)
        #     self.agent_simple = dso_utils_graph.TransformerDSOAgent_UseLayerLearning(self.operators_simple, **dso_agent_kwargs["transformer_kwargs"])

        # dso agent
        self.state_dict_dir, = logger.create_and_get_subdirs("state_dict")

        # use layer learning
        if not use_layer_learning and (not use_multi_layer):
            self.agent = dso_utils_graph.TransformerDSOAgent(self.operators, **dso_agent_kwargs["transformer_kwargs"])
        elif use_layer_learning:
            self.agent = dso_utils_graph.TransformerDSOAgent_UseLayerLearning(self.operators, **dso_agent_kwargs["transformer_kwargs"])
        else: # use_multi_layer
            self.agent = dso_utils_graph.TransformerDSOAgent_Multilayer(self.operators, **dso_agent_kwargs["transformer_kwargs"])
        # rl algo
        self.rl_algo = PPOAlgo(agent=self.agent, **rl_algo_kwargs["kwargs"])
        self.is_tensorboard = rl_algo_kwargs["kwargs"]["is_tensorboard"]

        # algo process variables
        self.train_iter = 0
        self.best_performance = - float("inf")
        self.best_writter = open(osp.join(logger.get_dir(), "best.txt"), "w")

        # tensorboard step
        self.env_step = 1

    def message_process(self, indices, node_message_sequences, node_message_lengths, node_message_log_probs, node_message_info_lists, constraint_message_sequences, constraint_message_lengths, constraint_message_log_probs, constraint_message_info_lists):
        node_sequences = [seq[indices] for seq in node_message_sequences]
        node_lengths = [lengths[indices] for lengths in node_message_lengths]
        node_log_probs = [log_prob[indices] for log_prob in node_message_log_probs]

        constraint_sequences = [seq[indices] for seq in constraint_message_sequences]
        constraint_lengths = [lengths[indices] for lengths in constraint_message_lengths]
        constraint_log_probs = [log_prob[indices] for log_prob in constraint_message_log_probs]

        for info_list in (node_message_info_lists, constraint_message_info_lists):
            scatter_degree_list, all_counters_lists, scatter_parent_lists, parent_child_pairs_lists, parent_child_length_lists, sibling_pairs_lists, sibling_length_lists = info_list
            for i in range(self.operators.T1):
                scatter_degree_list[i] = scatter_degree_list[i][indices]
                all_counters_lists[i] = [counters[indices] for counters in all_counters_lists[i]]
                scatter_parent_lists[i] = scatter_parent_lists[i][indices]

                # 处理父子关系和兄弟关系的数据，确保索引正确映射
                parent_useful_index = torch.any(parent_child_pairs_lists[i][:,0][:, None] == indices[None,:], dim=1)
                parent_child_pairs_lists[i] = parent_child_pairs_lists[i][parent_useful_index]
                parent_useful_cumsum = torch.cumsum(parent_useful_index.long(),dim=0)
                parent_child_length_lists[i][1:] = parent_useful_cumsum[parent_child_length_lists[i][1:]-1]
                parent_new_index0 = torch.full((self.batch_size,), fill_value=-1,dtype=torch.long)
                parent_new_index0[indices] = torch.arange(len(indices))
                parent_child_pairs_lists[i][:, 0] = parent_new_index0[parent_child_pairs_lists[i][:, 0]]

                sibling_useful_index = torch.any(sibling_pairs_lists[i][:,0][:, None] == indices[None,:], dim=1)
                sibling_pairs_lists[i] = sibling_pairs_lists[i][sibling_useful_index]
                sibling_useful_cumsum = torch.cumsum(sibling_useful_index.long(), dim=0)
                where_start_positive = torch.where(sibling_length_lists[i] > 0)[0][0]
                sibling_length_lists[i][where_start_positive:] = sibling_useful_cumsum[sibling_length_lists[i][where_start_positive:]-1]
                sibling_new_index0 = torch.full((self.batch_size,), fill_value=-1,dtype=torch.long)
                sibling_new_index0[indices] = torch.arange(len(indices))
                sibling_pairs_lists[i][:, 0] = sibling_new_index0[sibling_pairs_lists[i][:, 0]]

        return node_sequences, node_lengths, node_log_probs, node_message_info_lists, constraint_sequences, constraint_lengths, constraint_log_probs, constraint_message_info_lists

    def message_process_multi_layer(self, indices, message_sequences, message_lengths, message_log_probs, message_info_lists):
        for l in range(len(self.operators.T)):
            for i in range(self.operators.T[l]):
                message_sequences[l][i] = message_sequences[l][i][indices]
                message_lengths[l][i] = message_lengths[l][i][indices]
                message_log_probs[l][i] = message_log_probs[l][i][indices]

            scatter_degree_list, all_counters_lists, scatter_parent_lists, parent_child_pairs_lists, parent_child_length_lists, sibling_pairs_lists, sibling_length_lists = message_info_lists[l]
            for i in range(self.operators.T[l]):
                scatter_degree_list[i] = scatter_degree_list[i][indices]
                all_counters_lists[i] = [counters[indices] for counters in all_counters_lists[i]]
                scatter_parent_lists[i] = scatter_parent_lists[i][indices]

                # 处理父子关系和兄弟关系的数据，确保索引正确映射
                parent_useful_index = torch.any(parent_child_pairs_lists[i][:,0][:, None] == indices[None,:], dim=1)
                parent_child_pairs_lists[i] = parent_child_pairs_lists[i][parent_useful_index]
                parent_useful_cumsum = torch.cumsum(parent_useful_index.long(),dim=0)
                parent_child_length_lists[i][1:] = parent_useful_cumsum[parent_child_length_lists[i][1:]-1]
                parent_new_index0 = torch.full((self.batch_size,), fill_value=-1,dtype=torch.long)
                parent_new_index0[indices] = torch.arange(len(indices))
                parent_child_pairs_lists[i][:, 0] = parent_new_index0[parent_child_pairs_lists[i][:, 0]]

                sibling_useful_index = torch.any(sibling_pairs_lists[i][:,0][:, None] == indices[None,:], dim=1)
                sibling_pairs_lists[i] = sibling_pairs_lists[i][sibling_useful_index]
                sibling_useful_cumsum = torch.cumsum(sibling_useful_index.long(), dim=0)
                where_start_positive = torch.where(sibling_length_lists[i] > 0)[0][0]
                sibling_length_lists[i][where_start_positive:] = sibling_useful_cumsum[sibling_length_lists[i][where_start_positive:]-1]
                sibling_new_index0 = torch.full((self.batch_size,), fill_value=-1,dtype=torch.long)
                sibling_new_index0[indices] = torch.arange(len(indices))
                sibling_pairs_lists[i][:, 0] = sibling_new_index0[sibling_pairs_lists[i][:, 0]]
        return message_sequences, message_lengths, message_log_probs, message_info_lists
    
    def pad_to_match(self, a, b, fill=-1, dim=1):
        """
        把 a 和 b 在指定 dim 上补到相同长度，缺的地方用 fill 值填充。
        返回补完后的 (a, b)。
        """
        sz_a, sz_b = a.size(dim), b.size(dim)
        if sz_a == sz_b:
            return a, b
        if sz_a < sz_b:          # a 更短
            pad = [0] * (a.ndim * 2)
            pad[-dim * 2 - 1] = sz_b - sz_a   # dim 的右侧补
            a = torch.nn.functional.pad(a, pad, value=fill)
        else:                    # b 更短
            pad = [0] * (b.ndim * 2)
            pad[-dim * 2 - 1] = sz_a - sz_b
            b = torch.nn.functional.pad(b, pad, value=fill)
        return a, b

    def process(self, use_layer_learning=False, use_multi_layer=False):
        start_time = time()
        for self.train_iter in range(self.total_iter+1):
            if self.current_early_stop > self.early_stop:
                break
            iter_start_time = time()
            
            # train
            # use layer learning
            if not use_layer_learning and not use_multi_layer:
                sequences, all_lengths, log_probs, (scatter_degree, all_counters_list, scatter_parent_where_seq, 
                                                    parent_child_pairs, parent_child_length, sibling_pairs, sibling_length) = self.agent.sample_sequence_eval(self.batch_size)
            elif use_layer_learning:
                sequences, all_lengths, log_probs, (scatter_degree, all_counters_list, scatter_parent_where_seq,
                                                    parent_child_pairs, parent_child_length, sibling_pairs, sibling_length),\
                                                        (node_message_sequences, node_message_lengths, node_message_log_probs, node_message_info_lists,
                                                        constraint_message_sequences, constraint_message_lengths, constraint_message_log_probs, constraint_message_info_lists) = self.agent.sample_sequence_eval(self.batch_size)
                node_expr_seq, node_expr_length, node_expr_scatter_degree = [t.clone() for t in node_message_sequences], [t.clone() for t in node_message_lengths], [t.clone() for t in node_message_info_lists[0]]
                constraint_expr_seq, constraint_expr_length, constraint_expr_scatter_degree = [t.clone() for t in constraint_message_sequences], [t.clone() for t in constraint_message_lengths], [t.clone() for t in constraint_message_info_lists[0]]
            else: # use_multi_layer
                sequences, all_lengths, log_probs, (scatter_degree, all_counters_list, scatter_parent_where_seq,
                                                    parent_child_pairs, parent_child_length, sibling_pairs, sibling_length),\
                                                    (message_sequences, message_lengths, message_log_probs, message_info_lists) = self.agent.sample_sequence_eval(self.batch_size)
                message_expr_seq = copy.deepcopy(message_sequences)   # 正确深拷贝
                message_expr_length = copy.deepcopy(message_lengths)
                message_expr_info_lists = copy.deepcopy(message_info_lists)

            expression_list = [dso_utils_graph.Expression(sequence[1:length+1], scatter_degree_now[:length], self.operators) for sequence, length, scatter_degree_now in zip(sequences, all_lengths, scatter_degree)] # 将一个表达式（可能是某种计算图或数学公式）表示为一棵树
            # print(f"expression[0]:{expression_list[0].get_nlp()}")
            # print(f"expression[-1]:{expression_list[-1].get_nlp()}")

            expression_generation_time = time() - iter_start_time

            eval_expression_start_time = time()

            # 表达式评估
            ensemble_expressions = dso_utils_graph.EnsemBleExpression(expression_list) # 实现模型集成            
            pred_y, scores = get_score(ensemble_expressions, self.train_data, score_func_name=self.score_func_name)
            if len(self.operators.math_operators) > 7:
                n = self.batch_size//2
                ensemble_expressions2 = dso_utils_graph.EnsemBleExpression(expression_list[-n:])
                pred_y2, scores2 = get_score(ensemble_expressions2, self.train_data, score_func_name=self.score_func_name)
                valid_mask = torch.isfinite(scores)
                scores = torch.where(valid_mask, scores, torch.tensor(float('-inf'), dtype=scores.dtype, device=scores.device))
                scores[-n:] = scores2
            # print(scores)
            eval_expression_time = time() - eval_expression_start_time

            # 强化学习训练
            rl_start_time = time()
            # 根据精度选择表现最好的一部分表达式，并更新相关数据以便进行下一轮训练
            returns, indices = torch.topk(scores, self.eval_expression_num, sorted=False)

            sequences, all_lengths, log_probs = sequences[indices], all_lengths[indices], log_probs[indices]
            scatter_degree, all_counters_list, scatter_parent_where_seq = scatter_degree[indices],\
                                                                        [all_counters[indices] for all_counters in all_counters_list],\
                                                                        scatter_parent_where_seq[indices]
  
            # 处理父子关系和兄弟关系的数据，确保索引正确映射
            parent_useful_index = torch.any(parent_child_pairs[:,0][:, None] == indices[None,:], dim=1)
            parent_child_pairs = parent_child_pairs[parent_useful_index]
            parent_useful_cumsum = torch.cumsum(parent_useful_index.long(),dim=0)
            parent_child_length[1:] = parent_useful_cumsum[parent_child_length[1:]-1]
            parent_new_index0 = torch.full((self.batch_size,), fill_value=-1,dtype=torch.long)
            parent_new_index0[indices] = torch.arange(len(indices))
            parent_child_pairs[:, 0] = parent_new_index0[parent_child_pairs[:, 0]]

            sibling_useful_index = torch.any(sibling_pairs[:,0][:, None] == indices[None,:], dim=1)
            sibling_pairs = sibling_pairs[sibling_useful_index]
            sibling_useful_cumsum = torch.cumsum(sibling_useful_index.long(), dim=0)
            where_start_positive = torch.where(sibling_length > 0)[0][0]
            sibling_length[where_start_positive:] = sibling_useful_cumsum[sibling_length[where_start_positive:]-1]
            sibling_new_index0 = torch.full((self.batch_size,), fill_value=-1,dtype=torch.long)
            sibling_new_index0[indices] = torch.arange(len(indices))
            sibling_pairs[:, 0] = sibling_new_index0[sibling_pairs[:, 0]]

            # # use layer learning
            # if use_layer_learning:
            #     node_sequences, node_lengths, node_log_probs, node_info_list,\
            #         constraint_sequences, constraint_lengths, constraint_log_probs, constraint_info_lists = self.message_process(indices,
            #                                                                                                                      node_message_sequences, node_message_lengths, node_message_log_probs, node_message_info_lists,
            #                                                                                                                      constraint_message_sequences, constraint_message_lengths, constraint_message_log_probs, constraint_message_info_lists)

            assert (sibling_pairs[:, 0].min() == parent_child_pairs[:, 0].min() == 0) and (sibling_pairs[:, 0].max() == parent_child_pairs[:, 0].max() ==  len(indices) - 1)
            # print('********')
            index_useful = (torch.arange(sequences.shape[1]-1, dtype=torch.long)[None, :] < all_lengths[:, None]).type(torch.float32)
            # 调用强化学习算法的 train 方法进行训练，并返回结果
            # use layer learning
            if not use_layer_learning and not use_multi_layer:
                results_rl = self.rl_algo.train(sequences, all_lengths, log_probs, index_useful, (scatter_degree, all_counters_list, scatter_parent_where_seq, parent_child_pairs, parent_child_length, sibling_pairs, sibling_length), returns=returns, train_iter=self.train_iter,)
            elif use_layer_learning:
                node_sequences, node_lengths, node_log_probs, node_info_list,\
                    constraint_sequences, constraint_lengths, constraint_log_probs, constraint_info_lists = self.message_process(indices,
                                                                                                                                 node_message_sequences, node_message_lengths, node_message_log_probs, node_message_info_lists,
                                                                                                                                 constraint_message_sequences, constraint_message_lengths, constraint_message_log_probs, constraint_message_info_lists)
                results_rl = self.rl_algo.train(sequences, all_lengths, log_probs, index_useful, (scatter_degree, all_counters_list, scatter_parent_where_seq, parent_child_pairs, parent_child_length, sibling_pairs, sibling_length), returns=returns, train_iter=self.train_iter, use_layer_learning=use_layer_learning,
                                            node_message_sequences=node_sequences, node_message_lengths=node_lengths, node_message_info_lists=node_info_list, constraint_message_sequences=constraint_sequences, constraint_message_lengths=constraint_lengths, constraint_message_info_lists=constraint_info_lists)
            else: # use_multi_layer
                message_sequences, message_lengths, message_log_probs, message_info_lists = self.message_process_multi_layer(indices, message_sequences, message_lengths, message_log_probs, message_info_lists)
                results_rl = self.rl_algo.train(sequences, all_lengths, log_probs, index_useful, (scatter_degree, all_counters_list, scatter_parent_where_seq, parent_child_pairs, parent_child_length, sibling_pairs, sibling_length), returns=returns, train_iter=self.train_iter, use_layer_learning=use_layer_learning,
                                            use_multi_layer=use_multi_layer, message_sequences=message_sequences, message_lengths=message_lengths, message_info_lists=message_info_lists)

            _, where_to_test = torch.topk(scores, 1, sorted=True)
            best_expr_index = where_to_test[0].item()
            best_expr = expression_list[best_expr_index]
            best_pred_y = pred_y[best_expr_index]
            print(f'best_pred_y: {best_pred_y}')
            # print('#########')
            # 时间记录和日志记录
            iter_end_time = time()
            rl_time = iter_end_time - rl_start_time
            iter_time = iter_end_time - iter_start_time

            ## tensorboard record
            total_time = iter_end_time - start_time
            results = {
                       "train/best_loss": returns.max().item(),
                       "train/epsilon_mean_loss": returns.mean(),
                       "train/epsilon_var_loss": returns.std(),
                       "train/all_mean_loss": scores.mean(),
                       "train/all_var_loss": scores.std(),

                       "train/train_iteration": self.train_iter,
                       "train/iter_time": iter_time,
                       "train/iter_time_generation": expression_generation_time,
                       "train/iter_time_evaluation": eval_expression_time,
                       "train/iter_time_rl": rl_time,
                       "train/total_time": total_time,
                       }
            results.update(results_rl)
            results_hist = {"train/best_pred_y": best_pred_y,
                            }
            # ensemble_loss =  {"train/best_loss": returns.max().item(),
            #                 "train/topk_mean_loss": returns.mean(),
            #                 "train/topk_var_loss": returns.std(),
            #                 "train/all_mean_loss": scores.mean(),
            #                 "train/all_var_loss": scores.std(),
            #                 }
            # writer.add_scalars(f'Train Loss', ensemble_loss, global_step=self.train_iter+1)


            ## save expressions and models
            # 如果当前迭代次数符合记录表达式的条件，则对选定的表达式进行进一步验证，并根据验证结果决定是否更新最佳性能指标
            if self.train_iter % self.record_expression_freq == 0:
                # _, where_to_test = torch.topk(scores, self.record_expression_num, sorted=True)
                for N in self.best_of_N:
                    for test_data in self.test_data:
                        file_name = re.search(r"save_data_total_(.*?)\.", test_data.npy_data_path).group(1)
                        _, where_to_test = torch.topk(scores, N, sorted=True)
                        BON_exprs = [expression_list[i] for i in where_to_test]  # 取 top N 的表达式
                        recall_list = []
                        for expr in BON_exprs:
                            ensemble_expr = dso_utils_graph.EnsemBleExpression([expr])
                            _, recall = get_score(ensemble_expr, test_data, score_func_name="recall")
                            recall_list.append(recall)
                        print(f'The best of {N} test recall is', recall_list)
                        best_idx = recall_list.index(max(recall_list))
                        ensemble_best_expr = dso_utils_graph.EnsemBleExpression([BON_exprs[best_idx]])
                        _, scores_test = get_score(ensemble_best_expr, test_data, score_func_name=self.score_func_name)
                        _, recall_test = get_score(ensemble_best_expr, test_data, score_func_name="recall")
                        
                        if use_layer_learning:
                            best_used = where_to_test[best_idx].item()
                            # node_message_used, node_length_used, node_scatter_degree_used = [], [], []
                            # constraint_message_used, constraint_length_used, constraint_scatter_degree_used = [], [], []
                            node_message_used = [node_expr_seq[i][best_used] for i in range(self.operators.T1)]
                            node_length_used = [node_expr_length[i][best_used] for i in range(self.operators.T1)]
                            node_scatter_degree_used = [node_expr_scatter_degree[i][best_used] for i in range(self.operators.T1)]
                            constraint_message_used = [constraint_expr_seq[i][best_used] for i in range(self.operators.T2)]
                            constraint_length_used = [constraint_expr_length[i][best_used] for i in range(self.operators.T2)]
                            constraint_scatter_degree_used = [constraint_expr_scatter_degree[i][best_used] for i in range(self.operators.T2)]

                            node_used_expr = [dso_utils_graph.Expression(node_message_used_now[1:length_now+1], node_scatter_degree_used_now[:length_now], self.operators) for (node_message_used_now, length_now, node_scatter_degree_used_now) in zip(node_message_used, node_length_used, node_scatter_degree_used)]
                            constraint_used_expr = [dso_utils_graph.Expression(constraint_message_used_now[1:length_now+1], constraint_scatter_degree_used_now[:length_now], self.operators) for (constraint_message_used_now, length_now, constraint_scatter_degree_used_now) in zip(constraint_message_used, constraint_length_used, constraint_scatter_degree_used)]

                        if use_multi_layer:
                            best_used = where_to_test[best_idx].item()
                            message_used, length_used, scatter_degree_used = [], [], []
                            message_expr = []
                            for l in range(len(self.operators.T)):
                                message_used = [message_expr_seq[l][i][best_used] for i in range(self.operators.T[l])]
                                length_used = [message_expr_length[l][i][best_used] for i in range(self.operators.T[l])]
                                scatter_degree_used = [message_expr_info_lists[l][0][i][best_used] for i in range(self.operators.T[l])]
                                message_expr.append([dso_utils_graph.Expression(message_used_now[1:length_now+1], scatter_degree_used_now[:length_now], self.operators) for (message_used_now, length_now, scatter_degree_used_now) in zip(message_used, length_used, scatter_degree_used)])

                        best = f"""test_file_name:{file_name} \
                            iteration:{self.train_iter} \
                            best of N:{N} \
                            train_loss:{round(returns.max().item(), 4)} \
                            BON_test_loss:{round(scores_test.item(), 4)} \
                            BON_test_top50_recall:{round(recall_test.item(), 4)} \
                            BON_exp_nlp:{BON_exprs[best_idx].get_nlp()} \
                            BON_exp_expression:{BON_exprs[best_idx].get_expression()}\n"""
                        
                        # use layer learning
                        if use_layer_learning:
                            for i in range(self.operators.T1):
                                best += f"variable_generated_{i+1}_nlp:{node_used_expr[i].get_nlp()} \
                                    variable_generated_{i+1}_expression:{node_used_expr[i].get_expression()}\n"
                            for i in range(self.operators.T2):
                                best += f"constraint_generated_{i+1}_nlp:{constraint_used_expr[i].get_nlp()} \
                                    constraint_generated_{i+1}_expression:{constraint_used_expr[i].get_expression}\n"
                                
                        if use_multi_layer:
                            for l in range(len(self.operators.T)):
                                for i in range(self.operators.T[l]):
                                    best += f"layer{l}_message{i+1}_nlp:{message_expr[l][i].get_nlp()}\
                                        layer{l}_message{i+1}_expression:{message_expr[l][i].get_expression()}\n"
                                    
                        self.best_writter.write(best)
                    
                        # where_to_record = torch.where(loss_valid > self.best_performance)[0]
                        # if len(where_to_record) > 0:
                        #     self.current_early_stop = 0
                        #     pairs = [(expressions_to_valid[i], loss_valid[i].item(), scores_valid[i]) for i in where_to_record]
                        #     pairs.sort(key=lambda x: x[1])
                        #     self.best_performance = pairs[-1][1]
                        #     for (exp, value, recall_value) in pairs:
                        #         best = f"iteration:{self.train_iter}_loss:{round(value, 4)}_recall:{round(recall_value.item(), 4)}\t{exp.get_nlp()}\t{exp.get_expression()}\n"
                        #         self.best_writter.write(best)
                        #     logger.log(best)
                        #     self.best_writter.flush()
                        #     os.fsync(self.best_writter.fileno())
                        # else:
                        #     self.current_early_stop += self.record_expression_freq
                        results.update({
                            f"test/{file_name}_BON_{N}/BON_best_loss": scores_test.item(),
                            f"test/{file_name}_BON_{N}/BON_best_recall": recall_test.item(),
                            # "test/test_best_loss": loss_test.max().item(),
                            # "test/test_all_mean_loss": loss_test.mean(),
                            # "test/test_all_var_loss": loss_test.std(),

                            # "test/test_best_loss_recall": scores_test[torch.argmax(loss_test)].item(),
                            # "test/test_best_recall": scores_test.max().item(),
                            # "test/test_all_mean_recall": scores_test.mean(),
                            # "test/test_all_var_recall": scores_test.std(),
                            "test/test_iteration": self.train_iter,
                        })

                        # state_dict = self.agent.state_dict()
                        state_obj = {
                            'epoch' : self.train_iter,
                            'best_of_N': N,
                            'model_state': self.agent.state_dict(),
                            'best_expr': BON_exprs[best_idx]
                        }
                        state_dict_dir = os.path.join(self.state_dict_dir, f'{file_name}/BON_{N}')
                        os.makedirs(state_dict_dir, exist_ok=True)
                        state_dict_save_path = osp.join(state_dict_dir, f"train_iter_{self.train_iter}.pkl")
                        torch.save(state_obj, state_dict_save_path)
            results.update({"env_step": self.env_step})
            results_hist.update({"env_step": self.env_step})
            logger.logkvs_tb(results)
            logger.dumpkvs_tb()
            logger.log_hist(results_hist)
            self.env_step += 1

        end_time = time()
        total_time_training = end_time - start_time
        logger.logkvs_tb({"Total time of training process":total_time_training})
        logger.dumpkvs_tb()