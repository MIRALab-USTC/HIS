from typing import Any
import torch
import os
from os import path as osp
from collections import defaultdict
import hydra
# donot import numpy here as we have to set np threads later


IMPORTANT_INFO_SUFFIX = "*"*10
WARN_INFO_SUFFIX = "!"*10

# NODE_FEATURES = ['node_feature_fanin_num', 'node_feature_fanout_num', 'node_feature_lev', 'node_feature_levr', 'node_feature_Traverseid', 
# 'node_feature_5', 'node_feature_6', 'node_feature_7', 'node_feature_8', 'node_feature_9', 
# 'node_feature_10', 'node_feature_11', 'node_feature_12', 'node_feature_13', 'node_feature_14', 
# 'node_feature_15', 'node_feature_16', 'node_feature_17', 'node_feature_18', 'node_feature_19', 
# 'node_feature_20', 'node_feature_21', 'node_feature_22', 'node_feature_23', 'node_feature_24', 
# 'node_feature_25', 'node_feature_26', 'node_feature_27', 'node_feature_28', 'node_feature_29', 
# 'node_feature_30', 'node_feature_31', 'node_feature_32', 'node_feature_33', 'node_feature_34', 
# 'node_feature_35', 'node_feature_36', 'node_feature_37', 'node_feature_38', 'node_feature_39', 
# 'node_feature_40', 'node_feature_41', 'node_feature_42', 'node_feature_43', 'node_feature_44', 
# 'node_feature_45', 'node_feature_46', 'node_feature_47', 'node_feature_48', 'node_feature_49', 
# 'node_feature_50', 'node_feature_51', 'node_feature_52', 'node_feature_53', 'node_feature_54', 
# 'node_feature_55', 'node_feature_56', 'node_feature_57', 'node_feature_58', 'node_feature_59', 
# 'node_feature_60', 'node_feature_61', 'node_feature_62', 'node_feature_63', 'node_feature_64', 
# 'node_feature_65', 'node_feature_66', 'node_feature_67', 'node_feature_68']
# CONSTRAINT_FEATURES = ['constraint_feature_fanin_num', 'constraint_feature_fanout_num', 'constraint_feature_lev', 'constraint_feature_levr', 'constraint_feature_Traverseid', 
# 'constraint_feature_5', 'constraint_feature_6', 'constraint_feature_7', 'constraint_feature_8', 'constraint_feature_9', 
# 'constraint_feature_10', 'constraint_feature_11', 'constraint_feature_12', 'constraint_feature_13', 'constraint_feature_14', 
# 'constraint_feature_15', 'constraint_feature_16', 'constraint_feature_17', 'constraint_feature_18', 'constraint_feature_19', 
# 'constraint_feature_20', 'constraint_feature_21', 'constraint_feature_22', 'constraint_feature_23', 'constraint_feature_24', 
# 'constraint_feature_25', 'constraint_feature_26', 'constraint_feature_27', 'constraint_feature_28', 'constraint_feature_29', 
# 'constraint_feature_30', 'constraint_feature_31', 'constraint_feature_32', 'constraint_feature_33', 'constraint_feature_34', 
# 'constraint_feature_35', 'constraint_feature_36', 'constraint_feature_37', 'constraint_feature_38', 'constraint_feature_39', 
# 'constraint_feature_40', 'constraint_feature_41', 'constraint_feature_42', 'constraint_feature_43', 'constraint_feature_44', 
# 'constraint_feature_45', 'constraint_feature_46', 'constraint_feature_47', 'constraint_feature_48', 'constraint_feature_49', 
# 'constraint_feature_50', 'constraint_feature_51', 'constraint_feature_52', 'constraint_feature_53', 'constraint_feature_54', 
# 'constraint_feature_55', 'constraint_feature_56', 'constraint_feature_57', 'constraint_feature_58', 'constraint_feature_59', 
# 'constraint_feature_60', 'constraint_feature_61', 'constraint_feature_62', 'constraint_feature_63', 'constraint_feature_64', 
# 'constraint_feature_65', 'constraint_feature_66', 'constraint_feature_67', 'constraint_feature_68']
# no_lut
NODE_FEATURES = ['node_feature_fanin_num', 'node_feature_fanout_num', 'node_feature_lev', 'node_feature_levr', 'node_feature_Traverseid'] 
CONSTRAINT_FEATURES = ['constraint_feature_fanin_num', 'constraint_feature_fanout_num', 'constraint_feature_lev', 'constraint_feature_levr', 'constraint_feature_Traverseid']
EDGE_FEATURES = ['coef_normalized']
GRAPH_NAMES = CONSTRAINT_FEATURES + EDGE_FEATURES + NODE_FEATURES


# DEVICE = torch.device("cpu")
DEVICE = torch.device("cuda")
torch.set_default_device(DEVICE)

TRAIN_NAME_DICT = defaultdict(lambda : "")
TRAIN_NAME_DICT.update({"setcover": "500r_1000c_0.05d", "indset": "750_4", "facilities": "100_100_5", "cauctions": "100_500"})



WORK_DIR = osp.dirname(osp.dirname(__file__))
DATA_DIR = osp.join(WORK_DIR, "data")
RESULT_BASE_DIR = osp.join(WORK_DIR, "results")
INSTANCE_DIR = osp.join(DATA_DIR, "instances")
SAMPLE_DIR = osp.join(DATA_DIR, "samples")
RESULT_DIR = osp.join(RESULT_BASE_DIR, "results")



ITER_DICT = defaultdict(lambda : 2000)
GLOBAL_INFO_DICT = {}


SAFE_EPSILON = 1e-6
DETAILED_LOG_FREQ = 10
DETAILED_LOG = True

STATUS_DICT = {"optimal": 0, "timelimit":-1, "infeasible":-2, "unbounded":-3, "userinterrupt":-4, "unknown":-5}
STATUS_INDEX_DICT = {v:k for k,v in STATUS_DICT.items()}