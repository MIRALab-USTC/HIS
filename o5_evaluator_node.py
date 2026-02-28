import pandas as pd
import numpy as np
from numpy import *
import torch
import time
import json
import joblib
from torch_geometric import loader
np.seterr(divide='ignore', invalid='ignore')
# from utils.function_set import _AND, _OR, _NOT, _Recall_loss, _Recall_lut_loss, GNN_Recall_loss
import os
current_directory = os.getcwd()
os.chdir(current_directory)
FEATURE_CSV = os.path.join(current_directory, './features.csv')
TRUTHTABLE_CSV = os.path.join(current_directory, './truth_table.csv')
LABLES_CSV = os.path.join(current_directory, './labels.csv')
with open('./configs.json', 'r') as f:
    json_kwargs = json.load(f)
MODEL = json_kwargs['MODEL']
SEL_PERCENT = json_kwargs['SEL_PERCENT']
RANDOM = json_kwargs['RANDOM']
NORMALIZE = json_kwargs['NORMALIZE']
METHOD = json_kwargs['METHOD']
FEATURE_SELECTION = json_kwargs['feature_selection']
# for test
# SRMODEL_NO_LUT = '(x_2+(((x_0-cos(x_1))-exp(x_2))/cos((x_1-x_4))))'
# SRMODEL_LUT = '((1-(x_56*(x_59*x_5)))*(1-(x_23*(1-x_22))))'
# SEL_PERCENT = 0.5
# METHOD = 'MCTS'
# for test
def _normalize(features):
    max_vals = np.max(features, axis=0)
    min_vals = np.min(features, axis=0)
    normalized_features = (features - min_vals) / \
        (max_vals - min_vals + 1e-3)

    # print(f"debug log normalized features: {normalized_features}")
    return normalized_features

def _process_csv():
    # data_frame_feature = pd.read_csv(self.csv_feature_path)
    # data_frame_label = pd.read_csv(self.csv_label_path)
    # data_frame_truth_table = pd.read_csv(self.csv_truth_table_path)
    # numpy_data_feature = data_frame_feature.to_numpy()
    # numpy_label_data = data_frame_label.to_numpy()
    # numpy_data_truth_table = data_frame_truth_table.to_numpy()
    # # get features current node
    # features = numpy_data_feature[:, 1:6].astype(np.int32)
    # labels = self._get_labels(numpy_data_feature, numpy_label_data)    

    # process collected csv
    data_frame = pd.read_csv(FEATURE_CSV)
    # data_frame_label = pd.read_csv(LABLES_CSV)
    data_frame_truth_table = pd.read_csv(TRUTHTABLE_CSV)

    numpy_data = data_frame.to_numpy()
    # numpy_label_data = data_frame_label.to_numpy()
    numpy_data_truth_table = data_frame_truth_table.to_numpy()

    # get features current node
    features = numpy_data[:, 1:6].astype(np.float32)
    # labels = _get_labels(numpy_data, numpy_label_data)
     
    # get features truth table
    features_truth_table = numpy_data_truth_table[:, 1:].astype(np.float32)
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

# def _get_labels(numpy_data_feature, numpy_label_data):
#     # labels = np.zeros((numpy_data_feature.shape[0],1), dtype=np.int32)
#     # features_traverse_id = list(numpy_data_feature[:,5].astype(np.int32))
#     # labels_traverse_id = list(numpy_label_data[:,0].astype(np.int32))
#     # collected_labels = numpy_label_data[:,-2:-1].astype(np.int32)
#     # for i in range(len(features_traverse_id)):
#     #     try:
#     #         index = labels_traverse_id.index(features_traverse_id[i])
#     #         labels[i] = collected_labels[index]
#     #     except:
#     #         # print(f"debug log: {i}th id not in collected labels")
#     #         pass

#     # return labels

#     # implemented based on numpy faster
#     # generated labels
#     labels = np.zeros((numpy_data_feature.shape[0], 1), dtype=np.int32)

#     # features
#     features_traverse_id = numpy_data_feature[:, 5:6].astype(np.int32)
#     print(f"features shape: {features_traverse_id.shape}")
#     # collected labels
#     labels_traverse_id = numpy_label_data[:, 0:1].astype(np.int32)
#     collected_labels = numpy_label_data[:, -2:-1].astype(np.int32)
#     print(f"collected_labels shape: {collected_labels.shape}")
#     # get positive traverse id
#     label_nonzero = np.nonzero(collected_labels)
#     print(f"label_nonzero: {label_nonzero}")
#     positive_traverse_id = labels_traverse_id[label_nonzero]
#     print(f"positive_traverse_id: {positive_traverse_id}")
#     print(f"positive_traverse_id: {positive_traverse_id.shape}")
#     postive_indexes = [np.where(features_traverse_id == traverse_id)[
#         0] for traverse_id in positive_traverse_id]
#     postive_indexes = np.vstack(postive_indexes)
#     print(f"postive_indexes: {postive_indexes}")
#     print(f"postive_indexes shape: {postive_indexes.shape}")
#     labels[postive_indexes] = 1
#     labels = labels.astype(np.int32)

#     return labels
def get_MCTS_baseline_indexes(model, features, first_traverse_id):
    y_prediction = score_for_MCTS(model, features, 'all')
    y_prediction_indexes = np.argsort(y_prediction, kind = 'stable')
    y_prediction_indexes = np.array(y_prediction_indexes, dtype=np.float32) + first_traverse_id
    # print('y_prediction_indexes', y_prediction_indexes[1990:])
    total_num = len(y_prediction)
    sel_num = int(total_num * SEL_PERCENT)
    # print('sel_num', sel_num)
    sel_indexes = np.array(
        y_prediction_indexes[total_num-sel_num:], dtype=np.int32)
    # sel_indexes = sel_indexes[::-1]
    sel_indexes.sort()
    # print(sel_indexes)
    # print(sel_indexes.shape[0])
    return sel_indexes

def get_baseline_indexes(y_prediction, first_traverse_id):
    y_prediction_indexes = np.argsort(y_prediction, kind = 'stable')
    y_prediction_indexes = np.array(y_prediction_indexes, dtype=np.float32) + first_traverse_id
    # print('y_prediction_indexes', y_prediction_indexes[1990:])
    total_num = len(y_prediction)
    sel_num = int(total_num * SEL_PERCENT)
    # print('sel_num', sel_num)
    sel_indexes = np.array(
        y_prediction_indexes[total_num-sel_num:], dtype=np.int32)
    # sel_indexes = sel_indexes[::-1]
    sel_indexes.sort()
    # print(sel_indexes)
    # print(sel_indexes.shape[0])
    return sel_indexes

def get_LR_indexes(model, features, first_traverse_id):
    y_prediction = model.predict(features)
    y_prediction_indexes = np.argsort(y_prediction, kind = 'stable')
    y_prediction_indexes = np.array(y_prediction_indexes, dtype=np.float32) + first_traverse_id
    # print('y_prediction_indexes', y_prediction_indexes[1990:])
    total_num = len(y_prediction)
    sel_num = int(total_num * SEL_PERCENT)
    # print('sel_num', sel_num)
    sel_indexes = np.array(
        y_prediction_indexes[total_num-sel_num:], dtype=np.int32)
    # sel_indexes = sel_indexes[::-1]
    sel_indexes.sort()
    # print(sel_indexes)
    # print(sel_indexes.shape[0])
    return sel_indexes

def get_MCTS_our_indexes(est_gp_no_lut, est_gp_lut, features, first_traverse_id):
    def process(x):
        x[x!=0] = 1
        return x
    # print('est_gp_no_lut', est_gp_no_lut._program)
    # print('est_gp_lut', est_gp_lut._program)
    x_test_no_lut = features[:,:5]
    # print(x_test_no_lut)
    # print('x_test_no_lut', x_test_no_lut[0])
    # print('traverse ID', features[:,4])
    x_test_lut = features[:,5:]
    x_test_lut = process(x_test_lut)
    # print(x_test_lut)
    # print('x_test_lut', x_test_lut[0])
    #est_gp_no_lut
    # y_prediction_no_lut = normalization_no_lut(est_gp_no_lut._program.execute(x_test_no_lut))
    y_prediction_no_lut = score_for_MCTS(est_gp_no_lut, x_test_no_lut, 'no_lut')
    # y_prediction_indexes = np.argsort(y_prediction_no_lut, kind = 'stable')
    # print('y_prediction_indexes', y_prediction_indexes[1990:])
    # for i in range(len(y_prediction_no_lut)):
    #     print(y_prediction_no_lut[i])
    # y_prediction_no_lut = np.nan_to_num(y_prediction_no_lut, nan=np.max(y_prediction_no_lut))
    # print(y_prediction_no_lut)
    non_nan_mask = ~np.isnan(y_prediction_no_lut)
    # filter the NAN data
    y_prediction_without_nan = y_prediction_no_lut[non_nan_mask]
    weight = np.median(y_prediction_without_nan)
    # print('the weighted factor is', weight) 
    #est_gp_lut
    # y_prediction_lut = est_gp_lut._program.execute(x_test_lut)
    y_prediction_lut = score_for_MCTS(est_gp_lut, x_test_lut, 'lut')
    #est_gp_no_lut + weight * est_gp_lut
    # print('weight', weight)
    y_prediction = y_prediction_no_lut + weight * y_prediction_lut

    # print('y_prediction', np.sort(y_prediction))
    # print('y_prediction_shape', y_prediction.shape[0])
    y_prediction_indexes = np.argsort(y_prediction, kind = 'stable')
    y_prediction_indexes = np.array(y_prediction_indexes, dtype=np.float32) + first_traverse_id
    # print('y_prediction_indexes', y_prediction_indexes[1990:])
    total_num = len(y_prediction)
    sel_num = int(total_num * SEL_PERCENT)
    # print('sel_num', sel_num)
    sel_indexes = np.array(
        y_prediction_indexes[total_num-sel_num:], dtype=np.int32)
    # sel_indexes = sel_indexes[::-1]
    sel_indexes.sort()
    # print(sel_indexes)
    # print(sel_indexes.shape[0])
    return sel_indexes

def SVD_sort(svd_model, x_test):
    svd_vd = svd_model['svd_vd']
    svd_s = svd_model['svd_s']
    # sub_x_test = x_test[list(sub_prediction_indexes), :]
    ds = np.zeros((x_test.shape[0],1))
    coeff = -1
    for i in range(svd_vd.shape[0]):
        cur_ds = coeff * np.linalg.norm(x_test - svd_vd[i:i+1,:], axis=1).reshape(-1,1) # N*1
        cur_ds = cur_ds * svd_s[i]
        ds += cur_ds
    scores = ds.reshape(-1)#score(负数）越小相似度越低，顺位越靠前
    return scores 

def normalization_no_lut(vector):
    if np.min(vector) < 0:
        vector = vector - np.min(vector)
        vector = vector / vector.max(axis=0)
    else:
        vector = vector / vector.max(axis=0)
    return vector

# def load_SR_model():
#     # GPLearn
#     # est_gp_no_lut = joblib.load(SRMODEL_NO_LUT)
#     # est_gp_lut = joblib.load(SRMODEL_LUT)
#     # MCTS
#     est_gp_no_lut = SRMODEL_NO_LUT
#     est_gp_lut = SRMODEL_LUT
#     return est_gp_no_lut, est_gp_lut

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def score_for_MCTS(eq, x_test, feature_selection):
    ## define independent variables and dependent variable
    # print('eq is', eq)
    num_var = len(x_test[0])
    for i in range(num_var):
        globals()[f'x_{i}'] = x_test[:,i]
    f_pred = eval(eq)
    
    f_pred = f_pred.astype(np.float32)
    if feature_selection != 'lut':
        f_pred = sigmoid(f_pred)
    # print('f_pred is', f_pred)
    return f_pred

def get_random_indexes(num_samples, first_traverse_id):
    random_scores = [np.random.rand() for _ in range(num_samples)]
    ascending_indexes = np.argsort(random_scores)
    ascending_indexes += first_traverse_id.astype(int)
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
    if FEATURE_SELECTION == 'no_lut':
        features = features[:, 1:6]
    time_dict["process_csv"] = time.time()
    # second step: read models and inference
    model = joblib.load(MODEL)
    time_dict["load_model"] = time.time()
    if METHOD == 'CMO':
        if FEATURE_SELECTION == 'no_lut':
            y_prediction = score_for_MCTS(model, features, 'no_lut')
        elif FEATURE_SELECTION == 'all':
            y_prediction = score_for_MCTS(model, features, 'all')
        sel_indexes = get_baseline_indexes(y_prediction, first_traverse_id)
    elif METHOD == 'GPLearn':
        y_prediction = model._program.execute(features)
        y_prediction = sigmoid(y_prediction)
        sel_indexes = get_baseline_indexes(y_prediction, first_traverse_id)
    elif METHOD == 'DSR':
        y_prediction = model.predict(features)
        y_prediction = sigmoid(y_prediction)
        sel_indexes = get_baseline_indexes(y_prediction, first_traverse_id)
    elif METHOD == 'Random':
        y_prediction = model
        sel_indexes = get_baseline_indexes(y_prediction, first_traverse_id)
    elif METHOD == 'SVD':
        y_prediction = SVD_sort(model, features)
        sel_indexes = get_baseline_indexes(y_prediction, first_traverse_id)
    elif METHOD in ['RidgeLR', 'xgboost', 'lightGBM']:
        y_prediction = model.predict(features)
        sel_indexes = get_baseline_indexes(y_prediction, first_traverse_id)
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
