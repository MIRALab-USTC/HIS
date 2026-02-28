import numpy as np
import numpy as np
from gplearn.functions import _function_map, _Function
from gplearn.functions import make_function
from gplearn.fitness import make_fitness
def _protected_and(x1, x2):
    return np.multiply(x1, x2)

def _protected_or(x1, x2):
    return np.maximum(x1, x2)

def _protected_not(x1):
    return 1 - x1

def _protected_exponent(x1):
    with np.errstate(over='ignore'): # 忽略数值溢出的错误
        return np.where(x1 < 100, np.exp(x1), 0.0)

def _proteceted_exponent_negative(x1):
    with np.errstate(over='ignore'):
        return np.where(x1 > -100, np.exp(-x1), 0.0)

def _protected_n2(x1):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x1) < 1e6, np.square(x1), 0.0)


def _protected_n3(x1):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x1) < 1e6, np.power(x1, 3), 0.0)


def _protected_division_ignore_overflow(x1, x2):
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)


exp = _Function(function=_protected_exponent, name='exp', arity=1)
expneg = _Function(function=_proteceted_exponent_negative, name='expneg', arity=1)
n2 = _Function(function=_protected_n2, name='n2', arity=1)
n3 = _Function(function=_protected_n3, name='n3', arity=1)
tanh = _Function(function=np.tanh, name='tanh', arity=1)
div = _Function(function=_protected_division_ignore_overflow, name='div', arity=2)
logical_or = _Function(function=_protected_or, name='logical_or', arity=2)
logical_not = _Function(function=_protected_not, name='logical_not', arity=1)
logical_and = _Function(function=_protected_and, name='logical_and', arity=2)
# def AND(x1, x2):
#     return np.multiply(x1, x2)

# def OR(x1, x2):
#     return np.maximum(x1, x2)

# def NOT(x1):
#     return 1-x1

def recall_loss(y, y_pred, w):
    # 根据 y_pred 对样本排序，降序排列
    sorted_indices = np.argsort(y_pred)[::-1]
    
    #topk的k选择多少
    top_k = [0.4]
    positive_nodes_ratio = 0
    negative_nodes_ratio = 0
    # cnt = 0
    # true_label_num = 10
    # 计算前topk节点的数量
    for k in top_k:
        top_k_percent = int(k * len(y))

    # 获取前k节点的真实标签
        top_k_percent_labels = y[sorted_indices][:top_k_percent]

    # 计算前50%节点中真实节点的比例
    # for index in sorted_indices:
    #     if index <= true_label_num:
    #         cnt+=1
    # real_nodes_ratio = cnt / true_label_num
        positive_nodes_ratio += np.sum(top_k_percent_labels) / np.sum(y)
        negative_nodes_ratio += (int(k * len(y)) - np.sum(top_k_percent_labels)) / (len(y) - np.sum(y))
    avg_nodes_ratio = (positive_nodes_ratio - negative_nodes_ratio) / len(top_k)
#  + np.average(((y_pred - y) ** 2), weights=w)
    return 1 - avg_nodes_ratio  # 最小化 loss，等价于最大化真实节点比例

def recall_lut_loss(y, y_pred, w):
    # 根据 y_pred 对样本排序，降序排列
    positive_indexes = np.where(y==1)[0]
    predict_positive_indexes = np.where(y_pred==1)[0]
    intersection = np.intersect1d(positive_indexes, predict_positive_indexes)
    cnt = len(intersection)
    recall = cnt/len(positive_indexes)
    positive_ratio = len(predict_positive_indexes)/len(y_pred)
    return 1 - recall +  0.3 * positive_ratio

def gnn_recall_loss(y, y_pred, w):      # y按照先正节点后负节点排序
    # 根据 y_pred 对样本排序，升序排列
    sorted_indices = np.argsort(y_pred)

    # 正节点的数量
    true_label_num = 7738
    total_num = len(y)
    #topk取值
    k = 0.4
    
    #计数cnt
    cnt=0
    
    # 计算前topk节点的数量
    top_k_percent = int(k * total_num)

    # 获取前k节点的indexes
    top_k_percent_labels = sorted_indices[total_num-top_k_percent:]

    # 计算前topk节点中真实节点的比例
    for index in top_k_percent_labels:
        if index <= true_label_num:
            cnt+=1
    real_nodes_ratio = cnt / true_label_num
    return 1 - real_nodes_ratio # 最小化 loss，等价于最大化真实节点比例

def sigmoid(x):
    return 1/(1+np.exp(-x))

def bce_logictsloss(y, y_pred, w): 
    y_pred = sigmoid(y_pred)
    epsilon = 1e-15  # 避免 log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # 裁剪预测值
    loss = - np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return loss

def focal_logictsloss(y, y_pred, w):
    try:
        y_pred = sigmoid(y_pred)
        focal_gamma = 2
        # if self.single_domain_ensemble:
        #     y = y.repeat(1, y_pred.shape[1])
        num_negative_samples = len(np.where(y==0)[0])
        num_positive_samples = len(np.where(y>0)[0])
        epsilon = 1e-3
        alpha_positive = (num_negative_samples+1e-3) / (
            num_positive_samples+num_negative_samples)
        alpha_negative = (num_positive_samples+1e-3) / (
            num_positive_samples+num_negative_samples)
        # positive
        weight_positive = np.power(1. - y_pred + epsilon, focal_gamma)
        focal_positive = -alpha_positive * weight_positive * np.log(y_pred + epsilon)
        loss_positive = y * focal_positive
        # negative
        weight_negative = np.power(y_pred, focal_gamma)
        focal_negative = -alpha_negative * weight_negative * np.log(1. - y_pred + epsilon)
        loss_negative = (1. - y) * focal_negative

        loss = np.mean(loss_positive + loss_negative)
        if np.isnan(loss):
            raise ValueError("Computed r is NaN")
        return loss
    except Exception as e:
        # print(f'Error: {e}')
        # print('wrong return for score.py')
        return 1

def GLNN_loss(y, y_hat, w):
    def sigmoid(y):
        return 1/(1+np.exp(-y))
    def focal_loss(predict_y, target_y, focal_gamma=2, debug_log=False):
        # if self.single_domain_ensemble:
        #     target_y = target_y.repeat(1, predict_y.shape[1])
        num_negative_samples = len(np.where(target_y==0)[0])
        num_positive_samples = len(np.where(target_y>0)[0])
        epsilon = 1e-3
        alpha_positive = (num_negative_samples+1e-3) / (
            num_positive_samples+num_negative_samples)
        alpha_negative = (num_positive_samples+1e-3) / (
            num_positive_samples+num_negative_samples)
        # positive
        weight_positive = np.power(1. - predict_y + epsilon, focal_gamma)
        focal_positive = -alpha_positive * weight_positive * np.log(predict_y + epsilon)
        loss_positive = target_y * focal_positive
        if debug_log:
            print(
                f"debug log focal loss weight positive: {weight_positive.shape}")
            print(
                f"debug log focal loss focal_positive: {focal_positive.shape}")
            print(f"debug log focal loss loss_positive: {loss_positive.shape}")
        # negative
        weight_negative = np.power(predict_y, focal_gamma)
        focal_negative = -alpha_negative * weight_negative * np.log(1. - predict_y + epsilon)
        loss_negative = (1. - target_y) * focal_negative

        loss = np.mean(loss_positive + loss_negative)
        # print(f"debug log focal loss: {loss.shape}")
        return loss

    def MSE_loss(y_pred, y_true):
        mse = np.linalg.norm(y_pred - y_true, 2) ** 2 / y_true.shape[0]
        return mse
    'When using the GLNN loss, fix the epsilon as 0.5 for DSR'
    epsilon=0.5
    try:
        y_hat = sigmoid(y_hat)
        y_labels = (y >= 1).astype('float32')
        y_prediction = y - y_labels
        focal = focal_loss(y_hat, y_labels)
        mse = MSE_loss(y_hat, y_prediction)
        loss = epsilon * focal + (1 - epsilon) * mse
        if np.isnan(loss):
            raise ValueError("Computed r is NaN")
        return loss
    except Exception as e:
        return 1