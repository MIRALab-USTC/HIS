"""
File adapted from https://github.com/dandip/DSRPytorch
"""

from typing import List
import torch
from collections import OrderedDict, defaultdict

import settings.consts as consts
import utils.logger as logger

MATH_ARITY = OrderedDict([ # arity 2 -> arity 1
    ('+',2),
    ('-',2),
    ('*',2),
    ('/',2), 
    ('^', 2),
    ('exp',1),
    ('log',1),
    ('scatter_sum',1),
    ('scatter_mean',1),
    ('scatter_max',1),
    ('scatter_min',1),
])
# CONSTANT_OPERATORS = ["2.0", "5.0", "10.0", "0.1", "0.2", "0.5"] # @ means a place holder which will be optimized in the inner loop
CONSTANT_OPERATORS = ["0.1", "0.2", "0.5"] # @ means a place holder which will be optimized in the inner loop
INVERSE_OPERATOR_DICT = {"exp": "log", "log": "exp", "sqrt": "square", "square": "sqrt"}

def binary_f(x,a,b):
    return f"({a} {x} {b})"
def unary_f(x,a):
    return f"torch.{x}({a})"
def unary_f_nlp(x,a):
    return f"{x}({a})"
def nlp_power(x,a,b):
    return f"{a}^{b}"
def nlp_square(x,a):
    return f"{a}^2"
def nlp_sqrt(x,a):
    return f"{a}^0.5"

def scatter_func(x, a, scatter_degree, max_degree):
    assert 0 < scatter_degree <= max_degree

    b = "c_edge_index" if (scatter_degree % 2) == 0 else "v_edge_index"
    output = f"{x}({a},{b})"
    if scatter_degree > 1:
        output += f"[{b}]"
    elif scatter_degree == 1:
        output = f"{x}({a},{b},dim=0,dim_size=variable.size(0))"
    return output

def scatter_func_nlp(x, a, scatter_degree, max_degree):
    return f"{x}({a})"

def power(x,a,b):
    return f"torch.pow({a}, {b})"

TORCH_OPERATOR_DICT = {
    "^": power,
}

NLP_OPERATOR_DICT = {
    "^": nlp_power,
}


class Operators: # 用于管理表达式生成过程中使用的操作符
    # use_layer_learning和use_multi_layer务必只有一个True，且当use_multi_layer=True,需给出num_messages
    def __init__(self, const_list=None, math_list="simple", var_list="graph", scatter_max_degree=2, use_layer_learning=False, use_multi_layer=False, num_messages:List[int]=None): 
        """
        order: vars, consts, arity_two_operators, arity_one_operators
        """
        if var_list == "graph":
            if (not use_layer_learning) and (not use_multi_layer):
                self.var_operators = consts.GRAPH_NAMES
            elif use_multi_layer:
                self.T = num_messages
                MESSAGE_FEATURES = []
                for l in range(len(num_messages)): # len(num_messages)表示最大层数
                    for i in range(num_messages[l]):
                        MESSAGE_FEATURES.append(f'layer{l}_message_{i+1}')
                self.var_operators = consts.GRAPH_NAMES + MESSAGE_FEATURES
            else:
                self.T1 = 5
                self.T2 = 5
                MESSAGE_FEATURES = [f'node_message_{i+1}' for i in range(self.T1)] + [f'constraint_message_{i+1}' for i in range(self.T2)]
                self.var_operators = consts.GRAPH_NAMES + MESSAGE_FEATURES
        else:
            raise NotImplementedError

        if const_list is None:
            self.constant_operators = CONSTANT_OPERATORS[:]
        else:
            self.constant_operators = const_list

        if math_list == "simple":
            self.math_operators = ['+', '-', '*', 'scatter_sum', 'scatter_mean', 'scatter_max', 'scatter_min'] 
        else:
            assert math_list == 'all'
            self.math_operators = list(MATH_ARITY.keys())
        self.scatter_max_degree = scatter_max_degree
        if use_multi_layer:
            self.scatter_max_degree = len(num_messages)


        self.operator_list = self.var_operators + self.constant_operators + self.math_operators #将所有操作符（变量、常数、数学运算符）合并为一个列表
        self.operator_dict = {k:i for i,k in enumerate(self.operator_list)} # 创建一个字典，将操作符名称映射到其索引
        self.operator_length = len(self.operator_list)

        arity_dict = defaultdict(int, MATH_ARITY) #  从 MATH_ARITY 中获取每个操作符的元数（即操作符所需的参数数量）
        self.arity_list = [arity_dict[operator] for operator in self.operator_list]
        self.arity_tensor = torch.tensor(self.arity_list, dtype=torch.long)

        self.zero_arity_mask = torch.tensor([True if arity_dict[x]==0 else False for x in self.operator_list], dtype=torch.bool)[None, :] # 表示哪些操作符的元数为 0
        self.nonzero_arity_mask = torch.tensor([True if arity_dict[x]!=0 else False for x in self.operator_list], dtype=torch.bool)[None, :] # 表示哪些操作符的元数不为 0

        self.have_inverse = torch.tensor([((operator in INVERSE_OPERATOR_DICT) and (INVERSE_OPERATOR_DICT[operator] in self.operator_dict)) for operator in self.operator_list], dtype=torch.bool) # 表示哪些操作符有逆操作符
        self.where_inverse = torch.full(size=(self.operator_length,), fill_value=int(1e5), dtype=torch.long) # 存储每个操作符的逆操作符的索引。如果没有逆操作符，则值为 1e5
        self.where_inverse[self.have_inverse] = torch.tensor([self.operator_dict[INVERSE_OPERATOR_DICT[operator]] for i, operator in enumerate(self.operator_list) if self.have_inverse[i]], dtype=torch.long)

        variable_mask = torch.zeros(len(self.operator_list), dtype=torch.bool) # 表示哪些操作符是变量
        variable_mask[:len(self.var_operators)] = True
        self.variable_mask = variable_mask[None, :]
        self.non_variable_mask = torch.logical_not(self.variable_mask)

        const_mask = torch.zeros(len(self.operator_list), dtype=torch.bool) # 表示哪些操作符是常量
        const_mask[len(self.var_operators):-len(self.math_operators)] = True
        self.const_mask = const_mask[None, :]
        self.non_const_mask = torch.logical_not(self.const_mask)

        self.scatter_num = sum([("scatter" in x) for x in self.math_operators]) # 表示哪些操作符是聚合函数
        assert self.scatter_num > 0
        scatter_mask = torch.zeros(len(self.operator_list), dtype=torch.bool)
        scatter_mask[-self.scatter_num:] = True
        self.scatter_mask = scatter_mask[None, :]
        self.non_scatter_mask = torch.logical_not(self.scatter_mask)

        num_math_arity_two = sum([1 for x in self.math_operators if MATH_ARITY[x]==2])
        num_math_arity_one = len(self.math_operators) - num_math_arity_two
        self.arity_zero_begin, self.arity_zero_end = 0, len(self.var_operators) + len(self.constant_operators) # 无参数操作符的索引范围
        self.arity_two_begin, self.arity_two_end = len(self.var_operators) + len(self.constant_operators), len(self.var_operators) + len(self.constant_operators) + num_math_arity_two # 二元操作符的索引范围
        self.arity_one_begin, self.arity_one_end = len(self.operator_list) - num_math_arity_one, len(self.operator_list) # 一元操作符的索引范围

        self.variable_begin, self.variable_end = 0, len(self.var_operators) # 变量操作符的索引范围
        self.scatter_begin, self.scatter_end = len(self.operator_list) - self.scatter_num, len(self.operator_list) # 聚合函数的索引范围

        self.variable_constraint_begin, self.variable_constraint_end = 0, len(consts.CONSTRAINT_FEATURES)
        self.variable_variable_begin, self.variable_variable_end = self.variable_constraint_end + len(consts.EDGE_FEATURES), self.variable_constraint_end + len(consts.EDGE_FEATURES) + len(consts.NODE_FEATURES)

        if use_layer_learning:
            self.variable_message_begin, self.variable_message_end = self.variable_variable_end, self.variable_variable_end + self.T1 + self.T2
            self.variable_node_message_begin, self.variable_node_message_end = self.variable_variable_end, self.variable_variable_end + self.T1
            self.variable_constraint_message_begin, self.variable_constraint_message_end = self.variable_variable_end + self.T1, self.variable_variable_end + self.T1 + self.T2
        if use_multi_layer:
            self.variable_message_begin = [self.variable_variable_end]
            for l in range(1, len(num_messages)):
                self.variable_message_begin.append(self.variable_message_begin[l-1] + num_messages[l-1])
            self.variable_message_end = self.variable_message_begin[-1] + num_messages[-1]

        scatter_degree_0_mask = torch.zeros(len(self.operator_list), dtype=torch.bool)
        scatter_degree_0_mask[:self.variable_variable_begin] = True
        self.scatter_degree_0_mask = scatter_degree_0_mask[None, :]

        scatter_degree_1_mask = torch.zeros(len(self.operator_list), dtype=torch.bool)
        scatter_degree_1_mask[self.variable_constraint_end:-len(self.math_operators)] = True
        if use_layer_learning:
            scatter_degree_1_mask[self.variable_constraint_message_begin:self.variable_constraint_message_end] = False # c to v,除约束节点、更新后约束节点、运算符,其余为True
        self.scatter_degree_1_mask = scatter_degree_1_mask[None, :]


        scatter_degree_2_mask = scatter_degree_0_mask.clone()
        scatter_degree_2_mask[self.variable_variable_end:-len(self.math_operators)] = True
        self.scatter_degree_2_mask = scatter_degree_2_mask[None, :]

        # use_layer_learning
        if use_layer_learning:
            scatter_degree_layer_0_mask = scatter_degree_0_mask.clone()
            scatter_degree_layer_0_mask[self.variable_node_message_end:-len(self.math_operators)] = True # 除变量节点、更新后变量节点、运算符其余为True
            self.scatter_degree_layer_0_mask = scatter_degree_layer_0_mask[None, :] # (1, 162)
        if use_multi_layer:
            c_mask = torch.zeros(len(self.operator_list), dtype=torch.bool)
            c_mask[:self.variable_constraint_end] = True
            c_mask[-len(self.math_operators):] = True
            self.c_mask = c_mask[None, :]
            self.non_c_mask = torch.logical_not(self.c_mask)

            v_mask = torch.zeros(len(self.operator_list), dtype=torch.bool)
            v_mask[self.variable_variable_begin:self.variable_variable_end] = True
            v_mask[-len(self.math_operators):] = True
            self.v_mask = v_mask[None, :]
            self.non_v_mask = torch.logical_not(self.v_mask)

            self.message_mask = []
            for l in range(len(self.T)):
                self.message_mask.append(torch.zeros(len(self.operator_list), dtype=torch.bool))
                if l == len(self.T)-1:
                    self.message_mask[l][self.variable_message_begin[l]:self.variable_message_end] = True
                else:
                    self.message_mask[l][self.variable_message_begin[l]:self.variable_message_begin[l+1]] = True
                self.message_mask[l] = self.message_mask[l][None, :]




    def is_var_i(self, i):
        return self.variable_begin <= i < self.variable_end # 判断给定的索引 i 是否对应一个变量操作符
