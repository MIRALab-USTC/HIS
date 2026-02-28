"""
File adapted from https://github.com/dandip/DSRPytorch
"""

from typing import List
from turtle import forward
from scipy import special
from sympy import sequence
import torch.nn as nn
import torch.nn.functional as F
import torch

from settings.consts import SAFE_EPSILON
from dso_utils_graph.transformer_utils import TransformerDSOEncoder
                


class DSOAgent(nn.Module):
    def __init__(self, operators, min_length=4, max_length=128, hidden_size=128, num_layers=2, soft_length=64, two_sigma_square=16):
        super().__init__()

        self.input_size = 2 * (operators.operator_length+1) + operators.scatter_max_degree + 1 # One-hot encoded parent and sibling
        self.hidden_size = hidden_size
        self.output_size = operators.operator_length # Output is a softmax distribution over all operators
        self.num_layers = num_layers
        self.operators = operators

        # Initial cell optimization
        self.init_input = nn.Parameter(data=torch.rand(1, self.input_size), requires_grad=True)
        self.init_hidden = nn.Parameter(data=torch.rand(self.num_layers, 1, self.hidden_size), requires_grad=True)

        self.min_length = min_length
        self.max_length = max_length
        self.soft_length = soft_length
        self.two_sigma_square = two_sigma_square

        self.lstm = nn.LSTM(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            proj_size = self.output_size,
        )
        self.init_hidden_lstm = nn.Parameter(data=torch.rand(self.num_layers, 1, self.output_size), requires_grad=True)
        self.activation = nn.Softmax(dim=1)

    @torch.no_grad()
    def sample_sequence_eval(self, n): # 在评估模式下生成符号表达式序列
        sequences = torch.zeros(n, 0, dtype=torch.long)

        input_tensor = self.init_input.expand(n, -1).contiguous()
        hidden_tensor = self.init_hidden.expand(-1, n, -1).contiguous()
        hidden_lstm = self.init_hidden_lstm.expand(-1, n, -1).contiguous()

        sequence_mask = torch.ones(n, dtype=torch.bool)
        counters = torch.ones(n, 1, dtype=torch.long) # Number of tokens that must be sampled to complete expression

        length = 0
        all_lengths = torch.zeros(n, dtype=torch.long)

        # While there are still tokens left for sequences in the batch
        all_log_prob_list, all_counters_list, all_inputs_list, all_scatter_degree_list, scatter_parent_where_seq = [], [], [], [torch.zeros((n,1), dtype=torch.long)], torch.full((n,1), fill_value=-1, dtype=torch.long)
        while(sequence_mask.any()):
            output, hidden_tensor, hidden_lstm = self.forward(input_tensor, hidden_tensor, hidden_lstm, length)

            # Apply constraints and normalize distribution
            output = self.apply_constraints(output, counters, length, sequences, scatter_degree=all_scatter_degree_list[-1], scatter_parent_where_seq=scatter_parent_where_seq)
            output = output / torch.sum(output, dim=1, keepdim=True)

            # Sample from categorical distribution
            dist = torch.distributions.Categorical(output)
            token = dist.sample()

            # Add sampled tokens to sequences
            sequences = torch.cat((sequences, token[:, None]), dim=1)
            length += 1
            all_lengths[sequence_mask] += 1

            # Add log probability of current token
            all_log_prob_list.append(dist.log_prob(token)[:, None])

            # Add entropy of current token
            all_counters_list.append(counters)
            all_inputs_list.append(input_tensor)

            # Update counter
            counters = counters + (torch.logical_and(self.operators.arity_one_begin<=token, token<self.operators.arity_one_end).long() \
                        + torch.logical_and(self.operators.arity_two_begin<=token, token<self.operators.arity_two_end).long() * 2 - 1)[:, None]
            sequence_mask = torch.logical_and(sequence_mask, counters.squeeze(1) > 0)

            # Compute next parent and sibling; assemble next input tensor
            input_tensor, scatter_parent_where_seq = self.get_parent_sibling(n, sequences, length-1, sequence_mask, all_scatter_degree_list, scatter_parent_where_seq=scatter_parent_where_seq) # get input_tensor and update all_scatter_degree_list

        # Filter entropies log probabilities using the sequence_mask
        assert all_lengths.min() >= self.min_length and all_lengths.max() <= self.max_length+1 and all_lengths.max() == sequences.shape[1]
        log_probs = torch.cat(all_log_prob_list, dim=1)

        return sequences, all_lengths, log_probs, (all_counters_list, all_inputs_list, all_scatter_degree_list, scatter_parent_where_seq)

    def sample_sequence_train(self, sequences, info_lists): # 在训练模式下生成符号表达式序列
        all_counters_list, all_inputs_list, all_scatter_degree_list, scatter_parent_where_seq = info_lists
        assert sequences.shape[1] == len(all_counters_list) == len(all_inputs_list)
        n = len(sequences)

        all_inputs_list[0] = self.init_input.expand(n, -1).contiguous()
        hidden_tensor = self.init_hidden.expand(-1, n, -1).contiguous()
        hidden_lstm = self.init_hidden_lstm.expand(-1, n, -1).contiguous()

        all_log_prob_list, all_entropy_list = [], []
        for i, (token, counters, input_tensor, scatter_degree) in enumerate(zip(sequences.t(), all_counters_list, all_inputs_list, all_scatter_degree_list)):
            output, hidden_tensor, hidden_lstm = self.forward(input_tensor, hidden_tensor, hidden_lstm, i)

            output = self.apply_constraints(output, counters, i, sequences[:,:i], scatter_degree=scatter_degree, scatter_parent_where_seq=scatter_parent_where_seq[:, :i+1])
            output = output / torch.sum(output, dim=1, keepdim=True)

            dist = torch.distributions.Categorical(output)
            all_log_prob_list.append(dist.log_prob(token)[:, None])
            all_entropy_list.append(dist.entropy()[:, None])

        entropies = torch.cat(all_entropy_list, dim=1)
        log_probs = torch.cat(all_log_prob_list, dim=1)

        return entropies, log_probs


    def forward(self, input, hidden, hidden_lstm, cur_length): # LSTM的前向传播
        """Input should be [parent, sibling]
        """

        output, (hn, cn) = self.lstm(input.unsqueeze(0).float(), (hidden_lstm, hidden))
        output = output.squeeze(0)

        # ~ soft length constraint
        prior_vec = torch.zeros(self.output_size)
        if cur_length < self.soft_length:
            prior_vec[self.operators.arity_zero_begin:self.operators.arity_zero_end] = - (self.soft_length - cur_length) ** 2 / self.two_sigma_square
        elif cur_length > self.soft_length:
            prior_vec[self.operators.arity_two_begin:self.operators.arity_two_end] = - (cur_length - self.soft_length) ** 2 / self.two_sigma_square
        output = output + prior_vec[None, :]

        output = self.activation(output)
        return output, cn, hn

    def apply_constraints(self, output, counters, length, sequences, scatter_degree, scatter_parent_where_seq):
        """Applies in situ constraints to the distribution contained in output based on the current tokens
        """
        # Add small epsilon to output so that there is a probability of selecting
        # everything. Otherwise, constraints may make the only operators ones
        # that were initially set to zero, which will prevent us selecting
        # anything, resulting in an error being thrown
        
        output = output + SAFE_EPSILON

        # turn off column features if we set var_list = simple
        # if self.operators.column_var_mask is not None:
        #     output[:, self.operators.column_var_mask] = 0.

        # ~ Check that minimum length will be met ~
        # Explanation here
        min_boolean_mask = (counters + length) >= self.min_length
        min_length_mask = torch.logical_or(self.operators.nonzero_arity_mask, min_boolean_mask)
        output[~min_length_mask] = 0.

        # ~ Check that maximum length won't be exceed ~
        max_boolean_mask = (counters + length) < self.max_length
        max_length_mask = torch.logical_or(self.operators.zero_arity_mask, max_boolean_mask)
        output[~max_length_mask] = 0

        # forbid direct inverse function
        # last_token = sequences[:, -1]
        # output[xxx, ]


        # ~ Ensure that all expressions have a variable ~
        # nonvar_zeroarity_mask = ~torch.logical_and(self.operators.zero_arity_mask, self.operators.nonvariable_mask)
        if (length == 0): # First thing we sample can't be
            # output = torch.minimum(output, nonvar_zeroarity_mask)
            output[:, self.operators.const_mask.squeeze(0)] = 0
        else:
            last_counter_mask = (counters == 1)
            non_var_now_mask = torch.logical_not( (sequences < self.operators.variable_end).any(dim=1, keepdim=True) )
            last_token_and_no_var_mask = torch.logical_and(last_counter_mask, non_var_now_mask)
            const_and_last_token_and_no_var_mask = torch.logical_and(last_token_and_no_var_mask, self.operators.const_mask)
            output[const_and_last_token_and_no_var_mask] = 0
            # nonvar_zeroarity_mask = nonvar_zeroarity_mask.repeat(counters.shape[0], 1)
            # # Don't sample a nonvar zeroarity token if the counter is at 1 and
            # # we haven't sampled a variable yet
            # counter_mask = (counters == 1)
            # contains_novar_mask = ~(torch.isin(sequences, self.operators.variable_tensor).any(axis=1))
            # last_token_and_no_var_mask = (~torch.logical_and(counter_mask, contains_novar_mask)[:, None]).long()
            # nonvar_zeroarity_mask = torch.max(nonvar_zeroarity_mask, last_token_and_no_var_mask * torch.ones(nonvar_zeroarity_mask.shape)).long()
            # output = torch.minimum(output, nonvar_zeroarity_mask)

            # ~ forbid inverse unary
            last_token = sequences[:, -1]
            last_token_has_inverse = torch.where(self.operators.have_inverse[last_token])[0]
            last_token_inverse = self.operators.where_inverse[last_token[last_token_has_inverse]]
            output[last_token_has_inverse, last_token_inverse] = 0

            # degree 1,3,5,...
            scatter_mod = scatter_degree % 2
            where_same_sub_tree = scatter_parent_where_seq[:, :-1] == scatter_parent_where_seq[:,-1:]
            sub_tree_counter_is_degree_2 = torch.logical_and(torch.logical_and(sequences>=self.operators.arity_two_begin, sequences < self.operators.arity_two_end), where_same_sub_tree)
            sub_tree_counter_is_degree_0 = torch.logical_and(sequences < self.operators.arity_zero_end, where_same_sub_tree)
            sub_tree_counter = 1 + sub_tree_counter_is_degree_2.sum(dim=1, keepdim=True) - sub_tree_counter_is_degree_0.sum(dim=1, keepdim=True)
            sub_tree_last_counter_mask = (sub_tree_counter == 1)

            where_scatter_degree_1 = (scatter_mod == 1)
            where_scatter_degree_1_and_last_counter = torch.logical_and(where_scatter_degree_1, sub_tree_last_counter_mask)

            non_message_passing_1 = torch.logical_not( torch.logical_and(sequences < self.operators.variable_constraint_end, where_same_sub_tree).any(dim=1, keepdim=True) )

            where_scatter_degree_1_and_last_counter_and_non_message_passing = torch.logical_and(where_scatter_degree_1_and_last_counter, non_message_passing_1)
            scatter_degree_1_mask = torch.logical_and(where_scatter_degree_1_and_last_counter_and_non_message_passing, self.operators.scatter_degree_1_mask)
            output[scatter_degree_1_mask] = 0.

            # degree 2
            where_scatter_degree_2 = torch.logical_and(scatter_degree > 0, scatter_mod == 0)
            where_scatter_degree_2_and_last_counter = torch.logical_and(where_scatter_degree_2, sub_tree_last_counter_mask)
            non_message_passing_2 = torch.logical_not(torch.logical_and(torch.logical_and(sequences < self.operators.variable_variable_end, sequences >= self.operators.variable_variable_begin), where_same_sub_tree).any(dim=1, keepdim=True))

            where_scatter_degree_2_and_last_counter_and_non_message_passing = torch.logical_and(where_scatter_degree_2_and_last_counter, non_message_passing_2)
            scatter_degree_2_mask = torch.logical_and(where_scatter_degree_2_and_last_counter_and_non_message_passing, self.operators.scatter_degree_2_mask)
            output[scatter_degree_2_mask] = 0.

        # mask to avoid too many message passing layers
        scatter_should_mask = (scatter_degree >= self.operators.scatter_max_degree)
        scatter_mask = torch.logical_and(scatter_should_mask, self.operators.scatter_mask)
        output[scatter_mask] = 0.

        # mask constraint features when scatter_degree == 0
        where_scatter_degree_0 = (scatter_degree == 0)
        scatter_degree_0_mask = torch.logical_and(where_scatter_degree_0, self.operators.scatter_degree_0_mask)
        output[scatter_degree_0_mask] = 0.

        return output

    def get_parent_sibling(self, batch_size, sequences, recent, sequence_mask, all_scatter_degree_list, scatter_parent_where_seq):
        parent_sibling, parent_sibling_where = self._get_parent_sibling(batch_size, sequences, recent, sequence_mask)
        parent_where = parent_sibling_where[:,0]

        current_scatter_degree = (torch.as_tensor([all_scatter_degree_list[x][i] for i,x in enumerate(parent_where)], dtype=torch.long) + \
                                 (sequences[torch.arange(batch_size), parent_where] >= self.operators.scatter_begin).long())
        all_scatter_degree_list.append(current_scatter_degree[:, None])

        scatter_parent_where_now = scatter_parent_where_seq[torch.arange(batch_size), parent_where]
        scatter_parent_where_now[sequences[:, recent]>=self.operators.scatter_begin] = recent
        scatter_parent_where_seq = torch.cat((scatter_parent_where_seq, scatter_parent_where_now[:, None]), dim=1)

        input_tensor1 = F.one_hot(parent_sibling.reshape(-1), num_classes=self.output_size+1)
        input_tensor1 = input_tensor1.reshape(batch_size, -1)
        input_tensor2 = F.one_hot(current_scatter_degree, num_classes=self.operators.scatter_max_degree + 1)

        input_tensor = torch.cat((input_tensor1, input_tensor2), dim=1)

        return input_tensor, scatter_parent_where_seq


    def _get_parent_sibling(self, batch_size, sequences, recent, sequence_mask):
        """Returns parent, sibling for the most recent token in token_list
        """

        parent_sibling = torch.full(size=(batch_size, 2), fill_value=-1, dtype=torch.long)
        parent_sibling[~sequence_mask] = 0

        parent_sibling_where = torch.full(size=(batch_size,2), fill_value=-1, dtype=torch.long)
        parent_sibling_where[~sequence_mask] = 0

        token_last = sequences[:, recent]
        token_last_is_parent = self.operators.arity_tensor[token_last] > 0
        parent_sibling[token_last_is_parent, 0] = token_last[token_last_is_parent]
        parent_sibling[token_last_is_parent, 1] = self.output_size # means empty token

        parent_sibling_where[token_last_is_parent,0] = recent

        c = torch.zeros(batch_size, dtype=torch.long)
        for i in range(recent, -1, -1):
            unfinished_bool_index = (parent_sibling < 0).any(dim=1)
            if not unfinished_bool_index.any():
                break

            unfinished_token_i = sequences[:, i][unfinished_bool_index]
            c[unfinished_bool_index] += (torch.logical_and(self.operators.arity_one_begin<=unfinished_token_i, unfinished_token_i<self.operators.arity_one_end).long() \
                        + torch.logical_and(self.operators.arity_two_begin<=unfinished_token_i, unfinished_token_i<self.operators.arity_two_end).long() * 2 - 1)
            found_now = torch.logical_and(unfinished_bool_index, c==0)

            parent_sibling[found_now] = sequences[found_now, i:(i+2)]
            parent_sibling_where[found_now,0] = i
            parent_sibling_where[found_now,1] = i+1

        assert (torch.logical_and(parent_sibling >= 0, parent_sibling <= self.output_size)).all()
        assert (parent_sibling_where[:,0] >= 0).all() # or (recent == 0)

        return parent_sibling, parent_sibling_where


class TransformerDSOAgent(DSOAgent):
    def __init__(self, operators, min_length=4, max_length=128, soft_length=64, two_sigma_square=16, d_model=64, num_heads=8, d_ff=256, num_layers=6, structural_encoding=True, layer_encoding=True, logits_parentsib=False, **kwargs):
        '''
        operators: 操作符集合。
        min_length: 生成表达式的最小长度。
        max_length: 生成表达式的最大长度。
        soft_length: 软长度约束。
        two_sigma_square: 用于计算软长度约束的方差。
        d_model: Transformer的模型维度。
        num_heads: Transformer的头数。
        d_ff: Transformer的前馈网络维度。
        num_layers: Transformer的层数。
        structural_encoding: 是否使用结构编码。
        '''
        super(DSOAgent, self).__init__()
        self.operators, self.min_length, self.max_length, self.soft_length, self.two_sigma_square = operators, min_length, max_length, soft_length, two_sigma_square
        self.output_size = operators.operator_length
        self.transformer = TransformerDSOEncoder(self.output_size, operators.scatter_max_degree, max_length=max_length, d_model=d_model, num_heads=num_heads, d_ff=d_ff, num_layers=num_layers, structural_encoding=structural_encoding, layer_encoding=layer_encoding, logits_parentsib=logits_parentsib)
        self.activation = nn.Softmax(dim=1)

    def forward(self, raw_x, scatter_degree, parentchild_indices, parent_child_now, sibling_indices, sibling_now):
        cur_length = raw_x.shape[1]-1
        output = self.transformer(raw_x, scatter_degree, parentchild_indices, parent_child_now, sibling_indices, sibling_now) # only return the last logits

        # ~ soft length constraint
        prior_vec = torch.zeros(self.output_size)

        if cur_length < self.soft_length:
            prior_vec[self.operators.arity_zero_begin:self.operators.arity_zero_end] = - (self.soft_length - cur_length) ** 2 / self.two_sigma_square
        elif cur_length > self.soft_length:
            prior_vec[self.operators.arity_two_begin:self.operators.arity_two_end] = - (cur_length - self.soft_length) ** 2 / self.two_sigma_square
        output = output + prior_vec[None, :]

        output = self.activation(output)
        return output

    @torch.no_grad()
    def sample_sequence_eval(self, n):
        max_len = self.max_length + 3

        sequences = torch.zeros((n,max_len), dtype=torch.long)
        sequences[:,0] = self.output_size
        scatter_degree = torch.zeros_like(sequences)
        parent_child_pairs = torch.zeros((n*(max_len), 3), dtype=torch.long)
        parent_child_length = torch.zeros(max_len, dtype=torch.long)
        sibling_pairs = torch.zeros((n*(max_len), 3), dtype=torch.long)
        sibling_length = torch.zeros(max_len, dtype=torch.long)
        scatter_parent_where_seq = torch.full_like(sequences, fill_value=-1)

        length = 0
        all_lengths = torch.zeros(n, dtype=torch.long)
        sequence_mask = torch.ones(n, dtype=torch.bool)
        counters = torch.ones(n, 1, dtype=torch.long) # Number of tokens that must be sampled to complete expression
        all_log_prob_list, all_counters_list = [], []

        while(sequence_mask.any()):

            output = self.forward(sequences[:,:(length+1)], scatter_degree[:,:(length+1)], parent_child_pairs[:parent_child_length[max(length-1,0)]], parent_child_pairs[parent_child_length[max(length-1,0)]:parent_child_length[length]], sibling_pairs[:sibling_length[max(length-1,0)]], sibling_pairs[sibling_length[max(length-1,0)]:sibling_length[length]])

            # Apply constraints and normalize distribution
            output = self.apply_constraints(output, counters, length, sequences[:,1:(length+1)], scatter_degree=scatter_degree[:,length][:,None], scatter_parent_where_seq=scatter_parent_where_seq[:,:(length+1)])
            output = output / torch.sum(output, dim=1, keepdim=True)

            # Sample from categorical distribution
            dist = torch.distributions.Categorical(output)
            token = dist.sample()

            # Add sampled tokens to sequences
            sequences[:, length+1] = token
            length += 1
            all_lengths[sequence_mask] += 1

            # Add log probability of current token
            all_log_prob_list.append(dist.log_prob(token)[:, None])

            # Add entropy of current token
            all_counters_list.append(counters)

            # Update counter
            counters = counters + (torch.logical_and(self.operators.arity_one_begin<=token, token<self.operators.arity_one_end).long() \
                        + torch.logical_and(self.operators.arity_two_begin<=token, token<self.operators.arity_two_end).long() * 2 - 1)[:, None]
            sequence_mask = torch.logical_and(sequence_mask, counters.squeeze(1) > 0)

            # Compute next parent and sibling; assemble next input tensor
            self.get_parent_sibling(n, sequences[:,1:], length-1, sequence_mask, scatter_degree, scatter_parent_where_seq, parent_child_pairs, parent_child_length, sibling_pairs, sibling_length) # update all info

        assert all_lengths.min() >= self.min_length and all_lengths.max() <= self.max_length+1 and all_lengths.max() == length
        log_probs = torch.cat(all_log_prob_list, dim=1)

        return sequences[:,:(length+1)], all_lengths, log_probs, (scatter_degree[:,:(length+1)], all_counters_list, scatter_parent_where_seq[:,:(length+1)], parent_child_pairs[:parent_child_length[length]], parent_child_length[:length+1], sibling_pairs[:sibling_length[length]], sibling_length[:length+1])

    def get_parent_sibling(self, batch_size, sequences, recent, sequence_mask, scatter_degree, scatter_parent_where_seq, parent_child_pairs, parent_child_length, sibling_pairs, sibling_length):
        _, parent_sibling_where = self._get_parent_sibling(batch_size, sequences, recent, sequence_mask)
        parent_where, sibling_where = parent_sibling_where[:,0], parent_sibling_where[:,1]

        scatter_degree[:, recent+1] = ( scatter_degree[torch.arange(batch_size), parent_where] + \
                                 (sequences[torch.arange(batch_size), parent_where] >= self.operators.scatter_begin).long())

        scatter_parent_where_now = scatter_parent_where_seq[torch.arange(batch_size), parent_where]
        scatter_parent_where_now[sequences[:, recent]>=self.operators.scatter_begin] = recent

        scatter_parent_where_seq[:, recent+1] = scatter_parent_where_now

        parent_child_length[recent+1] = parent_child_length[recent] + len(parent_where) # recent指向当前生成的token数量
        parent_child_pairs[parent_child_length[recent]: parent_child_length[recent+1], 0] = torch.arange(len(parent_where))
        parent_child_pairs[parent_child_length[recent]: parent_child_length[recent+1], 2] = recent + 2 # 默认指向待生成token 的下一位置
        parent_child_pairs[parent_child_length[recent]: parent_child_length[recent+1], 1] = parent_where + 1 # 指向待生成token的父节点位置

        where_has_sibling = torch.where(sibling_where > 0)[0]
        sibling_length[recent+1] = sibling_length[recent] + len(where_has_sibling)
        sibling_pairs[sibling_length[recent]: sibling_length[recent+1], 0] = where_has_sibling
        sibling_pairs[sibling_length[recent]: sibling_length[recent+1], 2] = recent + 2
        sibling_pairs[sibling_length[recent]: sibling_length[recent+1], 1] = sibling_where[where_has_sibling] + 1


    def sample_sequence_train(self, sequences, info_lists):
        scatter_degree, all_counters_list, scatter_parent_where_seq, parent_child_pairs, parent_child_length, sibling_pairs, sibling_length = info_lists
        length_max = sequences.shape[1] - 1

        all_log_prob_list, all_entropy_list = [], []
        for length in range(length_max):
            output = self.forward(sequences[:,:(length+1)], scatter_degree[:,:(length+1)], parent_child_pairs[:parent_child_length[max(length-1,0)]], parent_child_pairs[parent_child_length[max(length-1,0)]:parent_child_length[length]], sibling_pairs[:sibling_length[max(length-1,0)]], sibling_pairs[sibling_length[max(length-1,0)]:sibling_length[length]])

            # Apply constraints and normalize distribution
            output = self.apply_constraints(output, all_counters_list[length], length, sequences[:,1:(length+1)], scatter_degree=scatter_degree[:,length][:,None], scatter_parent_where_seq=scatter_parent_where_seq[:,:(length+1)])
            output = output / torch.sum(output, dim=1, keepdim=True)

            # Sample from categorical distribution
            dist = torch.distributions.Categorical(output)
            all_log_prob_list.append(dist.log_prob(sequences[:, length+1])[:, None])
            all_entropy_list.append(dist.entropy()[:, None])

        entropies = torch.cat(all_entropy_list, dim=1)
        log_probs = torch.cat(all_log_prob_list, dim=1)

        return entropies, log_probs

class TransformerDSOAgent_UseLayerLearning(TransformerDSOAgent):
    def __init__(self, operators, layer_min_length=[4,4,4], layer_max_length=[48,16,8], layer_soft_length=[26,10,6], two_sigma_square=16, d_model=64, num_heads=8, d_ff=256, num_layers=6, structural_encoding=True, layer_encoding=True, logits_parentsib=False, special_encoding=False, **kwargs):
        super(DSOAgent, self).__init__()
        self.operators, self.layer_min_length, self.layer_max_length, self.layer_soft_length, self.two_sigma_square = operators, layer_min_length, layer_max_length, layer_soft_length, two_sigma_square
        self.output_size = operators.operator_length
        self.special_encoding = special_encoding
        self.transformer = TransformerDSOEncoder(self.output_size, operators.scatter_max_degree, max_length=layer_max_length[0], d_model=d_model, num_heads=num_heads, d_ff=d_ff, num_layers=num_layers, structural_encoding=structural_encoding, layer_encoding=layer_encoding, logits_parentsib=logits_parentsib)
        self.activation = nn.Softmax(dim=1)

    def apply_constraints(self, output, counters, length, sequences, scatter_degree, scatter_parent_where_seq, layer, node_message_lengths, constraint_message_lengths):
        min_boolen_mask = (counters + length) >= self.layer_min_length[layer]
        min_length_mask = torch.logical_or(self.operators.nonzero_arity_mask, min_boolen_mask)
        # output[~min_length_mask] = 0 # 把不满足最小长度限制的序列行中零元操作符概率置0
        output = output * min_length_mask.to(output.dtype)

        max_boolen_mask = (counters + length) < self.layer_max_length[layer]
        max_length_mask = torch.logical_or(self.operators.zero_arity_mask, max_boolen_mask)
        # output[~max_length_mask] = 0 # 把到达最后一个token的序列行中非零元操作符概率置0
        output = output * max_length_mask.to(output.dtype)

        active_spencoding = torch.zeros(output.shape[0], dtype=torch.bool, device=output.device)
        if len(self.operators.math_operators) > 7:
            mask = torch.zeros_like(output, dtype=torch.bool)
            n = int(output.shape[0]/2)
            mask[-n:, -8:-4] = 1
            output = output*(1-mask.to(output.dtype))

        if length == 0:
            # output[:, self.operators.const_mask.squeeze(0)] = 0
            const_mask = self.operators.const_mask.squeeze(0)
            output = output * (1 - const_mask.to(output.dtype))

            if layer > 0: # 在非0层，需要消息聚合，第一个token必须是scatter算子
                # output[:, self.operators.non_scatter_mask.squeeze(0)] = 0
                non_scatter_mask = self.operators.non_scatter_mask.squeeze(0)
                output = output * (1 - non_scatter_mask.to(output.dtype))
            else:
                # output[:, self.operators.scatter_degree_layer_0_mask.squeeze(0)] = 0 # 除变量节点、更新后变量节点、运算符其余为True # scatter算子也应该屏蔽
                # output[:, self.operators.scatter_mask.squeeze(0)] = 0
                degree_0_mask = self.operators.scatter_degree_layer_0_mask.squeeze(0)
                scatter_mask_0 = self.operators.scatter_mask.squeeze(0)
                output = output * (1 - degree_0_mask.to(output.dtype))
                output = output * (1 - scatter_mask_0.to(output.dtype))
                # output[:, self.operators.scatter_degree_layer_0_mask] = 0

        else:
            # 屏蔽无变量情况
            last_counter_mask = (counters == 1)
            non_var_now_mask = torch.logical_not((sequences < self.operators.variable_end).any(dim=1, keepdim=True))
            last_token_and_no_var_mask = torch.logical_and(last_counter_mask, non_var_now_mask)
            const_and_last_token_and_no_var_mask = torch.logical_and(self.operators.const_mask, last_token_and_no_var_mask)
            # output[const_and_last_token_and_no_var_mask] = 0
            output = output*(1 - const_and_last_token_and_no_var_mask.to(output.dtype))

            # 屏蔽相邻逆运算
            last_token = sequences[:, -1]
            last_token_has_inverse = torch.where(self.operators.have_inverse[last_token])[0]
            last_token_inverse = self.operators.where_inverse[last_token[last_token_has_inverse]]
            mask = torch.ones_like(output)
            mask[last_token_has_inverse, last_token_inverse] = 0
            # output[last_token_has_inverse, last_token_inverse] = 0 # 屏蔽逆运算(上一个token可逆,则屏蔽相邻逆运算)
            output = output*mask.to(output.dtype)

            where_same_sub_tree = scatter_parent_where_seq[:, :-1] == scatter_parent_where_seq[:,-1:] # (512, 1) 判断当前节点是否与其父节点在同一子树中
            sub_tree_counter_is_degree_2 = torch.logical_and(torch.logical_and(sequences>=self.operators.arity_two_begin, sequences < self.operators.arity_two_end), where_same_sub_tree)
            sub_tree_counter_is_degree_0 = torch.logical_and(sequences < self.operators.arity_zero_end, where_same_sub_tree)
            sub_tree_counter = 1 + sub_tree_counter_is_degree_2.sum(dim=1, keepdim=True) - sub_tree_counter_is_degree_0.sum(dim=1, keepdim=True) # 当前子树还需要生成的tokens
            sub_tree_last_counter_mask = (sub_tree_counter == 1)
            
            if self.special_encoding:
                active_spencoding = torch.logical_and(sequences == self.operators.variable_variable_begin + 2, where_same_sub_tree).any(dim=1)
            if layer == 1: # c to v
                # output[:, self.operators.variable_node_message_begin:self.operators.variable_node_message_end] = 0 # c to v的更新不应该有更新后的变量节点输入
                mask = torch.ones_like(output)
                mask[:, self.operators.variable_node_message_begin:self.operators.variable_node_message_end] = 0
                output = output*mask.to(output.dtype)

                # 无消息传递的情况，必须要有c，c'，或运算符
                non_message_passing_c2v = torch.logical_not(torch.logical_and(torch.logical_or(sequences < self.operators.variable_constraint_end,
                                                                                               torch.logical_and(sequences >= self.operators.variable_constraint_message_begin, sequences < self.operators.variable_constraint_message_end)),
                                                                                               where_same_sub_tree).any(dim=1, keepdim=True))
                last_token_and_no_message_passing_c2v = torch.logical_and(sub_tree_last_counter_mask, non_message_passing_c2v)
                c2v_mask = torch.logical_and(last_token_and_no_message_passing_c2v, self.operators.scatter_degree_1_mask)
                # output[c2v_mask] = 0
                output = output*(1 - c2v_mask.to(output.dtype))
                
                # 禁止采样超长的更新节点
                remaining = self.layer_max_length[layer] - (length+counters).squeeze(1) # 当前层最大长度减去已生成长度
                mask = torch.ones_like(output)
                for idx in range(self.operators.T2):
                    # if(constraint_length[idx].max() > remaining):
                    #     output[:, self.operators.variable_constraint_message_begin + idx] = 0.0
                    mask_constraint = (constraint_message_lengths[idx] > remaining)
                    mask[mask_constraint, self.operators.variable_constraint_message_begin+idx] = 0.0
                    # output[mask_constraint, self.operators.variable_constraint_message_begin+idx] = 0.0
                output = output*mask.to(output.dtype)

                scatter_should_mask = (scatter_degree >= 1)
                scatter_mask = torch.logical_and(scatter_should_mask, self.operators.scatter_mask)
                # output[scatter_mask] = 0
                output = output*(1 - scatter_mask.to(output.dtype))

            elif layer == 2: # v to c
                # output[:, self.operators.variable_message_begin:self.operators.variable_message_end] = 0 # v to c的更新不应该有更新后的任何节点输入
                mask = torch.ones_like(output)
                mask[:, self.operators.variable_message_begin:self.operators.variable_message_end] = 0
                output = output*mask.to(output.dtype)

                non_message_passing_v2c = torch.logical_not(torch.logical_and(torch.logical_and(sequences >= self.operators.variable_variable_begin,
                                                                                                sequences < self.operators.variable_variable_end),
                                                                                                where_same_sub_tree).any(dim=1, keepdim=True))
                last_token_and_no_message_passing_v2c = torch.logical_and(sub_tree_last_counter_mask, non_message_passing_v2c)
                v2c_mask = torch.logical_and(last_token_and_no_message_passing_v2c, self.operators.scatter_degree_2_mask)
                # output[v2c_mask] = 0
                output = output*(1 - v2c_mask.to(output.dtype))
                scatter_should_mask = (scatter_degree >= 1)
                scatter_mask = torch.logical_and(scatter_should_mask, self.operators.scatter_mask)
                # output[scatter_mask] = 0
                output = output*(1 - scatter_mask.to(output.dtype))

            else: # layer 0
                # output[:, self.operators.scatter_mask.squeeze(0)] = 0
                # output[:, :self.operators.variable_variable_begin] = 0
                # output[:, self.operators.variable_constraint_message_begin:self.operators.variable_constraint_message_end] = 0
                mask = torch.ones_like(output)
                mask[:, self.operators.scatter_mask.squeeze(0)] = 0
                mask[:, :self.operators.variable_variable_begin] = 0
                mask[:, self.operators.variable_constraint_message_begin:self.operators.variable_constraint_message_end] = 0
                output = output*mask.to(output.dtype)
                non_message_passing = torch.logical_not(torch.logical_and(torch.logical_and(sequences >= self.operators.variable_variable_begin,
                                                                                            sequences < self.operators.variable_node_message_end),
                                                                                            where_same_sub_tree).any(dim=1, keepdim=True))
                last_token_and_no_message_passing = torch.logical_and(sub_tree_last_counter_mask, non_message_passing)
                layer_0_mask = torch.logical_and(last_token_and_no_message_passing, self.operators.scatter_degree_layer_0_mask)
                # output[layer_0_mask] = 0
                output = output*(1 - layer_0_mask.to(output.dtype))

                # balance_mask = (scatter_degree < 1).squeeze(-1)
                # balance_logits = (output[:, self.operators.variable_node_message_begin:self.operators.variable_node_message_end].float().mean() - output[:, :].float().mean())**2 # /self.two_sigma_square
                # output[balance_mask, self.operators.variable_variable_begin:self.operators.variable_variable_end] = torch.clamp(output[balance_mask, self.operators.variable_variable_begin:self.operators.variable_variable_end]-balance_logits,min=0.0)
                # output[balance_mask, self.operators.variable_node_message_begin:self.operators.variable_node_message_end] += balance_logits

                m = (scatter_degree < 1).squeeze(-1)[:, None]          # (B,1)
                b = (output[:, self.operators.variable_node_message_begin:self.operators.variable_node_message_end].float().mean() - output[:, :].float().mean()) ** 2

                vv = slice(self.operators.variable_variable_begin, self.operators.variable_variable_end)
                vn = slice(self.operators.variable_node_message_begin, self.operators.variable_node_message_end)

                output = output.clone()
                output = output.clone()
                output[:, vv] = torch.where(m, torch.clamp(output[:, vv] - b, min=0), output[:, vv])
                output[:, vn] = torch.where(m, output[:, vn] + b, output[:, vn])

                remaining = self.layer_max_length[layer] - (length+counters).squeeze(1)
                mask = torch.ones_like(output)
                for idx in range(self.operators.T2):
                    # if(constraint_length[idx].max() > remaining):
                    #     output[:, self.operators.variable_constraint_message_begin + idx] = 0.0
                    mask_constraint = (constraint_message_lengths[idx] > remaining)
                    mask[mask_constraint, self.operators.variable_constraint_message_begin+idx] = 0.0
                for idx in range(self.operators.T1):
                    # if (node_length[idx].max() > remaining):
                    #     output[:, self.operators.variable_node_message_begin + idx] = 0.0
                    mask_node = (node_message_lengths[idx] > remaining)
                    mask[mask_node, self.operators.variable_node_message_begin+idx] = 0.0
                output = output*mask.to(output.dtype)

        return output, active_spencoding

    def forward(self, raw_x, scatter_degree, parentchild_indices, parent_child_now, sibling_indices, sibling_now, layer, active_spencoding=None):
        cur_length = raw_x.shape[1]-1
        output = self.transformer(raw_x, scatter_degree, parentchild_indices, parent_child_now, sibling_indices, sibling_now, active_spencoding) # only return the last logits

        # ~ soft length constraint
        prior_vec = torch.zeros(self.output_size)

        if cur_length < self.layer_soft_length[layer]:
            prior_vec[self.operators.arity_zero_begin:self.operators.arity_zero_end] = - (self.layer_soft_length[layer] - cur_length) ** 2 / self.two_sigma_square
        elif cur_length > self.layer_soft_length[layer]:
            prior_vec[self.operators.arity_two_begin:self.operators.arity_two_end] = - (cur_length - self.layer_soft_length[layer]) ** 2 / self.two_sigma_square
        output = output + prior_vec[None, :]

        output = self.activation(output)
        return output
    
    @torch.no_grad() # @torch.no_grad()
    def sample_sequence_layer(self, n, node_message_sequences, node_message_lengths, node_message_log_probs, constraint_message_sequences, constraint_message_lengths, constraint_message_log_probs, layer):
        max_len = self.layer_max_length[layer] + 3

        # sequences = torch.zeros((n,max_len), dtype=torch.long)
        sequences = torch.full((n, max_len), fill_value=-1)
        sequences[:,0] = self.output_size
        scatter_degree = torch.zeros_like(sequences)
        parent_child_pairs = torch.zeros((n*(max_len), 3), dtype=torch.long)
        parent_child_length = torch.zeros(max_len, dtype=torch.long)
        sibling_pairs = torch.zeros((n*(max_len), 3), dtype=torch.long)
        sibling_length = torch.zeros(max_len, dtype=torch.long)
        scatter_parent_where_seq = torch.full_like(sequences, fill_value=-1)

        length = 0
        all_lengths = torch.zeros(n, dtype=torch.long)
        sequence_mask = torch.ones(n, dtype=torch.bool)
        counters = torch.ones(n, 1, dtype=torch.long) # Number of tokens that must be sampled to complete expression
        # all_log_prob_list, all_counters_list = [], []
        all_counters_list = []
        all_log_probs = torch.zeros_like(sequences, dtype=torch.float32)
        active_spencoding = torch.zeros(n, dtype=torch.bool)

        while(sequence_mask.any()):

            output = self.forward(sequences[:, :(length+1)], scatter_degree[:, :(length+1)], parent_child_pairs[:parent_child_length[max(length-1, 0)]], parent_child_pairs[parent_child_length[max(length-1, 0)]:parent_child_length[length]], sibling_pairs[:sibling_length[max(length-1, 0)]], sibling_pairs[sibling_length[max(length-1, 0)]:sibling_length[length]], layer=layer, active_spencoding=active_spencoding)
            
            # Apply constraints and normalize distribution
            output, active_spencoding = self.apply_constraints(output, counters, length, sequences[:,1:(length+1)], scatter_degree=scatter_degree[:,length][:,None], scatter_parent_where_seq=scatter_parent_where_seq[:,:(length+1)], layer=layer, node_message_lengths=node_message_lengths, constraint_message_lengths=constraint_message_lengths)
            output = output / torch.sum(output, dim=1, keepdim=True)

            # Sample from categorical distribution
            dist = torch.distributions.Categorical(output)
            token = dist.sample()
            log_prob_token = dist.log_prob(token)

            mask = (sequences[:, length+1] == -1)
            sequences[mask, length+1] = token[mask]
            all_log_probs[mask, length] = log_prob_token[mask]
            token[~mask] = sequences[~mask, length+1]

            # 替换node_message,node_message_log_probs
            node_message_mask = torch.logical_and(token >= self.operators.variable_node_message_begin, token<self.operators.variable_node_message_end)
            node_message_mask = torch.logical_and(node_message_mask, sequence_mask)
            rows = node_message_mask.nonzero(as_tuple=True)[0]
            for i in rows:
                l = node_message_lengths[token[i] - self.operators.variable_node_message_begin][i].item()
                sequences[i, length+1:length+1+l] = node_message_sequences[token[i] - self.operators.variable_node_message_begin][i, 1:1+l]
                all_log_probs[i, length:length+l] = node_message_log_probs[token[i] - self.operators.variable_node_message_begin][i, :l]

            # 替换constraint_message,constraint_message_log_probs
            constraint_message_mask = torch.logical_and(token >= self.operators.variable_constraint_message_begin, token<self.operators.variable_constraint_message_end)
            constraint_message_mask = torch.logical_and(constraint_message_mask, sequence_mask)
            rows = constraint_message_mask.nonzero(as_tuple=True)[0]
            for i in rows:
                l = constraint_message_lengths[token[i] - self.operators.variable_constraint_message_begin][i].item()
                sequences[i, length+1:length+1+l] = constraint_message_sequences[token[i] - self.operators.variable_constraint_message_begin][i, 1:1+l]
                all_log_probs[i, length:length+l] = constraint_message_log_probs[token[i] - self.operators.variable_constraint_message_begin][i, :l]

            token = sequences[:, length+1]
            length += 1
            all_lengths[sequence_mask] += 1

            # all_log_prob_list.append(log_prob_token)
            all_counters_list.append(counters)

            # Update counter
            counters = counters + (torch.logical_and(self.operators.arity_one_begin<=token, token<self.operators.arity_one_end).long() \
                        + torch.logical_and(self.operators.arity_two_begin<=token, token<self.operators.arity_two_end).long() * 2 - 1)[:, None]
            sequence_mask = torch.logical_and(sequence_mask, counters.squeeze(1) > 0)

            # Compute next parent and sibling; assemble next input tensor
            self.get_parent_sibling(n, sequences[:,1:], length-1, sequence_mask, scatter_degree, scatter_parent_where_seq, parent_child_pairs, parent_child_length, sibling_pairs, sibling_length) # update all info
            assert scatter_degree.max() <= self.operators.scatter_max_degree, f"scatter_degree out of range: max={scatter_degree.max()}, allowed={self.operators.scatter_max_degree}"

        assert all_lengths.min() >= self.layer_min_length[layer] and all_lengths.max() <= self.layer_max_length[layer]+1 and all_lengths.max() == length

        return sequences[:,:(length+1)], all_lengths, all_log_probs[:, :length], (scatter_degree[:,:(length+1)], all_counters_list, scatter_parent_where_seq[:,:(length+1)], parent_child_pairs[:parent_child_length[length]], parent_child_length[:length+1], sibling_pairs[:sibling_length[length]], sibling_length[:length+1])

    def sample_sequence_eval(self, n):
        node_message_sequences, node_message_lengths, node_message_log_probs = [], [], []
        constraint_message_sequences, constraint_message_lengths, constraint_message_log_probs = [], [], []

        # 打印子表达式需要
        node_message_info_lists, constraint_message_info_lists = [[] for _ in range(7)], [[] for _ in range(7)]
        # n_scatter_degree, n_counters, c_scatter_parent_where_seq, n_parent_child_pairs, n_parent_child_length, n_sibling_pairs, n_sibling_length = []*7

        for i in range(self.operators.T2):
            # sequences, all_lengths, log_probs,(scatter_degree, all_counters_list, scatter_parent_where_seq, parent_child_pairs, parent_child_length, sibling_pairs, sibling_length) = self.sample_sequence_layer(n, node_message_sequences, node_message_lengths, node_message_log_probs, constraint_message_sequences, constraint_message_lengths, constraint_message_log_probs, 2)
            sequences, all_lengths, log_probs, info_list = self.sample_sequence_layer(n, node_message_sequences, node_message_lengths, node_message_log_probs, constraint_message_sequences, constraint_message_lengths, constraint_message_log_probs, 2)
            constraint_message_sequences.append(sequences)
            constraint_message_lengths.append(all_lengths)
            constraint_message_log_probs.append(log_probs)
            # constraint_message_info_lists.append((scatter_degree, all_counters_list, scatter_parent_where_seq, parent_child_pairs, parent_child_length, sibling_pairs, sibling_length))
            for i, info in enumerate(info_list):
                constraint_message_info_lists[i].append(info)

        for i in range(self.operators.T1):
            # sequences, all_lengths, log_probs,(scatter_degree, all_counters_list, scatter_parent_where_seq, parent_child_pairs, parent_child_length, sibling_pairs, sibling_length) = self.sample_sequence_layer(n, node_message_sequences, node_message_lengths, node_message_log_probs, constraint_message_sequences, constraint_message_lengths, constraint_message_log_probs, 1)
            sequences, all_lengths, log_probs, info_list = self.sample_sequence_layer(n, node_message_sequences, node_message_lengths, node_message_log_probs, constraint_message_sequences, constraint_message_lengths, constraint_message_log_probs, 1)
            node_message_sequences.append(sequences)
            node_message_lengths.append(all_lengths)
            node_message_log_probs.append(log_probs)
            # node_message_info_lists.append((scatter_degree, all_counters_list, scatter_parent_where_seq, parent_child_pairs, parent_child_length, sibling_pairs, sibling_length))
            for i, info in enumerate(info_list):
                node_message_info_lists[i].append(info)

        sequences, all_lengths, log_probs, (scatter_degree, all_counters_list, scatter_parent_where_seq,
                                            parent_child_pairs, parent_child_length, sibling_pairs, sibling_length) = self.sample_sequence_layer(n, node_message_sequences, node_message_lengths, node_message_log_probs, constraint_message_sequences, constraint_message_lengths, constraint_message_log_probs, 0)
        return sequences, all_lengths, log_probs, (scatter_degree, all_counters_list, scatter_parent_where_seq,
                                                   parent_child_pairs, parent_child_length, sibling_pairs, sibling_length),(node_message_sequences, node_message_lengths, node_message_log_probs, node_message_info_lists, constraint_message_sequences, constraint_message_lengths, constraint_message_log_probs, constraint_message_info_lists)
    
    def sample_sequence_train(self, sequences, info_lists, node_message_sequences, node_message_lengths, node_message_info_lists, constraint_message_sequences, constraint_message_lengths, constraint_message_info_lists):
        scatter_degree, all_counters_list, scatter_parent_where_seq, parent_child_pairs, parent_child_length, sibling_pairs, sibling_length = info_lists
        node_message_scatter_degree, node_message_counters, node_message_scatter_parent_where_seq, node_message_parent_child_pairs, node_message_parent_child_length, node_message_sibling_pairs, node_message_sibling_length = node_message_info_lists
        constraint_message_scatter_degree, constraint_message_counters, constraint_message_scatter_parent_where_seq, constraint_message_parent_child_pairs, constraint_message_parent_child_length, constraint_message_sibling_pairs, constraint_message_sibling_length = constraint_message_info_lists

        constraint_message_new_log_probs, node_message_new_log_probs = [], []
        constraint_message_entropy, node_message_entropy = [], []
        active_spencoding = torch.zeros(sequences.shape[0], dtype=torch.bool)
        # v2c
        for i in range(self.operators.T2):
            length_max = constraint_message_sequences[i].shape[1] - 1
            new_log_probs = torch.zeros_like(constraint_message_sequences[i], dtype=torch.float32)
            c_entropy = torch.zeros_like(constraint_message_sequences[i], dtype=torch.float32)

            for length in range(length_max):
                output = self.forward(constraint_message_sequences[i][:, :(length+1)], constraint_message_scatter_degree[i][:, :(length+1)],
                                      constraint_message_parent_child_pairs[i][:constraint_message_parent_child_length[i][max(length-1, 0)]], constraint_message_parent_child_pairs[i][constraint_message_parent_child_length[i][max(length-1, 0)]:constraint_message_parent_child_length[i][length]],
                                      constraint_message_sibling_pairs[i][:constraint_message_sibling_length[i][max(length-1, 0)]], constraint_message_sibling_pairs[i][constraint_message_sibling_length[i][max(length-1, 0)]:constraint_message_sibling_length[i][length]], layer=2, active_spencoding=active_spencoding)
                # with torch.no_grad():
                output, active_spencoding = self.apply_constraints(output, constraint_message_counters[i][length], length, constraint_message_sequences[i][:, 1:(length+1)], constraint_message_scatter_degree[i][:,length][:,None], constraint_message_scatter_parent_where_seq[i][:, :(length+1)], layer=2, node_message_lengths=node_message_lengths, constraint_message_lengths=constraint_message_lengths)
                output = output/torch.sum(output, dim=1, keepdim=True)

                dist = torch.distributions.Categorical(output)
                log_p = dist.log_prob(constraint_message_sequences[i][:, length + 1]).unsqueeze(1)
                ent_p = dist.entropy().unsqueeze(1)

                # new_log_probs[:, length] = dist.log_prob(constraint_message_sequences[i][:, length+1])
                # c_entropy[:, length] = dist.entropy()
                idx = torch.tensor([length], device=new_log_probs.device)
                new_log_probs = new_log_probs.index_copy(1, idx, log_p)
                c_entropy = c_entropy.index_copy(1, idx, ent_p)

            constraint_message_new_log_probs.append(new_log_probs)
            constraint_message_entropy.append(c_entropy)

        # c2v
        for i in range(self.operators.T1):
            length_max = node_message_sequences[i].shape[1] - 1
            # new_log_probs = torch.zeros_like(node_message_sequences[i])
            new_log_probs = torch.full_like(node_message_sequences[i], fill_value=1.0, dtype=torch.float32)
            n_entropy = torch.zeros_like(node_message_sequences[i], dtype=torch.float32)# 

            for length in range(length_max):
                output = self.forward(node_message_sequences[i][:, :(length+1)], node_message_scatter_degree[i][:, :(length+1)],
                                      node_message_parent_child_pairs[i][:node_message_parent_child_length[i][max(length-1, 0)]], node_message_parent_child_pairs[i][node_message_parent_child_length[i][max(length-1, 0)]:node_message_parent_child_length[i][length]],
                                      node_message_sibling_pairs[i][:node_message_sibling_length[i][max(length-1, 0)]], node_message_sibling_pairs[i][node_message_sibling_length[i][max(length-1, 0)]:node_message_sibling_length[i][length]], layer=1, active_spencoding=active_spencoding)
                # with torch.no_grad():
                output, active_spencoding = self.apply_constraints(output, node_message_counters[i][length], length, node_message_sequences[i][:, 1:(length+1)], node_message_scatter_degree[i][:,length][:,None], node_message_scatter_parent_where_seq[i][:, :(length+1)], layer=1, node_message_lengths=node_message_lengths, constraint_message_lengths=constraint_message_lengths)
                output = output/torch.sum(output, dim=1, keepdim=True)

                dist = torch.distributions.Categorical(output)

                token = node_message_sequences[i][:, length+1]
                log_p = dist.log_prob(token)
                entropy_p = dist.entropy()

                # no in-place
                # with torch.no_grad():
                mask = torch.logical_and(token >= self.operators.scatter_begin, token < self.operators.scatter_end)
                mask = torch.logical_and(mask, node_message_scatter_degree[i][:, length] > 0)
                mask_p = torch.logical_or(mask, new_log_probs[:, length] <= 0)

                rows = mask.nonzero(as_tuple=True)[0]
                
                # new_log_probs[~mask_p, length] = log_p[~mask_p]
                # n_entropy[~mask_p, length] = entropy_p[~mask_p]
                new_log_probs = new_log_probs.clone()
                n_entropy = n_entropy.clone()
                new_log_probs[:, length] = torch.where(mask_p, new_log_probs[:, length], log_p)
                n_entropy[:, length] = torch.where(mask_p, n_entropy[:, length], entropy_p)
               
               # no in-place
                mask_lp  = torch.zeros_like(new_log_probs, dtype=torch.bool)
                mask_ent = torch.zeros_like(n_entropy, dtype=torch.bool)
                value_lp = torch.zeros_like(new_log_probs)
                value_ent = torch.zeros_like(n_entropy)

                for r in rows:
                    l = length+1
                    while(node_message_scatter_degree[i][r, l] > 1):
                        l += 1
                    l = l - length
                    for j in range(self.operators.T2):
                        if (l == constraint_message_lengths[j][r]) and (torch.equal(node_message_sequences[i][r, (length+1):(length+1+l)], constraint_message_sequences[j][r, 1:1+l])):
                            # new_log_probs[r, length:length+l] = constraint_message_new_log_probs[j][r, :l]
                            # n_entropy[r, length:length+l] = constraint_message_entropy[j][r, :l]
                            mask_lp[r, length:length+l] = True
                            mask_ent[r, length:length+l] = True
                            value_lp[r, length:length+l] = constraint_message_new_log_probs[j][r, :l]
                            value_ent[r, length:length+l] = constraint_message_entropy[j][r, :l]
                            break
                new_log_probs = torch.where(mask_lp, value_lp, new_log_probs)
                n_entropy = torch.where(mask_ent, value_ent, n_entropy)

            node_message_new_log_probs.append(new_log_probs)
            node_message_entropy.append(n_entropy)

        length_max = sequences.shape[1] - 1
        # new_log_probs = torch.ones_like(sequences, dtype=torch.float32)
        new_log_probs = torch.ones((sequences.shape[0], length_max), dtype=torch.float32)
        entropy = torch.zeros_like(new_log_probs, dtype=torch.float32)
        for length in range(length_max):
            output = self.forward(sequences[:,:(length+1)], scatter_degree[:,:(length+1)], parent_child_pairs[:parent_child_length[max(length-1,0)]], parent_child_pairs[parent_child_length[max(length-1,0)]:parent_child_length[length]], sibling_pairs[:sibling_length[max(length-1,0)]], sibling_pairs[sibling_length[max(length-1,0)]:sibling_length[length]], 0, active_spencoding=active_spencoding)
            # output[:, self.operators.variable_message_begin:self.operators.variable_message_end] = 0

            # Apply constraints and normalize distribution
            # with torch.no_grad():
            output, active_spencoding = self.apply_constraints(output, all_counters_list[length], length, sequences[:,1:(length+1)], scatter_degree=scatter_degree[:,length][:,None], scatter_parent_where_seq=scatter_parent_where_seq[:,:(length+1)], layer=0, node_message_lengths=node_message_lengths, constraint_message_lengths=constraint_message_lengths)
            output = output / torch.sum(output, dim=1, keepdim=True)

            dist = torch.distributions.Categorical(output)
            token = sequences[:, length+1]
            log_p = dist.log_prob(token)
            entropy_p = dist.entropy()

            # no in-place
            # with torch.no_grad():
            mask = torch.logical_and(token >= self.operators.scatter_begin, token < self.operators.scatter_end)
            mask_p = torch.logical_or(mask, new_log_probs[:, length] <= 0)
            # new_log_probs[~mask_p, length] = log_p[~mask_p]
            # entropy[~mask_p, length] = entropy_p[~mask_p]
            new_log_probs = new_log_probs.clone()
            entropy = entropy.clone()
            new_log_probs[:, length] = torch.where(mask_p, new_log_probs[:, length], log_p)
            entropy[:, length] = torch.where(mask_p, entropy[:, length], entropy_p)

            mask_m = torch.logical_and(mask, scatter_degree[:, length] < 1)
            rows = mask_m.nonzero(as_tuple=True)[0]

            # no in-place
            mask_lp  = torch.zeros_like(new_log_probs, dtype=torch.bool)
            mask_ent = torch.zeros_like(entropy, dtype=torch.bool)
            value_lp = torch.zeros_like(new_log_probs)
            value_ent = torch.zeros_like(entropy)

            for r in rows:
                l = length + 1
                while(scatter_degree[r, l] > 0):
                    l += 1
                l = l - length
                for j in range(self.operators.T1):
                    if (l == node_message_lengths[j][r]) and (torch.equal(sequences[r, length+1:length+1+l], node_message_sequences[j][r, 1:1+l])):
                        # new_log_probs[r, length:length+l] = node_message_new_log_probs[j][r, :l]
                        # entropy[r, length:length+l] = node_message_entropy[j][r, :l]
                        # no in-place
                        mask_lp[r, length:length+l] = True
                        mask_ent[r, length:length+l] = True
                        value_lp[r, length:length+l] = node_message_new_log_probs[j][r, :l]
                        value_ent[r, length:length+l] = node_message_entropy[j][r, :l]
                        break
            new_log_probs = torch.where(mask_lp, value_lp, new_log_probs)
            entropy = torch.where(mask_ent, value_ent, entropy)

        return entropy, new_log_probs
    
class TransformerDSOAgent_Multilayer(TransformerDSOAgent):
    def __init__(self, operators, layer_min_length:List[int], layer_max_length:List[int], layer_soft_length:List[int], two_sigma_square=16, d_model=64, num_heads=8, d_ff=256, num_layers=6, structural_encoding=True, layer_encoding=True, logits_parentsib=False, **kwargs):
        super(DSOAgent, self).__init__()
        self.operators, self.layer_min_length, self.layer_max_length, self.layer_soft_length, self.two_sigma_square = operators, layer_min_length, layer_max_length, layer_soft_length, two_sigma_square
        self.output_size = operators.operator_length
        self.transformer = TransformerDSOEncoder(self.output_size, operators.scatter_max_degree, max_length=layer_max_length[0], d_model=d_model, num_heads=num_heads, d_ff=d_ff, num_layers=num_layers, structural_encoding=structural_encoding, layer_encoding=layer_encoding, logits_parentsib=logits_parentsib)
        self.activation = nn.Softmax(dim=1)

    def apply_constraints(self, output, counters, length, sequences, scatter_degree, scatter_parent_where_seq, layer, message_lengths):
        min_boolen_mask = (counters + length) >= self.layer_min_length[layer]
        min_length_mask = torch.logical_or(self.operators.nonzero_arity_mask, min_boolen_mask)
        # output[~min_length_mask] = 0 # 把不满足最小长度限制的序列行中零元操作符概率置0
        output = output * min_length_mask.to(output.dtype)

        max_boolen_mask = (counters + length) < self.layer_max_length[layer]
        max_length_mask = torch.logical_or(self.operators.zero_arity_mask, max_boolen_mask)
        # output[~max_length_mask] = 0 # 把到达最后一个token的序列行中非零元操作符概率置0
        output = output * max_length_mask.to(output.dtype)

        if length == 0:
            # const
            const_mask = self.operators.const_mask.squeeze(0)
            output = output*(1-const_mask.to(output.dtype))

            if layer > 0:
                scatter_mask = self.operators.scatter_mask.squeeze(0)
                output = output*scatter_mask.to(output.dtype)
            else:
                # layer0 的第一个token，变量:v,layer0_message,运算符:math
                degree_0_mask = self.operators.v_mask.squeeze(0).clone() # 应该除了v,layer0_message和运算符其余为True
                degree_0_mask[self.operators.variable_message_begin[0]:self.operators.variable_message_begin[0]+self.operators.T[0]]=True
                scatter_mask_0 = self.operators.scatter_mask.squeeze(0)
                output = output * degree_0_mask.to(output.dtype)
                output = output * (1 - scatter_mask_0.to(output.dtype))
        
        else:
            # 屏蔽无变量情况
            last_counter_mask = (counters == 1)
            non_var_now_mask = torch.logical_not((sequences < self.operators.variable_end).any(dim=1, keepdim=True))
            last_token_and_no_var_mask = torch.logical_and(last_counter_mask, non_var_now_mask)
            const_and_last_token_and_no_var_mask = torch.logical_and(self.operators.const_mask, last_token_and_no_var_mask)
            # output[const_and_last_token_and_no_var_mask] = 0
            output = output*(1 - const_and_last_token_and_no_var_mask.to(output.dtype))

            # 屏蔽相邻逆运算
            last_token = sequences[:, -1]
            last_token_has_inverse = torch.where(self.operators.have_inverse[last_token])[0]
            last_token_inverse = self.operators.where_inverse[last_token[last_token_has_inverse]]
            mask = torch.ones_like(output)
            mask[last_token_has_inverse, last_token_inverse] = 0
            # output[last_token_has_inverse, last_token_inverse] = 0 # 屏蔽逆运算(上一个token可逆,则屏蔽相邻逆运算)
            output = output*mask.to(output.dtype)

            where_same_sub_tree = scatter_parent_where_seq[:, :-1] == scatter_parent_where_seq[:,-1:] # (512, 1) 判断当前节点是否与其父节点在同一子树中
            sub_tree_counter_is_degree_2 = torch.logical_and(torch.logical_and(sequences>=self.operators.arity_two_begin, sequences < self.operators.arity_two_end), where_same_sub_tree)
            sub_tree_counter_is_degree_0 = torch.logical_and(sequences < self.operators.arity_zero_end, where_same_sub_tree)
            sub_tree_counter = 1 + sub_tree_counter_is_degree_2.sum(dim=1, keepdim=True) - sub_tree_counter_is_degree_0.sum(dim=1, keepdim=True) # 当前子树还需要生成的tokens
            sub_tree_last_counter_mask = (sub_tree_counter == 1)

            max_layer = len(self.operators.T)
            if layer == max_layer:
                mask = torch.ones_like(output)
                mask[:, self.operators.variable_message_begin[0]:self.operators.variable_message_end] = 0
                output = output*mask.to(output.dtype)

                if layer%2 == 0:
                    non_message_passing = torch.logical_not(torch.logical_and(torch.logical_and(sequences>=self.operators.variable_variable_begin,
                                                                                     sequences<self.operators.variable_variable_end),
                                                                                     where_same_sub_tree).any(dim=1, keepdim=True))
                    last_token_and_non_message_passing = torch.logical_and(sub_tree_last_counter_mask, non_message_passing)
                    mp_mask = torch.logical_and(last_token_and_non_message_passing, self.operators.scatter_degree_2_mask)
                    output = output*(1 - mp_mask.to(output.dtype))
                elif layer%2 == 1:
                    non_message_passing = torch.logical_not(torch.logical_and(torch.logical_and(sequences>=self.operators.variable_constraint_begin,
                                                                                     sequences<self.operators.variable_constraint_end),
                                                                                     where_same_sub_tree).any(dim=1, keepdim=True))
                    last_token_and_non_message_passing = torch.logical_and(sub_tree_last_counter_mask, non_message_passing)
                    mp_mask = torch.logical_and(last_token_and_non_message_passing, self.operators.non_c_mask)
                    output = output*(1 - mp_mask.to(output.dtype))

                scatter_should_mask = (scatter_degree >= 1)
                scatter_mask = torch.logical_and(scatter_should_mask, self.operators.scatter_mask)
                # output[scatter_mask] = 0
                output = output*(1 - scatter_mask.to(output.dtype))
            
            elif layer > 0:
                # 限制出现变量的类型
                mask = torch.ones_like(output)
                mask[:, self.operators.variable_variable_end: self.operators.variable_message_begin[layer]] = 0
                mask[:, self.operators.variable_message_begin[layer]+self.operators.T[layer]:self.operators.variable_message_end] = 0
                output = output*mask.to(output.dtype)
                # if layer <= max_layer - 3:
                #     mask[:, self.operators.variable_message_begin[layer+2]:self.operators.variable_message_end] = 0
                #     output = output*mask.to(output.dtype)

                # 避免无效消息传递
                if layer%2 == 0: # c侧更新，要有v
                    non_message_passing = torch.logical_not(input=torch.logical_and(torch.logical_or(torch.logical_and(sequences>=self.operators.variable_variable_begin,
                                                                                                           sequences<self.operators.variable_variable_end),
                                                                                    torch.logical_and(sequences>=self.operators.variable_message_begin[layer],
                                                                                                           sequences<self.operators.variable_message_begin[layer]+self.operators.T[layer])),
                                                                                                           where_same_sub_tree).any(dim=1, keepdim=True))
                    last_token_and_non_message_passing = torch.logical_and(sub_tree_last_counter_mask, non_message_passing)
                    m_mask = self.operators.v_mask.clone()
                    m_mask[:, self.operators.variable_message_begin[layer]:self.operators.variable_message_begin[layer]+self.operators.T[layer]] = True
                    mp_mask = torch.logical_not(m_mask)
                    mp_mask = torch.logical_and(last_token_and_non_message_passing, mp_mask)
                    output = output*(1-mp_mask.to(output.dtype))
                elif layer%2 == 1: # v侧更新，要有c
                    non_message_passing = torch.logical_not(input=torch.logical_and(torch.logical_or(torch.logical_and(sequences>=self.operators.variable_constraint_begin,
                                                                                                           sequences<self.operators.variable_constraint_end),
                                                                                    torch.logical_and(sequences>=self.operators.variable_message_begin[layer],
                                                                                                           sequences<self.operators.variable_message_begin[layer]+self.operators.T[layer])),
                                                                                                           where_same_sub_tree).any(dim=1, keepdim=True))
                    last_token_and_non_message_passing = torch.logical_and(sub_tree_last_counter_mask, non_message_passing)
                    m_mask = self.operators.c_mask.clone()
                    m_mask[:, self.operators.variable_message_begin[layer]:self.operators.variable_message_begin[layer]+self.operators.T[layer]] = True
                    mp_mask = torch.logical_not(m_mask)
                    mp_mask = torch.logical_and(last_token_and_non_message_passing, mp_mask)
                    output = output*(1-mp_mask.to(output.dtype))

                # 禁止采样超长的更新节点
                remaining = self.layer_max_length[layer] - (length+counters).squeeze(1) # 当前层最大长度减去已生成长度
                mask = torch.ones_like(output)
                for idx in range(self.operators.T[layer]):
                    # if(constraint_length[idx].max() > remaining):
                    #     output[:, self.operators.variable_constraint_message_begin + idx] = 0.0
                    mask_constraint = (message_lengths[layer][idx] > remaining)
                    mask[mask_constraint, self.operators.variable_message_begin[layer]+idx] = 0.0
                    # output[mask_constraint, self.operators.variable_constraint_message_begin+idx] = 0.0
                output = output*mask.to(output.dtype)

                scatter_should_mask = (scatter_degree >= 1)
                scatter_mask = torch.logical_and(scatter_should_mask, self.operators.scatter_mask)
                # output[scatter_mask] = 0
                output = output*(1 - scatter_mask.to(output.dtype))
                
            else: # layer == 0
                mask = torch.ones_like(output)
                mask[:, self.operators.scatter_mask.squeeze(0)] = 0
                mask[:, :self.operators.variable_variable_begin] = 0
                mask[:, self.operators.variable_message_begin[0]+self.operators.T[0]:self.operators.variable_message_end] = 0
                output = output*mask.to(output.dtype)
                non_message_passing = torch.logical_not(torch.logical_and(torch.logical_and(sequences >= self.operators.variable_variable_begin,
                                                                                            sequences < self.operators.variable_message_begin[0]+self.operators.T[0]),
                                                                                            where_same_sub_tree).any(dim=1, keepdim=True))
                last_token_and_non_message_passing = torch.logical_and(sub_tree_last_counter_mask, non_message_passing)
                m_mask = self.operators.v_mask.clone()
                m_mask[:, self.operators.variable_message_begin[0]:self.operators.variable_message_begin[0]+self.operators.T[0]] = True
                mp_mask = torch.logical_not(m_mask)
                mp_mask = torch.logical_and(last_token_and_non_message_passing, mp_mask)
                output = output*(1-mp_mask.to(output.dtype))

                m = (scatter_degree < 1).squeeze(-1)[:, None]          # (B,1)
                b = (output[:, self.operators.variable_message_begin[0]:self.operators.variable_message_begin[0]+self.operators.T[0]].float().mean() - output[:, :].float().mean()) ** 2

                vv = slice(self.operators.variable_variable_begin, self.operators.variable_variable_end)
                vn = slice(self.operators.variable_message_begin[0], self.operators.variable_message_begin[0]+self.operators.T[0])

                output = output.clone()
                output = output.clone()
                output[:, vv] = torch.where(m, torch.clamp(output[:, vv] - b, min=0), output[:, vv])
                output[:, vn] = torch.where(m, output[:, vn] + b, output[:, vn])

                remaining = self.layer_max_length[layer] - (length+counters).squeeze(1)
                mask = torch.ones_like(output)
                # 禁止采样超长的更新节点
                remaining = self.layer_max_length[layer] - (length+counters).squeeze(1) # 当前层最大长度减去已生成长度
                mask = torch.ones_like(output)
                for idx in range(self.operators.T[layer]):
                    # if(constraint_length[idx].max() > remaining):
                    #     output[:, self.operators.variable_constraint_message_begin + idx] = 0.0
                    mask_constraint = (message_lengths[layer][idx] > remaining)
                    mask[mask_constraint, self.operators.variable_message_begin[layer]+idx] = 0.0
                    # output[mask_constraint, self.operators.variable_constraint_message_begin+idx] = 0.0
                output = output*mask.to(output.dtype)
                output = output*mask.to(output.dtype)
        return output
    
    def forward(self, raw_x, scatter_degree, parentchild_indices, parent_child_now, sibling_indices, sibling_now, layer):
        cur_length = raw_x.shape[1]-1
        output = self.transformer(raw_x, scatter_degree, parentchild_indices, parent_child_now, sibling_indices, sibling_now) # only return the last logits

        # ~ soft length constraint
        prior_vec = torch.zeros(self.output_size)

        if cur_length < self.layer_soft_length[layer]:
            prior_vec[self.operators.arity_zero_begin:self.operators.arity_zero_end] = - (self.layer_soft_length[layer] - cur_length) ** 2 / self.two_sigma_square
        elif cur_length > self.layer_soft_length[layer]:
            prior_vec[self.operators.arity_two_begin:self.operators.arity_two_end] = - (cur_length - self.layer_soft_length[layer]) ** 2 / self.two_sigma_square
        output = output + prior_vec[None, :]

        output = self.activation(output)
        return output
    
    @torch.no_grad() # @torch.no_grad()
    def sample_sequence_layer(self, n, message_sequences, message_lengths, message_log_probs, layer):
        max_len = self.layer_max_length[layer] + 3

        # sequences = torch.zeros((n,max_len), dtype=torch.long)
        sequences = torch.full((n, max_len), fill_value=-1)
        sequences[:,0] = self.output_size
        scatter_degree = torch.zeros_like(sequences)
        parent_child_pairs = torch.zeros((n*(max_len), 3), dtype=torch.long)
        parent_child_length = torch.zeros(max_len, dtype=torch.long)
        sibling_pairs = torch.zeros((n*(max_len), 3), dtype=torch.long)
        sibling_length = torch.zeros(max_len, dtype=torch.long)
        scatter_parent_where_seq = torch.full_like(sequences, fill_value=-1)

        length = 0
        all_lengths = torch.zeros(n, dtype=torch.long)
        sequence_mask = torch.ones(n, dtype=torch.bool)
        counters = torch.ones(n, 1, dtype=torch.long) # Number of tokens that must be sampled to complete expression
        # all_log_prob_list, all_counters_list = [], []
        all_counters_list = []
        all_log_probs = torch.zeros_like(sequences, dtype=torch.float32)

        while(sequence_mask.any()):

            output = self.forward(sequences[:, :(length+1)], scatter_degree[:, :(length+1)], parent_child_pairs[:parent_child_length[max(length-1, 0)]], parent_child_pairs[parent_child_length[max(length-1, 0)]:parent_child_length[length]], sibling_pairs[:sibling_length[max(length-1, 0)]], sibling_pairs[sibling_length[max(length-1, 0)]:sibling_length[length]], layer=layer)
            
            # Apply constraints and normalize distribution
            output = self.apply_constraints(output, counters, length, sequences[:,1:(length+1)], scatter_degree=scatter_degree[:,length][:,None], scatter_parent_where_seq=scatter_parent_where_seq[:,:(length+1)], layer=layer, message_lengths=message_lengths)
            output = output / (torch.sum(output, dim=1, keepdim=True) + 1e-6)

            # Sample from categorical distribution
            dist = torch.distributions.Categorical(output)
            token = dist.sample()
            log_prob_token = dist.log_prob(token)

            mask = (sequences[:, length+1] == -1)
            sequences[mask, length+1] = token[mask]
            all_log_probs[mask, length] = log_prob_token[mask]
            token[~mask] = sequences[~mask, length+1]

            # for l in reversed(range(len(self.operators.T))):
                # message_end = self.operators.variable_message_begin[l+1] if (l < len(self.operators.T)-1) else self.operators.variable_message_end
            if layer < len(self.operators.T):
                message_mask = torch.logical_and(token >= self.operators.variable_message_begin[layer], token < self.operators.variable_message_begin[layer]+self.operators.T[layer])
                message_mask = torch.logical_and(message_mask, sequence_mask)
                rows = message_mask.nonzero(as_tuple=True)[0]
                for i in rows:
                    ll = message_lengths[layer][token[i]-self.operators.variable_message_begin[layer]][i].item()
                    sequences[i, length+1:length+1+ll] = message_sequences[layer][token[i] - self.operators.variable_message_begin[layer]][i, 1:1+ll]
                    all_log_probs[i, length:length+ll] = message_log_probs[layer][token[i] - self.operators.variable_message_begin[layer]][i, :ll]

            token = sequences[:, length+1]
            length += 1
            all_lengths[sequence_mask] += 1

            # all_log_prob_list.append(log_prob_token)
            all_counters_list.append(counters)

            # Update counter
            counters = counters + (torch.logical_and(self.operators.arity_one_begin<=token, token<self.operators.arity_one_end).long() \
                        + torch.logical_and(self.operators.arity_two_begin<=token, token<self.operators.arity_two_end).long() * 2 - 1)[:, None]
            sequence_mask = torch.logical_and(sequence_mask, counters.squeeze(1) > 0)

            # Compute next parent and sibling; assemble next input tensor
            self.get_parent_sibling(n, sequences[:,1:], length-1, sequence_mask, scatter_degree, scatter_parent_where_seq, parent_child_pairs, parent_child_length, sibling_pairs, sibling_length) # update all info
            assert scatter_degree.max() <= self.operators.scatter_max_degree, f"scatter_degree out of range: max={scatter_degree.max()}, allowed={self.operators.scatter_max_degree}"

        assert all_lengths.min() >= self.layer_min_length[layer] and all_lengths.max() <= self.layer_max_length[layer]+1 and all_lengths.max() == length

        return sequences[:,:(length+1)], all_lengths, all_log_probs[:, :length], (scatter_degree[:,:(length+1)], all_counters_list, scatter_parent_where_seq[:,:(length+1)], parent_child_pairs[:parent_child_length[length]], parent_child_length[:length+1], sibling_pairs[:sibling_length[length]], sibling_length[:length+1])
    
    def sample_sequence_eval(self, n):
        # node_message_sequences, node_message_lengths, node_message_log_probs = [], [], []
        # constraint_message_sequences, constraint_message_lengths, constraint_message_log_probs = [], [], []
        message_sequences = [[] for _ in range(len(self.operators.T))]
        message_lengths = [[] for _ in range(len(self.operators.T))]
        message_log_probs = [[] for _ in range(len(self.operators.T))]

        # 打印子表达式需要
        # node_message_info_lists, constraint_message_info_lists = [[] for _ in range(7)], [[] for _ in range(7)]
        message_info_lists = []
        for l in range(len(self.operators.T)):
            message_info_lists.append([[] for _ in range(7)])
        
        for l in reversed(range(len(self.operators.T))):
            for i in range(self.operators.T[l]):
                sequences, all_lengths, log_probs, info_list = self.sample_sequence_layer(n, message_sequences, message_lengths, message_log_probs, l+1)
                message_sequences[l].append(sequences)
                message_lengths[l].append(all_lengths)
                message_log_probs[l].append(log_probs)
                for k, info in enumerate(info_list):
                    message_info_lists[l][k].append(info)

        sequences, all_lengths, log_probs, (scatter_degree, all_counters_list, scatter_parent_where_seq,
                                            parent_child_pairs, parent_child_length, sibling_pairs, sibling_length) = self.sample_sequence_layer(n, message_sequences, message_lengths, message_log_probs, 0)
        return sequences, all_lengths, log_probs, (scatter_degree, all_counters_list, scatter_parent_where_seq,
                                                   parent_child_pairs, parent_child_length, sibling_pairs, sibling_length),(message_sequences, message_lengths, message_log_probs, message_info_lists)
    
    def sample_sequence_train(self, sequences, info_lists, message_sequences, message_lengths, message_info_lists):
        scatter_degree, all_counters_list, scatter_parent_where_seq, parent_child_pairs, parent_child_length, sibling_pairs, sibling_length = info_lists

        message_new_log_probs = [[] for _ in range(len(self.operators.T))]
        message_entropy = [[] for _ in range(len(self.operators.T))]
        for l in reversed(range(len(self.operators.T))):
            for i in range(self.operators.T[l]):
                length_max = message_sequences[l][i].shape[1] - 1
                new_log_probs = torch.full_like(message_sequences[l][i], fill_value=1.0, dtype=torch.float32)
                entropy = torch.zeros_like(message_sequences[l][i], dtype=torch.float32)

                for length in range(length_max): # parent_child_pairs[i]=message_info_lists[l][3][i],parent_child_length[i]=message_info_lists[l][4][i],sibling_pairs[i]=message_info_lists[l][5][i],sibling_length[i]=message_info_lists[l][6][i]
                    output = self.forward(message_sequences[l][i][:, :(length+1)], message_info_lists[l][0][i][:, :(length+1)],
                                      message_info_lists[l][3][i][:message_info_lists[l][4][i][max(length-1, 0)]], message_info_lists[l][3][i][message_info_lists[l][4][i][max(length-1, 0)]:message_info_lists[l][4][i][length]],
                                      message_info_lists[l][5][i][:message_info_lists[l][6][i][max(length-1, 0)]], message_info_lists[l][5][i][message_info_lists[l][6][i][max(length-1, 0)]:message_info_lists[l][6][i][length]], layer=l+1)
                    # with torch.no_grad():
                    # message_scatter_degree[i]=message_info_lists[l][0][i], message_counters[i]=message_info_lists[l][1][i], message_scatter_parent_where_seq[i]=message_info_lists[l][2][i]
                    output = self.apply_constraints(output, message_info_lists[l][1][i][length], length, message_sequences[l][i][:, 1:(length+1)], message_info_lists[l][0][i][:,length][:,None], message_info_lists[l][2][i][:, :(length+1)], layer=l+1, message_lengths=message_lengths)
                    output = output/(torch.sum(output, dim=1, keepdim=True) + 1e-6)

                    dist = torch.distributions.Categorical(output)
                    token = message_sequences[l][i][:, length+1]
                    log_p = dist.log_prob(token)
                    ent_p = dist.entropy()

                    # no in-place
                    # with torch.no_grad():
                    mask = torch.logical_and(token >= self.operators.scatter_begin, token < self.operators.scatter_end)
                    mask = torch.logical_and(mask, message_info_lists[l][0][i][:, length] > 0)
                    mask_p = torch.logical_or(mask, new_log_probs[:, length] <= 0)

                    rows = mask.nonzero(as_tuple=True)[0]
                    
                    # new_log_probs[~mask_p, length] = log_p[~mask_p]
                    # n_entropy[~mask_p, length] = entropy_p[~mask_p]
                    new_log_probs = new_log_probs.clone()
                    entropy = entropy.clone()
                    new_log_probs[:, length] = torch.where(mask_p, new_log_probs[:, length], log_p)
                    entropy[:, length] = torch.where(mask_p, entropy[:, length], ent_p)

                    # no in-place
                    mask_lp  = torch.zeros_like(new_log_probs, dtype=torch.bool)
                    mask_ent = torch.zeros_like(entropy, dtype=torch.bool)
                    value_lp = torch.zeros_like(new_log_probs)
                    value_ent = torch.zeros_like(entropy)

                    for r in rows:
                        ll = length+1
                        while(message_info_lists[l][0][i][r, ll] > 1):
                            ll += 1
                        ll = ll - length
                        # end = l + 3 if (l < len(self.operators.T)-2) else (l+2)
                        # for k in range(l+1, end): # 仅保留k=l+1
                        for j in range(self.operators.T[l+1]):
                            if (ll == message_lengths[l+1][j][r]) and (torch.equal(message_sequences[l][i][r, (length+1):(length+1+ll)], message_sequences[l+1][j][r, 1:1+ll])):
                                # new_log_probs[r, length:length+l] = constraint_message_new_log_probs[j][r, :l]
                                # n_entropy[r, length:length+l] = constraint_message_entropy[j][r, :l]
                                mask_lp[r, length:length+ll] = True
                                mask_ent[r, length:length+ll] = True
                                value_lp[r, length:length+ll] = message_new_log_probs[l+1][j][r, :ll]
                                value_ent[r, length:length+ll] = message_entropy[l+1][j][r, :ll]
                                break
                    
                    new_log_probs = torch.where(mask_lp, value_lp, new_log_probs)
                    entropy = torch.where(mask_ent, value_ent, entropy)
                message_new_log_probs[l].append(new_log_probs)
                message_entropy[l].append(entropy)
        
        length_max = sequences.shape[1] - 1
        # new_log_probs = torch.ones_like(sequences, dtype=torch.float32)
        new_log_probs = torch.ones((sequences.shape[0], length_max), dtype=torch.float32)
        entropy = torch.zeros_like(new_log_probs, dtype=torch.float32)
        for length in range(length_max):
            output = self.forward(sequences[:,:(length+1)], scatter_degree[:,:(length+1)], parent_child_pairs[:parent_child_length[max(length-1,0)]], parent_child_pairs[parent_child_length[max(length-1,0)]:parent_child_length[length]], sibling_pairs[:sibling_length[max(length-1,0)]], sibling_pairs[sibling_length[max(length-1,0)]:sibling_length[length]], 0)
            # output[:, self.operators.variable_message_begin:self.operators.variable_message_end] = 0

            # Apply constraints and normalize distribution
            # with torch.no_grad():
            output = self.apply_constraints(output, all_counters_list[length], length, sequences[:,1:(length+1)], scatter_degree=scatter_degree[:,length][:,None], scatter_parent_where_seq=scatter_parent_where_seq[:,:(length+1)], layer=0, message_lengths=message_lengths)
            output = output / (torch.sum(output, dim=1, keepdim=True) + 1e-6)

            dist = torch.distributions.Categorical(output)
            token = sequences[:, length+1]
            log_p = dist.log_prob(token)
            entropy_p = dist.entropy()

            # no in-place
            # with torch.no_grad():
            mask = torch.logical_and(token >= self.operators.scatter_begin, token < self.operators.scatter_end)
            mask_p = torch.logical_or(mask, new_log_probs[:, length] <= 0)
            # new_log_probs[~mask_p, length] = log_p[~mask_p]
            # entropy[~mask_p, length] = entropy_p[~mask_p]
            new_log_probs = new_log_probs.clone()
            entropy = entropy.clone()
            new_log_probs[:, length] = torch.where(mask_p, new_log_probs[:, length], log_p)
            entropy[:, length] = torch.where(mask_p, entropy[:, length], entropy_p)

            mask_m = torch.logical_and(mask, scatter_degree[:, length] < 1)
            rows = mask_m.nonzero(as_tuple=True)[0]

            # no in-place
            mask_lp  = torch.zeros_like(new_log_probs, dtype=torch.bool)
            mask_ent = torch.zeros_like(entropy, dtype=torch.bool)
            value_lp = torch.zeros_like(new_log_probs)
            value_ent = torch.zeros_like(entropy)

            for r in rows:
                l = length + 1
                while(scatter_degree[r, l] > 0):
                    l += 1
                l = l - length
                for j in range(self.operators.T[0]):
                    if (l == message_lengths[0][j][r]) and (torch.equal(sequences[r, length+1:length+1+l], message_sequences[0][j][r, 1:1+l])):
                        # new_log_probs[r, length:length+l] = node_message_new_log_probs[j][r, :l]
                        # entropy[r, length:length+l] = node_message_entropy[j][r, :l]
                        # no in-place
                        mask_lp[r, length:length+l] = True
                        mask_ent[r, length:length+l] = True
                        value_lp[r, length:length+l] = message_new_log_probs[0][j][r, :l]
                        value_ent[r, length:length+l] = message_entropy[0][j][r, :l]
                        break
            new_log_probs = torch.where(mask_lp, value_lp, new_log_probs)
            entropy = torch.where(mask_ent, value_ent, entropy)

        return entropy, new_log_probs   