import torch
from torch import nn
from torch import functional as F
from math import log

nn.Transformer
class TransformerDSOEncoder(nn.Module):
    def __init__(self, vocabulary_size, scatter_max_degree, max_length, d_model=64, num_heads=8, d_ff=256, num_layers=6, structural_encoding=True, layer_encoding=True, logits_parentsib=False):
        super().__init__()
        self.num_layers = num_layers
        self.structural_encoding = structural_encoding
        self.layer_encoding = layer_encoding
        self.logits_parentsib = logits_parentsib
        self.vocabulary_embedding = nn.Embedding(vocabulary_size+1, d_model) # nn.embedding随机初始化vocabulary_size+1个d_model维向量
        self.position_encoder = PositionalEncoding(d_model, max_length+2)

        if self.structural_encoding:
            self.scatter_degree_embedding = nn.Embedding(scatter_max_degree+1, d_model)
            self.relation_encoder = nn.Parameter(data=torch.rand(3), requires_grad=True)
            self.active_relation_encoder = nn.Embedding(2, d_model)
            self.active_special_encoder = nn.Embedding(1, d_model)

        self.self_attention = nn.ModuleList([MultiheadAttention(d_model, num_heads) for _ in range(num_layers)])
        self.feed_forward = nn.ModuleList([FeedForward(d_model, d_ff) for _ in range(num_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.logits = nn.Linear(d_model, vocabulary_size, bias=False)
        self._parent_sibling = torch.arange(2)

    def forward(self, raw_x, scatter_degree, parentchild_indices, parent_child_now, sibling_indices, sibling_now, active_spencoding=None):

        x = self.vocabulary_embedding(raw_x) # (512, len+1) 到 (512, len+1, 32)
        x = self.position_encoder(x) # (512, len+1, 32)

        if self.structural_encoding:
            if self.layer_encoding:
                x = x + self.scatter_degree_embedding(scatter_degree) # (512, len+1, 32) 到 (512, len+1, 32)
            if active_spencoding is not None and active_spencoding.any():
                special_vec = self.active_special_encoder.weight[0]
                idx = active_spencoding.nonzero(as_tuple=False).squeeze(1)
                x[idx, -1] += special_vec
            active_parent_sibling_embedding = self.active_relation_encoder(self._parent_sibling) # (2, 32)
            x[parent_child_now[:,0], parent_child_now[:,1]] += active_parent_sibling_embedding[0] # 加到x的父节点 # parent_child_now每个元素是一个三元组，比如(0, 1, 2)分别表示第0行表达式正在生成的token的父节点位置和自身位置(构成一对父子关系)
            x[sibling_now[:,0], sibling_now[:,1]] += active_parent_sibling_embedding[1] # 加到x的兄弟节点

            spatial = parentchild_indices, sibling_indices, self.relation_encoder

        else:
            spatial = None

        for i in range(self.num_layers):
            # Self-attention
            x = self.self_attention[i](x, x, x, spatial) + x
            x = self.layer_norms[i](x)
            
            # Feed forward
            x = self.feed_forward[i](x) + x
            x = self.layer_norms[i](x)
        
        if self.logits_parentsib:
            if parent_child_now.any():
                logits_input = x[parent_child_now[:, 0], parent_child_now[:, 1]]
                if (sibling_now[:,0] > 0).any():
                    logits_input[sibling_now[:, 0]] = (logits_input[sibling_now[:,0]] + x[sibling_now[:,0], sibling_now[:,1]])/2
            else:
                logits_input = x[:, -1,...]
        else:
            logits_input = x[:,-1,...]
        # x = self.logits(x[:,-1,...])
        x = self.logits(logits_input)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[0, :x.size(1), :] # pe=(1, max_len, d_model)
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiheadAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_per_head = self.d_model // self.num_heads
        self.norm_value = torch.sqrt(torch.tensor(self.d_per_head).float())
        
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

        self.activation = nn.Softmax(dim=-1)
        
    def forward(self, query, key, value, spatial):
        batch_size = query.size(0)
        
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)
        
        query = query.view(batch_size, -1, self.num_heads, self.d_per_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.d_per_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.d_per_head).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.norm_value

        if spatial is not None:
            parentchild_indices, sibling_indices, embedings = spatial

            scores[parentchild_indices[:,0],:,parentchild_indices[:,1],parentchild_indices[:,2]] += embedings[0]
            scores[parentchild_indices[:,0],:,parentchild_indices[:,2],parentchild_indices[:,1]] += embedings[1]
            scores[sibling_indices[:,0],:,sibling_indices[:,1],sibling_indices[:,2]] += embedings[2]
            scores[sibling_indices[:,0],:,sibling_indices[:,2],sibling_indices[:,1]] += embedings[2]

        attention = self.activation(scores)
        
        x = torch.matmul(attention, value)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        x = self.output_linear(x)
        
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
                
    def forward(self, x):        
        return self.model(x)