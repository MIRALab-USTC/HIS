import torch
from torch.optim import Adam
import utils.logger as logger
import settings.consts as consts
import numpy as np

def normalize_meanstd(returns): # 使用均值和标准差进行归一化
    return (returns - returns.mean()) / (returns.std() + consts.SAFE_EPSILON)

def normalize_minmax(returns): # 使用最小值和最大值进行归一化
    min_return = returns.min()
    delta = returns.max() - min_return
    if delta < consts.SAFE_EPSILON:
        delta = 1.
    return (returns - min_return) / delta

class PPOAlgo(): # 基于 PPO 算法的强化学习训练器
    def __init__(self,
                 agent, # 待优化的符号表达式生成模型

                 lr_actor=4e-4,
                 K_epochs=8, # PPO 算法的迭代次数
                 eps_clip=0.2, # PPO 算法的裁剪范围
                 entropy_coef=5e-3, # 熵正则化系数
                 entropy_gamma=0.8, # 熵正则化的衰减系数
                 entropy_decrease=False, # 是否逐渐减少熵正则化系数
                 lr_decrease=False, # 是否逐渐减少学习率
                 decrease_period=1000, # 学习率和熵系数衰减的周期
                 record_num_half=1, 

                 return_norm = "minmax",
                 detailed=consts.DETAILED_LOG, # 是否记录详细日志
                 detailed_freq=consts.DETAILED_LOG_FREQ, # 详细日志记录频率
                 is_tensorboard=True,
                 train_algo = "GRPO+PPO"
                 ):
        self.agent = agent

        self.K_epochs = K_epochs
        self.clip_low, self.clip_high = 1 - eps_clip, 1 + eps_clip

        self.optimizer = Adam([
                        {'params': agent.parameters(), 'lr': lr_actor},
                    ])
        
        self.entropy_coef = entropy_coef
        self.entropy_gamma = entropy_gamma

        self.entropy_decrease, self.lr_decrease = entropy_decrease, lr_decrease
        self.decrease_period = decrease_period
        self.record_num_half = record_num_half
        
        self.return_norm_func = globals()[f"normalize_{return_norm}"]

        self.detailed = detailed
        self.detailed_freq = detailed_freq
        
        self.env_step = 1

        self.train_algo = train_algo

    def train(self,
              sequences, all_lengths, log_probs, index_useful, info_lists, returns, train_iter,
              use_layer_learning=False, node_message_sequences=None, node_message_lengths=None, node_message_info_lists=None, constraint_message_sequences=None, constraint_message_lengths=None, constraint_message_info_lists=None,
              use_multi_layer=False, message_sequences=None, message_lengths=None, message_info_lists=None):
        torch.cuda.empty_cache()
        detailed_log = (self.detailed and (train_iter % self.detailed_freq == 0))
        if detailed_log:
            detailed_dict_list = []

        if (train_iter+1) % self.decrease_period == 0:
            if self.entropy_decrease:
                self.entropy_coef *= 0.8
            if self.lr_decrease:
                for g in self.optimizer.param_groups:
                    g["lr"] *= 0.8
        processed_advantages = self.return_norm_func(returns)

        for i in range(self.K_epochs):
            # use layer learning
            if not use_layer_learning and not use_multi_layer:
                new_entropies, new_log_probs = self.agent.sample_sequence_train(sequences, info_lists)
            elif use_layer_learning:
                new_entropies, new_log_probs = self.agent.sample_sequence_train(sequences, info_lists, node_message_sequences, node_message_lengths, node_message_info_lists, constraint_message_sequences, constraint_message_lengths, constraint_message_info_lists)
            else: # use_multi_layer
                new_entropies, new_log_probs = self.agent.sample_sequence_train(sequences, info_lists, message_sequences, message_lengths, message_info_lists)
            log_ratios = new_log_probs - log_probs

            joint_prob_ratios = (index_useful * log_ratios).sum(dim=1) #index_useful: 一个掩码，用于指示哪些数据是有效的。
            joint_ratios = joint_prob_ratios.exp()

            sign_advantage_index = torch.sign(processed_advantages)
            clipped_ratios = torch.min(joint_ratios * sign_advantage_index, torch.clamp(joint_ratios, self.clip_low, self.clip_high) * sign_advantage_index) * sign_advantage_index

            loss_actor = 0
            if self.train_algo == "GRPO":
                loss_actor = processed_advantages.mean()
            elif self.train_algo == "GRPO+PPO":
                loss_actor = (clipped_ratios * processed_advantages).mean()

            entropy_gamma = torch.pow(self.entropy_gamma, torch.arange(new_entropies.shape[1]))[None, :]
            loss_entropy = (index_useful * new_entropies * entropy_gamma).sum(dim=1).mean()

            loss = - loss_actor - self.entropy_coef * loss_entropy

            # with torch.autograd.set_detect_anomaly(True):
            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if detailed_log and (i<self.record_num_half or i>=self.K_epochs - self.record_num_half):
                detailed_dict = dict(
                    processed_advantages=processed_advantages,
                    joint_ratios=joint_ratios,
                    clipped_ratios=clipped_ratios,
                    log_probs=log_probs,
                    all_lengths=all_lengths
                )
                detailed_dict = {k:v.detach().cpu().numpy() for k,v in detailed_dict.items()}

                detailed_dict_list.append(detailed_dict)

        ratios = log_ratios.exp()
        with torch.no_grad():
            record_dict = dict(
                loss_actor=loss_actor,
                loss_entropy=loss_entropy,
                loss=loss,
                clip_frac=((joint_ratios - clipped_ratios).abs() > consts.SAFE_EPSILON).sum() / len(joint_ratios),
                approx_KL=(((ratios-1) - log_ratios) * index_useful).sum() / all_lengths.sum(),
                average_len=torch.mean(all_lengths.float()),
                std_len=torch.std(all_lengths.float())
            )
        record_dict = {f"rl_algo/{k}":v.item() for k,v in record_dict.items()}
        if detailed_log:
            self.detailed_log(detailed_dict_list)

        return record_dict

    def detailed_log(self, detailed_dict_list):
        for ith, detailed_dict in enumerate(detailed_dict_list):
            logger.log_hist({f"rl_algo/{ith}th_epoch/processed_advantages": detailed_dict["processed_advantages"],
                                f"rl_algo/{ith}th_epoch/joint_ratios": detailed_dict["joint_ratios"],
                                f"rl_algo/{ith}th_epoch/clipped_ratios": detailed_dict["clipped_ratios"],
                                f"env_step": self.env_step
                                })
        self.env_step+=1
