import settings.consts as consts
from os import path as osp
import hydra # 用于配置管理的库，支持从YAML文件中加载配置
from omegaconf import OmegaConf # 用于处理配置文件的库，支持嵌套配置和配置解析

from train_utils_gs4co import TrainDSOAgent
import utils.utilities as utilities


@hydra.main(config_path='settings', config_name='train_gs4co', version_base=None) 
def main(conf): 
    log_dir = utilities.get_log_dir(exp_type="train_graph", instance_type=conf.instance_kwargs.instance_type, exp_name=conf.exp_name) 

    for i in range(conf.exp_num):
        new_conf = OmegaConf.to_container(conf, resolve=True)
        train_kwargs, instance_kwargs, expression_kwargs, dso_agent_kwargs, rl_algo_kwargs= new_conf["train_kwargs"], new_conf["instance_kwargs"], new_conf["expression_kwargs"], new_conf["dso_agent_kwargs"], new_conf["rl_algo_kwargs"] 
        print(new_conf["train_data_loader"])
        logdir, _ = utilities.initial_logger_and_seed(log_dir, i, new_conf, original_seed=train_kwargs['seed'])
        train_agent = TrainDSOAgent(**train_kwargs, instance_kwargs=instance_kwargs, expression_kwargs=expression_kwargs, dso_agent_kwargs=dso_agent_kwargs, rl_algo_kwargs=rl_algo_kwargs, train_data_loader=new_conf["train_data_loader"], test_data_loader=new_conf["test_data_loader"]) 
        train_agent.process(train_kwargs["use_layer_learning"], train_kwargs["use_multi_layer"]) 
        del train_agent 

if __name__ == "__main__":
    main()
