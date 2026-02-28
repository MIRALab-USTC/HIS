"""
File adapted from https://github.com/ds4dm/learn2branch
"""
import os
import argparse
import pickle
import glob
import shutil
import gzip
import numpy as np
import multiprocessing as mp

import pyscipopt as scip
import utils.utilities as utilities

class VanillaFullstrongBranchingDataCollector(scip.Branchrule): # 自定义分支策略类
    """
    Implements branching policy to be used by SCIP such that data collection required for hybrid models is embedded in it.
    """
    def __init__(self, rng, query_expert_prob=0.60):
        self.khalil_root_buffer = {} # 用于缓存 Khalil 特征
        self.obss = [] # 存储每个节点的观测数据
        self.targets = [] # 存储每个节点的目标变量
        self.obss_feats = [] # 存储每个节点的额外特征
        self.exploration_policy = "pscost" # 默认的探索策略
        self.query_expert_prob = query_expert_prob # 调用全强分支策略的概率
        self.rng = rng # 随机数生成器
        self.iteration_counter = 0 # 迭代计数器

    def branchinit(self): # 初始化分支策略的状态变量
        self.ndomchgs = 0
        self.ncutoffs = 0
        self.khalil_root_buffer = {}

    def branchexeclp(self, allowaddcons): # 决定在当前节点选择哪个变量进行分支，记录数据，并执行分支操作。
        self.iteration_counter += 1

        query_expert = self.rng.rand() < self.query_expert_prob
        if query_expert or self.model.getNNodes() == 1:
            candidate_vars, *_ = self.model.getPseudoBranchCands()
            candidate_mask = [var.getCol().getLPPos() for var in candidate_vars]

            state = utilities.extract_state(self.model)
            state_khalil = utilities.extract_khalil_variable_features(self.model, candidate_vars, self.khalil_root_buffer)

            result = self.model.executeBranchRule('vanillafullstrong', allowaddcons)
            cands_, scores, npriocands, bestcand = self.model.getVanillafullstrongData()
            best_var = cands_[bestcand]

            self.add_obs(best_var, (state, state_khalil), (cands_, scores))
            if self.model.getNNodes() == 1:
                self.state = [state, state_khalil, self.obss[0]]

            self.model.branchVar(best_var)
            result = scip.SCIP_RESULT.BRANCHED
        else:
            result = self.model.executeBranchRule(self.exploration_policy, allowaddcons)

        # fair node counting
        if result == scip.SCIP_RESULT.REDUCEDDOM:
            self.ndomchgs += 1
        elif result == scip.SCIP_RESULT.CUTOFF:
            self.ncutoffs += 1

        return {'result':result}

    def add_obs(self, best_var, state_, cands_scores=None): # 在分支定界过程中添加观察数据。包括变量特征、约束特征、边特征以及目标值（即最佳分支变量的索引）
        """
        Adds sample to the `self.obs` to be processed later at the end of optimization.

        Parameters
        ----------
            best_var : pyscipopt.Variable
                object representing variable in LP
            state_ : tuple
                extracted features of constraints and variables at a node
            cands_scores : np.array
                scores of each of the candidate variable on which expert policy was executed

        Return
        ------
        (bool): True if sample is added succesfully. False o.w.
        """
        if self.model.getNNodes() == 1:
            self.obss = []
            self.targets = []
            self.obss_feats = []
            self.map = sorted([x.getCol().getIndex() for x in self.model.getVars(transformed=True)])

        cands, scores = cands_scores
        # Do not record inconsistent scores. May happen if SCIP was early stopped (time limit).
        if any([s < 0 for s in scores]):
            return False

        state, state_khalil = state_
        var_features = state[2]['values']
        cons_features = state[0]['values']
        edge_features = state[1]

        # add more features to variables
        cands_index = [x.getCol().getIndex() for x in cands]
        khalil_features = -np.ones((var_features.shape[0], state_khalil.shape[1]))
        cand_ind = np.zeros((var_features.shape[0],1))
        khalil_features[cands_index] = state_khalil
        cand_ind[cands_index] = 1
        var_features = np.concatenate([var_features, khalil_features, cand_ind], axis=1)

        tmp_scores = -np.ones(len(self.map))
        if scores:
            tmp_scores[cands_index] = scores

        self.targets.append(best_var.getCol().getIndex())
        self.obss.append([var_features, cons_features, edge_features])
        depth = self.model.getCurrentNode().getDepth()
        self.obss_feats.append({'depth':depth, 'scores':np.array(tmp_scores), 'iteration': self.iteration_counter})

        return True

# 从给定的实例中收集样本数据：初始化SCIP模型、设置参数、执行优化过程并记录分支决策过程中产生的观察数据
def make_samples(in_queue, out_queue):
    """
    in_queue: 输入队列，包含任务信息（如：episode编号、实例路径、随机种子等）。
    out_queue: 输出队列，用于发送任务状态和结果。
    Worker loop: fetch an instance, run an episode and record samples.

    Parameters
    ----------
    in_queue : multiprocessing.Queue
        Input queue from which orders are received.
    out_queue : multiprocessing.Queue
        Output queue in which to send samples.
    """
    while True: # 无限循环等待接收新任务，每次从输入队列获取一个任务，包含任务所需的所有参数
        try:
            episode, instance, seed, time_limit, outdir, rng = in_queue.get()

            print(f"[Worker {os.getpid()}] Processing episode {episode}, instance {instance}...")

            # 创建一个新的SCIP模型实例，并设置其参数，包括静默模式、读取问题实例文件、初始化参数（使用指定的随机种子）、时钟类型、时间限制和节点限制
            m = scip.Model()
            m.setIntParam('display/verblevel', 0)
            m.readProblem(f'{instance}')
            utilities.init_scip_params(m, seed=seed)
            m.setIntParam('timing/clocktype', 2)
            m.setRealParam('limits/time', time_limit)
            m.setLongintParam('limits/nodes', node_limit)

            # 创建一个自定义的分支规则实例 VanillaFullstrongBranchingDataCollector 并将其添加到模型中，以便在分支决策时收集数据
            branchrule = VanillaFullstrongBranchingDataCollector(rng, node_record_prob)
            m.includeBranchrule(
                branchrule=branchrule,
                name="Sampling branching rule", desc="",
                priority=666666, maxdepth=-1, maxbounddist=1)

            # 设置一系列分支策略参数以控制分支行为
            m.setBoolParam('branching/vanillafullstrong/integralcands', True)
            m.setBoolParam('branching/vanillafullstrong/scoreall', True)
            m.setBoolParam('branching/vanillafullstrong/collectscores', True)
            m.setBoolParam('branching/vanillafullstrong/donotbranch', True)
            m.setBoolParam('branching/vanillafullstrong/idempotent', True)

            # 发送开始通知
            out_queue.put({
                "type":'start',
                "episode":episode,
                "instance":instance,
                "seed": seed
            })

            m.optimize()
            print(f"[Worker {os.getpid()}] Episode {episode} done.")
            
            # data storage - root and node data are saved separately.
            # node data carries a reference to the root filename.
            # 收集和保存数据
            if m.getNNodes() >= 1 and len(branchrule.obss) > 0 :
                filenames = []
                max_depth = max(x['depth'] for x in branchrule.obss_feats)
                stats = {'nnodes':m.getNNodes(), 'time':m.getSolvingTime(), 'gap':m.getGap(), 'nobs':len(branchrule.obss)}

                # prepare root data
                sample_state, sample_khalil_state, root_obss = branchrule.state
                sample_cand_scores = branchrule.obss_feats[0]['scores']
                sample_cands = np.where(sample_cand_scores != -1)[0]
                sample_cand_scores = sample_cand_scores[sample_cands]
                cand_choice = np.where(sample_cands == branchrule.targets[0])[0][0]

                root_filename = f"{outdir}/sample_root_0_{episode}.pkl"

                filenames.append(root_filename)
                with gzip.open(root_filename, 'wb') as f:
                    pickle.dump({
                        'type':'root',
                        'episode':episode,
                        'instance': instance,
                        'seed': seed,
                        'stats': stats,
                        'root_state': [sample_state, sample_khalil_state, sample_cands, cand_choice, sample_cand_scores],
                        'obss': [branchrule.obss[0], branchrule.targets[0], branchrule.obss_feats[0], None],
                        'max_depth': max_depth
                        }, f)

                # node data
                for i in range(1, len(branchrule.obss)):
                    iteration_counter = branchrule.obss_feats[i]['iteration']
                    filenames.append(f"{outdir}/sample_node_{iteration_counter}_{episode}.pkl")
                    with gzip.open(filenames[-1], 'wb') as f:
                        pickle.dump({
                            'type' : 'node',
                            'episode':episode,
                            'instance': instance,
                            'seed': seed,
                            'stats': stats,
                            'root_state': f"{outdir}/sample_root_0_{episode}.pkl",
                            'obss': [branchrule.obss[i], branchrule.targets[i], branchrule.obss_feats[i], None],
                            'max_depth': max_depth
                        }, f)

                out_queue.put({
                    "type": "done",
                    "episode": episode,
                    "instance": instance,
                    "seed": seed,
                    "filenames":filenames,
                    "nnodes":len(filenames),
                })


            # 清理资源
            m.freeProb()
        except Exception as e:
            out_queue.put({
                "type": "error",
                "message": str(e),
                "episode": episode,
                "instance": instance,
                "seed": seed
            })
            print(f"[Worker {os.getpid()}] Error processing episode {episode}: {str(e)}")

def send_orders(orders_queue, instances, seed, time_limit, outdir, start_episode): # 任务（即实例解决请求）放入到 orders_queue 中供工作进程处理
    """
    Worker loop: fetch an instance, run an episode and record samples.

    Parameters
    ----------
    orders_queue : multiprocessing.Queue
        Input queue from which orders are received.
    instances : list
        list of filepaths of instances which are solved by SCIP to collect data
    seed : int
        initial seed to insitalize random number generator with
    time_limit : int
        maximum time for which to solve an instance while collecting data
    outdir : str
        directory where to save data
    start_episode : int
        episode to resume data collection. It is used if the data collection process was stopped earlier for some reason.
    """
    rng = np.random.RandomState(seed)
    episode = 0
    while True:
        instance = rng.choice(instances) # 从实例列表中随机选择实例
        seed = rng.randint(2**32) # 生成随机种子
        # already processed; for a broken process; for root dataset to not repeat instances and seed
        if episode <= start_episode: 
            episode += 1
            continue
        
        # 将包含任务信息的列表 [episode, instance, seed, time_limit, outdir, rng] 放入 orders_queue 中供工作者处理
        orders_queue.put([episode, instance, seed, time_limit, outdir, rng])
        episode += 1

def collect_samples(instances, outdir, rng, n_samples, n_jobs, time_limit):
    # 用于并行收集样本数据。它通过启动多个工作进程来解决给定实例，并记录每个实例的样本数据
    """
    Worker loop: fetch an instance, run an episode and record samples.

    Parameters
    ----------
    instances : list
        filepaths of instances which will be solved to collect data
    outdir : str
        directory where to save data
    rng : np.random.RandomState
        random number generator
    n_samples : int
        total number of samples to collect.
    n_jobs : int
        number of CPUs to utilize or number of instances to solve in parallel.
    time_limit : int
        maximum time for which to solve an instance while collecting data
    instances: 包含待解决问题文件路径的列表。
    outdir: 存储生成数据的目录。
    rng: 用于生成随机数的 np.random.RandomState 对象。
    n_samples: 需要收集的总样本数量。
    n_jobs: 并行处理的CPU核心数或同时解决的实例数。
    time_limit: 解决每个实例时的最大时间限制。
    """
    os.makedirs(outdir, exist_ok=True)

    # start workers
    # 创建两个队列：orders_queue（用于向工作者发送任务）和 answers_queue（用于接收工作者的结果）
    orders_queue = mp.Queue(maxsize=2*n_jobs)
    answers_queue = mp.SimpleQueue()
    workers = []
    # 根据 n_jobs 启动相应数量的工作进程，每个进程运行 make_samples 函数，负责从 orders_queue 获取任务并把结果放入 answers_queue。
    for i in range(n_jobs):
        p = mp.Process( # 创建工作进程和调度进程
                target=make_samples,
                args=(orders_queue, answers_queue),
                daemon=True)
        workers.append(p)
        p.start()

    # dir to keep samples temporarily; helps keep a prefect count
    tmp_samples_dir = f'{outdir}/tmp'
    os.makedirs(tmp_samples_dir, exist_ok=True)

    # if the process breaks due to some reason, resume from this last_episode.
    # 查找输出目录中已有的 .pkl 文件，从中提取最后一个已完成的 episode 和 sample number，以便在中断后恢复
    existing_samples = glob.glob(f"{outdir}/*.pkl")
    last_episode, last_i = -1, 0
    if existing_samples:
        last_episode = max(int(x.split("/")[-1].split(".pkl")[0].split("_")[-2]) for x in existing_samples) # episode is 2nd last
        last_i = max(int(x.split("/")[-1].split(".pkl")[0].split("_")[-1]) for x in existing_samples) # sample number is the last

    # start dispatcher
    # 启动一个调度器进程 dispatcher，负责将任务分配给工作者。调度器使用 send_orders 函数，将任务放入 orders_queue。
    dispatcher = mp.Process(
            target=send_orders,
            args=(orders_queue, instances, rng.randint(2**32), time_limit, tmp_samples_dir, last_episode),
            daemon=True)
    dispatcher.start()

    # 主循环
    i = last_i # for a broken process 初始化计数器 i（样本计数）和 in_buffer（缓冲区计数）
    in_buffer = 0
    while i <= n_samples:
        sample = answers_queue.get() # 在主循环中，不断从 answers_queue 获取样本

        # 如果样本类型为 'start'，表示一个新任务开始，增加缓冲区计数器
        if sample['type'] == 'start':
            in_buffer += 1

        # 如果样本类型为 'done'，表示一个任务完成，处理结果并将临时文件移动到最终输出目录，更新样本计数器 i。
        if sample['type'] == 'done':
            for filename in sample['filenames']:
                x = filename.split('/')[-1].split(".pkl")[0]
                os.rename(filename, f"{outdir}/{x}.pkl")
                i+=1
                print(f"[m {os.getpid()}] {i} / {n_samples} samples written, ep {sample['episode']} ({in_buffer} in buffer).")

                if  i == n_samples:
                    # early stop dispatcher (hard)
                    if dispatcher.is_alive(): # 如果达到所需的样本数量，则终止调度器
                        dispatcher.terminate()
                        print(f"[m {os.getpid()}] dispatcher stopped...")
                    break

        # 如果调度器不再活跃，则退出循环。
        if not dispatcher.is_alive():
            break

    # stop all workers (hard) 终止所有工作者进程
    for p in workers:
        p.terminate()

    shutil.rmtree(tmp_samples_dir, ignore_errors=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # problem parameters
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=utilities.valid_seed,
        default=0,
    )

    parser.add_argument(
        '-j', '--njobs',
        help='Number of parallel jobs.',
        type=int,
        default=1,
    )
    args = parser.parse_args()

    train_size = 10000
    valid_size = 2000
    test_size = 5000
    time_limit = 3600
    node_limit = 3000
    node_record_prob = 1.0

    basedir= "data/samples"
    # get instance filenames
    if args.problem == 'setcover':
        instances_train = glob.glob('data/instances/setcover/train_500r_1000c_0.05d/*.lp')
        instances_valid = glob.glob('data/instances/setcover/valid_500r_1000c_0.05d/*.lp')
        instances_test = glob.glob('data/instances/setcover/transfer_500r_1000c_0.05d/*.lp')
        out_dir = f'{basedir}/setcover/500r_1000c_0.05d'

    elif args.problem == 'cauctions':
        instances_train = glob.glob('data/instances/cauctions/train_100_500/*.lp')
        instances_valid = glob.glob('data/instances/cauctions/valid_100_500/*.lp')
        instances_test = glob.glob('data/instances/cauctions/transfer_100_500/*.lp')
        out_dir = f'{basedir}/cauctions/100_500'

    elif args.problem == 'indset':
        instances_train = glob.glob('data/instances/indset/train_750_4/*.lp')
        instances_valid = glob.glob('data/instances/indset/valid_750_4/*.lp')
        instances_test = glob.glob('data/instances/indset/transfer_750_4/*.lp')
        out_dir = f'{basedir}/indset/750_4'

    elif args.problem == 'facilities':
        instances_train = glob.glob('data/instances/facilities/train_100_100_5/*.lp')
        instances_valid = glob.glob('data/instances/facilities/valid_100_100_5/*.lp')
        instances_test = glob.glob('data/instances/facilities/transfer_100_100_5/*.lp')
        out_dir = f'{basedir}/facilities/100_100_5'
        time_limit = 600

    else:
        raise NotImplementedError

    print(f"{len(instances_train)} train instances for {train_size} samples")
    print(f"{len(instances_valid)} validation instances for {valid_size} samples")
    print(f"{len(instances_test)} test instances for {test_size} samples")

    rng = np.random.RandomState(args.seed + 1)
    collect_samples(instances_train, out_dir +"/train", rng, train_size, args.njobs, time_limit)
    print("Success: Train data collection")

    rng = np.random.RandomState(args.seed + 1)
    collect_samples(instances_valid, out_dir +"/valid", rng, valid_size, args.njobs, time_limit)
    print("Success: Valid data collection")

    rng = np.random.RandomState(args.seed + 1)
    collect_samples(instances_test, out_dir +"/test", rng, test_size, args.njobs, time_limit)
    print("Success: Test data collection")

    if args.problem == "indset":
        mediumvalid_size = 2000
        instances_mediumvalid = glob.glob('data/instances/indset/mediumvalid_1000_4/*.lp')
        out_dir = f'{basedir}/indset/1000_4'

        print(f"{len(instances_mediumvalid)} medium validation instances for {mediumvalid_size} samples")

        rng = np.random.RandomState(args.seed + 1)
        collect_samples(instances_mediumvalid, out_dir +"/mediumvalid", rng, mediumvalid_size, args.njobs, time_limit)
        print("Success: Medium validation data collection")
