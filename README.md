# OfflineRL-Kit: An elegant PyTorch offline reinforcement learning library.

![MIT](https://img.shields.io/badge/license-MIT-blue)

OfflineRL-Kit is an offline reinforcement learning library based on pure PyTorch. This library has some features which are friendly and convenient for researchers, including:

- Elegant framework, the code structure is very clear and easy to use
- State-of-the-art offline RL algorithms, including model-free and model-based approaches
- High scalability, you can build your new algorithm with few lines of code based on the components in our library
- Support parallel tuning, very covenient to researchers
- Clear and powerful log system, easy to manage experiments

## Supported algorithms
- Model-free
    - [Conservative Q-Learning (CQL)](https://arxiv.org/abs/2006.04779)
    - [TD3+BC](https://arxiv.org/abs/2106.06860)
    - [Implicit Q-Learning (IQL)](https://arxiv.org/abs/2110.06169)
    - [Mildly Conservative Q-Learning (MCQ)](https://arxiv.org/abs/2206.04745)
- Model-based
    - [Model-based Offline Policy Optimization (MOPO)](https://arxiv.org/abs/2005.13239)

## Examples
### train
```shell
python run_example/run_mopo.py
```
### tune
```shell
python tune_example/tune_mopo.py
```
### plot
```shell
python common/plotter.py --algos "mopo"
```

## Tutorials
More tutorial documentations are available soon.

## Log
Our logger supports a variant of record file types, including .txt(backup for stdout), .csv(records loss or performance or other metrics in training progress), .tfevents (tensorboard for visualizing the training curve), .json(backup for hyper-parameters).
Our logger also has a clear log structure:
```
└─log(root dir)
    └─task
        └─algo_0
        |   └─seed_0&timestamp_xxx
        |   |   ├─checkpoint
        |   |   ├─model
        |   |   ├─record
        |   |   │  ├─tb
        |   |   │  ├─consoleout_backup.txt
        |   |   │  ├─policy_training_progress.csv
        |   |   │  ├─hyper_param.json
        |   |   ├─result
        |   └─seed_1&timestamp_xxx
        └─algo_1
```
This is an example of logger.

First, import some relevant packages:
```py
from common.logger import Logger, make_log_dirs
```
Then initialize logger:
```py
log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args))
# key: output file name, value: output handler type
output_config = {
    "consoleout_backup": "stdout",
    "policy_training_progress": "csv",
    "dynamics_training_progress": "csv",
    "tb": "tensorboard"
}
logger = Logger(log_dirs, output_config)
logger.log_hyperparameters(vars(args))
```

Let's log some metrics:
```py
# log
logger.logkv("eval/normalized_episode_reward", norm_ep_rew_mean)
logger.logkv("eval/normalized_episode_reward_std", norm_ep_rew_std)
logger.logkv("eval/episode_length", ep_length_mean)
logger.logkv("eval/episode_length_std", ep_length_std)
# set timestep
logger.set_timestep(num_timesteps)
# dump results to the record files
logger.dumpkvs()
```

## Benchmark Results
Benchmark results are available soon.


