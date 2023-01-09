import numpy as np
import torch
import collections


def qlearning_dataset(env, dataset=None, terminate_on_end=False, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.

    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().

    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)
    
    has_next_obs = True if 'next_observations' in dataset.keys() else False

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        if has_next_obs:
            new_obs = dataset['next_observations'][i].astype(np.float32)
        else:
            new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue  
        if done_bool or final_timestep:
            episode_step = 0
            if not has_next_obs:
                continue

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_len, max_ep_len=1000, device="cpu"):
        super().__init__()

        self.obs_dim = dataset["observations"].shape[-1]
        self.action_dim = dataset["actions"].shape[-1]
        self.max_len = max_len
        self.max_ep_len = max_ep_len
        self.device = torch.device(device)
        self.input_mean = np.concatenate([dataset["observations"], dataset["actions"]], axis=1).mean(0)
        self.input_std = np.concatenate([dataset["observations"], dataset["actions"]], axis=1).std(0) + 1e-6

        data_ = collections.defaultdict(list)
        
        use_timeouts = False
        if 'timeouts' in dataset:
            use_timeouts = True

        episode_step = 0
        self.trajs = []
        for i in range(dataset["rewards"].shape[0]):
            done_bool = bool(dataset['terminals'][i])
            if use_timeouts:
                final_timestep = dataset['timeouts'][i]
            else:
                final_timestep = (episode_step == 1000-1)
            for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
                data_[k].append(dataset[k][i])
            if done_bool or final_timestep:
                episode_step = 0
                episode_data = {}
                for k in data_:
                    episode_data[k] = np.array(data_[k])
                self.trajs.append(episode_data)
                data_ = collections.defaultdict(list)
            episode_step += 1
        
        indices = []
        for traj_ind, traj in enumerate(self.trajs):
            end = len(traj["rewards"])
            for i in range(end):
                indices.append((traj_ind, i, i+self.max_len))

        self.indices = np.array(indices)
        

        returns = np.array([np.sum(t['rewards']) for t in self.trajs])
        num_samples = np.sum([t['rewards'].shape[0] for t in self.trajs])
        print(f'Number of samples collected: {num_samples}')
        print(f'Num trajectories: {len(self.trajs)}')
        print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        traj_ind, start_ind, end_ind = self.indices[idx]
        traj = self.trajs[traj_ind].copy()
        obss = traj['observations'][start_ind:end_ind]
        actions = traj['actions'][start_ind:end_ind]
        next_obss = traj['next_observations'][start_ind:end_ind]
        rewards = traj['rewards'][start_ind:end_ind].reshape(-1, 1)
        delta_obss = next_obss - obss
    
        # padding
        tlen = obss.shape[0]
        inputs = np.concatenate([obss, actions], axis=1)
        inputs = (inputs - self.input_mean) / self.input_std
        inputs = np.concatenate([inputs, np.zeros((self.max_len - tlen, self.obs_dim+self.action_dim))], axis=0)
        targets = np.concatenate([delta_obss, rewards], axis=1)
        targets = np.concatenate([targets, np.zeros((self.max_len - tlen, self.obs_dim+1))], axis=0)
        masks = np.concatenate([np.ones(tlen), np.zeros(self.max_len - tlen)], axis=0)

        inputs = torch.from_numpy(inputs).to(dtype=torch.float32, device=self.device)
        targets = torch.from_numpy(targets).to(dtype=torch.float32, device=self.device)
        masks = torch.from_numpy(masks).to(dtype=torch.float32, device=self.device)

        return inputs, targets, masks