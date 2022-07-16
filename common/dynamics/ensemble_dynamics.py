import os
import numpy as np
import torch
import torch.nn as nn

from typing import Callable, List, Tuple, Dict
from common.dynamics import BaseDynamics
from common.utils.scaler import StandardScaler
from common.logger import Logger


class EnsembleDynamics(BaseDynamics):
    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        scaler: StandardScaler,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        penalty_coef: float = 0.0,
        uncertainty_mode: str = "aleatoric"
    ) -> None:
        super().__init__(model, optim)
        self.scaler = scaler
        self.terminal_fn = terminal_fn
        self._penalty_coef = penalty_coef
        self._uncertainty_mode = uncertainty_mode

    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        "imagine single forward step"
        obs_act = np.concatenate([obs, action], axis=-1)
        obs_act = self.scaler.transform(obs_act)
        with torch.no_grad():
            mean, logvar = self.model(obs_act)
        mean = mean.cpu().numpy()
        logvar = logvar.cpu().numpy()
        mean[..., :-1] += obs
        std = np.sqrt(np.exp(logvar))
        samples = (mean + np.random.normal(size=mean.shape) * std).astype(np.float32)
        next_obss = samples[..., :-1]
        rewards = samples[..., -1:]

        select_indexes = np.random.randint(0, next_obss.shape[0], size=(obs.shape[0]))
        next_obs = next_obss[select_indexes, np.arange(obs.shape[0])]
        reward = rewards[select_indexes, np.arange(obs.shape[0])]
        terminal = self.terminal_fn(obs, action, next_obs)
        info = {}
        info["raw_reward"] = reward

        if self._penalty_coef:
            if self._uncertainty_mode == "aleatoric":
                penalty = np.amax(np.linalg.norm(std, axis=2), axis=0)
            elif self._uncertainty_mode == "pairwise-diff":
                next_obses_mean = mean[:, :, :-1]
                next_obs_mean = np.mean(next_obses_mean, axis=0)
                diff = next_obses_mean - next_obs_mean
                penalty = np.amax(np.linalg.norm(diff, axis=2), axis=0)
            else:
                raise ValueError
            penalty = np.expand_dims(penalty, 1).astype(np.float32)
            assert penalty.shape == reward.shape
            reward = reward - self._penalty_coef * penalty
            info["penalty"] = penalty
        
        return next_obs, reward, terminal, info

    def format_samples_for_training(self, data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        obss = data["observations"]
        actions = data["actions"]
        next_obss = data["next_observations"]
        rewards = data["rewards"]
        delta_obss = next_obss - obss
        inputs = np.concatenate((obss, actions), axis=-1)
        targets = np.concatenate((delta_obss, rewards), axis=-1)
        return inputs, targets

    def train(self, data: Dict, logger: Logger) -> None:
        inputs, targets = self.format_samples_for_training(data)
        data_size = inputs.shape[0]
        holdout_size = min(int(data_size * 0.2), 1000)
        train_size = data_size - holdout_size
        train_splits, holdout_splits = torch.utils.data.random_split(range(data_size), (train_size, holdout_size))
        train_inputs, train_targets = inputs[train_splits.indices], targets[train_splits.indices]
        holdout_inputs, holdout_targets = inputs[holdout_splits.indices], targets[holdout_splits.indices]

        self.fit(train_inputs)
        train_inputs = self.transform(train_inputs)
        holdout_inputs = self.transform(holdout_inputs)
        holdout_losses = [1e10 for i in range(self.model.num_ensemble)]

        data_idxes = np.random.randint(train_size, size=[self.model.num_ensemble, train_size])
        def shuffle_rows(arr):
            idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxes]

        epoch = 0
        cnt = 0
        logger.log("Training dynamics:")
        while True:
            epoch += 1
            self.model.train()
            train_loss = self.learn(train_inputs[data_idxes], train_targets[data_idxes])
            new_holdout_losses = self.validate(holdout_inputs, holdout_targets)
            holdout_loss = (np.sort(new_holdout_losses)[:self.model.num_elites]).mean()
            logger.logkv("loss/dynamics_train_loss", train_loss)
            logger.logkv("loss/dynamics_holdout_loss", holdout_loss)
            logger.set_timestep(epoch)
            logger.dumpkvs(exclude=["policy_training_progress"])

            # shuffle data for each base learner
            data_idxes = shuffle_rows(data_idxes)

            indexes = []
            for i, new_loss, old_loss in zip(range(len(holdout_losses)), new_holdout_losses, holdout_losses):
                improvement = (old_loss - new_loss) / old_loss
                if improvement > 0.01:
                    indexes.append(i)
                    holdout_losses[i] = new_loss
            
            if len(indexes) > 0:
                self.update_save(indexes)
                cnt = 0
            else:
                cnt += 1
            
            if cnt >= 5:
                break

        indexes = self.select_elites(holdout_losses)
        self.set_elites(indexes)
        self.save(logger.model_dir)
        self.model.eval()
        logger.log("elites:{} , holdout loss: {}".format(indexes, (np.sort(holdout_losses)[:self.model.num_elites]).mean()))
    
    def learn(self, inputs: np.ndarray, targets: np.ndarray, batch_size: int = 256) -> float:
        train_size = inputs.shape[1]
        losses = []

        for batch_num in range(int(np.ceil(train_size / batch_size))):
            inputs_batch = inputs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = targets[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = torch.as_tensor(targets_batch).to(self.model.device)
            
            mean, logvar = self.model(inputs_batch)
            inv_var = torch.exp(-logvar)
            # Average over batch and dim, sum over ensembles.
            mse_loss_inv = (torch.pow(mean - targets_batch, 2) * inv_var).mean(dim=(1, 2))
            var_loss = logvar.mean(dim=(1, 2))
            loss = mse_loss_inv.sum() + var_loss.sum()
            loss = loss + self.model.get_decay_loss()
            loss = loss + 0.01 * self.model.max_logvar.sum() - 0.01 * self.model.min_logvar.sum()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses.append(loss.item())
        return np.mean(losses)
    
    def validate(self, inputs: np.ndarray, targets: np.ndarray) -> List[float]:
        self.model.eval()
        targets = torch.as_tensor(targets).to(self.model.device)
        with torch.no_grad():
            mean, _ = self.model(inputs)
            loss = ((mean - targets) ** 2).mean(dim=(1, 2))
        val_loss = list(loss.cpu().numpy())
        return val_loss
    
    def fit(self, x: np.ndarray) -> None:
        self.scaler.fit(x)

    def transform(self, x: np.ndarray) -> None:
        return self.scaler.transform(x)
    
    def select_elites(self, metrics: List) -> List[int]:
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        elites = [pairs[i][1] for i in range(self.model.num_ensemble)]
        return elites
    
    def update_save(self, indexes: List[int]) -> None:
        self.model.update_save(indexes)
    
    def set_elites(self, indexes: List[int]) -> None:
        self.model.set_elites(indexes)

    def save(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), os.path.join(save_path, "dynamics.pth"))
        self.scaler.save_scaler(save_path)
    
    def load(self, load_path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=self.model.device))
        self.scaler.load_scaler(load_path)
