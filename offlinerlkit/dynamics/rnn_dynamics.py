import os
import numpy as np
import torch
import torch.nn as nn

from typing import Callable, List, Tuple, Dict
from torch.utils.data.dataloader import DataLoader
from offlinerlkit.dynamics import BaseDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.logger import Logger


class RNNDynamics(BaseDynamics):
    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        scaler: StandardScaler,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    ) -> None:
        super().__init__(model, optim)
        self.scaler = scaler
        self.terminal_fn = terminal_fn
    
    @ torch.no_grad()
    def step(
        self,
        obss: np.ndarray,
        actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        "imagine single forward step"
        inputs = np.concatenate([obss, actions], axis=-1)
        inputs = self.scaler.transform(inputs)
        preds, _ = self.model(inputs)
        # get last timestep pred
        preds = preds[:, -1]
        next_obss = preds[..., :-1].cpu().numpy() + obss[:, -1]
        rewards = preds[..., -1:].cpu().numpy()

        terminals = self.terminal_fn(obss[:, -1], actions[:, -1], next_obss)
        info = {}

        return next_obss, rewards, terminals, info

    def train(self, data: Dict, batch_size: int, max_iters: int, logger: Logger) -> None:
        self.model.train()
        loader = DataLoader(data, shuffle=True, batch_size=batch_size)
        for iter in range(max_iters):
            for batch in loader:
                train_loss = self.learn(batch)
                logger.logkv_mean("loss/model", train_loss)
            
            logger.set_timestep(iter)
            logger.dumpkvs(exclude=["policy_training_progress"])
        self.save(logger.model_dir)
        self.model.eval()
    
    def learn(self, batch) -> float:
        inputs, targets, masks = batch
        preds, _ = self.model.forward(inputs)

        loss = (((preds - targets) ** 2).mean(-1) * masks).mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item()
    
    def save(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), os.path.join(save_path, "dynamics.pth"))
        self.scaler.save_scaler(save_path)
    
    def load(self, load_path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=self.model.device))
        self.scaler.load_scaler(load_path)