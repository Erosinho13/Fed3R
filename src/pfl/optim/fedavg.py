# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import torch

from .base import FedBase
from .utils import get_client_optimizer


class FedAvg(FedBase):
    def __init__(self, train_fed_loader, available_clients, clients_to_cache, server_model,
                 server_optimizer, server_lr, server_momentum, max_grad_norm, clip_grad_norm,
                 save_dir, seed, save_client_params_to_disk=False, **kwargs):
        client_model = None  # FedAvg does not have a client model and it is stateless
        super().__init__(
            train_fed_loader, available_clients, clients_to_cache, server_model, client_model,
            server_optimizer, server_lr, server_momentum, max_grad_norm, clip_grad_norm, save_dir, seed,
            save_client_params_to_disk, stateless_clients=True, args=kwargs.get('args', None)
        )
        self.temperature = getattr(self.args, 'temperature', None) if self.args is not None else None
        self.ft_type = getattr(self.args, 'ft_type', None) if self.args is not None else None

    @torch.no_grad()
    def reset_combined_model(self):
        """Combine global_model and client_model into combined model to make predictions
        """
        # FedAvg has no client model so simply use the server model
        state_dict = self.server_model.server_state_dict()
        self.combined_model.load_state_dict(state_dict, strict=False)

    def run_local_updates(
            self, client_loader, num_local_epochs,
            client_optimizer, client_optimizer_args
    ):

        if self.ft_type is not None:
            if self.ft_type == 'feat':
                for param in self.combined_model.classifier.parameters():
                    param.requires_grad = False
            else:
                raise NotImplementedError

        avg_loss = 0.0
        count = 0
        device = next(self.combined_model.parameters()).device
        total_num_local_steps = num_local_epochs * len(client_loader)
        client_optimizer, client_scheduler = get_client_optimizer(
            client_optimizer, self.combined_model, total_num_local_steps, client_optimizer_args
        )
        for _ in range(num_local_epochs):
            for x, y in client_loader:
                x, y = x.to(device), y.to(device)
                client_optimizer.zero_grad()
                yhat = self.combined_model(x)
                if self.temperature is not None:
                    yhat = yhat / self.temperature
                loss = self.loss_fn(yhat, y)
                avg_loss = avg_loss * count / (count + 1) + loss.item() / (count + 1)
                count += 1
                loss.backward()
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.combined_model.parameters(), self.max_grad_norm)
                client_optimizer.step()
                client_scheduler.step()
        # Return number of batches in epoch as a proxy for dataset size
        return avg_loss, len(client_loader)

    def update_local_model_and_get_client_grad(self):
        """Update client_model based on combined_model and return the state_dict with the global model "grad".
        """
        # FedAvg does not have a client model. So, simply return the difference (old - new)
        old_params = self.server_model.server_state_dict()
        new_params = self.combined_model.server_state_dict()
        return OrderedDict((k, v - new_params[k]) for (k, v) in old_params.items())

