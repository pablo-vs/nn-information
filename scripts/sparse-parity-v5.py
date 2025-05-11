#!/usr/bin/env python
# coding: utf-8
"""
This script trains MLPs on multiple sparse parity problems at once.

Comments
    - now does sampling for everything except the test batch -- frequencies of subtasks are exactly distributed within test batch
    - now allows for early stopping
"""

from collections import defaultdict
from itertools import islice, product
import random
import time
from pathlib import Path

import numpy as np
# import scipy.stats
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
import matplotlib.pyplot as plt

ex = Experiment("sparse-parity-v4")
ex.captured_out_filter = apply_backspaces_and_linefeeds

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len, device=self.tensors[0].device)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.i:self.i+self.batch_size]
            batch = tuple(torch.index_select(t, 0, indices) for t in self.tensors)
        else:
            batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


def get_batch(n_tasks, n, Ss, codes, sizes, device='cpu', dtype=torch.float32):
    """Creates batch. 

    Parameters
    ----------
    n_tasks : int
        Number of tasks.
    n : int
        Bit string length for sparse parity problem.
    Ss : list of lists of ints
        Subsets of [1, ... n] to compute sparse parities on.
    codes : list of int
        The subtask indices which the batch will consist of
    sizes : list of int
        Number of samples for each subtask
    device : str
        Device to put batch on.
    dtype : torch.dtype
        Data type to use for input x. Output y is torch.int64.

    Returns
    -------
    x : torch.Tensor
        inputs
    y : torch.Tensor
        labels
    """
    batch_x = torch.zeros((sum(sizes), n_tasks+n), dtype=dtype, device=device)
    batch_y = torch.zeros((sum(sizes),), dtype=torch.int64, device=device)
    start_i = 0
    for (S, size, code) in zip(Ss, sizes, codes):
        if size > 0:
            x = torch.randint(low=0, high=2, size=(size, n), dtype=dtype, device=device)
            y = torch.sum(x[:, S], dim=1) % 2
            x_task_code = torch.zeros((size, n_tasks), dtype=dtype, device=device)
            x_task_code[:, code] = 1
            x = torch.cat([x_task_code, x], dim=1)
            batch_x[start_i:start_i+size, :] = x
            batch_y[start_i:start_i+size] = y
            start_i += size
    return batch_x, batch_y
    
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def cross_entropy_with_margin(logits, targets, margin=1.0):
    # Original cross entropy
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    
    # Get predicted class and its probability
    probs = F.softmax(logits, dim=1)
    predicted_probs = probs[torch.arange(probs.size(0)), targets]
    
    # Only penalize if confidence is below the margin
    margin_mask = (predicted_probs < margin).float()
    
    return (ce_loss * margin_mask).mean()

def get_weight_norm(model):
    norms = dict()
    for n,p in model.named_parameters():
        norms[n] = p.norm(2).item()
    norms['global'] = sum([v**2 for v in norms.values()])**0.5
    return norms

def get_param_vector(model):
    return torch.cat([param.clone().detach().view(-1) for param in model.parameters()])

def get_param_distance(p1, p2):
    return torch.sqrt(((p1 - p2)**2).sum()).item()

# Helper functions for parameter exploration
def freeze_all_except_one(model, target_layer_idx, target_param_type, target_idx):
    """Freezes all parameters except one specific parameter."""
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Find the target Linear layer (accounting for activation layers)
    linear_layers = [i for i, m in enumerate(model) if isinstance(m, nn.Linear)]
    actual_idx = linear_layers[target_layer_idx]
    target_layer = model[actual_idx]
    
    # Make sure it's a Linear layer
    assert isinstance(target_layer, nn.Linear), "Target layer must be Linear"
    
    # Get the target parameter
    if target_param_type == 'weight':
        row, col = target_idx
        # Create a mask of zeros with the same shape as weight
        mask = torch.zeros_like(target_layer.weight)
        # Set the target position to 1
        mask[row, col] = 1
        # Apply the mask to the gradients during backward pass
        target_layer.weight.register_hook(lambda grad: grad * mask)
        # Enable gradients for the whole weight tensor
        target_layer.weight.requires_grad = True
    elif target_param_type == 'bias':
        idx = target_idx
        # Create a mask of zeros with the same shape as bias
        mask = torch.zeros_like(target_layer.bias)
        # Set the target position to 1
        mask[idx] = 1
        # Apply the mask to the gradients during backward pass
        target_layer.bias.register_hook(lambda grad: grad * mask)
        # Enable gradients for the whole bias tensor
        target_layer.bias.requires_grad = True
    
    return target_layer, actual_idx


# --------------------------
#    ,-------------.
#   (_\  CONFIG     \
#      |    OF      |
#      |    THE     |
#     _| EXPERIMENT |
#    (_/_____(*)___/
#             \\
#              ))
#              ^
# --------------------------
@ex.config
def cfg():
    n_tasks = 10
    n = 50
    k = 3
    alpha = 0
    offset = 0

    D = -1 # -1 for infinite data

    width = 100
    depth = 2
    activation = 'ReLU'
    loss_margin = 0.9
    
    steps = 20000
    batch_size = 3000
    lr = 1e-3
    weight_decay = 1e-2
    test_points = 3000
    test_points_per_task = 100

    stop_early = False
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    log_freq = max(1, steps // 1000)
    verbose=False

# --------------------------
#  |-|    *
#  |-|   _    *  __
#  |-|   |  *    |/'   SEND
#  |-|   |~*~~~o~|     IT!
#  |-|   |  O o *|
# /___\  |o___O__|
# --------------------------
@ex.capture
def create_model(n_tasks, n, width, depth, activation, device):
    if activation == 'relu':
        activation_fn = nn.ReLU
    elif activation == 'tanh':
        activation_fn = nn.Tanh
    else:
        activation_fn = nn.ReLU  # Default
    
    layers = []
    for i in range(depth):
        if i == 0:
            layers.append(nn.Linear(n_tasks + n, width))
            layers.append(activation_fn())
        elif i == depth - 1:
            layers.append(nn.Linear(width, 2))
        else:
            layers.append(nn.Linear(width, width))
            layers.append(activation_fn())
    
    return nn.Sequential(*layers).to(device)

@ex.capture
def prepare_data(
        n_tasks,
        n,
        k,
        alpha,
        offset,
        D,
        test_points,
        test_points_per_task,
        batch_size,
        device,
        dtype,
        seed,
        _log):
    
    Ss = []
    for _ in range(n_tasks * 10):
        S = tuple(sorted(list(random.sample(range(n), k))))
        if S not in Ss:
            Ss.append(S)
        if len(Ss) == n_tasks:
            break
    assert len(Ss) == n_tasks, "Couldn't find enough subsets for tasks for the given n, k"
    ex.info['Ss'] = Ss

    probs = np.array([np.power(n, -alpha) for n in range(1+offset, n_tasks+offset+1)])
    probs = probs / np.sum(probs)
    cdf = np.cumsum(probs)

    test_batch_sizes = [int(prob * test_points) for prob in probs]
    # _log.debug(f"Total batch size = {sum(batch_sizes)}")

    if D != -1:
        samples = np.searchsorted(cdf, np.random.rand(D,))
        hist, _ = np.histogram(samples, bins=n_tasks, range=(0, n_tasks-1))
        train_x, train_y = get_batch(n_tasks=n_tasks, n=n, Ss=Ss, codes=list(range(n_tasks)), sizes=hist, device='cpu', dtype=dtype)
        train_x = train_x.to(device)
        train_y = train_y.to(device)
        train_loader = FastTensorDataLoader(train_x, train_y, batch_size=min(D, batch_size), shuffle=True)
        train_iter = cycle(train_loader)
        ex.info['D'] = D

        def get_batch_fn():
            return train_iter
    else:
        ex.info['D'] = steps * batch_size
        
        def get_batch_fn(batch_size):
            samples = np.searchsorted(cdf, np.random.rand(batch_size,))
            hist, _ = np.histogram(samples, bins=n_tasks, range=(0, n_tasks-1))
            return get_batch(n_tasks=n_tasks, n=n, Ss=Ss, codes=list(range(n_tasks)), sizes=hist, device=device, dtype=dtype)

    train_loader = get_batch_fn()
    test_loader = lambda: get_batch(n_tasks=n_tasks, n=n, Ss=Ss, codes=list(range(n_tasks)), sizes=test_batch_sizes, device=device, dtype=dtype)
    per_task_test_loaders = [lambda: get_batch(n_tasks=n_tasks, n=n, Ss=[Ss[i]], codes=[i], sizes=[test_points_per_task], device=device, dtype=dtype) for i in range(n_tasks)]
    return train_loader, test_loader, per_task_test_loaders

@ex.capture
def train_model(
        _run,
        model,
        train_loader,
        test_loader,
        per_task_test_loaders,
        loss_margin,
        steps,
        n_tasks,
        lr,
        weight_decay,
        stop_early,
        device,
        dtype,
        log_freq,
        verbose,
        seed,
        _log):

    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    mlp = model
    _log.debug("Created model.")
    _log.debug(f"Model has {sum(t.numel() for t in mlp.parameters())} parameters") 
    ex.info['P'] = sum(t.numel() for t in mlp.parameters())

    loss_fn = lambda *x: cross_entropy_with_margin(*x, loss_margin)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=weight_decay)
    ex.info['log_steps'] = list()
    ex.info['accuracies'] = list()
    ex.info['losses'] = list()
    ex.info['losses_subtasks'] = dict()
    ex.info['accuracies_subtasks'] = dict()
    ex.info['weight_norm'] = list()
    ex.info['param_distance'] = list()
    for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        ex.info[f'step_dist_{i}'] = list()

    param_queue = list()

    for i in range(n_tasks):
        ex.info['losses_subtasks'][str(i)] = list()
        ex.info['accuracies_subtasks'][str(i)] = list()
    early_stop_triggers = []
    for step in tqdm(range(steps), disable=not verbose):
        if step % log_freq == 0:
            with torch.no_grad():
                x_i, y_i = test_loader()
                y_i_pred = mlp(x_i)
                labels_i_pred = torch.argmax(y_i_pred, dim=1)
                ex.info['accuracies'].append(torch.sum(labels_i_pred == y_i).item() / y_i.shape[0]) 
                ex.info['losses'].append(loss_fn(y_i_pred, y_i).item())
                for i, loader in enumerate(per_task_test_loaders):
                    x_i, y_i = loader()
                    y_i_pred = mlp(x_i)
                    ex.info['losses_subtasks'][str(i)].append(loss_fn(y_i_pred, y_i).item())
                    labels_i_pred = torch.argmax(y_i_pred, dim=1)
                    ex.info['accuracies_subtasks'][str(i)].append(torch.sum(labels_i_pred == y_i).item() / test_points_per_task)
                ex.info['log_steps'].append(step)
            if stop_early:
                if step > 4000 and len(ex.info['losses']) >= 2 \
                    and ex.info['losses'][-1] > ex.info['losses'][-2]:
                    early_stop_triggers.append(True)
                else:
                    early_stop_triggers.append(False)
                if len(early_stop_triggers) > 10 and all(early_stop_triggers[-10:]):
                    break
                early_stop_triggers = early_stop_triggers[-10:]
        optimizer.zero_grad()

        x, y_target = train_loader()
        y_pred = mlp(x)
        loss = loss_fn(y_pred, y_target)
        ex.info['weight_norm'].append(get_weight_norm(mlp))

        param_queue.append(get_param_vector(mlp))
        for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
            if len(param_queue) > i:
                ex.info[f'step_dist_{i}'].append(get_param_distance(param_queue[-1], param_queue[-1-i]))
        if len(param_queue) > 4096:
            param_queue.pop(0)

        loss.backward()
        optimizer.step()

    weights_path = f"model_weights_{_run._id}.pt"
    torch.save(mlp.state_dict(), weights_path)
    _run.add_artifact(weights_path)

@ex.capture
def explore_parameter(_run, model, train_loader, target_layer_idx, target_param_type, 
                     target_idx, exploration_lr, exploration_epochs, temperature, device):
    """Explore a single parameter space while freezing others."""
    criterion = nn.CrossEntropyLoss()
    
    # Freeze all parameters except the target one
    target_layer, actual_layer_idx = freeze_all_except_one(model, target_layer_idx, target_param_type, target_idx)
    
    # Setup optimizer with only the unfrozen parameter
    if target_param_type == 'weight':
        optimizer = optim.Adam([target_layer.weight], lr=exploration_lr)
    else:  # bias
        optimizer = optim.Adam([target_layer.bias], lr=exploration_lr)
    
    # Lists to store parameter values and losses
    param_values = []
    losses = []
    
    # Get initial parameter value
    if target_param_type == 'weight':
        row, col = target_idx
        init_value = target_layer.weight[row, col].item()
    else:  # bias
        init_value = target_layer.bias[target_idx].item()
    
    print(f"Starting parameter exploration. Initial value: {init_value:.6f}")
    
    # Training loop
    for epoch in range(exploration_epochs):
        model.train()
        epoch_losses = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Record the current value of the parameter and loss
            if target_param_type == 'weight':
                row, col = target_idx
                current_param_value = target_layer.weight[row, col].item()
            else:  # bias
                current_param_value = target_layer.bias[target_idx].item()
            
            param_values.append(current_param_value)
            epoch_losses.append(loss.item())
        
        # Average loss for this epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        
        # Log to Sacred
        _run.log_scalar('exploration.parameter_value', current_param_value, epoch)
        _run.log_scalar('exploration.loss', avg_loss, epoch)
        
        print(f"Epoch {epoch+1}/{exploration_epochs}, Parameter: {current_param_value:.6f}, Loss: {avg_loss:.6f}")
    
    # Calculate and visualize Bayesian posterior
    param_grid, posterior = calculate_bayesian_posterior(param_values, losses, temperature)
    
    # Create plots
    fig = plt.figure(figsize=(12, 10))
    
    # Plot parameter trajectory
    plt.subplot(2, 2, 1)
    plt.plot(param_values)
    plt.title(f'Parameter Value vs. Iteration (Layer {target_layer_idx}, {"Weight" if target_param_type=="weight" else "Bias"} {target_idx})')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Value')
    
    # Plot loss trajectory
    plt.subplot(2, 2, 2)
    plt.plot(losses)
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot parameter distribution
    plt.subplot(2, 2, 3)
    plt.hist(param_values, bins=30, density=True, alpha=0.7)
    plt.title('Parameter Value Distribution')
    plt.xlabel('Parameter Value')
    plt.ylabel('Frequency')
    
    # Plot Bayesian posterior
    plt.subplot(2, 2, 4)
    plt.plot(param_grid, posterior)
    plt.title('Bayesian Posterior Distribution')
    plt.xlabel('Parameter Value')
    plt.ylabel('Posterior Probability')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = f"parameter_exploration_{_run._id}.png"
    plt.savefig(fig_path)
    _run.add_artifact(fig_path)
    
    # Save exploration data
    exploration_data = {
        'layer_idx': target_layer_idx,
        'param_type': target_param_type,
        'param_idx': target_idx,
        'param_values': param_values,
        'losses': losses,
        'param_grid': param_grid.tolist(),
        'posterior': posterior.tolist()
    }
    
    import json
    with open(f"exploration_data_{_run._id}.json", 'w') as f:
        json.dump(exploration_data, f)
    _run.add_artifact(f"exploration_data_{_run._id}.json")
    
    return exploration_data

def calculate_bayesian_posterior(param_values, losses, temperature=1.0):
    """Calculate Bayesian posterior distribution from parameter values and losses."""
    # Convert to numpy arrays
    param_array = np.array(param_values)
    loss_array = np.array(losses)
    
    # Create a grid of parameter values for smooth plotting
    param_min, param_max = np.min(param_array), np.max(param_array)
    param_grid = np.linspace(param_min, param_max, 1000)
    
    # Interpolate losses to the grid
    from scipy.interpolate import interp1d
    loss_interpolator = interp1d(param_array, loss_array, kind='cubic', 
                                fill_value='extrapolate')
    loss_grid = loss_interpolator(param_grid)
    
    # Calculate unnormalized posterior using Boltzmann distribution
    unnormalized_posterior = np.exp(-loss_grid / temperature)
    
    # Normalize to get probabilities
    posterior = unnormalized_posterior / np.sum(unnormalized_posterior)
    
    return param_grid, posterior


# Main experiment
@ex.automain
def run(_run, explore_params):
    # Create model and prepare data
    model = create_model()
    train_loader = prepare_data()  # You'll need to implement this
    
    # Train the model
    trained_model = train_model(model=model, train_loader=train_loader)
    
    # Optionally run parameter exploration
    if explore_params:
        explore_parameter(model=trained_model, train_loader=train_loader)
    
    return "Experiment completed successfully!"

