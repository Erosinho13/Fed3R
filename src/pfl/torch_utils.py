from collections import OrderedDict
import torch


def get_device(model):
    return next(model.parameters()).device


@torch.no_grad()
def add_(model_r: torch.nn.Module, model_l1: torch.nn.Module, model_l2: torch.nn.Module) -> None:
    """Perform the inplace_update r = l1 + l2, each of which is a torch.nn.Module.
        Assume that each model has *exactly* the same parameters.
    """
    for r, l1, l2 in zip(model_r.parameters(), model_l1.parameters(), model_l2.parameters()):
        torch.add(l1, l2, out=r)


@torch.no_grad()
def interpolate_(model_r: torch.nn.Module,
                 model_l1: torch.nn.Module, model_l2: torch.nn.Module,
                 alpha: float) -> None:
    """Perform the inplace_update r = a*l1 + (1-a)*l2, each of which is a torch.nn.Module.
        Assume that each model has *exactly* the same parameters.
    """
    for r, l1, l2 in zip(model_r.parameters(), model_l1.parameters(), model_l2.parameters()):
        torch.add(l2, l1 - l2, alpha=alpha, out=r)  # r = l2 + alpha * (l1 - l2)

def norm(model: torch.nn.Module) -> torch.Tensor:
    return torch.linalg.norm(torch.stack([torch.linalg.norm(v.view(-1)) for v in model.parameters()]).view(-1))


def weighted_average_of_state_dicts(state_dict_lst, weights):
    """Compute a weighted-average of state dicts with weights.
        Assume that the keys match exactly and that all items are floating point.

    Args:
        state_dict_lst (list): list of OrderedDict
        weights (list): list of Python floats or ints

    Returns:
        OrderedDict: state dict denoting average
    """
    s = sum(weights)
    weights = [w / s for w in weights]  # renormalize
    out = OrderedDict()
    for name in state_dict_lst[0].keys():
        params = [s[name] for s in state_dict_lst]
        out[name] = sum(w * t for (w, t) in zip(weights, params))
    return out

