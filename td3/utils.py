import torch


def reset_weights(m: torch.nn.Module):
    if hasattr(m, "reset_parameters"):
        m.reset_parameteres()