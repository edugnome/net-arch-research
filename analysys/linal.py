import torch

def get_eigenvalues(tensor: torch.Tensor):
    return torch.linalg.eigvalsh(tensor)

def get_rank(tensor: torch.Tensor):
    return torch.linalg.matrix_rank(tensor)
