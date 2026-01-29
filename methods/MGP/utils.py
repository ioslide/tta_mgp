import math
import random
from loguru import logger as log
from copy import deepcopy
from typing import Dict, Tuple, Optional, Iterable, List
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict
from torch.distributed import ReduceOp
import contextlib
import torch.jit

def compute_input_gradients(model, imgs):
    imgs.requires_grad = True
    logits = model(imgs)
    entropies = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    _entropy_idx = torch.where(entropies < math.log(1000) * 0.4)[0]
    entropies = entropies[_entropy_idx]
    loss = entropies.mean(0)
    input_gradients = torch.autograd.grad(outputs=loss, inputs=imgs, create_graph=True)[0].detach()
    imgs.requires_grad = False
    model.zero_grad()
    input_gradients = torch.norm(input_gradients, p=2, dim=(1, 2, 3))
    return input_gradients, entropies, logits

def symmetric_cross_entropy(x: torch.Tensor, x_ema: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    return -(1 - alpha) * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - alpha * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)

def collect_params(model: nn.Module) -> Tuple[list, list]:
    params, names = [], []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ["weight", "bias"] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def configure_model(model: nn.Module) -> nn.Module:
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
            if hasattr(m, "track_running_stats"):
                m.track_running_stats = False
            if hasattr(m, "running_mean"):
                m.running_mean = None
            if hasattr(m, "running_var"):
                m.running_var = None
    return model

 
@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def softmax_clamp(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.softmax(1)
    logits = torch.clamp(logits, min=0.0, max=0.99)
    return logits

def slr(logits: torch.Tensor, target_logits: torch.Tensor, gamma: float = 0.001, eps: float = 0.001) -> torch.Tensor:
    probs = softmax_clamp(logits)
    target_probs = softmax_clamp(target_logits)
    return - (probs * torch.log((target_probs  * (1- gamma)) / ((1 - target_probs) + eps)  + gamma) / (1- gamma)).sum(1)

def slr1(logits: torch.Tensor, clip: float = 0.99, eps: float = 1e-5) -> torch.Tensor:
    probs = logits.softmax(1)
    probs = torch.clamp(probs, min=0.0, max=clip)
    return - (probs * torch.log((probs / (torch.ones_like(probs) - probs)) + eps)).sum(1)

def update_model_probs1(current_model_probs, new_probs):
    """使用指数移动平均（EMA）更新模型的平均输出概率。"""
    if new_probs.size(0) == 0:
        return current_model_probs
    
    with torch.no_grad():
        if current_model_probs is None:
            return new_probs.mean(0)
        else:
            return 0.9 * current_model_probs + 0.1 * new_probs.mean(0)

# https://www.arxiv.org/pdf/2510.25480
def compute_gwa_for_norm_layers(params):
    alignment_scores = []
    
    for p in params:
        if p.grad is not None:
            grad = -p.grad.detach()
            param_val = p.detach()

            dot_product = torch.sum(grad * param_val)
            grad_norm = torch.norm(grad)
            param_norm = torch.norm(param_val)
            
            if grad_norm > 1e-12 and param_norm > 1e-12:
                cosine_sim = dot_product / (grad_norm * param_norm)
                alignment_scores.append(cosine_sim)
    
    if not alignment_scores:
        return torch.tensor(0.0, device=params[0].device)
        
    return torch.stack(alignment_scores).mean()
    
def update_model_probs(x_ema, x, momentum=0.9):
    return momentum * x_ema + (1 - momentum) * x

