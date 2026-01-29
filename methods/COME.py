from copy import deepcopy
from loguru import logger as log
import torch
import torch.nn as nn
import torch.jit
import torch.optim as optim
import numpy as np
from typing import Tuple

__all__ = ["setup"]

def dirichlet_entropy(x: torch.Tensor):#key component of COME
    x = x / torch.norm(x, p=2, dim=-1, keepdim=True) * torch.norm(x, p=2, dim=-1, keepdim=True).detach()
    brief = torch.exp(x)/(torch.sum(torch.exp(x), dim=1, keepdim=True) + 1000)
    uncertainty = 1000 / (torch.sum(torch.exp(x), dim=1, keepdim=True) + 1000)
    probability = torch.cat([brief, uncertainty], dim=1) + 1e-7
    entropy = -(probability * torch.log(probability)).sum(1)
    return entropy

class COME(nn.Module):
    def __init__(self, cfg, model, optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.steps = cfg.OPTIM.STEPS
        assert self.steps > 0, "COME requires >= 1 step(s) to forward and update"

    @torch.enable_grad()
    def forward(self, x, y=None, adapt=True):
        if not adapt:
            self.model.eval()
            outputs = self.model(x)
            self.model.train()
            return outputs

        outputs = self.forward_and_adapt(x)

        return outputs

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        for _ in range(self.steps):
            outputs = self.model(x)
            # 使用 dirichlet_entropy 替代标准的 softmax_entropy
            loss = dirichlet_entropy(outputs).mean(0)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return outputs

def collect_params(model: nn.Module) -> Tuple[list, list]:
    """收集可训练的归一化层参数 (与 Tent 一致)"""
    params, names = [], []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ["weight", "bias"] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def configure_model(model: nn.Module) -> nn.Module:
    """配置模型：训练模式，只有归一化层可训练 (与 Tent 一致)"""
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
            # 强制使用 batch statistics (对于 BN)
            if hasattr(m, "track_running_stats"):
                m.track_running_stats = False
            if hasattr(m, "running_mean"):
                m.running_mean = None
            if hasattr(m, "running_var"):
                m.running_var = None
    return model

def setup(model, cfg):
    log.info("Setup TTA method: COME (Tent with Conservative Entropy Minimization)")
    
    # 1. 配置模型 (开启 Norm 层训练，冻结其他层)
    model_configured = configure_model(deepcopy(model))
    
    # 2. 收集参数
    params, param_names = collect_params(model_configured)
    if not params:
        raise ValueError("No adaptable parameters found in the model.")

    # 3. 配置优化器 (通常使用 SGD)
    if cfg.OPTIM.METHOD == "SGD":
        optimizer = optim.SGD(
            params,
            lr=float(cfg.OPTIM.LR),
            momentum=float(getattr(cfg.OPTIM, "MOMENTUM", 0.9)),
            weight_decay=float(getattr(cfg.OPTIM, "WD", 0.0)),
            nesterov=bool(getattr(cfg.OPTIM, "NESTEROV", False)),
        )
    elif cfg.OPTIM.METHOD == "Adam":
        optimizer = optim.Adam(
            params,
            lr=float(cfg.OPTIM.LR),
            betas=(float(getattr(cfg.OPTIM, "BETA", 0.9)), 0.999),
            weight_decay=float(getattr(cfg.OPTIM, "WD", 0.0)),
        )
    else:
        raise ValueError(f"Unsupported optimizer method: {cfg.OPTIM.METHOD}")

    # 4. 初始化 COME 模块
    tta_model = COME(cfg, model_configured, optimizer)
    
    return tta_model