from copy import deepcopy
from loguru import logger as log
import torch
import torch.nn as nn
import torch.jit
import torch.optim as optim
import numpy as np
from collections import defaultdict 
from typing import Dict, Tuple, Optional, Iterable


__all__ = ["setup"]

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

class TENT(nn.Module):
    def __init__(self, cfg, model, optimizer):
        super().__init__()
        self.model = model # This model is already configured by setup
        self.optimizer = optimizer
        self.cfg = cfg
        self.steps = cfg.OPTIM.STEPS # Assuming steps is 1 from your simplified forward
        self.adapter_cfg = cfg.ADAPTER.Tent
        assert self.steps > 0, "tent requires >= 1 step(s) to forward and update"

        self.batch_index = 0

    @torch.enable_grad()
    def forward(self, x, y=None, adapt=True):
        if not adapt:
            self.model.eval()
            outputs = self.model(x)
            self.model.train()
            return outputs
        outputs = self.forward_and_adapt(x) if adapt else None
        return outputs

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)
        return outputs

    @torch.enable_grad()
    def forward_and_adapt(self,x):
        self.optimizer.zero_grad()

        outputs = self.model(x)
        loss = softmax_entropy(outputs).mean(0)
        loss.backward()
        self.optimizer.step()
        return outputs

    @torch.no_grad()
    def get_adaptable_vector(self) -> torch.Tensor:
        """
        将当前所有可适配的归一化层参数 (BN/LN/GN) 展平为一个 1D 向量 (在 CPU 上)。
        这个向量就是我们做子空间分析的样本点。
        """
        params, names = collect_params(self.model)
        if not params:
            raise ValueError("No adaptable parameters found in the model.")
        return torch.cat([p.detach().flatten().cpu() for p in params], dim=0)

    @torch.no_grad()
    def load_adaptable_vector(self, vec: torch.Tensor) -> None:
        """
        （可选）从一个 1D 向量恢复所有可适配参数。
        方便做“投影到前 k 个主成分再还原”的功能。
        """
        params, names = collect_params(self.model)
        offset = 0
        vec = vec.cpu()
        for p in params:
            numel = p.numel()
            chunk = vec[offset: offset + numel].view_as(p)
            p.copy_(chunk)
            offset += numel
        assert offset == vec.numel(), "Vector length does not match total adaptable params."

def collect_params(model: nn.Module) -> Tuple[list, list]:
    """收集可训练的归一化层参数"""
    params, names = [], []
    for nm, m in model.named_modules():
        if isinstance(
            m,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.LayerNorm,
                nn.GroupNorm,
            ),
        ):
            for np, p in m.named_parameters():
                if np in ["weight", "bias"] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def configure_model(model: nn.Module) -> nn.Module:
    """
    配置模型：训练模式，只有归一化层可训练。
    """
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(
            m,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.LayerNorm,
                nn.GroupNorm,
            ),
        ):
            m.requires_grad_(True)
            if hasattr(m, "track_running_stats"):
                m.track_running_stats = False
            if hasattr(m, "running_mean"):
                m.running_mean = None
            if hasattr(m, "running_var"):
                m.running_var = None
    return model

def setup(model, cfg):
    log.info("Setup TTA method: Tent (targeting BN layers, direct hooks)")
    model_configured = configure_model(deepcopy(model)) 
    params, param_names = collect_params(model_configured)
    
    if not params:
        raise ValueError("No adaptable parameters found in the model.")

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


    tta_model = TENT(cfg, model_configured, optimizer)
    return tta_model