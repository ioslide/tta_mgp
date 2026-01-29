from copy import deepcopy
from loguru import logger as log
from torch.nn.utils.weight_norm import WeightNorm
from einops import rearrange
import torch
import torchvision
import torch.nn as nn
import torch.jit
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import os

__all__ = ["setup"]


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

class DeYO(nn.Module):

    def __init__(self, cfg,model, optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.steps = cfg.OPTIM.STEPS
        assert self.steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

        self.reweight_ent = cfg.ADAPTER.DeYO.REWEIGHT_ENT
        self.reweight_plpd = cfg.ADAPTER.DeYO.REWEIGHT_PLPD

        self.plpd_threshold = cfg.ADAPTER.DeYO.PLPD
        self.deyo_margin = cfg.ADAPTER.DeYO.MARGIN * math.log(cfg.CORRUPTION.NUM_CLASS)
        self.margin_e0 = cfg.ADAPTER.DeYO.MARGIN_E0 * math.log(cfg.CORRUPTION.NUM_CLASS)

        self.aug_type = cfg.ADAPTER.DeYO.AUG_TYPE
        self.occlusion_size = cfg.ADAPTER.DeYO.OCCLUSION_SIZE
        self.row_start = cfg.ADAPTER.DeYO.ROW_START
        self.column_start = cfg.ADAPTER.DeYO.COLUMN_START
        self.patch_len = cfg.ADAPTER.DeYO.PATCH_LEN

    def forward(self, x, y=None, adapt=True):
        if not adapt:
            self.model.eval()
            outputs = self.model(x)
            self.model.train()
            return outputs

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)


        return outputs
    
    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)
        return outputs
        
    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    @torch.jit.script
    def softmax_entropy(self, x: torch.Tensor) -> torch.Tensor:
        return -(x.softmax(1) * x.log_softmax(1)).sum(1)
        
    @torch.enable_grad()
    def forward_and_adapt(self, x):
        if x.size(0) < 2:
            self.model.eval() # 切换到 eval 模式使用 running stats
            with torch.no_grad():
                outputs = self.model(x)
            self.model.train() # 切回 train 模式
            return outputs
        outputs, loss = self.loss_calculation(x)
        # update model only if not all instances have been filtered
        if loss is not None:
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad()
        return outputs

    def loss_calculation(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        imgs_test = x
        outputs = self.model(imgs_test)

        entropys = softmax_entropy(outputs)
        filter_ids_1 = torch.where((entropys < self.deyo_margin))
        if len(filter_ids_1[0]) < 2:
            loss = None 
            return outputs, loss
        entropys = entropys[filter_ids_1]
        if len(entropys) == 0:
            loss = None  # set loss to None, since all instances have been filtered
            return outputs, loss

        x_prime = imgs_test[filter_ids_1]
        x_prime = x_prime.detach()
        if self.aug_type == 'occ':
            first_mean = x_prime.view(x_prime.shape[0], x_prime.shape[1], -1).mean(dim=2)
            final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
            occlusion_window = final_mean.expand(-1, -1, self.occlusion_size, self.occlusion_size)
            x_prime[:, :, self.row_start:self.row_start + self.occlusion_size, self.column_start:self.column_start + self.occlusion_size] = occlusion_window
        elif self.aug_type == 'patch':
            resize_t = torchvision.transforms.Resize(((imgs_test.shape[-1] // self.patch_len) * self.patch_len, (imgs_test.shape[-1] // self.patch_len) * self.patch_len))
            resize_o = torchvision.transforms.Resize((imgs_test.shape[-1], imgs_test.shape[-1]))
            x_prime = resize_t(x_prime)
            x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=self.patch_len, ps2=self.patch_len)
            perm_idx = torch.argsort(torch.rand(x_prime.shape[0], x_prime.shape[1]), dim=-1)
            x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1), perm_idx]
            x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=self.patch_len, ps2=self.patch_len)
            x_prime = resize_o(x_prime)
        elif self.aug_type == 'pixel':
            x_prime = rearrange(x_prime, 'b c h w -> b c (h w)')
            x_prime = x_prime[:, :, torch.randperm(x_prime.shape[-1])]
            x_prime = rearrange(x_prime, 'b c (ps1 ps2) -> b c ps1 ps2', ps1=imgs_test.shape[-1], ps2=imgs_test.shape[-1])

        with torch.no_grad():
            outputs_prime = self.model(x_prime)

        prob_outputs = outputs[filter_ids_1].softmax(1)
        prob_outputs_prime = outputs_prime.softmax(1)

        cls1 = prob_outputs.argmax(dim=1)

        plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1, 1)) - torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1, 1))
        plpd = plpd.reshape(-1)

        filter_ids_2 = torch.where(plpd > self.plpd_threshold)
        entropys = entropys[filter_ids_2]
        if len(entropys) == 0:
            loss = None  # set loss to None, since all instances have been filtered
            return outputs, loss

        plpd = plpd[filter_ids_2]

        if self.reweight_ent or self.reweight_plpd:
            coeff = (float(self.reweight_ent) * (1. / (torch.exp(((entropys.clone().detach()) - self.margin_e0)))) +
                     float(self.reweight_plpd) * (1. / (torch.exp(-1. * plpd.clone().detach())))
                     )
            entropys = entropys.mul(coeff)

        loss = entropys.mean(0)
        return outputs, loss

    def save(self, ): pass
    def print(self, ): pass 


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def timeliness_reweighting(ages):
    if isinstance(ages, list):
        ages = torch.tensor(ages).float().cuda()
    return torch.exp(-ages) / (1 + torch.exp(-ages))


def configure_model(model):
    """Configure model for use with DeYO."""
    # train mode, because DeYO optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what DeYO updates
    model.requires_grad_(False)
    # configure norm for DeYO updates: enable grad + force batch statisics (this only for BN models)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
        if isinstance(m, nn.BatchNorm1d):
            m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
            m.requires_grad_(True)
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def collect_params(model):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        if 'layer4' in nm:
            continue
        if 'blocks.9' in nm:
            continue
        if 'blocks.10' in nm:
            continue
        if 'blocks.11' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue

        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names



def setup(model, cfg):
    model = configure_model(model)
    params, param_names = collect_params(model)
    if cfg.OPTIM.METHOD == "SGD":
        optimizer = optim.SGD(
            params, 
            lr=float(cfg.OPTIM.LR),
            dampening=cfg.OPTIM.DAMPENING,
            momentum=float(cfg.OPTIM.MOMENTUM),
            weight_decay=float(cfg.OPTIM.WD),
            nesterov=cfg.OPTIM.NESTEROV
        )
    elif cfg.OPTIM.METHOD == "Adam":
        optimizer = optim.Adam(
            params, 
            lr=float(cfg.OPTIM.LR),
            betas=(cfg.OPTIM.BETA, 0.999),
            weight_decay=float(cfg.OPTIM.WD)
        )
    TTA_model = DeYO(
        cfg,
        model, 
        optimizer
    )
    return TTA_model