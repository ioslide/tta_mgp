"""
Builds upon: https://github.com/mr-eggplant/ETA
Corresponding paper: https://arxiv.org/abs/2204.02610
"""
from torch.nn.utils.weight_norm import WeightNorm
import math
import logging
from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
from loguru import logger as log
from core.data.data_loading import get_source_loader
from tqdm import tqdm
__all__ = ["setup"]


class ETA_model(nn.Module):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        

        self.model, self.optimizer = prepare_eata_model_and_optimizer(model, cfg)
        self.num_samples_update_1 = 0 
        self.num_samples_update_2 = 0
        self.e_margin = math.log(cfg.CORRUPTION.NUM_CLASS) * 0.40
        self.d_margin = cfg.ADAPTER.ETA.D_MARGIN

        self.current_model_probs = None
        self.fisher_alpha = cfg.ADAPTER.ETA.FISHER_ALPHA
        self.steps = cfg.OPTIM.STEPS
        assert self.steps > 0, "tent requires >= 1 step(s) to forward and update"

        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, y=None, adapt=True):
        outputs = self.forward_and_adapt(x)

        return outputs

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        outputs = self.model(x)
        entropys = softmax_entropy(outputs)

        filter_ids_1 = torch.where(entropys < self.e_margin)
        ids1 = filter_ids_1
        ids2 = torch.where(ids1[0] > -0.1)
        entropys = entropys[filter_ids_1]

        if self.current_model_probs is not None:
            cosine_similarities = F.cosine_similarity(self.current_model_probs.unsqueeze(dim=0),
                                                      outputs[filter_ids_1].softmax(1), dim=1)
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
            entropys = entropys[filter_ids_2]
            ids2 = filter_ids_2
            updated_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
        else:
            updated_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1].softmax(1))
        coeff = 1 / (torch.exp(entropys.clone().detach() - self.e_margin))

        entropys = entropys.mul(coeff)  # reweight entropy losses for diff. samples
        loss = entropys.mean(0)


        if x[ids1][ids2].size(0) != 0:
            loss.backward()
            self.optimizer.step()
        else:
            outputs = outputs.detach()
        self.optimizer.zero_grad()

        self.num_samples_update_1 += filter_ids_1[0].size(0)
        self.num_samples_update_2 += entropys.size(0)
        self.reset_model_probs(updated_probs)
        return outputs

    def reset_steps(self, new_steps):
        self.steps = new_steps

    def reset_model_probs(self, probs):
        self.current_model_probs = probs
    

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def prepare_eata_model_and_optimizer(model, cfg):
    model = configure_model(model)
    params, _ = collect_params(model)

    optimizer = torch.optim.SGD(params, lr=cfg.OPTIM.LR, momentum=cfg.OPTIM.BETA)
    return model, optimizer

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())

    return model_state, optimizer_state


def collect_params(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names



def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x


def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)


def setup(model, cfg):
    log.info("Setup TTA method: ETA")
    model = ETA_model(
        cfg,
        model
    )
    log.info(" > model for adaptation: %s", model)
    return model

