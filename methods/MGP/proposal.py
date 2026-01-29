import math
import random
import contextlib
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Iterable
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.jit
import torchvision
from torch.distributed import ReduceOp
from loguru import logger as log
from einops import rearrange
from methods.MGP.transformers_cotta import get_tta_transforms

def collect_params(model: nn.Module) -> Tuple[List, List]:
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

def update_model_probs1(current_model_probs: Optional[torch.Tensor],
                        new_probs: torch.Tensor) -> Optional[torch.Tensor]:
    if new_probs.size(0) == 0:
        return current_model_probs
    
    with torch.no_grad():
        if current_model_probs is None:
            return new_probs.mean(0)
        else:
            return 0.9 * current_model_probs + 0.1 * new_probs.mean(0)


def update_model_probs(x_ema: torch.Tensor, x: torch.Tensor,
                       momentum: float = 0.9) -> torch.Tensor:
    return momentum * x_ema + (1 - momentum) * x

class Entropy(nn.Module):
    def __init__(self, clip: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.clip = clip

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        probs = logits.softmax(1)
        probs = torch.clamp(probs, min=0.0, max=self.clip)
        return -(probs * torch.log(
            (probs / (torch.ones_like(probs) - probs)) + self.eps
        )).sum(1)


class TrustedGradientGenerator:
    def __init__(self, cfg, num_classes: int):
        self.cfg = cfg
        self.num_classes = num_classes
        self.entropy_margin = math.log(num_classes) * 0.4
        self.pred_dist_ema = None
        self.dist_ema_momentum = 0.99
        self.trusted_grad_direction = None
        self.grad_ema_momentum = 0.9

    def _compute_gradient_alignments(self,
                                      per_sample_grads: List[torch.Tensor]) -> torch.Tensor:
        if self.trusted_grad_direction is None:
            return torch.ones(len(per_sample_grads), device=per_sample_grads[0].device)
        
        alignments = []
        for g in per_sample_grads:
            g_flat = g.view(-1)
            g_norm = g_flat / (g_flat.norm() + 1e-8)
            alignment = torch.abs(g_norm @ self.trusted_grad_direction)
            alignments.append(alignment)
        
        return torch.stack(alignments)

    def update_trusted_direction(self, trusted_grad: torch.Tensor):
        g_flat = trusted_grad.view(-1)
        g_norm = g_flat / (g_flat.norm() + 1e-8)
        
        if self.trusted_grad_direction is None:
            self.trusted_grad_direction = g_norm.detach()
        else:
            self.trusted_grad_direction = (
                self.grad_ema_momentum * self.trusted_grad_direction +
                (1 - self.grad_ema_momentum) * g_norm.detach()
            )
            self.trusted_grad_direction /= (self.trusted_grad_direction.norm() + 1e-8)


class RobustSubspaceTracker:
    def __init__(self, tracked_param_names: List[str], cfg):
        self.cfg = cfg
        self.tracked = set(tracked_param_names)
        self.buffer_size = cfg.ADAPTER.MGP.BUFFER_SIZE
        self.buffers: Dict[str, List[torch.Tensor]] = {name: [] for name in self.tracked}
        self.bases: Dict[str, Optional[torch.Tensor]] = {name: None for name in self.tracked}
        self.spectral_multiplier = 1.0
        self.min_rank = 1
        self.max_rank = cfg.ADAPTER.MGP.MAX_RANK

    def add_trusted_gradient(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in self.tracked and p.grad is not None:
                g = p.grad.detach().view(-1).cpu()
                self.buffers[name].append(g)
                while len(self.buffers[name]) > self.buffer_size:
                    self.buffers[name].pop(0)

    def distill_subspace_robust(self) -> Dict[str, any]:
        for name in self.tracked:
            buf = self.buffers.get(name, [])
            if len(buf) < max(2, self.min_rank + 1):
                continue
            
            G = torch.stack(buf, dim=0).cuda()
            n, d = G.shape
            G_centered = G - G.mean(dim=0, keepdim=True)
            
            _, S, Vh = torch.linalg.svd(G_centered, full_matrices=False)
            eigenvalues = S ** 2
            threshold = self._compute_mp_threshold(eigenvalues, n, d)
            significant_mask = eigenvalues > threshold
            r_new = max(self.min_rank, min(significant_mask.sum().item(), self.max_rank))
            B_new = Vh[:r_new, :].T
            
            B_old = self.bases.get(name, None)
            if B_old is not None:
                B_old = B_old.cuda()
                B_combined = self._inertial_fusion(B_old, B_new)
            else:
                B_combined = B_new
            
            Q, _ = torch.linalg.qr(B_combined)
            self.bases[name] = Q.cpu()

    def _compute_mp_threshold(self, eigenvalues: torch.Tensor,
                               n: int, d: int) -> torch.Tensor:
        gamma = d / n
        mp_upper_edge = (1 + math.sqrt(gamma)) ** 2
        n_noise = max(len(eigenvalues) // 2, 1)
        noise_eigenvalues = eigenvalues[-n_noise:]
        sigma_sq_estimate = noise_eigenvalues.median()
        threshold = sigma_sq_estimate * mp_upper_edge
        return threshold

    def _inertial_fusion(self, B_old: torch.Tensor,
                         B_new: torch.Tensor) -> torch.Tensor:
        B_new_residual = B_new - B_old @ (B_old.T @ B_new)
        residual_norms = torch.norm(B_new_residual, dim=0)
        threshold = self.cfg.ADAPTER.MGP.RESIDUAL_NORM_THRESHOLD
        significant_new = residual_norms > threshold
        
        if significant_new.sum() > 0:
            B_novel = B_new_residual[:, significant_new]
            B_novel = B_novel / (torch.norm(B_novel, dim=0, keepdim=True) + 1e-8)
            B_combined = torch.cat([B_old, B_novel], dim=1)
        else:
            B_combined = B_old
        
        if B_combined.shape[1] > self.max_rank:
            B_combined = B_combined[:, :self.max_rank]
        
        return B_combined

    def _subspace_angle(self, B1: torch.Tensor, B2: torch.Tensor) -> float:
        P1 = B1 @ B1.T
        P2 = B2 @ B2.T
        drift = torch.norm(P1 - P2, p='fro').item()
        return drift

    @torch.no_grad()
    def project_gradient(self, model: nn.Module):
        if not any(len(buf) > self.min_rank for buf in self.buffers.values()):
            return
        
        for name, p in model.named_parameters():
            if name in self.tracked and p.grad is not None:
                basis = self.bases.get(name, None)
                if basis is None:
                    continue
                
                B = basis.to(p.grad.device)
                g = p.grad.view(-1)
                g_parallel = B @ (B.T @ g)
                g_perp = g - g_parallel
                p.grad.copy_(g_perp.view_as(p.grad))


class RobustMGP(nn.Module):
    def __init__(self, cfg, model: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.model = configure_model(deepcopy(model))
        params, param_names = collect_params(self.model)
        
        self.optimizer = self._setup_optimizer(params, cfg)
        self.trusted_generator = TrustedGradientGenerator(cfg, num_classes=cfg.CORRUPTION.NUM_CLASS)
        self.subspace_tracker = RobustSubspaceTracker(param_names, cfg)
        
        self.batch_index = 0
        self.slow_freq = cfg.ADAPTER.MGP.DISTILL_FREQ
        self.softmax_entropy_clamp = Entropy()
        self.tta_transform = get_tta_transforms(cfg, padding_mode="reflect", cotta_augs=False)
        
        self.num_classes = cfg.CORRUPTION.NUM_CLASS
        self.class_probs_ema = torch.ones(self.num_classes).cuda() / self.num_classes
        self.batch_size = cfg.TEST.BATCH_SIZE
        
        self.loss_type = cfg.ADAPTER.MGP.LOSS_TYPE
        self.collect_freq = getattr(cfg.ADAPTER.MGP, 'COLLECT_FREQ', 40)

        # ============ ETA Specific Parameters ============
        self.current_model_probs = None
        self.e_margin = math.log(self.num_classes) * getattr(cfg.ADAPTER.ETA, 'E_MARGIN', 0.4)
        self.d_margin = getattr(cfg.ADAPTER.ETA, 'D_MARGIN', 0.05)
        self.num_samples_update_1 = 0
        self.num_samples_update_2 = 0
        # ============ DeYO Specific Parameters ============
        self.reweight_ent = getattr(cfg.ADAPTER.DeYO, 'REWEIGHT_ENT', True)
        self.reweight_plpd = getattr(cfg.ADAPTER.DeYO, 'REWEIGHT_PLPD', False)
        self.plpd_threshold = getattr(cfg.ADAPTER.DeYO, 'PLPD', 0.0)
        self.deyo_margin = getattr(cfg.ADAPTER.DeYO, 'MARGIN', 0.5) * math.log(self.num_classes)
        self.margin_e0 = getattr(cfg.ADAPTER.DeYO, 'MARGIN_E0', 0.4) * math.log(self.num_classes)
        self.aug_type = getattr(cfg.ADAPTER.DeYO, 'AUG_TYPE', 'occ')
        self.occlusion_size = getattr(cfg.ADAPTER.DeYO, 'OCCLUSION_SIZE', 112)
        self.row_start = getattr(cfg.ADAPTER.DeYO, 'ROW_START', 56)
        self.column_start = getattr(cfg.ADAPTER.DeYO, 'COLUMN_START', 56)
        self.patch_len = getattr(cfg.ADAPTER.DeYO, 'PATCH_LEN', 4)

    def _setup_optimizer(self, params: List, cfg) -> torch.optim.Optimizer:
        if cfg.OPTIM.METHOD == "SGD":
            return torch.optim.SGD(params, lr=float(cfg.OPTIM.LR), momentum=0.9)
        return torch.optim.Adam(params, lr=float(cfg.OPTIM.LR))

    def loss_calculation(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.loss_type == 'eta':
            return self._eta_loss_calculation(x)
        elif self.loss_type == 'deyo':
            return self._deyo_loss_calculation(x)
        else:
            return self._default_loss_calculation(x)

    def _default_loss_calculation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(x)
        entropys = self.softmax_entropy_clamp(outputs, outputs)
        loss = entropys.mean(0)
        return outputs, loss

    def _eta_loss_calculation(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        outputs = self.model(x)
        entropys = softmax_entropy(outputs)
        
        filter_ids_1 = torch.where(entropys < self.e_margin)
        ids1 = filter_ids_1
        ids2 = torch.where(ids1[0] > -0.1)
        entropys_filtered = entropys[filter_ids_1]
        
        if entropys_filtered.size(0) == 0:
            return outputs, None
        
        if self.current_model_probs is not None:
            cosine_similarities = F.cosine_similarity(
                self.current_model_probs.unsqueeze(dim=0),
                outputs[filter_ids_1].softmax(1),
                dim=1
            )
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
            entropys_filtered = entropys_filtered[filter_ids_2]
            ids2 = filter_ids_2
            updated_probs = update_model_probs1(
                self.current_model_probs,
                outputs[filter_ids_1][filter_ids_2].softmax(1)
            )
        else:
            updated_probs = update_model_probs1(
                self.current_model_probs,
                outputs[filter_ids_1].softmax(1)
            )
        
        if entropys_filtered.size(0) == 0:
            self.current_model_probs = updated_probs
            return outputs, None
        
        coeff = 1.0 / (torch.exp(entropys_filtered.clone().detach() - self.e_margin))
        entropys_weighted = entropys_filtered.mul(coeff)
        loss = entropys_weighted.mean(0)
        
        self.num_samples_update_1 += filter_ids_1[0].size(0)
        self.num_samples_update_2 += entropys_filtered.size(0)
        self.current_model_probs = updated_probs
        
        return outputs, loss

    def _deyo_loss_calculation(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if x.size(0) < 2:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)
            self.model.train()
            return outputs, None
        
        outputs = self.model(x)
        entropys = softmax_entropy(outputs)
        
        filter_ids_1 = torch.where(entropys < self.deyo_margin)
        if len(filter_ids_1[0]) < 2:
            return outputs, None
        
        entropys_filtered = entropys[filter_ids_1]
        if len(entropys_filtered) == 0:
            return outputs, None
        
        x_prime = x[filter_ids_1].detach()
        x_prime = self._create_deyo_augmentation(x_prime, x.shape[-1])
        
        with torch.no_grad():
            outputs_prime = self.model(x_prime)
        
        prob_outputs = outputs[filter_ids_1].softmax(1)
        prob_outputs_prime = outputs_prime.softmax(1)
        cls1 = prob_outputs.argmax(dim=1)
        
        plpd = (torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1, 1)) - 
                torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1, 1)))
        plpd = plpd.reshape(-1)
        
        filter_ids_2 = torch.where(plpd > self.plpd_threshold)
        entropys_filtered = entropys_filtered[filter_ids_2]
        
        if len(entropys_filtered) == 0:
            return outputs, None
        
        plpd = plpd[filter_ids_2]
        
        if self.reweight_ent or self.reweight_plpd:
            coeff = (
                float(self.reweight_ent) * (1.0 / (torch.exp(entropys_filtered.clone().detach() - self.margin_e0))) +
                float(self.reweight_plpd) * (1.0 / (torch.exp(-1.0 * plpd.clone().detach())))
            )
            entropys_filtered = entropys_filtered.mul(coeff)
        
        loss = entropys_filtered.mean(0)
        return outputs, loss

    def _create_deyo_augmentation(self, x: torch.Tensor, img_size: int) -> torch.Tensor:
        x = x.clone()
        
        if self.aug_type == 'occ':
            first_mean = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
            final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
            occlusion_window = final_mean.expand(-1, -1, self.occlusion_size, self.occlusion_size)
            x[:, :, self.row_start:self.row_start + self.occlusion_size,
              self.column_start:self.column_start + self.occlusion_size] = occlusion_window
              
        elif self.aug_type == 'patch':
            resize_t = torchvision.transforms.Resize(
                ((img_size // self.patch_len) * self.patch_len,
                 (img_size // self.patch_len) * self.patch_len)
            )
            resize_o = torchvision.transforms.Resize((img_size, img_size))
            x = resize_t(x)
            x = rearrange(x, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w',
                         ps1=self.patch_len, ps2=self.patch_len)
            perm_idx = torch.argsort(torch.rand(x.shape[0], x.shape[1], device=x.device), dim=-1)
            x = x[torch.arange(x.shape[0], device=x.device).unsqueeze(-1), perm_idx]
            x = rearrange(x, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)',
                         ps1=self.patch_len, ps2=self.patch_len)
            x = resize_o(x)
            
        elif self.aug_type == 'pixel':
            x = rearrange(x, 'b c h w -> b c (h w)')
            x = x[:, :, torch.randperm(x.shape[-1], device=x.device)]
            x = rearrange(x, 'b c (ps1 ps2) -> b c ps1 ps2', ps1=img_size, ps2=img_size)
            
        return x

    @torch.enable_grad()
    def forward(self, x: torch.Tensor, y: torch.Tensor = None,
                adapt: bool = True) -> torch.Tensor:
        if not adapt:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)
            self.model.train()
            return outputs
        
        self.optimizer.zero_grad()
        
        outputs, loss = self.loss_calculation(x)
        
        if loss is not None:
            loss.backward()
            
            self.subspace_tracker.project_gradient(self.model)
            
            if self.batch_index % self.collect_freq == 0:
                self.subspace_tracker.add_trusted_gradient(self.model)
                self._update_trusted_direction()
            
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        if self.batch_index % self.slow_freq == 0 and self.batch_index > 0:
            self.subspace_tracker.distill_subspace_robust()
        
        self.batch_index += 1
        return outputs

    def _update_trusted_direction(self):
        all_grads = []
        for name, p in self.model.named_parameters():
            if name in self.subspace_tracker.tracked and p.grad is not None:
                all_grads.append(p.grad.view(-1))
        
        if all_grads:
            concat_grad = torch.cat(all_grads)
            self.trusted_generator.update_trusted_direction(concat_grad)

    def reset(self):
        self.current_model_probs = None
        self.num_samples_update_1 = 0
        self.num_samples_update_2 = 0
        self.batch_index = 0


def setup(model: nn.Module, cfg) -> RobustMGP:
    return RobustMGP(cfg, model)
