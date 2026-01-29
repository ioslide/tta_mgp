from copy import deepcopy
from loguru import logger as log
import torch
import torch.nn as nn
import torch.jit
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import PIL
import torchvision.transforms as transforms
from typing import Dict, Tuple, Optional, List
import methods.ViDA.cotta_transforms as my_transforms

__all__ = ["setup"]

class ViDAInjectedMultiheadAttention(nn.Module):
    """
    Wrap torchvision's nn.MultiheadAttention to support ViDA adapters on:
      - packed QKV projection (in_proj_weight: [3E, E])
      - output projection (out_proj.weight: [E, E])
    This matches timm's injecting into `qkv` and `proj`.
    """
    def __init__(self, mha: nn.MultiheadAttention, r: int = 4, r2: int = 64):
        super().__init__()
        assert isinstance(mha, nn.MultiheadAttention)
        self.mha = mha

        # Basic attrs
        self.embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        self.dropout = mha.dropout
        self.batch_first = getattr(mha, "batch_first", False)

        # ViDA scales (HKA will set them dynamically)
        self.scale1 = 1.0
        self.scale2 = 1.0

        E = self.embed_dim

        # ----- Adapters for packed QKV (E -> 3E) -----
        self.vida_in_down  = nn.Linear(E, r,  bias=False)
        self.vida_in_up    = nn.Linear(r,  3 * E, bias=False)
        self.vida_in_down2 = nn.Linear(E, r2, bias=False)
        self.vida_in_up2   = nn.Linear(r2, 3 * E, bias=False)

        # ----- Adapters for out_proj (E -> E) -----
        self.vida_out_down  = nn.Linear(E, r,  bias=False)
        self.vida_out_up    = nn.Linear(r,  E, bias=False)
        self.vida_out_down2 = nn.Linear(E, r2, bias=False)
        self.vida_out_up2   = nn.Linear(r2, E, bias=False)

        # Init like your ViDAInjectedLinear
        nn.init.normal_(self.vida_in_down.weight,  std=1 / (r**2))
        nn.init.zeros_(self.vida_in_up.weight)
        nn.init.normal_(self.vida_in_down2.weight, std=1 / (r2**2))
        nn.init.zeros_(self.vida_in_up2.weight)

        nn.init.normal_(self.vida_out_down.weight,  std=1 / (r**2))
        nn.init.zeros_(self.vida_out_up.weight)
        nn.init.normal_(self.vida_out_down2.weight, std=1 / (r2**2))
        nn.init.zeros_(self.vida_out_up2.weight)

    def _delta_weight(self, up: nn.Linear, down: nn.Linear) -> torch.Tensor:
        # adapter(x) = up(down(x))  <=>  x @ (up.weight @ down.weight)^T
        # so delta_W = up.weight @ down.weight
        return up.weight @ down.weight  # [out, in]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask=None,
        need_weights: bool = False,
        attn_mask=None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        mha = self.mha

        # torchvision ViT uses self_attention(x, x, x) with batch_first=True typically.
        # F.multi_head_attention_forward expects (L, N, E) unless we transpose.
        if self.batch_first:
            # (N, L, E) -> (L, N, E)
            query_t = query.transpose(0, 1)
            key_t   = key.transpose(0, 1)
            value_t = value.transpose(0, 1)
        else:
            query_t, key_t, value_t = query, key, value

        # Only handle the common ViT case (same embed_dim, packed in_proj_weight)
        assert mha.in_proj_weight is not None, "Expected packed in_proj_weight for ViT-style MHA."
        assert mha.in_proj_bias is not None or True  # bias may be None in some configs

        # ----- Build delta weights -----
        delta_in  = self._delta_weight(self.vida_in_up,  self.vida_in_down)  * self.scale1 \
                  + self._delta_weight(self.vida_in_up2, self.vida_in_down2) * self.scale2   # [3E, E]
        delta_out = self._delta_weight(self.vida_out_up,  self.vida_out_down)  * self.scale1 \
                  + self._delta_weight(self.vida_out_up2, self.vida_out_down2) * self.scale2 # [E, E]

        in_proj_weight = mha.in_proj_weight + delta_in.to(mha.in_proj_weight.dtype)
        out_proj_weight = mha.out_proj.weight + delta_out.to(mha.out_proj.weight.dtype)

        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query_t,
            key_t,
            value_t,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=in_proj_weight,
            in_proj_bias=mha.in_proj_bias,
            bias_k=mha.bias_k,
            bias_v=mha.bias_v,
            add_zero_attn=mha.add_zero_attn,
            dropout_p=self.dropout,
            out_proj_weight=out_proj_weight,
            out_proj_bias=mha.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=False,
            average_attn_weights=average_attn_weights,
        )

        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        # 始终返回两个值，匹配 nn.MultiheadAttention 的接口
        return attn_output, attn_output_weights

class ViDAInjectedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, r=4, r2=64):
        super().__init__()
        self.linear_vida = nn.Linear(in_features, out_features, bias)
        # Low-rank adapter (Domain-shared)
        self.vida_down = nn.Linear(in_features, r, bias=False)
        self.vida_up = nn.Linear(r, out_features, bias=False)
        # High-rank adapter (Domain-specific)
        self.vida_down2 = nn.Linear(in_features, r2, bias=False)
        self.vida_up2 = nn.Linear(r2, out_features, bias=False)
        
        self.scale1 = 1.0 # Dynamic scale for low-rank
        self.scale2 = 1.0 # Dynamic scale for high-rank

        # Initialization
        nn.init.normal_(self.vida_down.weight, std=1 / r**2)
        nn.init.zeros_(self.vida_up.weight)
        nn.init.normal_(self.vida_down2.weight, std=1 / r2**2)
        nn.init.zeros_(self.vida_up2.weight)

    def forward(self, input):
        # Original output + Scaled Low-rank + Scaled High-rank
        return (self.linear_vida(input) + 
                self.vida_up(self.vida_down(input)) * self.scale1 + 
                self.vida_up2(self.vida_down2(input)) * self.scale2)

def inject_trainable_vida_torchvision_vit(model: nn.Module, r: int = 1, r2: int = 128):
    """
    For torchvision VisionTransformer (vit_b_16):
      - Replace EncoderBlock.self_attention (nn.MultiheadAttention) with ViDAInjectedMultiheadAttention
      - Replace EncoderBlock.mlp's Linear with ViDAInjectedLinear
    """
    require_grad_params = []
    names = []

    for module in model.modules():
        if module.__class__.__name__ == "EncoderBlock":
            # 1) self_attention
            if hasattr(module, "self_attention") and isinstance(module.self_attention, nn.MultiheadAttention):
                old_mha = module.self_attention
                new_mha = ViDAInjectedMultiheadAttention(old_mha, r=r, r2=r2)
                module.self_attention = new_mha

                for p in new_mha.parameters():
                    if "vida_" in p.__repr__():
                        pass  # not reliable; we collect below by name in named_parameters anyway

                # Collect adapter parameters explicitly
                for n, p in new_mha.named_parameters():
                    if "vida_" in n:
                        p.requires_grad = True
                        require_grad_params.append(p)
                        names.append(f"self_attention.{n}")

            # 2) MLPBlock (indices 0 and 3 are Linear in torchvision)
            if hasattr(module, "mlp"):
                for child_name, child in list(module.mlp.named_children()):
                    if isinstance(child, nn.Linear):
                        weight = child.weight
                        bias = child.bias
                        tmp = ViDAInjectedLinear(
                            child.in_features,
                            child.out_features,
                            bias is not None,
                            r,
                            r2,
                        )
                        tmp.linear_vida.weight = weight
                        if bias is not None:
                            tmp.linear_vida.bias = bias

                        module.mlp._modules[child_name] = tmp

                        for param in tmp.vida_up.parameters():   require_grad_params.append(param)
                        for param in tmp.vida_down.parameters(): require_grad_params.append(param)
                        for param in tmp.vida_up2.parameters():  require_grad_params.append(param)
                        for param in tmp.vida_down2.parameters():require_grad_params.append(param)

                        tmp.vida_up.weight.requires_grad = True
                        tmp.vida_down.weight.requires_grad = True
                        tmp.vida_up2.weight.requires_grad = True
                        tmp.vida_down2.weight.requires_grad = True

                        names.append(f"mlp.{child_name}")

    return require_grad_params, names

class Clip(object):
    def __init__(self, min_val=0.0, max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val
    def __call__(self, tensor):
        return torch.clamp(tensor, self.min_val, self.max_val)

class GaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()).to(tensor.device) * self.std + self.mean


def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False):
    img_shape = (224, 224, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5
    try:
        # 新版 torchvision 写法
        interpolation = transforms.InterpolationMode.BILINEAR
    except AttributeError:
        # 旧版兼容或者是 PIL 写法
        import PIL
        interpolation = PIL.Image.BILINEAR

    tta_transforms = transforms.Compose([
        my_transforms.Clip(0.0, 1.0), 
        my_transforms.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),  
        
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            interpolation=interpolation,
            fill=None 
        ),
        
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        my_transforms.GaussianNoise(0, gaussian_std),
        my_transforms.Clip(clip_min, clip_max)
    ])
    return tta_transforms

@torch.jit.script
def symmetric_consistency_loss(x: torch.Tensor, x_ema: torch.Tensor) -> torch.Tensor:
    return -0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum(1) - 0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum(1)

def update_ema_variables(ema_model, model, alpha_teacher, alpha_vida):
    for ema_param, (name, param) in zip(ema_model.parameters(), model.named_parameters()):
        if "vida_" in name:
            ema_param.data[:] = alpha_vida * ema_param[:].data[:] + (1 - alpha_vida) * param[:].data[:]
        else:
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]

class TTAMethod(nn.Module):
    def __init__(self, cfg, model, optimizer):
        super().__init__()
        self.model = model # Student model (with injected ViDA)
        self.optimizer = optimizer
        self.cfg = cfg
        self.steps = cfg.OPTIM.STEPS
        
        # Hyperparameters
        self.unc_thr = cfg.ADAPTER.ViDA.UNC_THR 
        self.alpha_teacher = cfg.ADAPTER.ViDA.MT
        self.alpha_vida = cfg.ADAPTER.ViDA.MT_ViDA

        # Setup Teacher Model (EMA)
        self.model_ema = deepcopy(self.model)
        for param in self.model_ema.parameters():
            param.detach_()
        
        # Transforms for uncertainty estimation
        self.transform = get_tta_transforms()

    @torch.enable_grad()
    def forward(self, x, y=None, adapt=True):
        if adapt:
            outputs = self.forward_and_adapt(x)
        else:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)
            self.model.train()
        return outputs

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)
        return outputs

    def set_scale(self, update_model, high, low):
        for _, module in update_model.named_modules():
            if isinstance(module, (ViDAInjectedLinear, ViDAInjectedMultiheadAttention)):
                module.scale1 = low.item() if torch.is_tensor(low) else low
                module.scale2 = high.item() if torch.is_tensor(high) else high
            
    @torch.enable_grad()
    def forward_and_adapt(self, x):
        self.model_ema.eval()
        self.model.train() # Student needs to be in train mode for grads

        # 1. Uncertainty Estimation (using Teacher)
        N = 10 
        outputs_uncs = []
        with torch.no_grad():
            for i in range(N):
                # Apply transforms for variance calculation
                outputs_ = self.model_ema(self.transform(x)).detach()
                outputs_uncs.append(outputs_)
            outputs_unc = torch.stack(outputs_uncs)
            variance = torch.var(outputs_unc, dim=0)
            uncertainty = torch.mean(variance) * 0.1 # Scaling factor from ViDA code

        # 2. Homeostatic Knowledge Allotment (HKA)
        # Determine weights based on uncertainty
        if uncertainty >= self.unc_thr:
            lambda_high = 1 + uncertainty
            lambda_low = 1 - uncertainty
        else:
            lambda_low = 1 + uncertainty
            lambda_high = 1 - uncertainty
        
        # Apply scales to both models
        self.set_scale(update_model=self.model, high=lambda_high, low=lambda_low)
        self.set_scale(update_model=self.model_ema, high=lambda_high, low=lambda_low)

        # 3. Forward Pass & Loss
        standard_ema = self.model_ema(x) # Teacher prediction (anchor)
        outputs = self.model(x)          # Student prediction

        loss = symmetric_consistency_loss(outputs, standard_ema.detach()).mean(0)

        # 4. Optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 5. Update Teacher (EMA)
        update_ema_variables(self.model_ema, self.model, self.alpha_teacher, self.alpha_vida)

        return outputs # Return student output or standard_ema depending on preference, ViDA returns ema usually

    @torch.no_grad()
    def get_adaptable_vector(self) -> torch.Tensor:
        """
        Flattens all trainable ViDA parameters into a 1D vector.
        """
        params, names = collect_params(self.model)
        if not params:
            raise ValueError("No adaptable parameters found in the model.")
        return torch.cat([p.detach().flatten().cpu() for p in params], dim=0)

    @torch.no_grad()
    def load_adaptable_vector(self, vec: torch.Tensor) -> None:
        """
        Restores ViDA parameters from a 1D vector.
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
    """Collect trainable ViDA parameters (up/down projections)."""
    params, names = [], []
    for name, param in model.named_parameters():
        if 'vida_' in name and param.requires_grad:
            params.append(param)
            names.append(name)
    return params, names

def configure_model(model: nn.Module, cfg) -> nn.Module:
    """
    Configure model: Freeze original params, inject ViDA adapters, set adapters to trainable.
    """
    model.eval() # Base model is frozen
    model.requires_grad_(False)
    
    # Inject ViDA (Rank defaults to 4 and 16/64 based on ViDA paper/code)
    r1 = cfg.ADAPTER.ViDA.RANK1 
    r2 = cfg.ADAPTER.ViDA.RANK2
    
    # inject_trainable_vida(model, r=r1, r2=r2)
    inject_trainable_vida_torchvision_vit(model, r=r1, r2=r2)

    # Ensure only ViDA params are trainable
    for n, p in model.named_parameters():
        if 'vida_' not in n:
            p.requires_grad = False
            
    return model

def setup(model, cfg):
    log.info("Setup TTA method: ViDA (Visual Domain Adapter)")
    
    # 1. Configure Model (Inject Adapters)
    model_configured = configure_model(deepcopy(model), cfg) 
    
    # 2. Collect Params
    # ViDA often separates LRs for backbone (if adapted) and ViDA params. 
    # Since we froze backbone, we only have ViDA params here.
    params, param_names = collect_params(model_configured)
    
    if not params:
        raise ValueError("No adaptable parameters found in the model.")
    
    # 3. Setup Optimizer
    # Note: ViDA paper uses specific LRs for adapters. 
    # If cfg.OPTIM.ViDALR is present, use it, otherwise use standard LR.
    lr = cfg.OPTIM.LR
    
    if cfg.OPTIM.METHOD == "SGD":
        optimizer = optim.SGD(
            params,
            lr=lr,
            momentum=float(getattr(cfg.OPTIM, "MOMENTUM", 0.9)),
            weight_decay=float(getattr(cfg.OPTIM, "WD", 0.0)),
            nesterov=bool(getattr(cfg.OPTIM, "NESTEROV", False)),
        )
    elif cfg.OPTIM.METHOD == "Adam":
        optimizer = optim.Adam(
            params,
            lr=lr,
            betas=(float(getattr(cfg.OPTIM, "BETA", 0.9)), 0.999),
            weight_decay=float(getattr(cfg.OPTIM, "WD", 0.0)),
        )
    else:
        raise ValueError(f"Unsupported optimizer method: {cfg.OPTIM.METHOD}")

    # 4. Initialize Method
    tta_model = TTAMethod(cfg, model_configured, optimizer)
    return tta_model