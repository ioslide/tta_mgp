from copy import deepcopy

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
import torchvision.transforms as transforms
from PIL import ImageFilter, ImageOps
import torch
import math
import warnings
import torchvision.transforms.functional as TF
import torch.nn as nn
from typing import Tuple, List, Optional
import numpy as np
import math

__all__ = ["setup"]

class LearnablePatchErasing(torch.nn.Module):
    def __init__(self, patch_size=16, img_size=224, noise_ratio=0.4):
        super().__init__()
        self.img_size = img_size

        self.patch_size = patch_size

        h = w = self.img_size // self.patch_size
        self.alpha = nn.Parameter(torch.ones(h * w)* math.sqrt(noise_ratio), requires_grad=True)

    def forward(self, img):
        """
        Args:
            img (Tensor): Tensor image to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
        """
        assert img.shape[-1] == img.shape[-2] and img.shape[-1] % self.patch_size == 0

        img_c = img.shape[-3]
        h = w = img.shape[-1] // self.patch_size # h 是个数
        patches = [(i*self.patch_size, j*self.patch_size) for i in range(h) for j in range(w)]
        new_img = torch.empty_like(img)
        for _, patch in enumerate(patches):
            v = torch.empty([img_c, self.patch_size, self.patch_size], dtype=torch.float32, device=img.device).normal_()
            new_img[...,patch[0]:patch[0]+self.patch_size, patch[1]:patch[1]+self.patch_size] = \
                (1 - self.alpha[_]**2) * img[..., patch[0]:patch[0]+self.patch_size, patch[1]:patch[1]+self.patch_size] + self.alpha[_]**2 * v
        new_img.clip_(-1, 1)
        return new_img

class SPA(nn.Module):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, cfg, model, optimizer, noise_ratio=0.4, freq_mask_ratio=0.2, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        
        self.prompts = Prompt(prompt_alpha=0.2).cuda()

        self.learnable_patch_erasing = LearnablePatchErasing(patch_size=16).cuda()
        self.patch_optimizer = torch.optim.SGD([self.learnable_patch_erasing.alpha], 1, momentum=0.9)


        if cfg.MODEL.ARCH in ['vit_base_patch16_224', 'vit_b_16']:
            freq_mask_ratio, noise_ratio = 0.2, 0.4        
            self.noise_ratio = noise_ratio # noise
            self.keep_prob = 1 - freq_mask_ratio # freq
        elif cfg.MODEL.ARCH in ['resnet50', 'resnet101', 'resnet152']:
            freq_mask_ratio, noise_ratio = 0.1, 0.1
            self.noise_ratio = noise_ratio # noise
            self.keep_prob = 1 - freq_mask_ratio # freq
        else:
            freq_mask_ratio, noise_ratio = 0.1, 0.1
            self.noise_ratio = noise_ratio # noise
            self.keep_prob = 1 - freq_mask_ratio # freq

        self.use_eata = False
        self.use_tent = False
        self.use_actmad = False
        self.e_margin = math.log(1000)*0.40
        self.d_margin = 0.05

    def forward(self, x, y=None,adapt=True):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt_eata(x)

        return outputs
    
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt_eata(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        Return: 
        1. model outputs; 
        2. the number of reliable and non-redundant samples; 
        3. the number of reliable samples;
        4. the moving average  probability vector over all previous samples
        """
        # forward
        outputs = self.model(x)

        loss = 0
        noise_x = self.prompts.masking(x, keep_prob=self.keep_prob)
        loss += self.get_low_consist_loss(noise_x, outputs)

        noise_x = self.learnable_patch_erasing(x)
        loss += self.get_low_consist_loss(noise_x, outputs)

        loss += self.get_auxiliary_loss(outputs)

        loss.backward()
        self.learnable_patch_erasing.alpha.grad = -self.learnable_patch_erasing.alpha.grad

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.patch_optimizer.step()
        self.patch_optimizer.zero_grad()
        self.fix_patch_alpha()
        return outputs
    
    def get_auxiliary_loss(self, outputs):
        loss = 0
        # if self.use_tent:
        loss += 0.1 * softmax_entropy(outputs).mean()
        # elif self.use_eata:
        #     loss += 0.1 * self.get_eta_loss(outputs)
        return loss
    
    def get_eta_loss(self, outputs):
        # adapt
        entropys = softmax_entropy(outputs)
        # filter unreliable samples
        filter_ids_1 = torch.where(entropys < self.e_margin)
        ids1 = filter_ids_1
        ids2 = torch.where(ids1[0]>-0.1)
        entropys = entropys[filter_ids_1] 
        # filter redundant samples
        if self.current_model_probs is not None: 
            cosine_similarities = F.cosine_similarity(self.current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
            entropys = entropys[filter_ids_2]
            ids2 = filter_ids_2
            updated_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
        else:
            updated_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1].softmax(1))
        self.current_model_probs = updated_probs
        
        coeff = 1 / (torch.exp(entropys.clone().detach() - self.e_margin))
        # implementation version 1, compute loss, all samples backward (some unselected are masked)
        entropys = entropys.mul(coeff) # reweight entropy losses for diff. samples
        
        loss = 0
        if ids2[0].shape[0] > 0:
            loss += entropys.mean(0)
        
        if self.fishers is not None:
            ewc_loss = 0
            for name, param in self.model.named_parameters():
                if name in self.fishers:
                    ewc_loss += self.fisher_alpha * (self.fishers[name][0] * (param - self.fishers[name][1])**2).sum()
            loss += ewc_loss
        return loss
    
    def get_low_consist_loss(self, x, outputs):
        low_outputs = self.model(x, use_predictor=True)

        conf, labels = outputs.softmax(dim=-1).max(dim=-1)
        low_conf, low_labels = low_outputs.softmax(dim=-1).max(dim=-1)
        conf_ids = torch.where(conf > low_conf)
        return get_loss_kl(low_outputs[conf_ids], outputs[conf_ids]).mean()
    
    @torch.no_grad()
    def fix_patch_alpha(self): # control the average noise ratio
        current_mean = (self.learnable_patch_erasing.alpha ** 2).mean()
        div = current_mean / self.noise_ratio
        self.learnable_patch_erasing.alpha /= torch.sqrt(div)

        torch.clip_(self.learnable_patch_erasing.alpha, -1, 1)
    
    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.current_model_probs = None
        self.learnable_patch_erasing.alpha.data = torch.ones_like(self.learnable_patch_erasing.alpha) * math.sqrt(self.noise_ratio)
        self.patch_optimizer = torch.optim.SGD([self.learnable_patch_erasing.alpha], 1, momentum=0.9)

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

criterion_kl = nn.KLDivLoss(reduction='batchmean').cuda()
def get_loss_kl(outputs, targets):
    return criterion_kl(outputs.log_softmax(dim=-1), targets.softmax(dim=-1).clone().detach())

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    if hasattr(model, 'predictor'):
        model.predictor.requires_grad_(True)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"

class Prompt(nn.Module):
    def __init__(self, prompt_alpha=0.5, num_range=200):
        transforms.ColorJitter
        super().__init__()
        imageH, imageW = 224, 224
        promptH = int(imageH * prompt_alpha) if int(imageH * prompt_alpha) > 1 else 1
        promptW = int(imageW * prompt_alpha) if int(imageW * prompt_alpha) > 1 else 1
        paddingH = (imageH - promptH) // 2
        paddingW = (imageW - promptW) // 2
        self.source_frequency, self.target_frequency = None, None

        # all kind of masking
        self.init_para = torch.ones((1, 3, promptH, promptW)).cuda()
        self.low_mask = F.pad(self.init_para, [imageH - paddingH - promptH, paddingH,
                                               imageW - paddingW - promptW, paddingW],
                       mode='constant', value=0.).contiguous()
        self.high_mask = 1 - self.low_mask


    def iFFT(self, amp_src_, pha_src, imgH, imgW):
        # recompose fft
        real = torch.cos(pha_src) * amp_src_
        imag = torch.sin(pha_src) * amp_src_
        fft_src_ = torch.complex(real=real, imag=imag)

        src_in_trg = torch.fft.ifft2(fft_src_, dim=(-2, -1), s=[imgH, imgW]).real
        return src_in_trg
    
    # def masking(self, x, keep_prob=0.9):

    #     _, _, imgH, imgW = x.size()

    #     fft = torch.fft.fft2(x.clone(), dim=(-2, -1))

    #     # extract amplitude and phase of both ffts
    #     amp_src, pha_src = torch.abs(fft), torch.angle(fft)
    #     amp_src = torch.fft.fftshift(amp_src)

    #     amp_src = self._random_mask_on_low_frequency_only(amp_src, keep_prob)

    #     amp_src = torch.fft.ifftshift(amp_src)
    #     x = self.iFFT(amp_src, pha_src, imgH, imgW)
    #     return x
    def masking(self, x, keep_prob=0.9):
        _, _, imgH, imgW = x.size()

        # 1. 确保输入是 float32 且连续
        x_input = x.clone().contiguous().float()

        # 2. 尝试在 GPU 上执行 FFT，如果失败则回退到 CPU
        # 这是解决 CUFFT_INTERNAL_ERROR 最暴力的有效方法
        try:
            fft = torch.fft.fft2(x_input, dim=(-2, -1))
        except RuntimeError as e:
            if "cuFFT" in str(e):
                # 出现 cuFFT 错误，将数据转到 CPU 计算 FFT，然后再转回 GPU
                # 虽然慢一点，但能保证代码不崩
                fft = torch.fft.fft2(x_input.cpu(), dim=(-2, -1)).to(x.device)
            else:
                raise e

        # extract amplitude and phase of both ffts
        amp_src, pha_src = torch.abs(fft), torch.angle(fft)
        amp_src = torch.fft.fftshift(amp_src)

        amp_src = self._random_mask_on_low_frequency_only(amp_src, keep_prob)

        amp_src = torch.fft.ifftshift(amp_src)
        
        # 3. 同样的逻辑应用到 iFFT (逆变换也可能报错)
        try:
             x_rec = self.iFFT(amp_src, pha_src, imgH, imgW)
        except RuntimeError as e:
            if "cuFFT" in str(e):
                # 注意：iFFT 需要手动处理一下设备回传
                # 这里假设 self.iFFT 内部调用了 torch.fft.ifft2
                # 为了简单起见，如果这里报错，我们手动在外面做 CPU 版 iFFT
                # 但由于 iFFT 是封装在 self.iFFT 里的，建议修改下面的 iFFT 方法，或者直接在这里重写
                
                # 手动执行 CPU 版 iFFT 逻辑:
                real = torch.cos(pha_src) * amp_src
                imag = torch.sin(pha_src) * amp_src
                fft_src_ = torch.complex(real=real, imag=imag)
                x_rec = torch.fft.ifft2(fft_src_.cpu(), dim=(-2, -1), s=[imgH, imgW]).real.to(x.device)
            else:
                raise e

        return x_rec.type_as(x)

    def _random_mask_on_low_frequency_only(self, amp_src, keep_prob):
        mask = self.low_mask * torch.empty_like(amp_src).bernoulli_(keep_prob) # random keep low frequency
        mask = mask + self.high_mask #  keep all high frequency
        mask = self.__convert_symmertry_mask(mask) ## make it symmertry
        amp_src = amp_src * mask
        return amp_src

    def get_frequency(self, x):
        fft = torch.fft.fft2(x.clone(), dim=(-2, -1))
        amp_src = torch.abs(fft)
        return torch.fft.fftshift(amp_src)
    
    def __convert_symmertry_mask(self, mask):
        startH = 1 - (mask.shape[-2] % 2)
        startW = 1 - (mask.shape[-1] % 2)

        sub_mask = mask[:, :, startH:, startW:]
        assert(sub_mask.shape[-1] % 2 == 1 and sub_mask.shape[-2] % 2 == 1)
        # converting
        maskH, maskW = sub_mask.shape[-2] // 2, sub_mask.shape[-1] // 2
        sub_mask[:, :, -maskH:, :] = 1
        sub_mask[:, :, maskH, :maskW] = 1
        sub_mask = sub_mask * torch.flip(sub_mask, dims=[-1, -2])
        # end
        mask[:, :, startH:, startW:] = sub_mask

        return mask

from core.model.build import split_up_model

class BYOLWrapper(nn.Module):
    def __init__(self, cfg, model, projector_dim, feature_extractor, classifier):
        super().__init__()
        self.model = model
        self.feature_extractor = feature_extractor
        self.classifier = classifier

        self.predictor = nn.Linear(projector_dim, projector_dim, bias=False)
        nn.init.eye_(self.predictor.weight)

    def forward(self, x, use_predictor=False):
        if not use_predictor:
            return self.model(x)

        feats = self.feature_extractor(x)
        if feats.dim() > 2:
            feats = torch.flatten(feats, 1)

        z = self.predictor(feats)
        logits = self.classifier(z)
        return logits


def unwrap_model(m: nn.Module) -> nn.Module:
    """递归剥掉常见 wrapper，拿到真正的 backbone（有 head/fc 的那个）。"""
    for attr in ["module", "model", "net", "backbone", "encoder"]:
        if hasattr(m, attr):
            inner = getattr(m, attr)
            if isinstance(inner, nn.Module) and inner is not m:
                return unwrap_model(inner)
    return m

def get_linear_in_features(layer: nn.Module) -> int:
    if isinstance(layer, nn.Linear):
        return layer.in_features
    if isinstance(layer, nn.Sequential):
        for l in layer:
            if isinstance(l, nn.Linear):
                return l.in_features
            if isinstance(l, nn.Sequential):
                for ll in l:
                    if isinstance(ll, nn.Linear):
                        return ll.in_features
    raise AttributeError(f"Cannot infer in_features from {type(layer)}")

def setup(model, cfg):
    feat_extractor, classifier = split_up_model(model, cfg.MODEL.ARCH, cfg.CORRUPTION.DATASET)
    projector_dim = get_linear_in_features(classifier)

    model = BYOLWrapper(cfg, model, projector_dim, feat_extractor, classifier).cuda()
    model_configured = configure_model(deepcopy(model))
    params, param_names = collect_params(model_configured)

    if not params:
        raise ValueError("No adaptable parameters found in the model.")

    if cfg.OPTIM.METHOD == "SGD":
        optimizer = torch.optim.SGD(
            params,
            lr=float(cfg.OPTIM.LR),
            momentum=float(getattr(cfg.OPTIM, "MOMENTUM", 0.9)),
            weight_decay=float(getattr(cfg.OPTIM, "WD", 0.0)),
            nesterov=bool(getattr(cfg.OPTIM, "NESTEROV", False)),
        )
    elif cfg.OPTIM.METHOD == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=float(cfg.OPTIM.LR),
            betas=(float(getattr(cfg.OPTIM, "BETA", 0.9)), 0.999),
            weight_decay=float(getattr(cfg.OPTIM, "WD", 0.0)),
        )
    else:
        raise ValueError(f"Unsupported optimizer method: {cfg.OPTIM.METHOD}")


    tta_model = SPA(
        cfg,
        model_configured,
        optimizer,
        noise_ratio=cfg.ADAPTER.SPA.NOISE_RATIO,
        freq_mask_ratio=cfg.ADAPTER.SPA.FREQ_MASK_RATIO,
        steps=1,
        episodic=False
    )
    return tta_model