from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
import torch.optim as optim
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple

__all__ = ["setup"]


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x / temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

@torch.jit.script
def compute_gradient(x):
    x.requires_grad_(True)
    entropy = softmax_entropy(x)
    grad_outputs = torch.jit.annotate(List[Optional[torch.Tensor]], [torch.ones_like(entropy)])
    grad = torch.autograd.grad(outputs=[entropy], inputs=[x], grad_outputs=grad_outputs, create_graph=False, retain_graph=False)[0]
    return grad

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

def get_distances(X, Y, dist_type="euclidean"):
    """
    Args:
        X: (N, D) tensor
        Y: (M, D) tensor
    """
    if dist_type == "euclidean":
        distances = torch.cdist(X, Y)
    elif dist_type == "cosine":
        # Note: Official implementation uses this formula
        distances = 1 - torch.matmul(F.normalize(X, dim=1), F.normalize(Y, dim=1).T)
    else:
        raise NotImplementedError(f"{dist_type} distance not implemented.")

    return distances

def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

# -----------------------------------------------------------------------------
# 2. 官方 ptta_tent.py 中的 FeatureBank 类
# -----------------------------------------------------------------------------

class FeatureBank:
    def __init__(self, queue_size, neighbor=1):
        self.queue_size = queue_size if queue_size > 0 else 64
        self.features = None
        self.probs = None
        self.ptr = 0
        # accurally, we use the "farthest" samples
        self.refine_method = "nearest_neighbors"
        self.dist_type = "cosine"
        self.num_neighbors = neighbor
        self.num_features_stored = 0
        self.image_bank = None

    def reset(self):
        self.features = None
        self.probs = None
        self.ptr = 0
        self.num_features_stored = 0
        self.image_bank = None

    def update(self, x, features, logits):
        probs = F.softmax(logits, dim=1)

        start = self.ptr
        end = start + features.size(0)
        # Official code: Specific check for queue_size == 64 re-initialization logic
        if (self.features is None or self.probs is None) or self.queue_size == 64:
            self.features = torch.zeros(self.queue_size, features.size(1)).cuda()
            self.probs = torch.zeros(self.queue_size, probs.size(1)).cuda()
            self.image_bank = torch.zeros(self.queue_size, x.size(1), x.size(2), x.size(3))
        
        # Handle wrap-around
        idxs_replace = torch.arange(start, end).cuda() % self.features.size(0)
        self.features[idxs_replace, :] = features
        self.probs[idxs_replace, :] = probs
        self.image_bank[idxs_replace, :, :, :] = x.cpu()
        self.ptr = end % len(self.features)
        self.num_features_stored += features.size(0)

    def refine_predictions(self, features):
        if self.refine_method == "nearest_neighbors":
            pred_labels, probs, images, grads = self.soft_k_nearest_neighbors(features)
        elif self.refine_method == "hard_nearest_neighbors":
            pred_labels, probs, images = self.hard_k_nearest_neighbors(features)
        elif self.refine_method is None:
            pred_labels = probs.argmax(dim=1)
            images = None
        else:
            raise NotImplementedError(f"{self.refine_method} not implemented.")

        return pred_labels, probs, images, grads

    def soft_k_nearest_neighbors(self, features):
        pred_probs = []
        pred_images = []
        grads = []
        # Official code hardcodes split(64)
        for feats in features.split(64):
            distances = get_distances(feats, self.features[:self.num_features_stored], self.dist_type)
            _, idxs = distances.topk(self.num_neighbors, dim=1, largest=True)
            # gathered_distances = torch.gather(distances, 1, idxs)
            grad = self.features[idxs].mean(1)
            probs = self.probs[idxs, :].mean(1)
            images = self.image_bank[idxs.cpu()].mean(1)
            # random_indices code commented out in official...
            pred_probs.append(probs)
            pred_images.append(images)
            grads.append(grad)
        pred_probs = torch.cat(pred_probs)
        pred_images = torch.cat(pred_images)
        grads = torch.cat(grads)
        _, pred_labels = pred_probs.max(dim=1)

        return pred_labels, pred_probs, pred_images, grads
    
    def hard_k_nearest_neighbors(self, features):
        pred_probs = []
        pred_labels = []
        pred_images = []
        for feats in features.split(64):
            distances = get_distances(feats, self.features[:self.num_features_stored], self.dist_type)
            _, idxs = distances.topk(self.num_neighbors, dim=1, largest=False)

            topk_probs = self.probs[idxs, :]
            topk_one_hot = F.one_hot(topk_probs.argmax(dim=2), num_classes=self.probs.size(1)).float()

            weights = 1.0 / (torch.gather(distances, 1, idxs) + 1e-12)
            weighted_one_hot = topk_one_hot * weights.unsqueeze(-1)
            sample_pred_prob = weighted_one_hot.sum(dim=1) / weights.sum(dim=1, keepdim=True)
            pred_probs.append(sample_pred_prob)

            sample_pred_label = sample_pred_prob.argmax(dim=1)
            pred_labels.append(sample_pred_label)

            images = self.image_bank[idxs.cpu()].mean(1)
            pred_images.append(images)

        pred_probs = torch.cat(pred_probs)
        pred_labels = torch.cat(pred_labels)
        pred_images = torch.cat(pred_images)

        return pred_labels, pred_probs, pred_images

    def get_nearest_or_farthest_features(self, features, nearest=True):
        if nearest:
            distances = get_distances(features, self.features[:self.num_features_stored], self.dist_type)
            _, idxs = distances.topk(1, dim=1, largest=False)
            selected_features = self.features[idxs]
            selected_features = selected_features.squeeze(1)
        else:
            distances = get_distances(features, self.features[:self.num_features_stored], self.dist_type)
            _, idxs = distances.topk(self.num_neighbors, dim=1, largest=True)
            selected_features = self.features[idxs]
        return selected_features

# -----------------------------------------------------------------------------
# 3. 官方 ptta_tent.py 中的 PTTA 类
# -----------------------------------------------------------------------------

class PTTA(nn.Module):
    def __init__(self, model, optimizer, e_margin=math.log(1000)*0.40, d_margin=0.05, loss2_weight=3, 
                 queue_size=1000, fisher_alpha=2000, neighbor=1):
        super().__init__()
        self.model = model
        # self.model = ResNet50FeatureExtractor(model)
        
        self.optimizer = optimizer
        self.steps = 1
        self.episodic = False
        self.fishers = None

        self.num_samples_update_1 = 0  # number of samples after First filtering
        self.num_samples_update_2 = 0  # number of samples after Second filtering
        self.e_margin = e_margin
        self.d_margin = d_margin

        self.current_model_probs = None # the moving average of probability vector (Eqn. 4)

        self.batch_counter = 0
        self.judge = False

        # Official: Save state dicts for reset
        self.model_state = deepcopy(model.state_dict())
        self.optimizer_state = deepcopy(optimizer.state_dict())

        self.memory_bank = FeatureBank(queue_size, neighbor)
        self.alpha = 1 / (neighbor + 1)
        self.loss2_weight = loss2_weight
        self.fisher_alpha = fisher_alpha

    
    def forward(self, x, y=None, adapt=True):
        if self.episodic:
            self.reset()
        if self.steps > 0:
            for _ in range(self.steps):
                outputs = self.forward_and_adapt(x)
        else:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)
        return outputs

    def reset(self, alpha=0.3):
        # Official: Load saved state dicts
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.memory_bank.reset()
        # Comments in official code about BN averaging are omitted as they were commented out

    def reset_steps(self, new_steps):
        self.steps = new_steps

    def reset_model_probs(self, probs):
        self.current_model_probs = probs
    
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        outputs = self.model(x)
        entropys = softmax_entropy(outputs)
        grads = compute_gradient(outputs.clone().detach())
        filter_ids_1 = torch.where(entropys < self.e_margin)
        
        if self.current_model_probs is not None:
            cosine_similarities = F.cosine_similarity(self.current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
            self.current_model_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
            if outputs[filter_ids_1][filter_ids_2].size(0) > 0:
                self.memory_bank.update(x[filter_ids_1][filter_ids_2], grads[filter_ids_1][filter_ids_2].clone().detach(), outputs[filter_ids_1][filter_ids_2].clone().detach())
        else:
            self.current_model_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1].softmax(1))
            if outputs[filter_ids_1].size(0) > 0:
                self.memory_bank.update(x[filter_ids_1], grads[filter_ids_1].clone().detach(), outputs[filter_ids_1].clone().detach())
                        
        if self.memory_bank.num_features_stored > 0:
            pred_labels, probs, img, grads_m = self.memory_bank.refine_predictions(grads.clone().detach())
            alpha = self.alpha
            # Official uses .cuda() explicitly
            x = x * alpha + img.cuda() * (1 - alpha)
            probs = outputs.softmax(dim=-1) * alpha + probs.cuda() * (1 - alpha)
            outputs2 = self.model(x)
            loss2 = F.kl_div(outputs2.log_softmax(dim=-1), probs, reduction='batchmean') * self.loss2_weight
        
        loss = entropys.mean()
        # loss = entropys[filter_ids_1].mean()
        if self.memory_bank.num_features_stored > 0:
            loss += loss2
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.batch_counter += 1
        return outputs

def configure_model(model: nn.Module) -> nn.Module:
    """Configure model: train mode, disable grad for non-norm layers.
    (This is standard Tent/PTTA setup logic, typically in tent.py or main.py)
    """
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
            # Force batch norm to use instance stats during adaptation
            if hasattr(m, "track_running_stats"):
                m.track_running_stats = False
            if hasattr(m, "running_mean"):
                m.running_mean = None
            if hasattr(m, "running_var"):
                m.running_var = None
    return model

def setup(model, cfg):
    """
    Setup PTTA method using configuration.
    Extracts params from cfg to pass to official PTTA class arguments.
    """
    print("Setup TTA method: PTTA (Purifying Malicious Samples)")
    
    # 1. Configure Model
    model_configured = configure_model(deepcopy(model))
    
    # 2. Collect Parameters
    params, param_names = collect_params(model_configured)
    if not params:
        raise ValueError("No adaptable parameters found in the model.")

    # 3. Setup Optimizer
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

    # 4. Initialize PTTA (Mapping cfg to official arguments)
    # Note: e_margin in official defaults to math.log(1000)*0.40, here we can override if cfg has it
    
    # Assuming cfg structure from user context:
    e_margin = getattr(cfg.ADAPTER.PTTA, "e_margin", math.log(1000) * 0.40)
    d_margin = cfg.ADAPTER.PTTA.d_margin
    loss2_weight = cfg.ADAPTER.PTTA.loss2_weight
    queue_size = cfg.ADAPTER.PTTA.queue_size
    neighbor = cfg.ADAPTER.PTTA.neighbor
    
    # Create the instance
    tta_model = PTTA(
        model=model_configured,
        optimizer=optimizer,
        e_margin=e_margin,
        d_margin=d_margin,
        loss2_weight=loss2_weight,
        queue_size=queue_size,
        neighbor=neighbor
    )
    
    # Handle 'steps' if it exists in cfg (official PTTA defaults to 1, has reset_steps method)
    steps = getattr(cfg.OPTIM, "STEPS", 1)
    if steps != 1:
        tta_model.reset_steps(steps)

    return tta_model