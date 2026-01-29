import timm
import torch
import torch.nn as nn
import core.model.resnet as Resnet
from core.model.imagenet_subsets import IMAGENET_A_MASK, IMAGENET_R_MASK, IMAGENET_V2_MASK, IMAGENET_D109_MASK

from copy import deepcopy
from robustbench.utils import load_model
from packaging import version
from loguru import logger as log
from robustbench.model_zoo.architectures.utils_architectures import normalize_model, ImageNormalizer
import torchvision
from transformers import BeitFeatureExtractor, Data2VecVisionForImageClassification
import getpass
username = getpass.getuser()
from typing import Union
from typing import Tuple, Optional

class NormalizedModel(nn.Module):
    def __init__(self, model: nn.Module, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
        super().__init__()
        self.model = model
        self.normalizer = ImageNormalizer(mean, std)

    def forward(self, x: torch.Tensor, **kwargs):
        x_normalized = self.normalizer(x)
        return self.model(x_normalized, **kwargs)

def build_model(cfg):

    if cfg.CORRUPTION.DATASET in ["imagenet_r"]:
        log.info(f"Wrapping model with mask for dataset {cfg.CORRUPTION.DATASET}")
        base_model = get_torchvision_model(cfg.MODEL.ARCH, 'IMAGENET1K_V1')
        mask = eval(f"{cfg.CORRUPTION.DATASET.upper()}_MASK")
        base_model = ImageNetXWrapper(base_model, mask=mask)
        return base_model.cuda()

    try:
        if cfg.MODEL.ARCH in ['resnet18', 'resnet34', 'resnet50', 'resnet101','resnet152']:
            base_model = Resnet.__dict__[cfg.MODEL.ARCH](pretrained=True,num_classes=cfg.CORRUPTION.NUM_CLASS).cuda()
        elif cfg.MODEL.ARCH in ['resnet50_gn','resnetv2_50d_gn']:
            base_model = timm.create_model('resnet50_gn', pretrained=False,num_classes=cfg.CORRUPTION.NUM_CLASS).cuda()
            checkpoint = torch.load('/gemini/data-1/models/imagenet/resnet50_gn_a1h2-8fe6c4d0.pth')
            base_model.load_state_dict(checkpoint)
            log.info("Model created successfully!")

        elif cfg.MODEL.ARCH in ['vit_b_16']:
            base_model = get_torchvision_model(cfg.MODEL.ARCH,'IMAGENET1K_V1')
        else:
            base_model = load_model(
                model_name=cfg.MODEL.ARCH,
                dataset=cfg.CORRUPTION.DATASET.split('_')[0], 
                threat_model='corruptions'
            )
    except ValueError:
        base_model = load_model(
            model_name=cfg.MODEL.ARCH,
            dataset=cfg.CORRUPTION.DATASET.split('_')[0], 
            threat_model='corruptions'
        )
        
    return base_model.cuda()

        
class TransformerWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.__dict__ = model.__dict__.copy()

    def forward(self, x):
        x = self.normalize(x)
        x = self.model._process_input(x)
        n = x.shape[0]

        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)

        x = x[:, 0]
        return x

class D2VWrapper(torch.nn.Module):
    def __init__(self, model, feature_extractor=None):
        super().__init__()
        self.model = model
        self.feature_extractor = feature_extractor
        self.normalize = None

    def forward(self, x):
        inputs = {"pixel_values": x}

        x = self.model(**inputs)

        x = x.logits

        return x
    
class D2VSplitWrapper(torch.nn.Module):
    def __init__(self, model, feature_extractor=None):
        super().__init__()
        self.model = model
        self.feature_extractor = feature_extractor
        self.normalize = None

    def forward(self, x):
        inputs = {
            "pixel_values": x,
            "head_mask": None,
            "output_attentions": None,
            "output_hidden_states": None,
            "return_dict": False
        }

        outputs = self.model.model.data2vec_vision(
            **inputs
        )
        
        outputs = outputs[1]

        return outputs

class ImageNetXMaskingLayer(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def forward(self, x):
        return x[:, self.mask]

class ImageNetXWrapper(torch.nn.Module):
    def __init__(self, model, mask):
        super().__init__()
        self.__dict__ = model.__dict__.copy()

        self.masking_layer = ImageNetXMaskingLayer(mask)

    def forward(self, x):
        logits = self.model(self.normalize(x))
        return self.masking_layer(logits)

def split_up_model(model, arch_name, dataset_name):
        
    if dataset_name in ["imagenet_a", "imagenet_r"]:
        norm = getattr(model, "normalize", nn.Identity())
        backbone = getattr(model, "model", model)
        mask = eval(f"{dataset_name.upper()}_MASK")

        is_torchvision_vit = all([
            hasattr(backbone, "_process_input"),
            hasattr(backbone, "class_token"),
            hasattr(backbone, "encoder"),
            hasattr(backbone, "heads"),
        ])
        if is_torchvision_vit:
            class TorchvisionViTFeature(nn.Module):
                def __init__(self, vit):
                    super().__init__()
                    self.vit = vit

                def forward(self, x):
                    x = self.vit._process_input(x)
                    b = x.shape[0]
                    cls = self.vit.class_token.expand(b, -1, -1)
                    x = torch.cat((cls, x), dim=1)
                    x = self.vit.encoder(x) 
                    x = x[:, 0]  
                    return x

            encoder = nn.Sequential(norm, TorchvisionViTFeature(backbone))

            head = backbone.heads.head if hasattr(backbone.heads, "head") else backbone.heads
            classifier = nn.Sequential(head, ImageNetXMaskingLayer(mask))
            return encoder, classifier

        encoder = nn.Sequential(norm, *list(backbone.children())[:-1], nn.Flatten())

        if hasattr(backbone, "fc"):
            classifier = backbone.fc
        elif hasattr(backbone, "classifier"):
            classifier = backbone.classifier
        elif hasattr(backbone, "head"):
            classifier = backbone.head
        elif hasattr(backbone, "heads"):
            classifier = backbone.heads.head if hasattr(backbone.heads, "head") else backbone.heads
        else:
            raise AttributeError(f"Cannot find classifier on backbone type {type(backbone)}")

        classifier = nn.Sequential(classifier, ImageNetXMaskingLayer(mask))
        return encoder, classifier

    if hasattr(model, "model") and hasattr(model.model, "pretrained_cfg") and hasattr(model.model, model.model.pretrained_cfg["classifier"]):
        classifier = deepcopy(getattr(model.model, model.model.pretrained_cfg["classifier"]))
        encoder = model
        encoder.model.reset_classifier(0)
        if isinstance(model, ImageNetXWrapper):
            encoder = nn.Sequential(encoder.normalize, encoder.model)
    elif "wide_resnet50_2" in arch_name or "resnext50_32x4d" in arch_name:
        encoder = nn.Sequential(*list(model.model.children())[:-1], nn.Flatten())
        classifier = model.model.fc
    elif "resnet" in arch_name or arch_name in {"Hendrycks2020AugMix", "Hendrycks2020Many", "Geirhos2018_SIN"}:
        encoder = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        classifier = model.fc
    elif "vit_" in arch_name and not "maxvit_" in arch_name:
        encoder = TransformerWrapper(model)
        classifier = model.model.heads.head
    else:
        raise ValueError(f"The model architecture '{arch_name}' is not supported for dataset '{dataset_name}'.")

    return encoder, classifier

def get_torchvision_model(model_name: str, weight_version: str = "IMAGENET1K_V1"):
    assert version.parse(torchvision.__version__) >= version.parse("0.13"), "Torchvision version has to be >= 0.13"

    available_models = torchvision.models.list_models(module=torchvision.models)
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' is not available in torchvision. Choose from: {available_models}")

    model_weights = torchvision.models.get_model_weights(model_name)
    available_weights = [init_name for init_name in dir(model_weights) if "IMAGENET1K" in init_name]

    if weight_version not in available_weights:
        raise ValueError(f"Weight type '{weight_version}' is not supported for torchvision model '{model_name}'."
                         f" Choose from: {available_weights}")

    model_weights = getattr(model_weights, weight_version)

    model = torchvision.models.get_model(model_name, weights=model_weights)

    transform = model_weights.transforms()
    model = normalize_model(model, transform.mean, transform.std)
    log.info(f"Successfully restored '{weight_version}' pre-trained weights"
                f" for model '{model_name}' from torchvision!")

    return model