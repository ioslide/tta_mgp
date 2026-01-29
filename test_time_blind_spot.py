import os
import sys
import shutil
import warnings
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms 
from torch.utils.data import TensorDataset, DataLoader
import pickle 
import math
import methods
import numpy as np
warnings.filterwarnings("ignore")
from copy import deepcopy
from einops import rearrange
from conf import cfg
from core.model import build_model
from wilds import get_dataset
from core.data.build import build_imagenet_k_r_v2
from core.utils import seed_everything, save_df
from robustbench.utils import clean_accuracy as accuracy
from robustbench.data import load_imagenet3dcc, load_imagenetc, load_imagenet_c_bar, load_cifar100c, load_cifar10c
from setproctitle import setproctitle
from loguru import logger as log
from tqdm import tqdm, trange
from prefetch_generator import BackgroundGenerator

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
        
_DATASET_ROOT_MAP = {
    'imagenet': 'ImageNet-C',
    'imagenet_3dcc': 'ImageNet-3DCC',
    'imagenet_c_bar': 'ImageNet-C-Bar',
    'imagenet_k': 'ImageNet-K',
    'imagenet_r': 'ImageNet-R',
}

SPECIAL_DATASETS = ['imagenet_k', 'imagenet_r']
# ------------------------------------------------

def load_custom_dataset_to_memory(cfg):
    img_list = []
    label_list = []

    if cfg.CORRUPTION.DATASET in ['imagenet_k', 'imagenet_r']:
        loader = build_imagenet_k_r_v2(cfg)
        log.info(f"Loading {cfg.CORRUPTION.DATASET} into memory...")
        for data in tqdm(loader, desc="Loading Data"):
            if isinstance(data, dict):
                imgs = data['images']
                labels = data['labels'][0]
            else:
                imgs, labels = data
            img_list.append(imgs)
            label_list.append(labels)
    else:
        raise ValueError(f"Dataset {cfg.CORRUPTION.DATASET} not supported in custom loader.")

    if len(img_list) == 0:
        raise ValueError("No data loaded! Check dataset path or filtering logic.")

    x_full = torch.cat(img_list)
    y_full = torch.cat(label_list)
    return x_full, y_full

def prepare_blind_spot_dataset(cfg, source_model):
    dataset_loaders = {
        "imagenet_3dcc": load_imagenet3dcc,
        "imagenet": load_imagenetc,
        "imagenet_c_bar": load_imagenet_c_bar,
    }
    
    is_special = cfg.CORRUPTION.DATASET in SPECIAL_DATASETS
    
    if not is_special:
        load_image = dataset_loaders.get(cfg.CORRUPTION.DATASET)
        loop_severities = cfg.CORRUPTION.SEVERITY
        loop_types = cfg.CORRUPTION.TYPE
        dataset_folder = _DATASET_ROOT_MAP.get(cfg.CORRUPTION.DATASET, 'ImageNet-C')
    else:
        loop_severities = [0] 
        loop_types = ['natural']
        dataset_folder = cfg.CORRUPTION.DATASET 

    log.info(">>> Stage 1: Filtering Blind-Spot samples (Source model errors only)...")
    source_model.eval()
    
    data_path = os.path.join(cfg.DATA_DIR, f'{dataset_folder}/source_error_data_{cfg.MODEL.ARCH}')
    os.makedirs(data_path, exist_ok=True)
    
    for severity in loop_severities:
        for corruption_type in loop_types:
            save_path = os.path.join(data_path, f'source_error_{cfg.CORRUPTION.DATASET}_S{severity}_{corruption_type}.pt')
            if os.path.exists(save_path):
                log.info(f"Source-Error data for {corruption_type} (Sev: {severity}) already exists. Skipping.")
                continue

            if is_special:
                x_full, y_full = load_custom_dataset_to_memory(cfg)
            else:
                x_full, y_full = load_image(cfg.CORRUPTION.NUM_EX, severity, cfg.DATA_DIR, False, [corruption_type])
            
            blind_indices = []
            with torch.no_grad():
                for i in trange(0, len(x_full), cfg.TEST.BATCH_SIZE, desc=f"Filtering {corruption_type}"):
                    batch_x = x_full[i:i+cfg.TEST.BATCH_SIZE].cuda()
                    batch_y = y_full[i:i+cfg.TEST.BATCH_SIZE].cuda()
                    output = source_model(batch_x)
                    pred = output.argmax(dim=1)

                    incorrect_mask = (pred != batch_y).cpu()
                    idx = torch.where(incorrect_mask)[0] + i
                    blind_indices.append(idx)
            
            blind_indices = torch.cat(blind_indices)
            if len(blind_indices) > 0:
                x_blind = x_full[blind_indices]
                y_blind = y_full[blind_indices]
                blind_spot_data = {"data": (x_blind, y_blind)}
                log.info(f"Captured {len(blind_indices)} Source-Error samples for {corruption_type} (Sev: {severity})")
                torch.save(blind_spot_data, save_path)
            else:
                log.warning(f"No errors found for {corruption_type}! This won't contribute to stress test.")


def eval(cfg):
    source_model = build_model(cfg).cuda()
    mode = cfg.INPUT.BLIND_SPOT_MODE
    if mode == 'SOURCE_ERROR':
        prepare_blind_spot_dataset(cfg, source_model)
        data_folder_name = f'source_error_data_{cfg.MODEL.ARCH}'
        data_prefix = 'source_error'
    else:
        raise ValueError(f"Unknown BLIND_SPOT_MODE: {mode}")

    if cfg.CORRUPTION.DATASET in SPECIAL_DATASETS:
        dataset_root = cfg.CORRUPTION.DATASET
        loop_severities = [0]
        loop_types = ['natural']
    else:
        dataset_root = _DATASET_ROOT_MAP.get(cfg.CORRUPTION.DATASET, cfg.CORRUPTION.DATASET)
        loop_severities = cfg.CORRUPTION.SEVERITY
        loop_types = cfg.CORRUPTION.TYPE

    data_base_path = os.path.join(cfg.DATA_DIR, f'{dataset_root}/{data_folder_name}')
    
    tta_model = getattr(methods, cfg.ADAPTER.NAME).setup(source_model, cfg)
    tta_model.cuda()

    base_results = {
        "method": cfg.ADAPTER.NAME,
        'dataset':cfg.CORRUPTION.DATASET,
        'model': cfg.MODEL.ARCH,
        'batch_size':cfg.TEST.BATCH_SIZE,
        'seed': cfg.SEED,
        'note': f"BlindSpot_{mode}_{cfg.NOTE}",
        'order': cfg.CORRUPTION.ORDER_NUM,
        'Avg': 0
    }
    log.info(f"Base Results Dict: {base_results}")
    summary_results = base_results.copy()
    rounds = 40
    if cfg.ADAPTER.NAME == "Source":
        rounds = 1
        
    log.info(f">>> Stage 2: Starting {rounds}-round Stress Test on {cfg.ADAPTER.NAME} with '{mode}' data...")
    
    low_acc_counter = 0     
    LOW_ACC_THRESHOLD = 5.0 
    PATIENCE_LIMIT = 2
    real_rounds = 0
    stop_experiment = False

    for _round in trange(rounds, desc="Stress Test Rounds"):
        avg_acc_round = 0.0
        real_rounds += 1
        round_key = f'round_{_round}'
        summary_results[round_key] = 0
        
        round_corruption_results = base_results.copy()
        
        for corruption_type in loop_types:
            for severity in loop_severities:
                db_path = os.path.join(data_base_path, f'{data_prefix}_{cfg.CORRUPTION.DATASET}_S{severity}_{corruption_type}.pt')
                
                if not os.path.exists(db_path):
                    log.warning(f"Data file not found, skipping: {db_path}")
                    continue

                blind_spot_db = torch.load(db_path, map_location='cpu')
                x_adapt, y_adapt = blind_spot_db.get("data")              

                correct = 0
                total = 0
                ds = TensorDataset(x_adapt, y_adapt)
                dl = DataLoaderX(ds, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                                num_workers=4, pin_memory=False)

                tta_model.eval()
                with torch.no_grad():
                    for xb, yb in dl:
                        xb = xb.cuda()
                        yb = yb.cuda()
                        out = tta_model(xb)
                        pred = out.argmax(1)
                        correct += (pred == yb).sum().item()
                        total += yb.size(0)

                acc = correct / total
                acc_pct = acc * 100
                log.debug(f"Round {_round} | {corruption_type} | Sev {severity} | Acc: {acc_pct:.2f}% | Samples: {len(x_adapt)}")
                round_corruption_results[corruption_type] = acc_pct


        valid_corruptions = [k for k in loop_types if k in round_corruption_results]
        if len(valid_corruptions) > 0:
            avg_acc_round = sum(round_corruption_results[c] for c in valid_corruptions) / len(valid_corruptions)
            summary_results[round_key] = avg_acc_round
            log.info(f"Round {_round} | Avg Acc: {avg_acc_round:.2f}%")
        else:
            summary_results[round_key] = 0
            log.warning(f"Round {_round} | No valid results to average.")

        if avg_acc_round < LOW_ACC_THRESHOLD:
            low_acc_counter += 1
            log.warning(f"Low accuracy detected ({avg_acc_round:.2f}%). Consecutive low rounds: {low_acc_counter}/{PATIENCE_LIMIT}")
        else:
            low_acc_counter = 0 

        if low_acc_counter >= PATIENCE_LIMIT:
            log.info(f"Accuracy has been below {LOW_ACC_THRESHOLD}% for {PATIENCE_LIMIT} consecutive rounds. Skipping remaining rounds.")
            stop_experiment = True
            break 
    
        if stop_experiment:
            break
            
    avg_total = sum([summary_results[f'round_{r}'] for r in range(real_rounds)])
    summary_results['Avg'] = avg_total / real_rounds if real_rounds > 0 else 0
    log.info(f"Final Protocol IV Avg Acc ({rounds} rounds): {avg_total:.2f}%")
    
    save_df(summary_results, f'./results/BlindSpot_Summary_{mode}_{cfg.CORRUPTION.DATASET}.csv')

def main():
    parser = argparse.ArgumentParser("Blind-Spot Stress Test (Protocol IV)")
    parser.add_argument("-acfg", "--adapter-config-file", metavar="FILE", default="", type=str)
    parser.add_argument("-dcfg", "--dataset-config-file", metavar="FILE", default="", type=str)
    parser.add_argument("-ocfg", "--order-config-file", metavar="FILE", default="", type=str)
    parser.add_argument("-mcfg", "--model-config-file", metavar="FILE", default="", type=str)
    parser.add_argument("opts", help="modify the configuration by command line", nargs=argparse.REMAINDER, default=None)

    args = parser.parse_args()
    cfg.merge_from_file(args.adapter_config_file)
    cfg.merge_from_file(args.dataset_config_file)
    if args.order_config_file != "": cfg.merge_from_file(args.order_config_file)
    if args.model_config_file != "": cfg.merge_from_file(args.model_config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    seed_everything(cfg.SEED)
    setproctitle(f"StressTest:{cfg.ADAPTER.NAME}")    
    
    log.remove()
    log.add(sys.stderr, level="DEBUG" if cfg.DEBUG else "INFO")

    try:
        log.info(f"METHOD config:\n{cfg.ADAPTER[cfg.ADAPTER.NAME]}")
    except:
        pass

    try:
        eval(cfg)
    except Exception as e:
        log.exception(f"Stress Test Failed: {e}")
        raise

if __name__ == "__main__":
    main()