import torch
import sys
import gc
import shutil
import os
import warnings
warnings.filterwarnings("ignore")
import argparse
import methods
from conf import cfg
from core.model import build_model
from core.data.build import build_imagenet_k_r_v2
from core.utils import seed_everything, save_df
from robustbench.utils import clean_accuracy as accuracy
from robustbench.data import load_imagenet3dcc, load_imagenetc, load_imagenet_c_bar
from setproctitle import setproctitle
from loguru import logger as log
from tqdm import tqdm, trange


def eval_imagenet_k_r(cfg, tta_model):
    """用于加载 imagenet_r, imagenet_k 的评估函数"""
    base_results = {
        "method": cfg.ADAPTER.NAME,
        'dataset': cfg.CORRUPTION.DATASET,
        'model': cfg.MODEL.ARCH,
        'batch_size': cfg.TEST.BATCH_SIZE,
        'seed': cfg.SEED,
        'note': cfg.NOTE,
        'order': cfg.CORRUPTION.ORDER_NUM,
        'Avg': 0,
    }

    loader = build_imagenet_k_r_v2(cfg)

    summary_results = base_results.copy()
    log.info(summary_results)

    rounds = cfg.TEST.ROUNDS
    
    low_acc_counter = 0     
    LOW_ACC_THRESHOLD = 5
    PATIENCE_LIMIT = 2

    real_rounds = 0
    for _round in range(rounds):
        real_rounds += 1
        base_results[f'round_{_round}'] = 0

        num_correct = 0.
        num_samples = 0
        for i, data in enumerate(loader):
            imgs, labels = data
            imgs, labels = imgs.cuda(), labels.cuda()
            output = tta_model(imgs, labels)
            predictions = output.argmax(1)
            num_correct += (predictions == labels.cuda()).float().sum()
            num_samples += imgs[0].shape[0] if isinstance(imgs, list) else imgs.shape[0]

        acc = num_correct.item() / num_samples
        err = 1. - acc
        current_acc_percentage = acc * 100
        base_results[f'round_{_round}'] += current_acc_percentage
        log.info(f"[round_{_round}]: Acc {acc:.2%} || Error {err:.2%}")

        if current_acc_percentage < LOW_ACC_THRESHOLD:
            low_acc_counter += 1
            log.warning(f"Low accuracy detected ({current_acc_percentage:.2f}%). Consecutive low rounds: {low_acc_counter}/{PATIENCE_LIMIT}")
        else:
            low_acc_counter = 0 

        if low_acc_counter >= PATIENCE_LIMIT:
            log.info(f"Accuracy has been below {LOW_ACC_THRESHOLD}% for {PATIENCE_LIMIT} consecutive rounds. Skipping remaining rounds.")
            break

    total_acc_sum = sum([base_results.get(f'round_{r}', 0) for r in range(rounds)])
    base_results['Avg'] = total_acc_sum / real_rounds
    
    save_df(base_results, f'./results/L-CS-IN_{cfg.CORRUPTION.DATASET}.csv')


def eval_imagenet_c(cfg, tta_model):
    dataset_loaders = {
        "imagenet_3dcc": load_imagenet3dcc,
        "imagenet": load_imagenetc,
        "imagenet_c_bar": load_imagenet_c_bar,
    }
    load_image = dataset_loaders.get(cfg.CORRUPTION.DATASET)
    
    base_results = {
        "method": cfg.ADAPTER.NAME,
        'dataset': cfg.CORRUPTION.DATASET,
        'model': cfg.MODEL.ARCH,
        'batch_size': cfg.TEST.BATCH_SIZE,
        'seed': cfg.SEED,
        'note': cfg.NOTE,
        'order': cfg.CORRUPTION.ORDER_NUM,
        'Avg': 0
    }
    summary_results = base_results.copy()
    log.info(summary_results)
    rounds = cfg.TEST.ROUNDS
    
    low_acc_counter = 0     
    LOW_ACC_THRESHOLD = 5.0
    PATIENCE_LIMIT = 2
    real_rounds = 0
    stop_experiment = False

    for _round in trange(rounds):
        real_rounds += 1
        
        for severity in cfg.CORRUPTION.SEVERITY:
            summary_results[f'round_{_round}'] = 0
            round_corruption_results = base_results.copy()
            for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
                x_test, y_test = load_image(
                    cfg.CORRUPTION.NUM_EX,
                    severity, 
                    cfg.DATA_DIR, 
                    False,
                    [corruption_type]
                )
                acc = accuracy(
                    model=tta_model, 
                    x=x_test.cuda(),
                    y=y_test.cuda(), 
                    batch_size=cfg.TEST.BATCH_SIZE,
                    is_enable_progress_bar=False
                )
                err = 1. - acc
                summary_results[f'round_{_round}'] += acc * 100
                round_corruption_results[corruption_type] = acc * 100
                round_corruption_results['Avg'] += acc * 100
                gc.collect()
                torch.cuda.empty_cache()

            round_corruption_results['Avg'] = round_corruption_results['Avg'] / len(cfg.CORRUPTION.TYPE)
            summary_results[f'round_{_round}'] = summary_results[f'round_{_round}'] / len(cfg.CORRUPTION.TYPE)
            
            save_df(round_corruption_results, f'./results/detail/L-CS_{_round}_{cfg.CORRUPTION.DATASET}_{cfg.CORRUPTION.ORDER_NUM}_severity_{severity}.csv')

            round_acc = summary_results[f'round_{_round}']
            round_error = 100 - round_acc
            log.info(f"[round_{_round}]: Acc {round_acc:.2f} || Error {round_error:.2f}")

            if round_acc < LOW_ACC_THRESHOLD:
                low_acc_counter += 1
                log.warning(f"Low accuracy detected ({round_acc:.2f}%). Consecutive low rounds: {low_acc_counter}/{PATIENCE_LIMIT}")
            else:
                low_acc_counter = 0

            if low_acc_counter >= PATIENCE_LIMIT:
                log.info(f"Accuracy has been below {LOW_ACC_THRESHOLD}% for {PATIENCE_LIMIT} consecutive rounds. Skipping remaining rounds.")
                stop_experiment = True
                break
        
        if stop_experiment:
            break

    total_acc_sum = sum([summary_results.get(f'round_{r}', 0) for r in range(real_rounds)])
    summary_results['Avg'] = total_acc_sum / real_rounds if real_rounds > 0 else 0

    current_severity = severity if 'severity' in locals() else cfg.CORRUPTION.SEVERITY[0]
    
    log.info(f"[Avg {current_severity}]: Acc {summary_results['Avg']:.2f} || Error {100-summary_results['Avg']:.2f}")
    save_df(summary_results, f'./results/L-CS_{cfg.CORRUPTION.DATASET}_{cfg.CORRUPTION.ORDER_NUM}.csv')


def eval(cfg):
    model = build_model(cfg).cuda()
    tta_model = getattr(methods, cfg.ADAPTER.NAME).setup(model, cfg)
    tta_model.cuda()
    
    if cfg.CORRUPTION.DATASET in ["imagenet_r", "imagenet_k"]:
        eval_imagenet_k_r(cfg, tta_model)
    elif cfg.CORRUPTION.DATASET in ["imagenet", "imagenet_3dcc", "imagenet_c_bar"]:
        eval_imagenet_c(cfg, tta_model)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.CORRUPTION.DATASET}. "
                        f"Supported datasets: imagenet_r, imagenet_k, imagenet, imagenet_3dcc, imagenet_c_bar")


def main():
    parser = argparse.ArgumentParser(
        "Pytorch Implementation for Continual Test Time Adaptation!"
    )
    parser.add_argument(
        "-acfg",
        "--adapter-config-file",
        metavar="FILE",
        default="",
        help="path to adapter config file",
        type=str,
    )
    parser.add_argument(
        "-dcfg",
        "--dataset-config-file",
        metavar="FILE",
        default="",
        help="path to dataset config file",
        type=str,
    )
    parser.add_argument(
        "-ocfg",
        "--order-config-file",
        metavar="FILE",
        default="",
        help="path to order config file",
        type=str,
    )
    parser.add_argument(
        '-mcfg',
        '--model-config-file',
        metavar="FILE",
        default="",
        help="path to model config file",
        type=str
    )
    parser.add_argument(
        "opts",
        help="modify the configuration by command line",
        nargs=argparse.REMAINDER,
        default=None,
    )
    

    args = parser.parse_args()
    if len(args.opts) > 0:
        args.opts[-1] = args.opts[-1].strip("\r\n")

    cfg.merge_from_file(args.adapter_config_file)
    cfg.merge_from_file(args.dataset_config_file)
    if args.order_config_file != "":
        cfg.merge_from_file(args.order_config_file)
    if args.model_config_file != "":
        cfg.merge_from_file(args.model_config_file)
    cfg.merge_from_list(args.opts)
    
    cfg.freeze()

    seed_everything(cfg.SEED)
    current_file_name = os.path.basename(__file__)
    setproctitle(f"{current_file_name}:{cfg.CORRUPTION.DATASET}:{cfg.ADAPTER.NAME}")    
    
    log.info(f"Loaded configuration file: \n"
             f"\tadapter: {args.adapter_config_file}\n"
             f"\tdataset: {args.dataset_config_file}\n"
             f"\torder: {args.order_config_file}\n"
             f"\tmodel: {args.model_config_file}")
    log.info(f"LOADER config:\n{cfg.LOADER}")
    log.info(f"OPTIM config:\n{cfg.OPTIM}")
    log.info(f"MODEL config:\n{cfg.MODEL}")
    
    log.remove()
    log_level = "DEBUG" if cfg.DEBUG else "INFO"
    log.add(sys.stderr, level=log_level)

    try:
        log.info(f"METHOD config:\n{cfg.ADAPTER[cfg.ADAPTER.NAME]}")
    except:
        pass

    try:
        eval(cfg)
    except Exception as e:
        log.info(
            f"Error in TTA {e} \n {cfg.SEED} TTA: {cfg.ADAPTER.NAME} DATASET: {cfg.CORRUPTION.DATASET} "
            f"BS: {cfg.TEST.BATCH_SIZE} MODEL: {cfg.MODEL.ARCH} ORDER: {cfg.CORRUPTION.ORDER_NUM} "
            f"SEVERITY: {cfg.CORRUPTION.SEVERITY} \n {cfg.NOTE}"
        )
        raise


if __name__ == "__main__":
    main()