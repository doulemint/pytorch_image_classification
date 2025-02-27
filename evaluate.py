#!/usr/bin/env python

import argparse
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import pandas as pd

from fvcore.common.checkpoint import Checkpointer
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit

from pytorch_image_classification import (
    apply_data_parallel_wrapper,
    create_dataloader,create_dataset,
    create_loss,create_transform,
    create_model,get_files,
    get_default_config,
    update_config,
)
from pytorch_image_classification.utils import (
    AverageMeter,
    create_logger,
    get_rank,
)
from pytorch_image_classification.datasets import MyDataset


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    update_config(config)
    config.freeze()
    return config


def evaluate(config, model, test_loader, loss_func, logger):
    device = torch.device(config.device)

    model.eval()

    loss_meter = AverageMeter()
    correct_meter = AverageMeter()
    start = time.time()

    pred_raw_all = []
    pred_prob_all = []
    pred_label_all = []
    gt_label_all = []
    with torch.no_grad():
        for data, targets in tqdm.tqdm(test_loader):
            data = data.to(device)
            gt_label_all.extend(targets)
            targets = targets.to(device)

            outputs = model(data)
            loss = loss_func(outputs, targets)

            pred_raw_all.append(outputs.cpu().numpy())
            pred_prob_all.append(F.softmax(outputs, dim=1).cpu().numpy())

            _, preds = torch.max(outputs, dim=1)
            pred_label_all.append(preds.cpu().numpy())

            loss_ = loss.item()
            correct_ = preds.eq(targets).sum().item()
            num = data.size(0)

            loss_meter.update(loss_, num)
            correct_meter.update(correct_, 1)

        accuracy = correct_meter.sum / len(test_loader.dataset)

        elapsed = time.time() - start
        logger.info(f'Elapsed {elapsed:.2f}')
        logger.info(f'Loss {loss_meter.avg:.4f} Accuracy {accuracy:.4f}')

    preds = np.concatenate(pred_raw_all)
    probs = np.concatenate(pred_prob_all)
    labels = np.concatenate(pred_label_all)
    return preds, probs, labels, loss_meter.avg, accuracy,gt_label_all


def main():
    config = load_config()

    if config.test.output_dir is None:
        output_dir = pathlib.Path(config.test.checkpoint).parent
    else:
        output_dir = pathlib.Path(config.test.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    logger = create_logger(name=__name__, distributed_rank=get_rank())

    model = create_model(config)
    model = apply_data_parallel_wrapper(config, model)
    checkpointer = Checkpointer(model)#,checkpoint_dir=output_dir,
                                # logger=logger,
                                # distributed_rank=get_rank()
    checkpointer.load(config.test.checkpoint)
    if config.augmentation.use_albumentations:
            if config.dataset.type=='dir':
                test_clean = get_files(config.dataset.dataset_dir+'val/','train',output_dir/'label_map.pkl')
                test_dataset = MyDataset(test_clean,config.dataset.dataset_dir+'val/',
                        transforms=create_transform(config, is_train=False),is_df=config.dataset.type=='df')
                test_loader = DataLoader(test_dataset, batch_size=config.train.batch_size, num_workers=config.train.dataloader.num_workers)
            else: 
                data_root = config.dataset.dataset_dir
                batch_size=config.train.batch_size
                num_workers = 2
                train_dataset, val_dataset = create_dataset(config, True)
                labeled_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)

                # test_dataset = MyDataset(test_clean, data_root, transforms=create_transform(config, is_train=False),data_type=config.dataset.subname)
                test_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    else:
      labeled_dataloader,test_loader = create_dataloader(config, is_train=True)
    _, test_loss = create_loss(config)

    preds, probs, labels, loss, acc,gt = evaluate(config, model, test_loader,
                                               test_loss, logger)

    output_path = output_dir / f'predictions_test.npz'
    np.savez(output_path,
             preds=preds,
             probs=probs,
             labels=labels,
             loss=loss,
             acc=acc,
             gt=gt)

    preds, probs, labels, loss, acc,gt = evaluate(config, model, labeled_dataloader,
                                               test_loss, logger)

    output_path = output_dir / f'predictions_train.npz'
    np.savez(output_path,
             preds=preds,
             probs=probs,
             labels=labels,
             loss=loss,
             acc=acc,
             gt=gt)


if __name__ == '__main__':
    main()
