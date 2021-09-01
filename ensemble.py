#!/usr/bin/env python

import argparse
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from fvcore.common.checkpoint import Checkpointer

from pytorch_image_classification import (
    apply_data_parallel_wrapper,
    create_dataloader,
    create_loss,
    create_model,
    get_default_config,
    update_config,
)
from pytorch_image_classification.utils import (
    AverageMeter,
    create_logger,
    get_rank,
)

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

def main():
    config = load_config()

    npz_files = ['/content/pytorch_image_classification/outputs/imagenet/efficientnet-b5/exp_Aug4/predictions.npz',
                '/content/pytorch_image_classification/outputs/imagenet/efficientnet-b5/exp_Aug6/predictions.npz']

    test_loader = create_dataloader(config, is_train=False)
    _, test_loss = create_loss(config)

    probs=[[0]*len(np.load(npz_files[0])['preds'][0])]*len(np.load(npz_files[0])['preds'])
    for f in npz_files:
        probs+= np.load(f)['preds'] 

    device = torch.device(config.device)
    loss_meter = AverageMeter()
    correct_meter = AverageMeter()

    gt=[]
    for data, targets in tqdm.tqdm(test_loader):

        targets = targets.to(device)
        gt.extend(targets)
    probs=torch.tensor(probs)
    gt=torch.tensor(gt)
    loss = test_loss(probs, gt)
    _, preds = torch.max(probs, dim=1)
    # pred_prob_all=F.softmax(outputs, dim=1)
    correct_ = preds.eq(gt).sum().item()
    correct_meter.update(correct_, 1)

    accuracy = correct_meter.sum / len(test_loader.dataset)
    print("new acc: ",accuracy,"preds: ",preds)

if __name__ == '__main__':
    main()
