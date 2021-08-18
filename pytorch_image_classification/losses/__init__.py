from typing import Callable, Tuple

import torch.nn as nn
import yacs.config

from .cutmix import CutMixLoss
from .mixup import MixupLoss
from .ricap import RICAPLoss
from .focal_loss import FocalLoss
from .dual_cutout import DualCutoutLoss
from .label_smoothing import LabelSmoothingLoss

class MultitaskLoss:
    def __init__(self, reduction: str, main_weight: int):
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        self.main_weight = main_weight

    def __call__(self, predictions,targets):
        other_weight=(1-self.main_weight)/(len(predictions)-1)
        loss=0
        weight=1
        for i,[pred,tar] in enumerate(zip(predictions,targets)):
            weight = other_weight
            if i==0:
                weight=self.main_weight
            loss += weight*self.criterion(pred,tar)
        return loss

def create_loss(config: yacs.config.CfgNode) -> Tuple[Callable, Callable]:
    if  config.model.multitask:
        train_loss = MultitaskLoss(reduction='mean',main_weight=0.6)
        val_loss = MultitaskLoss(reduction='mean',main_weight=0.6)
        return train_loss, val_loss
    if config.augmentation.use_mixup:
        train_loss = MixupLoss(reduction='mean')
    elif config.augmentation.use_ricap:
        train_loss = RICAPLoss(reduction='mean')
    elif config.augmentation.use_cutmix:
        train_loss = CutMixLoss(reduction='mean')
    elif config.augmentation.use_label_smoothing:
        train_loss = LabelSmoothingLoss(config, reduction='mean')
    elif config.augmentation.use_dual_cutout:
        train_loss = DualCutoutLoss(config, reduction='mean')
    elif config.augmentation.use_focal_loss:
        train_loss = FocalLoss(alpha=[0.75]*config.dataset.n_classes,num_classes=config.dataset.n_classes)
    else:
        train_loss = nn.CrossEntropyLoss(reduction='mean')
    val_loss = nn.CrossEntropyLoss(reduction='mean')
    return train_loss, val_loss
