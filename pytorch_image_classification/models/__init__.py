import importlib

import torch
import torch.nn as nn
import torch.distributed as dist
import yacs.config

from torchvision import models


def create_model(config: yacs.config.CfgNode) -> nn.Module:
    if config.model.pretrain:
        if config.model.pretrain_pth is not None:
            model = models.resnet34(pretrained=False)
            model.load_state_dict(torch.load(config.model.pretrain_pth))
            model.fc = nn.Linear(512, config.dataset.n_classes)
        elif config.model.name.startswith("resnet34"):
            model = models.resnet34(pretrained=True)
            model.avgpool = nn.AdaptiveAvgPool2d(1)
            model.fc = nn.Linear(512, config.dataset.n_classes)
        else:
            raise Exception('pretrain model not aviliable')
            
    else:
        module = importlib.import_module(
            'pytorch_image_classification.models'
            f'.{config.model.type}.{config.model.name}')
        model = getattr(module, 'Network')(config)
    device = torch.device(config.device)
    model.to(device)
    return model


def apply_data_parallel_wrapper(config: yacs.config.CfgNode,
                                model: nn.Module) -> nn.Module:
    local_rank = config.train.dist.local_rank
    if dist.is_available() and dist.is_initialized():
        if config.train.dist.use_sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model,
                                                    device_ids=[local_rank],
                                                    output_device=local_rank)
    else:
        model.to(config.device)
    return model
