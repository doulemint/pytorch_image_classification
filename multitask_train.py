from pytorch_image_classification import (
    apply_data_parallel_wrapper,
    create_loss,
    create_model,create_transform,
    get_files,
    create_optimizer,
    create_scheduler,
    get_default_config,
    update_config,
    worker_init_fn,
)
from pytorch_image_classification.utils import (
    AverageMeter,
    DummyWriter,
    compute_accuracy,
    count_op,
    create_logger,
    create_tensorboard_writer,
    find_config_diff,
    get_env_info,
    get_rank,
    save_config,
    set_seed,
    setup_cudnn,
)
from pytorch_image_classification import create_transform
from pytorch_image_classification import create_collator
from PIL import Image
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from fvcore.common.checkpoint import Checkpointer
import torch.distributed as dist
import pandas as pd

import argparse
import pathlib
import time
import torch

from typing import Tuple, Union
import torch, yacs
import numpy as np

global_step = 0

class LabelData(Dataset):
    def __init__(self, df: pd.DataFrame,configs, transforms=None):
        self.files = [configs.dataset.dataset_dir +"/"+ file for file in df["filename"].values]
        self.y1 = df["artist"].values.tolist()
        self.y2 = df["style"].values.tolist()
        self.y3 = df["genre"].values.tolist()
        self.transforms = transforms
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert('RGB')
        label = self.y[i]
        if self.transforms is not None:
            img = self.transforms(img)
            
        return img, label

def create_dataloader(config: yacs.config.CfgNode,is_train: bool) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:
    if is_train:
        df = pd.read_csv(config.dataset.cvsfile_train)
        train_df, valid_df = train_test_split(df, stratify=df["label"].values)
        train_dataset = LabelData(train_df,config,create_transform(config, is_train=True))
        val_dataset = LabelData(valid_df,config,create_transform(config, is_train=False))

        if dist.is_available() and dist.is_initialized():
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset)
        else:
            train_sampler = torch.utils.data.sampler.RandomSampler(
                train_dataset, replacement=False)
            val_sampler = torch.utils.data.sampler.SequentialSampler(
                val_dataset)

        train_collator = create_collator(config)

        train_batch_sampler = torch.utils.data.sampler.BatchSampler(
            train_sampler,
            batch_size=config.train.batch_size,
            drop_last=config.train.dataloader.drop_last)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=config.train.dataloader.num_workers,
            collate_fn=train_collator,
            pin_memory=config.train.dataloader.pin_memory,
            worker_init_fn=worker_init_fn)

        val_batch_sampler = torch.utils.data.sampler.BatchSampler(
            val_sampler,
            batch_size=config.validation.batch_size,
            drop_last=config.validation.dataloader.drop_last)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_sampler=val_batch_sampler,
            num_workers=config.validation.dataloader.num_workers,
            pin_memory=config.validation.dataloader.pin_memory,
            worker_init_fn=worker_init_fn)
        return train_loader, val_loader
    else:
        test_df = pd.read_csv(config.dataset.cvsfile_test)
        dataset = LabelData(test_df,config,create_transform(config, is_train=False))
        test_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config.test.batch_size,
                num_workers=config.test.dataloader.num_workers,
                # sampler=sampler,
                shuffle=False,
                drop_last=False,
                pin_memory=config.test.dataloader.pin_memory)
                
        return test_loader
def compute_multi_accuracy(outputs, targets):
    acc = []
    for out,target in zip(outputs,targets):
        _, preds = torch.max(out, dim=1)
        correct_ = preds.eq(target).sum().item()
        acc.append(correct_)
    return acc
    

def train(epoch, config, model, optimizer, scheduler, loss_func, train_loader,logger):

    global global_step
    logger.info(f'Train {epoch}/{global_step}')#
    device = torch.device(config.device)

    model.train()

    loss_meter = AverageMeter()
    correct_meter1 = AverageMeter()
    correct_meter2 = AverageMeter()
    correct_meter3 = AverageMeter()
    start = time.time()
    losses = []
    with torch.no_grad():
        for step, (data, targets) in enumerate(train_loader):
            global_step += 1
            
            data = data.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(data)
            loss_ = loss_func(outputs, targets)

            loss_.backward()
            optimizer.step()
            correct_ = compute_multi_accuracy(outputs,targets)

            num = data.size(0)
            loss = sum(loss_)
            
            loss_meter.update(loss, num)
            correct_meter1.update(correct_[0], 1)
            correct_meter2.update(correct_[1], 1)
            correct_meter3.update(correct_[2], 1)

        accuracy = correct_meter1.sum / len(train_loader.dataset)
        accuracy2 = correct_meter1.sum / len(train_loader.dataset)
        accuracy3 = correct_meter1.sum / len(train_loader.dataset)
        elapsed = time.time() - start
        logger.info(f'Elapsed {elapsed:.2f}')
        logger.info(f'Loss {loss_meter.avg:.4f} Accuracy {accuracy:.4f} \
            Accuracy2 {accuracy2:.4f} Accuracy3 {accuracy3:.4f}')

    # return      
def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    
    if args.config is not None:
        config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    if not torch.cuda.is_available():
        config.device = 'cpu'
        config.train.dataloader.pin_memory = False
    if not config.train.use_kfold:
        config.train.fold_num = 1
    if args.resume != '':
        config_path = pathlib.Path(args.resume) / 'config.yaml'
        config.merge_from_file(config_path.as_posix())
        config.merge_from_list(['train.resume', True])
    config.merge_from_list(['train.dist.local_rank', args.local_rank])

    config.model.multitask = True
    df = pd.read_csv(config.dataset.cvsfile_train)
    datasets=[df['artist'].nunique(),df['style'].nunique(),df['genre'].nunique()]
    config.dataset.multi_task = datasets
    config = update_config(config)
    config.freeze()
    return config

def main():
    global global_step
    print(1)
    config = load_config()

    set_seed(config)
    setup_cudnn(config)

    epoch_seeds = np.random.randint(np.iinfo(np.int32).max // 2,
                                    size=config.scheduler.epochs)
    
    output_dir = pathlib.Path(config.train.output_dir)
    print(output_dir)
    if get_rank() == 0:
        if not config.train.resume and output_dir.exists():
            raise RuntimeError(
                f'Output directory `{output_dir.as_posix()}` already exists')
        output_dir.mkdir(exist_ok=True, parents=True)
        if not config.train.resume:
            save_config(config, output_dir / 'config.yaml')
            save_config(get_env_info(config), output_dir / 'env.yaml')
            diff = find_config_diff(config)
            if diff is not None:
                save_config(diff, output_dir / 'config_min.yaml')
    
    logger = create_logger(name=__name__,
                           distributed_rank=get_rank(),
                           output_dir=output_dir,
                           filename='log.txt')
    logger.info(config)
    logger.info(get_env_info(config))

    print(2)
    model = create_model(config)
    optimizer = create_optimizer(config, model)
    if config.device != 'cpu' and config.train.use_apex:
        model, optimizer = apex.amp.initialize(
            model, optimizer, opt_level=config.train.precision)

    
    model = apply_data_parallel_wrapper(config, model)
    train_loader, val_loader = create_dataloader(config, is_train=True)

    print(3)
    scheduler = create_scheduler(config,
                                  optimizer,
                                  steps_per_epoch=len(train_loader))
    checkpointer = Checkpointer(model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                save_dir=output_dir,
                                save_to_disk=get_rank() == 0)
    train_loss, val_loss = create_loss(config)
    print(4)
    for epoch in range(config.train.epoch):
        train(epoch, config, model, optimizer, scheduler, train_loss, train_loader,logger)

if __name__ == '__main__':
    main()