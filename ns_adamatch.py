import argparse
import pathlib
import time,os

try:
    import apex
except ImportError:
    pass
import pandas as pd
import numpy as np

from train import train,validate,load_config
from pytorch_image_classification import (
    apply_data_parallel_wrapper,
    create_dataloader,
    create_loss,
    create_model,
    get_files,
    create_optimizer,
    create_scheduler,discriminative_lr_params,
    prepare_dataloader,
    get_default_config,
    update_config,
)
from pytorch_image_classification.config.config_node import ConfigNode
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
from pytorch_image_classification.models import get_model
from pytorch_image_classification.datasets import MyDataset, pesudoMyDataset

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from fvcore.common.checkpoint import Checkpointer

from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit

def  generate_pseudo_labels(weak_images_train, weak_images_test, teacher_models, confidence_thres):
       
    for model in teacher_models:   
       model.eval()

    with torch.no_grad():
        # pass train images into models
        preds_1 = teacher_models[0](weak_images_train)
        preds_2 = teacher_models[1](weak_images_test)
        final_predictions_train = torch.stack((preds_1, preds_2), dim=0).mean(0)

        # pass test images into models
        preds_1 = teacher_models[0](weak_images_train)
        preds_2 = teacher_models[1](weak_images_test)
        final_predictions_test = torch.stack((preds_1, preds_2), dim=0).mean(0)
        final_predictions_test_, _ = torch.nn.Softmax(dim=1)(
            torch.tensor(final_predictions_test)
        ).max(1)

        # compute thresholding mask
        test_mask = torch.sum(
            final_predictions_test_ > confidence_thres, dim=(1, 2)
        ) 

        # concatenate all predictions
        all_predictions = torch.cat(
            (final_predictions_train, final_predictions_test), dim=0
        )

    return all_predictions, test_mask

def compute_loss_target(predictions, pseudo_labels,gt, alpha):
    loss_func = nn.CrossEntropyLoss(reduction='mean')
    if gt is not None:
        target_loss = loss_func(predictions,pseudo_labels)
        student_loss = loss_func(predictions,gt)
        return ((1 - alpha) * target_loss) + (alpha * student_loss)
    else:
        student_loss = loss_func(predictions,gt)
        return student_loss
def get_alpha(epoch, total_epochs):
    initial_alpha = 0.1
    final_alpha = 0.5
    modified_alpha = (
        final_alpha - initial_alpha
    ) / total_epochs * epoch + initial_alpha
    return modified_alpha

def main():
    config = load_config()

    set_seed(config)
    setup_cudnn(config)

    # epoch_seeds = np.random.randint(np.iinfo(np.int32).max // 2,
    #                                 size=config.scheduler.epochs)

    if config.train.distributed:
        dist.init_process_group(backend=config.train.dist.backend,
                                init_method=config.train.dist.init_method,
                                rank=config.train.dist.node_rank,
                                world_size=config.train.dist.world_size)
        torch.cuda.set_device(config.train.dist.local_rank)

    output_dir = pathlib.Path(config.train.output_dir)
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
    
    data_root = config.dataset.dataset_dir+'train/'
    batch_size=config.train.batch_size

    if config.dataset.type=='dir':
        train_clean = get_files(data_root,'train',output_dir/'label_map.yaml')
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        for trn_idx, val_idx in sss.split(train_clean['filename'], train_clean['label']):
            train_frame = train_clean.loc[trn_idx]
            val_frame  = train_clean.loc[val_idx]
        test_clean=get_files(config.dataset.dataset_dir+'val/','train',output_dir/'label_map.yaml')
    elif config.dataset.type=='df':
        train_clean =  pd.read_csv(config.dataset.cvsfile_train)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        for trn_idx, val_idx in sss.split(train_clean['image'], train_clean['label']):
            train_frame = train_clean.loc[trn_idx]
            val_frame  = train_clean.loc[val_idx]
        test_clean =  pd.read_csv(config.dataset.cvsfile_test)
    
    soft = False
    
    weak_labeled_dataset = MyDataset(train_frame, data_root, transforms=create_transform(config, is_train=False), output_label=True,soft=soft,
                        n_class=config.dataset.n_classes,label_smooth=config.augmentation.use_label_smoothing,
                        epsilon=config.augmentation.label_smoothing.epsilon,is_df=config.dataset.type=='df')
    strong_labeled_dataset = MyDataset(train_frame, data_root, transforms=create_transform(config, is_train=True), output_label=True,soft=soft,
                        n_class=config.dataset.n_classes,label_smooth=config.augmentation.use_label_smoothing,
                        epsilon=config.augmentation.label_smoothing.epsilon,is_df=config.dataset.type=='df')

    weak_unlabeled_dataset = MyDataset(val_frame, data_root, transforms=create_transform(config, is_train=False), output_label=True,soft=soft,
                        n_class=config.dataset.n_classes,label_smooth=config.augmentation.use_label_smoothing,
                        epsilon=config.augmentation.label_smoothing.epsilon,is_df=config.dataset.type=='df')
    strong_unlabeled_dataset = MyDataset(val_frame, data_root, transforms=create_transform(config, is_train=True), output_label=True,soft=soft,
                        n_class=config.dataset.n_classes,label_smooth=config.augmentation.use_label_smoothing,
                        epsilon=config.augmentation.label_smoothing.epsilon,is_df=config.dataset.type=='df')

    num_workers=config.train.dataloader.num_workers
    weak_labeled_dataloader = DataLoader(weak_labeled_dataset, batch_size=batch_size, num_workers=num_workers)
    strong_labeled_dataloader = DataLoader(strong_labeled_dataset, batch_size=batch_size, num_workers=num_workers)
    weak_unlabeled_dataloader = DataLoader(weak_unlabeled_dataset, batch_size=batch_size, num_workers=num_workers)
    strong_unlabeled_dataloader = DataLoader(strong_unlabeled_dataset, batch_size=batch_size, num_workers=num_workers)

    
    student_model_opt = "resnet50"
    teacher_model_opt = ["resnet50","efficientnet-b5"]
    device = config.device
    num_epochs = config.scheduler.epochs
    teacher_model = []
    config.defrost()
    for opt in teacher_model_opt:
        config.model.name=opt
        teacher_model.append(get_model(config))
        ckp_pth= config.test.checkpoint+f'/checkpoint_{opt}.pth'
        if os.path.exists(ckp_pth):
                checkpoint = torch.load(ckp_pth, map_location='cpu')
                if isinstance(teacher_model[-1],
                      (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                    teacher_model[-1].module.load_state_dict(checkpoint['model'])
                    print(f"load model from {str(ckp_pth)}")
                else:
                    teacher_model[-1].load_state_dict(checkpoint['model'])
                    print(f"load model from {str(ckp_pth)}")
        teacher_model[-1].to(device)
    config.model.name=student_model_opt
    student_model=get_model(config)
    student_model.to(device)
    config.freeze()

    for j in range(config.scheduler.epochs):
        loss_meter = AverageMeter()
        student_model.train()
        for (weak_batch_train,weak_batch_test,strong_batch_train,strong_batch_test,) in zip(
            weak_labeled_dataloader, weak_unlabeled_dataloader, strong_labeled_dataloader, strong_unlabeled_dataloader): 
            weak_image_train = weak_batch_train[0].to(device)
            target = weak_batch_train[1].to(device)
            weak_image_test  = weak_batch_test[0].to(device)
            strong_image_train = strong_batch_train[0].to(device)
            strong_image_test = strong_batch_test[0].to(device)

            num_train = weak_image_train.size(0)

            student_prediction_train=student_model(strong_image_train)
            student_prediction_test=student_model(strong_image_test)

            #calcutate c_tau
            row_wise_max = torch.nn.Softmax(dim=1)(student_prediction_train)
            row_wise_max = torch.max(row_wise_max)
            final_sum=torch.mean(row_wise_max)
            # final_sum = row_wise_max.mean(0)
            c_tau = 0.8 * final_sum

            pseudo_labels,test_mask=generate_pseudo_labels(
                weak_image_train,weak_image_test,teacher_model,c_tau
            )

            alpha = get_alpha(j, num_epochs)
            train_loss = compute_loss_target(student_prediction_train,pseudo_labels[num_train:],target,alpha)
            test_loss = compute_loss_target(student_prediction_test,pseudo_labels[:num_train],None,alpha)

            loss = train_loss + (test_loss[test_mask]).mean()
            loss_meter.update(loss.cpu().item(),(num_train+test_loss.size(0)))

if __name__ == '__main__':
    main()

        
