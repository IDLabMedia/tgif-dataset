# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
#
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt

"""
Created in September 2022
@author: fabrizio.guillaro

Modified by Paschalis Giakoumoglou (@p-giakoumoglou)
August 2024 @ ITI-CERTH
Added checkpoint resume capability and additional metrics

Usage:
python trufor_train.py -exp "path/to/experiment.yaml"


TXT file format:
    {img_path}{delimiter}{mask_path}{delimiter}{label}
    Example with space: /path/img.jpg /path/mask.png 1
    Example with comma: /path/img.jpg,/path/mask.png,1
    
Note:
    - Balanced sampling is performed with respect to each txt file
    - The smallest dataset determines the number of samples per epoch from each dataset
"""

import sys, os
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
import logging

import torch
from torch.nn import functional as F

from config import update_config
from config import _C as config

from data.datasets import MixDataset
from torch.utils.data import DataLoader

from common.losses import TruForLoss
from torch.utils.tensorboard import SummaryWriter
from common.split_params import group_weight
from common.lr_schedule import WarmUpPolyLR
from common.utils import AverageMeter
import torchvision.transforms.functional as TF
from common.metrics import computeLocalizationMetrics, computeLocF1_th


parser = argparse.ArgumentParser(description='Training script for TruFor')
parser.add_argument('-gpu', '--gpu', type=int, default=0, help='device, use -1 for cpu')
parser.add_argument('-log', '--log', type=str, default='INFO', help='logging level')
parser.add_argument('-exp', '--exp', type=str, default=None, help='Yaml experiment file')
parser.add_argument('opts', help="other options", default=None, nargs=argparse.REMAINDER)

args = parser.parse_args()
config = update_config(config, args.exp)


gpu = args.gpu
loglvl = getattr(logging, args.log.upper())
logging.basicConfig(level=loglvl, format='%(message)s')

device = 'cuda:%d' % gpu if gpu >= 0 else 'cpu'
np.set_printoptions(formatter={'float': '{: 7.3f}'.format})

torch.set_flush_denormal(True)
if device != 'cpu':
    # cudnn setting
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    
    
from models.cmx.builder_np_conf import myEncoderDecoder as confcmx
model = confcmx(cfg=config)
ckpt_path = './ckpt/{}/last_ckpt.pth'.format(config.MODEL.NAME)
print(f"ckpt path: {ckpt_path}")
if os.path.exists('./ckpt/{}/last_ckpt.pth'.format(config.MODEL.NAME)):
    print(f"Loading from {ckpt_path}")
    checkpoint = torch.load('./ckpt/{}/last_ckpt.pth'.format(config.MODEL.NAME), map_location=torch.device(device))
    last_epoch = checkpoint['epoch']+1
    model.load_state_dict(checkpoint['state_dict'])
else:
    last_epoch = 0
model.to(device)


def freeze_dncnn(model):
    for name, param in model.named_parameters():
        if 'dncnn' in name:
            param.requires_grad = False
    print("dncnn parameters have been frozen.")

# Call this if you need to freeze dncnn
freeze_dncnn(model)

train = MixDataset(config.DATASET.TRAIN,
                   config.DATASET.IMG_SIZE,
                   train=True,
                   class_weight=config.DATASET.CLASS_WEIGHTS,
                   delimiter=config.DATASET.get('DELIMITER', ' '))

val = MixDataset(config.DATASET.VAL,
                 config.DATASET.IMG_SIZE,
                 train=False,
                 delimiter=config.DATASET.get('DELIMITER', ' '))

logging.info(train.get_info())
train_loader = DataLoader(train,
                          batch_size=config.BATCH_SIZE,
                          shuffle=True,
                          num_workers=config.WORKERS,
                          pin_memory=True)

val_loader = DataLoader(val,
                        batch_size=1,
                        shuffle=False,
                        num_workers=config.WORKERS,
                        pin_memory=True)

criterion = TruForLoss(weights=train.class_weights.to(device), ignore_index=-1)

os.makedirs('./ckpt/{}'.format(config.MODEL.NAME), exist_ok=True)
logdir = './{}/{}'.format(config.LOG_DIR, config.MODEL.NAME)
os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter('./{}/{}'.format(config.LOG_DIR, config.MODEL.NAME))


params = []
cmnext_params = []
modal_extract_params = []
cmnext_params = group_weight(cmnext_params, model, torch.nn.BatchNorm2d, config.LEARNING_RATE)

params.append(dict(params=cmnext_params[0]['params'], lr=config.LEARNING_RATE))
params.append(dict(params=cmnext_params[1]['params'], weight_decay=.0,
                   lr=config.LEARNING_RATE))

optimizer = torch.optim.SGD(params,
                            lr=config.LEARNING_RATE,
                            momentum=config.SGD_MOMENTUM,
                            weight_decay=config.WD
                            )

iters_per_epoch = len(train_loader)
iters = 0
max_iters = config.EPOCHS * iters_per_epoch
min_loss = 100

lr_schedule = WarmUpPolyLR(optimizer,
                           start_lr=config.LEARNING_RATE,
                           lr_power=config.POLY_POWER,
                           total_iters=max_iters,
                           warmup_steps=iters_per_epoch * config.WARMUP_EPOCHS)

scaler = torch.cuda.amp.GradScaler()

for epoch in range(last_epoch, config.EPOCHS):
    train.shuffle()  # for balanced sampling
    model.train()

    avg_loss = AverageMeter()
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(train_loader, desc='Training Epoch {}/{}'.format(epoch + 1, config.EPOCHS), unit='steps')
    for step, (images, _, masks, _) in enumerate(pbar):

        images = images.to(device, non_blocking=True)
        masks = masks.squeeze(1).to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            #images_norm = TF.normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            #inp = images_norm

            pred, _, _, _ = model(images)

            loss = criterion(pred, masks) / config.ACCUMULATE_ITERS
        scaler.scale(loss).backward()
        if ((step + 1) % config.ACCUMULATE_ITERS == 0) or (step + 1 == len(train_loader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        avg_loss.update(loss.detach().item())

        curr_iters = epoch * iters_per_epoch + step
        lr_schedule.step(cur_iter=curr_iters)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], curr_iters)

        if step == 0:
            maps = torch.nn.functional.softmax(pred, dim=1)[:, 1, :, :]
            writer.add_images('Images-Masks-Preds',
                              torch.cat((
                                  images,
                                  torch.tile(masks.unsqueeze(1), (1, 3, 1, 1)),
                                  torch.tile(maps.unsqueeze(1), (1, 3, 1, 1))), -2)
                              , epoch)

        pbar.set_postfix({"last_loss": loss.detach().item(), "epoch_loss": avg_loss.average()})
    writer.add_scalar('Training Loss', avg_loss.average(), epoch)
    
    #if (epoch + 1) % 10 == 0 or epoch == config.EPOCHS - 1:
    if 1:
        f1 = []
        f1th = []
        f1mods = []
        val_loss_avg = AverageMeter()
        model.eval()
        pbar = tqdm(val_loader, desc='Validating Epoch {}/{}'.format(epoch + 1, config.EPOCHS), unit='steps')
        for step, (images, _, masks, lab) in enumerate(pbar):
            with torch.no_grad():
                images = images.to(device, non_blocking=True)
                masks = masks.squeeze(1).to(device, non_blocking=True)
                
                pred, _, _, _ = model(images)
                
    
                val_loss = criterion(pred, masks)
                val_loss_avg.update(val_loss.detach().item())
                gt = masks.squeeze().cpu().numpy()
                map = torch.nn.functional.softmax(pred, dim=1)[:, 1, :, :].squeeze().cpu().numpy()
                F1_best, F1_th = computeLocalizationMetrics(map, gt)
                f1_mod = computeLocF1_th(map, gt)
                
                f1.append(F1_best)
                f1th.append(F1_th)
                f1mods.append(f1_mod)
    
        # Calculate values
        val_loss = val_loss_avg.average()
        val_f1_best = np.nanmean(f1)
        val_f1_fixed = np.nanmean(f1th)
        val_f1_mod = np.nanmean(f1mods)
        
        # Add values to the writer
        writer.add_scalar('Val Loss', val_loss, epoch)
        writer.add_scalar('Val F1 best', val_f1_best, epoch)
        writer.add_scalar('Val F1 fixed', val_f1_fixed, epoch)
        
        # Print values to the console
        print(f"Epoch {epoch+1} - Val Loss: {val_loss}, Val F1 best: {val_f1_best}, Val F1 fixed: {val_f1_fixed}, Val F1 mod: {val_f1_mod}")
        
        
        result = {'epoch': epoch, 'val_loss': val_loss, 'val_f1_best': val_f1_best,
                  'val_f1_fixed': val_f1_fixed, 'val_f1_mod': val_f1_mod, 'state_dict': model.state_dict()}
        torch.save(result, './ckpt/{}/last_ckpt.pth'.format(config.MODEL.NAME))
        
        if val_loss_avg.average() < min_loss:
            min_loss = val_loss_avg.average()
            result = {'epoch': epoch, 'val_loss': val_loss, 'val_f1_best': val_f1_best,
                      'val_f1_fixed': val_f1_fixed, 'val_f1_mod': val_f1_mod, 'state_dict': model.state_dict()}
            torch.save(result, './ckpt/{}/best_val_loss.pth'.format(config.MODEL.NAME))

result = {'epoch': epoch, 'val_loss': val_loss, 'val_f1_best': val_f1_best,
          'val_f1_fixed': val_f1_fixed, 'val_f1_mod': val_f1_mod, 'state_dict': model.state_dict()}
torch.save(result, './ckpt/{}/final.pth'.format(config.MODEL.NAME))
