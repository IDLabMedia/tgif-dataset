"""
Modified by Paschalis Giakoumoglou (@p-giakoumoglou)
August 2024 @ ITI-CERTH
Added checkpoint resume capability and additional metrics
Based on original work by Kostas Triaridis (@kostino)
August 2023 @ ITI-CERTH

Usage:
python ec_train.py --exp /path/to/experiment.yaml

TXT file format:
    {img_path}{delimiter}{mask_path}{delimiter}{label}
    Example with space: /path/img.jpg /path/mask.png 1
    Example with comma: /path/img.jpg,/path/mask.png,1
    
Note:
    - Balanced sampling is performed with respect to each txt file
    - The smallest dataset determines the number of samples per epoch from each dataset
"""
import os
import argparse
import numpy as np
from tqdm import tqdm
from common.utils import AverageMeter
from common.losses import TruForLoss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import logging
import torch
import torchvision.transforms.functional as TF

from data.datasets import MixDataset
from common.metrics import computeLocalizationMetrics, computeLocF1_th
from models.cmnext_conf import CMNeXtWithConf
from common.split_params import group_weight
from common.lr_schedule import WarmUpPolyLR
from models.modal_extract import ModalitiesExtractor
from configs.cmnext_init_cfg import _C as config, update_config

import cv2
cv2.setNumThreads(1)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-gpu', '--gpu', type=int, default=0, help='device, use -1 for cpu')
    parser.add_argument('-log', '--log', type=str, default='INFO', help='logging level')
    parser.add_argument('-train_bayar', '--train_bayar', action='store_true', help='finetune bayar conv')
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
    
    
    modal_extractor = ModalitiesExtractor(config.MODEL.MODALS[1:], config.MODEL.NP_WEIGHTS)
    if 'bayar' in config.MODEL.MODALS:
        modal_extractor.load_state_dict(torch.load('pretrained/modal_extractor/bayar_mhsa.pth'), strict=False)
        if not args.train_bayar:
            modal_extractor.bayar.eval()
            for param in modal_extractor.bayar.parameters():
                param.requires_grad = False
    
    model = CMNeXtWithConf(config.MODEL)
    if os.path.exists('./ckpt/{}/last_ckpt.pth'.format(config.MODEL.NAME)):
        checkpoint = torch.load('./ckpt/{}/last_ckpt.pth'.format(config.MODEL.NAME), map_location=torch.device(device))
        last_epoch = checkpoint['epoch']+1
        model.load_state_dict(checkpoint['state_dict'])
        #modal_extractor.load_state_dict(checkpoint['extractor_state_dict'])
    else:
        last_epoch = 0
    modal_extractor.to(device)
    model = model.to(device)
    
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
    modal_extract_params = group_weight(modal_extract_params, modal_extractor, torch.nn.BatchNorm2d, config.LEARNING_RATE)
    
    params.append(dict(params=cmnext_params[0]['params'] + modal_extract_params[0]['params'], lr=config.LEARNING_RATE))
    params.append(dict(params=cmnext_params[1]['params'] + modal_extract_params[1]['params'], weight_decay=.0,
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
        model.set_train()
        if args.train_bayar:
            modal_extractor.set_train()
        avg_loss = AverageMeter()
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc='Training Epoch {}/{}'.format(epoch + 1, config.EPOCHS), unit='steps')
        for step, (images, _, masks, _) in enumerate(pbar):
    
            images = images.to(device, non_blocking=True)
            masks = masks.squeeze(1).to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                modals = modal_extractor(images)
    
                images_norm = TF.normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                inp = [images_norm] + modals
    
                pred = model(inp)
    
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
            model.set_val()
            modal_extractor.set_val()
            pbar = tqdm(val_loader, desc='Validating Epoch {}/{}'.format(epoch + 1, config.EPOCHS), unit='steps')
            for step, (images, _, masks, lab) in enumerate(pbar):
                with torch.no_grad():
                    images = images.to(device, non_blocking=True)
                    masks = masks.squeeze(1).to(device, non_blocking=True)
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        modals = modal_extractor(images)
        
                        images_norm = TF.normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        inp = [images_norm] + modals
        
                        pred = model(inp)
        
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
            print(f"Epoch {epoch} - Val Loss: {val_loss}, Val F1 best: {val_f1_best}, Val F1 fixed: {val_f1_fixed}, Val F1 mod: {val_f1_mod}")
            result = {'epoch': epoch, 'val_loss': val_loss, 'val_f1_best': val_f1_best,
                      'val_f1_fixed': val_f1_fixed, 'val_f1_mod': val_f1_mod, 'state_dict': model.state_dict(),
                      'extractor_state_dict': modal_extractor.state_dict()}
            torch.save(result, './ckpt/{}/last_ckpt.pth'.format(config.MODEL.NAME))
            
            if val_loss < min_loss:
                min_loss = val_loss
                result = {'epoch': epoch, 'val_loss': val_loss, 'val_f1_best': val_f1_best,
                          'val_f1_fixed': val_f1_fixed, 'val_f1_mod': val_f1_mod, 'state_dict': model.state_dict(),
                          'extractor_state_dict': modal_extractor.state_dict()}
                torch.save(result, './ckpt/{}/best_val_loss.pth'.format(config.MODEL.NAME))
    
    result = {'epoch': config.EPOCHS - 1, 'val_loss': val_loss_avg.average(), 'val_f1_best': np.nanmean(f1),
              'val_f1_fixed': np.nanmean(f1th), 'val_f1_mod': np.nanmean(f1mods), 'state_dict': model.state_dict(),
              'extractor_state_dict': modal_extractor.state_dict()}
    torch.save(result, './ckpt/{}/final.pth'.format(config.MODEL.NAME))
