# -*- coding: utf-8 -*-
# @Time    : 2020-02-26 17:44
# @Author  : Zonas
# @Email   : zonas.wang@gmail.com
# @File    : train.py
"""

"""
import argparse
import logging
import os
import os.path as osp
import glob
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch import optim
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import read_image, ImageReadMode
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from unet import NestedUNet
from unet import UNet
from utils.dataset import BasicDataset
from config import UNetConfig

from losses import LovaszLossSoftmax
from losses import LovaszLossHinge
from losses import DiceLoss

import random
# from LovaszSoftmax.pytorch.lovasz_losses import binary_xloss, StableBCELoss

cfg = UNetConfig()

def train_net(net, cfg):
    # dataset = BasicDataset(cfg.images_dir, cfg.masks_dir, cfg.scale)
    # customize the dataset
    class SemiconDataset(Dataset):
        def __init__(self, images_dir, masks_dir, scale):
            self.images_dir = images_dir
            self.masks_dir = masks_dir
            self.scale = scale

            self.images = glob.glob('data/images_copy/*')
            self.b_mask = glob.glob('data/b_mask_copy/*')
            self.l_mask = glob.glob('data/l_mask_copy/*')
            self.s_mask = glob.glob('data/s_mask_copy/*')
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            image = read_image(self.images[idx])
            image = image/255 # TODO check if this is needed
            # three_channel_image = torch.stack((image, image, image), dim=0).squeeze(1)

            b_mask = read_image(self.b_mask[idx])
            l_mask = read_image(self.l_mask[idx])
            s_mask = read_image(self.s_mask[idx])

            # ramdom crop
            i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(400, 400))
            image = TF.crop(image, i, j, h, w)
            b_mask = TF.crop(b_mask, i, j, h, w)
            l_mask = TF.crop(l_mask, i, j, h, w)
            s_mask = TF.crop(s_mask, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                b_mask = TF.hflip(b_mask)
                l_mask = TF.hflip(l_mask)
                s_mask = TF.hflip(s_mask)

            # Random vertical flipping
            if random.random() > 0.5:
                image = TF.vflip(image)
                b_mask = TF.vflip(b_mask)
                l_mask = TF.vflip(l_mask)
                s_mask = TF.vflip(s_mask)

            combined_mask = torch.stack((b_mask, l_mask, s_mask), dim=0).squeeze(1)
            
            return image, combined_mask

    dataset = SemiconDataset(cfg.images_dir, cfg.masks_dir, cfg.scale)

    val_percent = cfg.validation / 100
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=False) # True triggers warning
    val_loader = DataLoader(val,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=False) # True triggers warning

    writer = SummaryWriter(comment=f'LR_{cfg.lr}_BS_{cfg.batch_size}_SCALE_{cfg.scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {cfg.epochs}
        Batch size:      {cfg.batch_size}
        Learning rate:   {cfg.lr}
        Optimizer:       {cfg.optimizer}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {cfg.save_cp}
        Device:          {device.type}
        Images scaling:  {cfg.scale}
    ''')

    if cfg.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(),
                               lr=cfg.lr)
    elif cfg.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(net.parameters(),
                                  lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    else:
        optimizer = optim.SGD(net.parameters(),
                              lr=cfg.lr,
                              momentum=cfg.momentum,
                              weight_decay=cfg.weight_decay,
                              nesterov=cfg.nesterov)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg.lr_decay_milestones,
                                               gamma = cfg.lr_decay_gamma)
    if cfg.n_classes > 1:
        criterion_1 = F.binary_cross_entropy#(inputs, targets, reduction='mean')
        criterion_2 = DiceLoss()
        # criterion = CrossEntropyLoss()
        # criterion = LovaszLossSoftmax()
        # criterion = dice_coeff
    else:
        criterion = LovaszLossHinge()

    for epoch in range(cfg.epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{cfg.epochs}', unit='img') as pbar:
            for batch_imgs, batch_masks in train_loader:

            # for batch in train_loader:
                # batch_imgs = batch['image']
                # batch_masks = batch['mask']
                assert batch_imgs.shape[1] == cfg.n_channels, \
                        f'Network has been defined with {cfg.n_channels} input channels, ' \
                        f'but loaded images have {batch_imgs.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                batch_imgs = batch_imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 #if cfg.n_classes == 1 else torch.long
                batch_masks = batch_masks.to(device=device, dtype=mask_type)

                inference_masks = net(batch_imgs)
                if cfg.n_classes == 1:
                    inferences = inference_masks
                    masks = batch_masks
                    # inferences = inference_masks.squeeze(1)
                    # masks = batch_masks.squeeze(1)
                else:
                    inferences = inference_masks
                    masks = batch_masks

                if cfg.deepsupervision:
                    bce_loss = 0
                    dice_loss = 0
                    loss = 0
                    for inference_mask in inferences:
                        bce_loss += criterion_1(F.sigmoid(inference_mask), masks)
                        dice_loss += criterion_2(inference_mask, masks)
                    bce_loss /= len(inferences)
                    dice_loss /= len(inferences)
                    loss += dice_loss
                    loss += bce_loss
                else:
                    loss = criterion(inferences, masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('BCELoss/train', bce_loss.item(), global_step)
                writer.add_scalar('DiceLoss/train', dice_loss.item(), global_step)
                writer.add_scalar('model/lr', optimizer.param_groups[0]['lr'], global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                pbar.update(batch_imgs.shape[0])
                global_step += 1

                if global_step % (len(dataset) // (10 * cfg.batch_size)) == 0:
                    val_score, val_bce, val_dice = eval_net(net, val_loader, device, n_val, cfg)
                    if cfg.n_classes > 1:
                        logging.info('Validation loss: {}'.format(val_score))
                        writer.add_scalar('Validation loss', val_score, global_step)
                        writer.add_scalar('Validation BCE', val_bce, global_step)
                        writer.add_scalar('Validation Dice', val_dice, global_step)

                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', batch_imgs, global_step)
                    if cfg.deepsupervision:
                            inference_masks = inference_masks[-1]
                    if cfg.n_classes == 1:
                        # writer.add_images('masks/true', batch_masks, global_step)
                        inference_mask = torch.sigmoid(inference_masks) > cfg.out_threshold
                        writer.add_images('masks/inference',
                                          inference_mask,
                                          global_step)
                    else:
                        # writer.add_images('masks/true', batch_masks, global_step)
                        ids = inference_masks.shape[1]  # N x C x H x W
                        inference_masks = torch.chunk(inference_masks, ids, dim=1)
                        for idx in range(0, len(inference_masks)):
                            inference_mask = torch.sigmoid(inference_masks[idx]) > cfg.out_threshold
                            writer.add_images('masks/inference_'+str(idx),
                                              inference_mask,
                                              global_step)

        if cfg.save_cp:
            try:
                os.mkdir(cfg.checkpoints_dir)
                logging.info('Created checkpoint directory')
            except OSError:
                pass

            ckpt_name = 'epoch_' + str(epoch + 1) + '.pth'
            torch.save(net.state_dict(),
                       osp.join(cfg.checkpoints_dir, ckpt_name))
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def eval_net(net, loader, device, n_val, cfg):
    """
    Evaluation without the densecrf with the dice coefficient

    """
    net.eval()
    tot = 0
    bce_loss = 0
    dice_coef = 0
    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        for imgs, true_masks in loader:
        # for batch in loader:
            # imgs = batch['image']
            # true_masks = batch['mask']

            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 #if cfg.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)
            criterion_1 = BCE = F.binary_cross_entropy #(inputs, targets, reduction='mean')#StableBCELoss()#binary_xloss
            criterion_2 = DiceLoss()
            # compute loss
            if cfg.deepsupervision:
                masks_preds = net(imgs)
                for masks_pred in masks_preds:
                    tot_ = 0
                    tot_bce = 0
                    tot_dice = 0
                    for true_mask, pred in zip(true_masks, masks_pred):
                        pred = (pred > cfg.out_threshold).float()
                        if cfg.n_classes > 1:
                            bce = criterion_1(F.sigmoid(pred), true_mask) 
                            dice = criterion_2(pred, true_mask.float())
                            # sub_cross_entropy = F.cross_entropy(pred, true_mask).item()
                            # sub_cross_entropy = F.cross_entropy(pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0).squeeze(1)).item()
                        else:
                            sub_cross_entropy = dice_coeff(pred, true_mask.squeeze(dim=1)).item()
                        tot_bce += bce
                        tot_dice += dice
                    tot_bce = tot_bce / len(masks_preds)
                    tot_dice = tot_dice / len(masks_preds)
                    bce_loss += tot_bce
                    dice_coef += tot_dice
                    tot_ += tot_bce
                    tot_ += tot_dice
                    tot += tot_
            else:
                masks_pred = net(imgs)
                for true_mask, pred in zip(true_masks, masks_pred):
                    pred = (pred > cfg.out_threshold).float()
                    if cfg.n_classes > 1:
                        tot += F.cross_entropy(pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0).squeeze(1)).item()
                    else:
                        tot += dice_coeff(pred, true_mask.squeeze(dim=1)).item()

            pbar.update(imgs.shape[0])

    return tot / n_val, bce_loss / n_val, dice_coef / n_val, 


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    logging.info(f'Using device {device}')

    net = eval(cfg.model)(cfg)
    logging.info(f'Network:\n'
                 f'\t{cfg.model} model\n'
                 f'\t{cfg.n_channels} input channels\n'
                 f'\t{cfg.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if cfg.bilinear else "Dilated conv"} upscaling')

    if cfg.load:
        net.load_state_dict(
            torch.load(cfg.load, map_location=device)
        )
        logging.info(f'Model loaded from {cfg.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net, cfg=cfg)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
