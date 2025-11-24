"""
Training script for MedVision.
Usage example:
    python train.py --config config.yaml

This system is for research and educational purposes only, not for clinical use.
"""

import os
import yaml
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import logging

from src.data import (
    APTOSDataset, ISICDataset, BreastCancerDataset, create_breast_cancer_datasets,
    get_train_transforms, get_val_transforms, create_dataloader, create_train_val_split
)
from src.models.classifier import Classifier, get_loss_function
from src.utils.metrics import compute_metrics
from src.utils.io import save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--dataset', type=str, choices=['aptos', 'isic', 'breast_cancer'], default='aptos')
    parser.add_argument('--resume', type=str, default=None)
    return parser.parse_args()


def load_config(path: str):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def setup_logging(log_dir: str, level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('medvision')


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    preds = []
    targets = []
    pbar = tqdm(loader, desc='Train')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy().tolist()
        preds.extend(probs)
        targets.extend(labels.detach().cpu().numpy().tolist())
        pbar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(targets, preds)
    return epoch_loss, metrics


def validate(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    preds = []
    targets = []
    with torch.no_grad():
        pbar = tqdm(loader, desc='Val')
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy().tolist()
            preds.extend(probs)
            targets.extend(labels.detach().cpu().numpy().tolist())
            pbar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(targets, preds)
    return epoch_loss, metrics


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Logging
    logger = setup_logging(cfg.get('logging', {}).get('log_dir', 'logs'))
    logger.info('Loaded configuration')

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and cfg['training']['device'] != 'cpu' else 'cpu')
    logger.info(f'Using device: {device}')

    # Dataset selection
    if args.dataset == 'breast_cancer':
        # Use the breast cancer dataset
        train_transform = get_train_transforms(cfg['data']['image_size'])
        val_transform = get_val_transforms(cfg['data']['image_size'])
        train_ds, val_ds = create_breast_cancer_datasets(cfg, train_transform, val_transform)
        logger.info(f'Loaded breast cancer dataset: {len(train_ds)} train, {len(val_ds)} val')
    elif args.dataset == 'aptos':
        csv_path = cfg['data']['aptos_train_csv']
        image_dir = cfg['data']['aptos_train_images']
        dataset = APTOSDataset(csv_path, image_dir, transform=get_train_transforms(cfg['data']['image_size']), preprocessing_config={
            'target_size': (cfg['data']['image_size'], cfg['data']['image_size']),
            'crop_borders': True,
            'use_clahe': cfg['data'].get('use_clahe', True),
            'hair_removal': cfg['data'].get('remove_hair', False),
            'clahe_clip_limit': cfg['data'].get('clahe_clip_limit', 2.0),
            'clahe_tile_size': cfg['data'].get('clahe_tile_size', 8)
        })
        # Train/val split
        train_ds, val_ds = create_train_val_split(dataset, val_size=cfg['data'].get('train_val_split', 0.2), random_state=cfg['data'].get('random_seed', 42))
    else:
        csv_path = cfg['data']['isic_train_csv']
        image_dir = cfg['data']['isic_train_images']
        dataset = ISICDataset(csv_path, image_dir, transform=get_train_transforms(cfg['data']['image_size']), preprocessing_config={
            'target_size': (cfg['data']['image_size'], cfg['data']['image_size']),
            'crop_borders': True,
            'use_clahe': cfg['data'].get('use_clahe', True),
            'hair_removal': cfg['data'].get('remove_hair', False),
            'clahe_clip_limit': cfg['data'].get('clahe_clip_limit', 2.0),
            'clahe_tile_size': cfg['data'].get('clahe_tile_size', 8)
        })
        # Train/val split
        train_ds, val_ds = create_train_val_split(dataset, val_size=cfg['data'].get('train_val_split', 0.2), random_state=cfg['data'].get('random_seed', 42))

    # Dataloaders
    train_loader = create_dataloader(train_ds, batch_size=cfg['training'].get('batch_size', 16), shuffle=True, num_workers=cfg['training'].get('num_workers', 4), pin_memory=cfg['training'].get('pin_memory', True))
    val_loader = create_dataloader(val_ds, batch_size=cfg['evaluation'].get('batch_size', 32), shuffle=False, num_workers=cfg['training'].get('num_workers', 4), pin_memory=cfg['training'].get('pin_memory', True))

    # Model
    model = Classifier(architecture=cfg['model']['architecture'], num_classes=cfg['model']['num_classes'], pretrained=cfg['model'].get('pretrained', True), dropout=cfg['model'].get('dropout', 0.3))
    model = model.to(device)

    # Loss
    class_weights = None
    if cfg['training'].get('use_class_weights', False):
        # compute weights from train_ds
        from src.data.dataset import get_class_weights
        class_weights = get_class_weights(train_ds).to(device)
    loss_fn = get_loss_function(cfg['model'].get('loss_type', 'weighted_ce'), class_weights=class_weights, focal_alpha=cfg['model'].get('focal_alpha', 0.25), focal_gamma=cfg['model'].get('focal_gamma', 2.0))

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=cfg['training'].get('learning_rate', 1e-4), weight_decay=cfg['training'].get('weight_decay', 1e-4))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, cfg['training'].get('num_epochs', 50)))

    best_val_auc = -float('inf')
    checkpoint_dir = Path(cfg['training'].get('checkpoint_dir', 'checkpoints'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    if args.resume:
        chk = torch.load(args.resume, map_location=device)
        model.load_state_dict(chk['model_state'])
        optimizer.load_state_dict(chk.get('optimizer_state', optimizer.state_dict()))
        start_epoch = chk.get('epoch', 0) + 1
        logger.info(f"Resumed from checkpoint: {args.resume} (start_epoch={start_epoch})")

    num_epochs = cfg['training'].get('num_epochs', 50)
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_metrics = validate(model, val_loader, loss_fn, device)

        logger.info(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
        logger.info(f"Train metrics: {train_metrics}")
        logger.info(f"Val metrics: {val_metrics}")

        # Scheduler step
        try:
            scheduler.step()
        except Exception:
            pass

        # Checkpoint
        monitor = val_metrics.get('auc', -float('inf'))
        if monitor > best_val_auc:
            best_val_auc = monitor
            ckpt_path = checkpoint_dir / 'best_model.pth'
            save_checkpoint(model, optimizer, str(ckpt_path), epoch, metadata={'val_auc': float(best_val_auc)})
            logger.info(f"Saved best model to {ckpt_path} (val_auc={best_val_auc:.4f})")

    logger.info("Training complete")


if __name__ == '__main__':
    main()
