"""
Evaluation script for MedVision.
Computes image- and patient-level metrics and saves ROC/confusion plots.
"""

import argparse
import yaml
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import APTOSDataset, ISICDataset, get_val_transforms, create_dataloader
from src.models.classifier import Classifier
from src.utils.metrics import compute_metrics, aggregate_patient_scores
from src.utils.io import load_image_rgb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--dataset', type=str, choices=['aptos', 'isic'], default='aptos')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    return parser.parse_args()


def load_config(path: str):
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def evaluate(model, loader, device):
    model.eval()
    preds = []
    targets = []
    image_ids = []
    patient_ids = []
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy().tolist()
            preds.extend(probs)
            targets.extend(labels.detach().cpu().numpy().tolist())
            image_ids.extend(batch.get('image_id', [''] * len(probs)))
            patient_ids.extend(batch.get('patient_id', [''] * len(probs)))
    return targets, preds, image_ids, patient_ids


def plot_roc(y_true, y_probs, out_path: Path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_confusion(y_true, y_pred, out_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = Classifier(architecture=cfg['model']['architecture'], num_classes=cfg['model']['num_classes'], pretrained=False)
    chk = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(chk['model_state'])
    model = model.to(device)

    # Dataset
    if args.dataset == 'aptos':
        ds = APTOSDataset(cfg['data']['aptos_train_csv'], cfg['data']['aptos_train_images'], transform=get_val_transforms(cfg['data']['image_size']), preprocessing_config={
            'target_size': (cfg['data']['image_size'], cfg['data']['image_size']),
            'crop_borders': True,
            'use_clahe': cfg['data'].get('use_clahe', True),
            'hair_removal': cfg['data'].get('remove_hair', False),
            'clahe_clip_limit': cfg['data'].get('clahe_clip_limit', 2.0),
            'clahe_tile_size': cfg['data'].get('clahe_tile_size', 8)
        })
    else:
        ds = ISICDataset(cfg['data']['isic_train_csv'], cfg['data']['isic_train_images'], transform=get_val_transforms(cfg['data']['image_size']), preprocessing_config={
            'target_size': (cfg['data']['image_size'], cfg['data']['image_size']),
            'crop_borders': True,
            'use_clahe': cfg['data'].get('use_clahe', True),
            'hair_removal': cfg['data'].get('remove_hair', False),
            'clahe_clip_limit': cfg['data'].get('clahe_clip_limit', 2.0),
            'clahe_tile_size': cfg['data'].get('clahe_tile_size', 8)
        })

    loader = create_dataloader(ds, batch_size=cfg['evaluation'].get('batch_size', 32), shuffle=False, num_workers=cfg['training'].get('num_workers', 4), pin_memory=cfg['training'].get('pin_memory', True))

    y_true, y_probs, image_ids, patient_ids = evaluate(model, loader, device)

    # Image-level metrics
    metrics = compute_metrics(y_true, y_probs)
    print("Image-level metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Patient-level aggregation
    patient_scores = aggregate_patient_scores(patient_ids, y_probs, method=cfg['evaluation'].get('patient_aggregation', 'mean'))
    # Map patient true labels by majority voting from dataset
    # For simplicity, compute patient labels from dataset df
    patient_labels = {}
    for i, pid in enumerate(patient_ids):
        if pid not in patient_labels:
            patient_labels[pid] = []
        patient_labels[pid].append(y_true[i])
    patient_true = {pid: int(np.round(np.mean(vals))) for pid, vals in patient_labels.items()}

    p_true = list(patient_true.values())
    p_scores = [patient_scores[pid] for pid in patient_true.keys()]

    p_metrics = compute_metrics(p_true, p_scores)
    print("Patient-level metrics:")
    for k, v in p_metrics.items():
        print(f"{k}: {v}")

    # Plots
    plot_roc(y_true, y_probs, Path('outputs/roc_image_level.png'))
    preds = [1 if p >= cfg['evaluation'].get('confidence_threshold', 0.5) else 0 for p in y_probs]
    plot_confusion(y_true, preds, Path('outputs/confusion_image_level.png'))

    # Patient ROC
    plot_roc(p_true, p_scores, Path('outputs/roc_patient_level.png'))

    print('Evaluation complete')


if __name__ == '__main__':
    main()
