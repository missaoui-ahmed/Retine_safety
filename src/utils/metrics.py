"""
Metrics utilities for MedVision.
Contains functions for sensitivity, specificity, AUC, F1, and aggregation.
"""

from typing import Tuple, List, Dict
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support, accuracy_score


def sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Compute sensitivity (recall for positive class) and specificity.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sensitivity, specificity


def compute_metrics(y_true: List[int], y_probs: List[float], threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute standard metrics given true labels and predicted probabilities.
    Returns accuracy, auc, precision, recall, f1, sensitivity, specificity
    """
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    # Binarize predictions
    y_pred = (y_probs >= threshold).astype(int)
    
    # Accuracy
    acc = float(accuracy_score(y_true, y_pred))
    
    # AUC
    try:
        auc = float(roc_auc_score(y_true, y_probs))
    except Exception:
        auc = float('nan')
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    sensitivity, specificity = sensitivity_specificity(y_true, y_pred)
    
    return {
        'accuracy': acc,
        'auc': auc,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity)
    }


def aggregate_patient_scores(patient_ids: List[str], scores: List[float], method: str = 'mean'):
    """
    Aggregate image-level probabilities into patient-level scores.
    method: mean, max
    """
    from collections import defaultdict
    agg = defaultdict(list)
    for pid, s in zip(patient_ids, scores):
        agg[pid].append(s)
    
    patient_scores = {}
    for pid, vals in agg.items():
        if method == 'mean':
            patient_scores[pid] = float(np.mean(vals))
        elif method == 'max':
            patient_scores[pid] = float(np.max(vals))
        else:
            patient_scores[pid] = float(np.mean(vals))
    
    return patient_scores
