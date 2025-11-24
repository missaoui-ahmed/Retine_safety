"""
I/O helpers for MedVision: image loading/saving and checkpoint helpers.
"""

import os
import base64
from typing import Tuple
from pathlib import Path
import cv2
import numpy as np
import torch


def load_image_rgb(path: str) -> np.ndarray:
    """
    Load image and convert to RGB
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(path: str, image: np.ndarray) -> None:
    """
    Save RGB image (uint8) to disk as PNG
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, rgb)


def tensor_to_base64(tensor: np.ndarray) -> str:
    """
    Convert a numpy image (H,W,3) uint8 to base64 PNG string
    """
    import io
    from PIL import Image

    if tensor.dtype != np.uint8:
        tensor = tensor.astype(np.uint8)
    img = Image.fromarray(tensor)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return b64


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str, epoch: int, metadata: dict = None):
    """
    Save model checkpoint with optimizer state.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'metadata': metadata or {}
    }
    torch.save(state, path)


def load_checkpoint(path: str, model: torch.nn.Module = None, optimizer: torch.optim.Optimizer = None, map_location=None):
    """
    Load checkpoint and optionally restore model and optimizer.
    Returns checkpoint dict.
    """
    if map_location is None:
        map_location = 'cpu'
    chk = torch.load(path, map_location=map_location)
    if model is not None:
        model.load_state_dict(chk['model_state'])
    if optimizer is not None and 'optimizer_state' in chk:
        optimizer.load_state_dict(chk['optimizer_state'])
    return chk
