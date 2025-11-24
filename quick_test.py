"""[X][[X]O[X]K[X]][X]
Quick test script to verify the MedVision system is set up correctly.
"""

import sys
from pathlib import Path

print("=" * 60)
print("MedVision Training - Quick Start")
print("=" * 60)

print("\n1. Testing imports...")

# Test imports
try:
    import torch
    import yaml
    from src.data import APTOSDataset, get_train_transforms, get_val_transforms, create_train_val_split, create_dataloader, get_class_weights
    from src.models.classifier import Classifier, get_loss_function
    print("[OK] All imports successful")
except Exception as e:
    print(f"[X] Import error: {e}")
    exit(1)

# Load config
print("\n2. Loading configuration...")
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
print(f"[OK] Config loaded")
print(f"  - Image size: {cfg['data']['image_size']}")
print(f"  - Model: {cfg['model']['architecture']}")
print(f"  - Batch size: {cfg['training']['batch_size']}")

# Check device
print(f"\n3. Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
if torch.cuda.is_available():
    print(f"  - GPU: {torch.cuda.get_device_name(0)}")
    print(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Load dataset
print("\n4. Loading dataset...")

# Check which dataset to use
csv_path = cfg['data']['aptos_train_csv']
img_dir = cfg['data']['aptos_train_images']

try:
    from src.data.dataset import APTOSDataset
    import pandas as pd
    
    # Check if paths exist
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not Path(img_dir).exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Get transforms for proper image sizing
    val_transform = get_val_transforms(image_size=cfg['data']['image_size'])
    
    # Preprocessing config to resize images
    preprocessing_config = {
        'target_size': (cfg['data']['image_size'], cfg['data']['image_size']),
        'crop_borders': cfg['data'].get('crop_margin', 0) > 0,
        'use_clahe': cfg['data'].get('use_clahe', True)
    }
    
    dataset = APTOSDataset(
        csv_path=csv_path,
        image_dir=img_dir,
        transform=val_transform,
        preprocessing_config=preprocessing_config,
        binary=False  # Use all 5 classes
    )
    print(f"[OK] Dataset loaded: {len(dataset)} samples")
    
    # Check class distribution
    if 'diagnosis' in df.columns:
        class_dist = df['diagnosis'].value_counts().sort_index().to_dict()
        print(f"  - Class distribution: {class_dist}")
    
except Exception as e:
    print(f"[X] Error loading dataset: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Create train/val split
print("\n5. Splitting train/validation...")
train_ds, val_ds = create_train_val_split(dataset, val_size=0.2, random_state=42)
print(f"[OK] Train: {len(train_ds)}, Val: {len(val_ds)}")

# Create dataloaders
print("\n6. Testing data loading...")
train_loader = create_dataloader(
    train_ds,
    batch_size=8,
    shuffle=True,
    num_workers=0
)
print(f"[OK] Dataloaders created")

try:
    batch = next(iter(train_loader))
    print(f"[OK] Batch loaded: images {batch['image'].shape}, labels {batch['label'].shape}")
except Exception as e:
    print(f"[X] Error loading batch: {e}")
    exit(1)

# Create model
print("\n7. Creating model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    model = Classifier(
        architecture=cfg['model']['architecture'],
        num_classes=cfg['model']['num_classes'],
        pretrained=cfg['model']['pretrained'],
        dropout=cfg['model']['dropout']
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"[OK] Model created: {cfg['model']['architecture']}")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
except Exception as e:
    print(f"[X] Error creating model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test forward pass
print("\n8. Testing forward pass...")
try:
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(train_loader))
        images = test_batch['image'].to(device)
        outputs = model(images)
        print(f"[OK] Forward pass successful: {outputs.shape}")
        print(f"  - Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
except Exception as e:
    print(f"[X] Error in forward pass: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Summary
print("\n" + "=" * 60)
print("[OK] ALL TESTS PASSED!")
print("=" * 60)

print("\nYour system is ready to train!")
print("\nTo start full training, run:")
print("  python train.py --config config.yaml --dataset aptos")

print(f"\nRecommended settings for your dataset:")
print(f"  - Adjust model.num_classes to {cfg['model']['num_classes']} in config.yaml (you have {cfg['model']['num_classes']} DR grades)")
print(f"  - Use weighted_ce loss (already configured)")
print(f"  - Train for 30-50 epochs")
print(f"  - Monitor validation AUC")
