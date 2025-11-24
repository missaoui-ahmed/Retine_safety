"""[X][[X]O[X]K[X]][X]
Test single image inference with the trained model
"""

import torch
import cv2
import yaml
from pathlib import Path
import numpy as np
from src.models.classifier import Classifier
from src.data.transforms import preprocess_medical_image
import pandas as pd

def test_single_image():
    """Test inference on a single random image from the dataset"""
    
    print("\n" + "="*60)
    print("Testing Single Image Inference")
    print("="*60)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n1. Device: {device}")
    
    # Load model
    print("\n2. Loading trained model...")
    model = Classifier(
        architecture=config['model']['architecture'],
        num_classes=config['model']['num_classes'],
        pretrained=False  # We're loading trained weights
    )
    
    checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
            epoch = checkpoint.get('epoch', 'N/A')
            metadata = checkpoint.get('metadata', {})
            val_auc = metadata.get('val_auc', 0.0)
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'N/A')
            val_auc = checkpoint.get('val_auc', 0.0)
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            epoch = checkpoint.get('epoch', 'N/A')
            val_auc = checkpoint.get('val_auc', 0.0)
        else:
            # Checkpoint is just the state dict
            model.load_state_dict(checkpoint)
            epoch = 'N/A'
            val_auc = 0.0
    else:
        model.load_state_dict(checkpoint)
        epoch = 'N/A'
        val_auc = 0.0
    
    model.to(device)
    model.eval()
    print(f"   [OK] Model loaded (epoch {epoch}, val_auc={val_auc:.4f})")
    
    # Get a random test image
    print("\n3. Loading test image...")
    train_df = pd.read_csv(config['data']['aptos_train_csv'])
    test_row = train_df.sample(1).iloc[0]
    
    image_id = test_row['id_code']
    true_label = test_row['diagnosis']
    image_path = Path(config['data']['aptos_train_images']) / f"{image_id}.png"
    
    print(f"   - Image ID: {image_id}")
    print(f"   - True Label: {true_label} (DR Grade {true_label})")
    
    # Load and preprocess image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"   [X] Failed to load image: {image_path}")
        return
    
    print(f"   - Original size: {image.shape}")
    
    # Preprocess
    preprocessed = preprocess_medical_image(
        image,
        target_size=(config['data']['image_size'], config['data']['image_size']),
        crop_borders=True,
        use_clahe=True,
        hair_removal=False
    )
    
    # Convert to tensor
    tensor = torch.from_numpy(preprocessed).permute(2, 0, 1).float() / 255.0
    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    tensor = tensor.unsqueeze(0).to(device)
    
    print(f"   - Preprocessed size: {tensor.shape}")
    
    # Run inference
    print("\n4. Running inference...")
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"True Label:      DR Grade {true_label}")
    print(f"Predicted:       DR Grade {predicted_class}")
    print(f"Confidence:      {confidence*100:.2f}%")
    match_str = "[OK] CORRECT" if predicted_class == true_label else "[X] INCORRECT"
    print(f"Match:           {match_str}")
    
    print("\nClass Probabilities:")
    print("-" * 40)
    class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    for i, prob in enumerate(probabilities[0]):
        bar = "=" * int(prob.item() * 50)
        print(f"  Grade {i} ({class_names[i]:15}): {prob.item()*100:5.2f}% {bar}")
    
    print("="*60)
    print("\n[OK] Inference test complete!")
    
    return predicted_class == true_label

if __name__ == "__main__":
    try:
        correct = test_single_image()
        # Exit 0 regardless of prediction accuracy - we're just testing the pipeline works
        print("\n[OK] Inference pipeline test completed successfully!")
        exit(0)
    except Exception as e:
        print(f"\n[X] Error during inference test: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
