# MedVision: Early Disease Screening via Image Biomarkers

This repository implements an end-to-end research pipeline for early disease screening from medical images (e.g., diabetic retinopathy, skin lesion classification).

IMPORTANT: This system is for research and educational purposes only, not for clinical use.

## Project Overview

- Data preprocessing (OpenCV + albumentations)
- PyTorch models (transfer learning with ResNet)
- Training pipeline with AdamW and cosine scheduler
- Explainability: Grad-CAM and Integrated Gradients
- FastAPI service for online inference and explainability
- Dockerized for easy deployment
 
## Quickstart

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Update `config.yaml` with dataset paths and training options.

3. Train (example for APTOS subset):

```powershell
python train.py --config config.yaml --dataset aptos
```

4. Evaluate:

```powershell
python eval.py --config config.yaml --dataset aptos --checkpoint checkpoints/best_model.pth
```

5. Run the API locally:

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

- GET /health — service status
- POST /predict — single file upload (returns probability + class)
- POST /batch_predict — JSON body with base64 images and optional patient_ids; returns per-image and per-patient scores
- POST /explain — file upload and method (gradcam or ig) returns base64 heatmap image

## Explainability

- Grad-CAM: uses `pytorch-grad-cam` to compute class activation maps
- Integrated Gradients: uses Captum to compute attributions

Both methods return overlayed heatmaps as base64 PNG strings via the API.

## Notes & Ethics

This work is intended for prototyping and research. It is NOT cleared for clinical use. Models should not be used to make medical decisions.

Please respect dataset licenses (APTOS, ISIC) and follow their citation requirements.

## License

MIT
