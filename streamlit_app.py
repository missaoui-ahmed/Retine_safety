import streamlit as st
import torch
import cv2
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Diabetic Retinopathy Detection - MedVision",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-text {
        color: #28a745;
        font-weight: bold;
    }
    .error-text {
        color: #dc3545;
        font-weight: bold;
    }
    .info-text {
        color: #0066cc;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'config' not in st.session_state:
    st.session_state.config = None
if 'device' not in st.session_state:
    st.session_state.device = None
if 'test_results' not in st.session_state:
    st.session_state.test_results = []

@st.cache_resource
def load_config():
    """Load configuration file"""
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        st.error(f"Failed to load config: {e}")
        return None

@st.cache_resource
def load_model(config):
    """Load trained model"""
    try:
        from src.models.classifier import Classifier
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = Classifier(
            architecture=config['model']['architecture'],
            num_classes=config['model']['num_classes'],
            pretrained=False
        )
        
        checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
        
        if isinstance(checkpoint, dict):
            if 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'])
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        return model, device
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

def preprocess_image(image_path, config):
    """Preprocess image for inference"""
    try:
        from src.data.transforms import preprocess_medical_image
        
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        preprocessed = preprocess_medical_image(
            image,
            target_size=(config['data']['image_size'], config['data']['image_size']),
            crop_borders=True,
            use_clahe=True,
            hair_removal=False
        )
        
        tensor = torch.from_numpy(preprocessed).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        
        return tensor.unsqueeze(0)
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return None

def is_likely_retinal_image(image_path):
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False, 0.0, "Could not load image"
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Check 1: Est que l'image est circulaire walla le 
        h, w = gray.shape
        aspect_ratio = max(h, w) / min(h, w)
        is_square_ish = aspect_ratio < 1.3
        
        # Check 2: Bordure En generale tkoun Black
        border_mean = np.mean([gray[0,:].mean(), gray[-1,:].mean(), gray[:,0].mean(), gray[:,-1].mean()])
        has_dark_borders = border_mean < 50
        
        # Check 3: Intensity range (retinal images have good contrast)
        intensity_range = gray.max() - gray.min()
        has_good_range = intensity_range > 100
        
        # Check 4: Color Variation
        color_variance = np.std([img_rgb[:,:,0].std(), img_rgb[:,:,1].std(), img_rgb[:,:,2].std()])
        is_colored = color_variance > 5  # Has color variation
        
        # Scoring
        score = 0
        reasons = []
        
        if is_square_ish:
            score += 25
            reasons.append(f"✓ Square/circular shape (aspect: {aspect_ratio:.2f})")
        else:
            reasons.append(f"✗ Non-square shape (aspect: {aspect_ratio:.2f})")
        
        if has_dark_borders:
            score += 30
            reasons.append(f"✓ Dark borders (mean={border_mean:.0f})")
        else:
            reasons.append(f"✗ No dark borders (mean={border_mean:.0f})")
        
        if has_good_range:
            score += 25
            reasons.append(f"✓ Good intensity range ({intensity_range})")
        else:
            reasons.append(f"✗ Limited range ({intensity_range})")
        
        if is_colored:
            score += 20
            reasons.append(f"✓ Color image (variance: {color_variance:.1f})")
        else:
            reasons.append(f"✗ Grayscale image")
        
        is_retinal = score >= 60
        confidence = score / 100.0
        reason = "; ".join(reasons)
        
        return is_retinal, confidence, reason
        
    except Exception as e:
        return False, 0.0, f"Error: {str(e)}"

def run_inference(image_tensor, model, device, config):
    """Run inference on preprocessed image"""
    try:
        with torch.no_grad():
            tensor = image_tensor.to(device)
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            probs_dict = {i: prob.item() for i, prob in enumerate(probabilities[0])}
        
        return predicted_class, confidence, probs_dict
    except Exception as e:
        st.error(f"Inference error: {e}")
        return None, None, None

def test_system():
    """Test system components"""
    results = {}
    
    # Test imports
    try:
        import torch
        import cv2
        import yaml
        import pandas as pd
        from src.models.classifier import Classifier
        from src.data import APTOSDataset
        results['Imports'] = {'status': True, 'message': 'All imports successful'}
    except Exception as e:
        results['Imports'] = {'status': False, 'message': str(e)}
    
    # Test GPU
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            results['GPU'] = {'status': True, 'message': f'{gpu_name} ({gpu_mem:.2f}GB)'}
        else:
            results['GPU'] = {'status': False, 'message': 'CUDA not available'}
    except Exception as e:
        results['GPU'] = {'status': False, 'message': str(e)}
    
    # Test dataset
    try:
        config = load_config()
        csv_path = config['data']['aptos_train_csv']
        df = pd.read_csv(csv_path)
        results['Dataset'] = {'status': True, 'message': f'{len(df)} retinal images loaded, {df["diagnosis"].nunique()} DR grades'}
    except Exception as e:
        results['Dataset'] = {'status': False, 'message': str(e)}
    
    # Test model
    try:
        config = load_config()
        if Path('checkpoints/best_model.pth').exists():
            model_path = Path('checkpoints/best_model.pth')
            model_size = model_path.stat().st_size / (1024**2)
            results['Model'] = {'status': True, 'message': f'{model_size:.2f}MB checkpoint found'}
        else:
            results['Model'] = {'status': False, 'message': 'Checkpoint not found'}
    except Exception as e:
        results['Model'] = {'status': False, 'message': str(e)}
    
    return results

# Sidebar
st.sidebar.title("🏥 MedVision Control Panel")
page = st.sidebar.radio(
    "Select Test Module",
    ["📊 Dashboard", "🔧 System Test", "🖼️ Single Image Test", "📁 Batch Test", "📈 Model Analysis", "⚙️ Settings"]
)

# Load config globally
config = load_config()
if config is None:
    st.error("Configuration file not found!")
    st.stop()

# CLASS NAMES FOR DIABETIC RETINOPATHY
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
CLASS_COLORS = {
    0: "#28a745",  # Green - No DR
    1: "#90ee90",  # Light Green - Mild
    2: "#ffc107",  # Yellow - Moderate
    3: "#ff8c00",  # Orange - Severe
    4: "#dc3545",  # Red - Proliferative DR
}

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================
if page == "📊 Dashboard":
    st.title("🏥 MedVision Testing Dashboard")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Architecture", config['model']['architecture'].upper())
    
    with col2:
        st.metric("Classes", config['model']['num_classes'])
    
    with col3:
        st.metric("Image Size", f"{config['data']['image_size']}×{config['data']['image_size']}")
    
    with col4:
        if torch.cuda.is_available():
            st.metric("GPU", "Available ✓")
        else:
            st.metric("GPU", "Not Available ✗")
    
    st.markdown("---")
    
    # System Status
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 System Status")
        results = test_system()
        
        for component, result in results.items():
            if result['status']:
                st.success(f"✓ {component}: {result['message']}")
            else:
                st.error(f"✗ {component}: {result['message']}")
    
    with col2:
        st.subheader("📋 Configuration")
        st.json({
            "Architecture": config['model']['architecture'],
            "Num Classes": config['model']['num_classes'],
            "Image Size": config['data']['image_size'],
            "Batch Size": config['training']['batch_size'],
            "Learning Rate": config['training']['learning_rate'],
            "Loss Type": config['model']['loss_type']
        })
    
    st.markdown("---")
    st.subheader("📊 Quick Stats")
    
    try:
        csv_path = config['data']['aptos_train_csv']
        df = pd.read_csv(csv_path)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Retinal Images", len(df))
        
        with col2:
            st.metric("DR Grades", "5 (0-4)")
        
        with col3:
            if 'diagnosis' in df.columns:
                benign_count = sum(df['diagnosis'] == 0)
                total = len(df)
                st.metric("No DR Cases", f"{benign_count} ({benign_count/total*100:.1f}%)")
        
        # Class distribution chart
        st.subheader("Class Distribution")
        class_dist = df['diagnosis'].value_counts().sort_index()
        
        fig = go.Figure(data=[
            go.Bar(
                x=[CLASS_NAMES[int(i)] for i in class_dist.index],
                y=class_dist.values,
                marker=dict(color=[CLASS_COLORS.get(int(i), "#0066cc") for i in class_dist.index]),
                text=class_dist.values,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Dataset Class Distribution",
            xaxis_title="DR Grade",
            yaxis_title="Number of Images",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

# ============================================================================
# PAGE: SYSTEM TEST
# ============================================================================
elif page == "🔧 System Test":
    st.title("🔧 System Verification Test")
    st.markdown("Run comprehensive system diagnostics")
    st.markdown("---")
    
    if st.button("▶️ Run System Tests", key="run_system_tests"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = {}
        
        # Test 1: Imports
        status_text.text("Testing imports...")
        try:
            import torch
            import cv2
            import yaml
            import pandas as pd
            from src.models.classifier import Classifier
            from src.data import APTOSDataset, get_val_transforms
            results['Imports'] = True
            progress_bar.progress(20)
        except Exception as e:
            results['Imports'] = False
            st.error(f"Import failed: {e}")
        
        # Test 2: GPU
        status_text.text("Checking GPU...")
        try:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                results['GPU'] = f"{gpu_name} ({gpu_mem:.2f}GB)"
            else:
                device = torch.device('cpu')
                results['GPU'] = "CPU Only"
            progress_bar.progress(40)
        except Exception as e:
            results['GPU'] = f"Error: {e}"
        
        # Test 3: Dataset
        status_text.text("Loading dataset...")
        try:
            csv_path = config['data']['aptos_train_csv']
            img_dir = config['data']['aptos_train_images']
            df = pd.read_csv(csv_path)
            results['Dataset'] = f"{len(df)} retinal images, {df['diagnosis'].nunique()} DR grades"
            progress_bar.progress(60)
        except Exception as e:
            results['Dataset'] = f"Error: {e}"
        
        # Test 4: Model
        status_text.text("Loading model...")
        try:
            model, device = load_model(config)
            if model is not None:
                total_params = sum(p.numel() for p in model.parameters())
                results['Model'] = f"{config['model']['architecture']} ({total_params:,} params)"
                progress_bar.progress(80)
            else:
                results['Model'] = "Failed to load"
        except Exception as e:
            results['Model'] = f"Error: {e}"
        
        # Test 5: Forward pass
        status_text.text("Testing forward pass...")
        try:
            if model is not None:
                dummy_input = torch.randn(1, 3, config['data']['image_size'], config['data']['image_size']).to(device)
                with torch.no_grad():
                    output = model(dummy_input)
                results['Forward Pass'] = f"Success ({output.shape})"
            progress_bar.progress(100)
        except Exception as e:
            results['Forward Pass'] = f"Error: {e}"
        
        status_text.text("Tests completed!")
        st.markdown("---")
        
        # Results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✓ Passed Tests")
            for test, result in results.items():
                if result and "Error" not in str(result):
                    st.success(f"**{test}**: {result}")
        
        with col2:
            st.subheader("✗ Failed Tests")
            has_failed = False
            for test, result in results.items():
                if not result or "Error" in str(result):
                    st.error(f"**{test}**: {result}")
                    has_failed = True
            if not has_failed:
                st.info("All tests passed!")

# ============================================================================
# PAGE: SINGLE IMAGE TEST
# ============================================================================
elif page == "🖼️ Single Image Test":
    st.title("🖼️ Single Image Inference Test")
    st.markdown("Upload or select an image for classification")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    true_label = None  # Track true label for dataset selection
    
    with col1:
        st.subheader("📤 Input Options")
        input_mode = st.radio("Select input mode:", ["Upload Image", "Select from Dataset"])
        
        if input_mode == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
            image_path = None
            if uploaded_file is not None:
                # Save uploaded file temporarily (Windows-compatible)
                temp_dir = Path("tmp")
                temp_dir.mkdir(exist_ok=True)  # Create tmp directory if it doesn't exist
                temp_path = temp_dir / uploaded_file.name
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                image_path = temp_path
        
        else:  # Select from dataset
            try:
                csv_path = config['data']['aptos_train_csv']
                df = pd.read_csv(csv_path)
                img_dir = config['data']['aptos_train_images']
                
                # Filter by class if needed
                selected_class = st.selectbox("Filter by DR Grade:", [-1, 0, 1, 2, 3, 4], 
                                             format_func=lambda x: "All Classes" if x == -1 else CLASS_NAMES[x])
                
                if selected_class >= 0:
                    df_filtered = df[df['diagnosis'] == selected_class]
                else:
                    df_filtered = df
                
                if len(df_filtered) > 0:
                    selected_idx = st.selectbox("Select image:", range(min(len(df_filtered), 100)))
                    selected_row = df_filtered.iloc[selected_idx]
                    image_id = selected_row['id_code']
                    true_label = int(selected_row['diagnosis'])
                    
                    image_path = Path(img_dir) / f"{image_id}.png"
                    st.info(f"📍 Image ID: {image_id}")
                    st.info(f"📊 True Label: Grade {true_label} ({CLASS_NAMES[true_label]})")
                else:
                    st.warning("No images found in this class")
                    image_path = None
            
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
                image_path = None
    
    with col2:
        if image_path and Path(image_path).exists():
            st.subheader("🖼️ Original Image")
            original_img = cv2.imread(str(image_path))
            if original_img is not None:
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                st.image(original_img, use_container_width=True)
                st.caption(f"Size: {original_img.shape[1]}×{original_img.shape[0]}")
            
            # Validate if image looks like a retinal image
            if input_mode == "Upload Image":
                is_retinal, val_confidence, reason = is_likely_retinal_image(image_path)
                
                if not is_retinal:
                    st.warning(f"⚠️ **Warning: This may not be a retinal fundus image!**")
                    st.info(f"**Validation Score: {val_confidence*100:.0f}%**\n\n{reason}")
                    st.markdown("""
                    **Why this matters:**
                    - This model was trained **only on retinal fundus images** (diabetic retinopathy dataset)
                    - It expects circular, color images of the eye fundus
                    - **Non-medical images** (dogs, cats, landscapes, etc.) will produce **meaningless predictions**
                    - The model only knows DR grades 0-4!
                    
                    **For accurate results, please upload:**
                    - Retinal fundus photographs
                    - Eye fundus images
                    - Diabetic retinopathy screening images
                    """)
                else:
                    st.success(f"✓ Image validation: {val_confidence*100:.0f}% confidence this is a mammogram")
            
            # Run inference button
            if st.button("🚀 Run Inference", key="run_inference"):
                with st.spinner("Loading model..."):
                    model, device = load_model(config)
                
                if model is not None:
                    with st.spinner("Preprocessing image..."):
                        image_tensor = preprocess_image(image_path, config)
                    
                    if image_tensor is not None:
                        with st.spinner("Running inference..."):
                            predicted_class, confidence, probs = run_inference(image_tensor, model, device, config)
                        
                        if predicted_class is not None:
                            st.success("✓ Inference completed!")
                            st.markdown("---")
                            
                            # Results
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Predicted Grade", CLASS_NAMES[predicted_class])
                            
                            with col2:
                                st.metric("Confidence", f"{confidence*100:.2f}%")
                            
                            with col3:
                                if input_mode == "Select from Dataset":
                                    match = predicted_class == true_label
                                    st.metric("Accuracy", "✓ Correct" if match else "✗ Wrong")
                            
                            st.markdown("---")
                            st.subheader("📊 Class Probabilities")
                            
                            # Probability bar chart
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=[CLASS_NAMES[i] for i in range(len(probs))],
                                    y=[probs[i] for i in range(len(probs))],
                                    marker=dict(color=[CLASS_COLORS.get(i, "#0066cc") for i in range(len(probs))]),
                                    text=[f"{probs[i]*100:.2f}%" for i in range(len(probs))],
                                    textposition='auto'
                                )
                            ])
                            
                            fig.update_layout(
                                title="Prediction Probabilities",
                                xaxis_title="DR Grade",
                                yaxis_title="Probability",
                                height=400,
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Pie chart
                            fig_pie = go.Figure(data=[go.Pie(
                                labels=[CLASS_NAMES[i] for i in range(len(probs))],
                                values=[probs[i] for i in range(len(probs))],
                                marker=dict(colors=[CLASS_COLORS.get(i, "#0066cc") for i in range(len(probs))])
                            )])
                            
                            st.plotly_chart(fig_pie, use_container_width=True)
        
        else:
            st.info("👈 Select an image from the left panel")

# ============================================================================
# PAGE: BATCH TEST
# ============================================================================
elif page == "📁 Batch Test":
    st.title("📁 Batch Testing")
    st.markdown("Test multiple images at once")
    st.markdown("---")
    
    # Load model once
    with st.spinner("Loading model..."):
        model, device = load_model(config)
    
    if model is None:
        st.error("Failed to load model")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Batch Settings")
            num_samples = st.slider("Number of random samples:", 1, 50, 10)
            
            selected_class = st.selectbox("Filter by class:", [-1, 0, 1, 2, 3, 4], 
                                         format_func=lambda x: "All Classes" if x == -1 else CLASS_NAMES[x],
                                         key="batch_class_filter")
        
        with col2:
            if st.button("▶️ Run Batch Test", key="run_batch_test"):
                try:
                    csv_path = config['data']['aptos_train_csv']
                    img_dir = config['data']['aptos_train_images']
                    df = pd.read_csv(csv_path)
                    
                    if selected_class >= 0:
                        df = df[df['diagnosis'] == selected_class]
                    
                    if len(df) < num_samples:
                        st.warning(f"Only {len(df)} images available in selected class")
                        num_samples = len(df)
                    
                    samples = df.sample(min(num_samples, len(df)), random_state=42)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results_list = []
                    correct_count = 0
                    
                    for idx, (_, row) in enumerate(samples.iterrows()):
                        status_text.text(f"Processing image {idx+1}/{num_samples}")
                        
                        image_id = row['id_code']
                        true_label = int(row['diagnosis'])
                        image_path = Path(img_dir) / f"{image_id}.png"
                        
                        if image_path.exists():
                            image_tensor = preprocess_image(image_path, config)
                            
                            if image_tensor is not None:
                                pred_class, confidence, _ = run_inference(image_tensor, model, device, config)
                                
                                if pred_class is not None:
                                    is_correct = pred_class == true_label
                                    if is_correct:
                                        correct_count += 1
                                    
                                    results_list.append({
                                        'Image ID': image_id,
                                        'True Grade': true_label,
                                        'Predicted Grade': pred_class,
                                        'Confidence': f"{confidence*100:.2f}%",
                                        'Correct': '✓' if is_correct else '✗'
                                    })
                        
                        progress_bar.progress((idx + 1) / num_samples)
                    
                    status_text.text("Batch test completed!")
                    
                    st.markdown("---")
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Images", len(results_list))
                    
                    with col2:
                        accuracy = (correct_count / len(results_list) * 100) if len(results_list) > 0 else 0
                        st.metric("Accuracy", f"{accuracy:.2f}%")
                    
                    with col3:
                        st.metric("Errors", len(results_list) - correct_count)
                    
                    st.markdown("---")
                    
                    # Results table
                    st.subheader("📋 Detailed Results")
                    results_df = pd.DataFrame(results_list)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv_results = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results (CSV)",
                        csv_results,
                        "batch_test_results.csv",
                        "text/csv"
                    )
                
                except Exception as e:
                    st.error(f"Error during batch test: {e}")
                    import traceback
                    st.error(traceback.format_exc())

# ============================================================================
# PAGE: MODEL ANALYSIS
# ============================================================================
elif page == "📈 Model Analysis":
    st.title("📈 Model Analysis")
    st.markdown("Analyze model architecture and performance")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏗️ Model Architecture")
        model_info = {
            "Architecture": config['model']['architecture'],
            "Input Size": f"{config['data']['image_size']}×{config['data']['image_size']}×3",
            "Output Classes": config['model']['num_classes'],
            "Loss Function": config['model']['loss_type'],
            "Dropout": config['model']['dropout'],
            "Pretrained": config['model']['pretrained']
        }
        st.json(model_info)
    
    with col2:
        st.subheader("📊 Training Configuration")
        train_info = {
            "Batch Size": config['training']['batch_size'],
            "Learning Rate": config['training']['learning_rate'],
            "Epochs": config['training']['num_epochs'],
            "Optimizer": "AdamW",
            "Scheduler": "Cosine Annealing",
            "Class Weights": "Enabled"
        }
        st.json(train_info)
    
    st.markdown("---")
    
    # Model parameters
    st.subheader("📈 Model Parameters")
    
    with st.spinner("Calculating model statistics..."):
        model, _ = load_model(config)
        
        if model is not None:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            non_trainable_params = total_params - trainable_params
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Parameters", f"{total_params:,}")
            
            with col2:
                st.metric("Trainable", f"{trainable_params:,}")
            
            with col3:
                st.metric("Non-trainable", f"{non_trainable_params:,}")
            
            with col4:
                param_size_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
                st.metric("Model Size", f"{param_size_mb:.2f}MB")
            
            # Parameter distribution
            st.markdown("---")
            st.subheader("🔍 Layer-wise Analysis")
            
            layer_info = []
            for name, param in model.named_parameters():
                layer_info.append({
                    'Layer': name,
                    'Shape': str(list(param.shape)),
                    'Parameters': param.numel(),
                    'Trainable': param.requires_grad
                })
            
            layers_df = pd.DataFrame(layer_info).head(20)
            st.dataframe(layers_df, use_container_width=True)

# ============================================================================
# PAGE: SETTINGS
# ============================================================================
elif page == "⚙️ Settings":
    st.title("⚙️ Settings & Configuration")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🗂️ Data Paths")
        st.text_input("Train CSV:", config['data']['aptos_train_csv'], disabled=True)
        st.text_input("Train Images:", config['data']['aptos_train_images'], disabled=True)
        st.text_input("Validation CSV:", config['data']['aptos_valid_csv'], disabled=True)
        st.text_input("Validation Images:", config['data']['aptos_valid_images'], disabled=True)
    
    with col2:
        st.subheader("📊 Data Configuration")
        st.metric("Image Size", f"{config['data']['image_size']}×{config['data']['image_size']}")
        st.metric("Crop Margin", f"{config['data']['crop_margin']}")
        st.metric("CLAHE Enabled", "Yes" if config['data']['use_clahe'] else "No")
        st.metric("Clahe Clip Limit", config['data']['clahe_clip_limit'])
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Model Configuration")
        st.json({
            "Architecture": config['model']['architecture'],
            "Num Classes": config['model']['num_classes'],
            "Pretrained": config['model']['pretrained'],
            "Dropout": config['model']['dropout'],
            "Loss Type": config['model']['loss_type'],
            "Focal Alpha": config['model'].get('focal_alpha', 'N/A'),
            "Focal Gamma": config['model'].get('focal_gamma', 'N/A')
        })
    
    with col2:
        st.subheader("🎓 Training Configuration")
        st.json({
            "Batch Size": config['training']['batch_size'],
            "Num Epochs": config['training']['num_epochs'],
            "Learning Rate": config['training']['learning_rate'],
            "Warmup Epochs": config['training'].get('warmup_epochs', 0),
            "Weight Decay": config['training'].get('weight_decay', 'N/A'),
            "Gradient Clipping": config['training'].get('gradient_clipping', 'N/A')
        })
    
    st.markdown("---")
    
    st.subheader("📄 Full Configuration")
    st.json(config)
    
    # System info
    st.markdown("---")
    st.subheader("💻 System Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**PyTorch Version:**")
        st.write(torch.__version__)
    
    with col2:
        st.write("**CUDA Available:**")
        st.write("Yes" if torch.cuda.is_available() else "No")
    
    with col3:
        st.write("**Device:**")
        st.write(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🏥 <b>MedVision</b> - Diabetic Retinopathy Detection System</p>
    <p>Testing Interface v1.0 | Powered by Streamlit</p>
    <p><small>For research and educational purposes only</small></p>
</div>
""", unsafe_allow_html=True)
