# Remote GPU Training Setup Guide - Indonesian License Plate Detection

This guide explains how to set up and run the Indonesian license plate training pipeline on a remote GPU server (Google Colab, Paperspace, AWS, etc.) where Claude Code is not available.

## 📋 Prerequisites

- Remote GPU server with Jupyter support
- Python 3.8+ environment
- At least 8GB GPU memory (recommended: 16GB+)
- ~5GB disk space for dataset and models

## 🚀 Quick Setup (Copy-Paste Commands)

### Step 1: Clone or Upload Project Files

**Option A: If you have Git access on remote server:**
```bash
# Clone the repository (if publicly available)
git clone <your-repo-url>
cd license-plate-training
```

**Option B: Manual Upload (Recommended):**
Upload these essential files/folders to your remote server:
```
license-plate-training/
├── notebooks/                    # All 5 notebooks (01-05)
├── scripts/
│   └── pipeline.py              # Essential production pipeline
├── .env                         # Your Roboflow API key
├── requirements_training.txt    # Python dependencies
├── CLAUDE.md                   # Project specifications
└── README.md                   # Project overview
```

### Step 2: Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements_training.txt

# Alternative: Install core packages manually if requirements file missing
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
pip install roboflow
pip install albumentations
pip install pandas numpy matplotlib seaborn
pip install pillow opencv-python
pip install python-dotenv
pip install jupyter ipywidgets
```

### Step 3: Verify GPU Setup

```python
# Run this in a notebook cell or Python shell
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "No GPU")
```

## 📝 Essential Files Checklist

Make sure you have these files uploaded to your remote server:

### Required Files:
- [ ] `notebooks/01_setup_and_dataset.ipynb`
- [ ] `notebooks/02_data_exploration.ipynb`
- [ ] `notebooks/03_data_preparation.ipynb`
- [ ] `notebooks/04_model_training.ipynb`
- [ ] `notebooks/05_evaluation_export.ipynb`
- [ ] `scripts/pipeline.py`
- [ ] `.env` (with your `ROBOFLOW_API_KEY=your_key_here`)
- [ ] `CLAUDE.md` (project specifications)

### Optional but Recommended:
- [ ] `requirements_training.txt`
- [ ] `README.md`
- [ ] Any existing dataset files (if already downloaded)

## 🔧 Environment Variables Setup

Create a `.env` file in the project root:
```bash
# Create .env file
echo "ROBOFLOW_API_KEY=your_roboflow_api_key_here" > .env

# Or create manually with your text editor
```

**Get your Roboflow API key:**
1. Go to https://roboflow.com
2. Sign in to your account
3. Go to Settings → API
4. Copy your API key

## 🏃‍♂️ Running the Training Pipeline

### Method 1: Sequential Notebook Execution

Run notebooks in order (recommended for first-time setup):

```bash
# Start Jupyter
jupyter notebook

# Then run notebooks in this order:
# 1. 01_setup_and_dataset.ipynb    - Downloads dataset
# 2. 02_data_exploration.ipynb     - Analyzes dataset
# 3. 03_data_preparation.ipynb     - Prepares and augments data
# 4. 04_model_training.ipynb       - Trains YOLOv8 model (main training)
# 5. 05_evaluation_export.ipynb    - Evaluates and exports model
```

### Method 2: Direct Training (Skip to Training)

If you already have the dataset and want to start training immediately:

```python
# Create this as train_direct.py
import os
from pathlib import Path
from ultralytics import YOLO
import torch

# Check GPU
print(f"GPU available: {torch.cuda.is_available()}")

# Paths
dataset_path = "dataset/data.yaml"  # Adjust if needed
models_dir = Path("models/final")
models_dir.mkdir(parents=True, exist_ok=True)

# Training configuration (from CLAUDE.md specifications)
model = YOLO('yolov8n.pt')  # Download pretrained model

# Train
results = model.train(
    data=dataset_path,
    epochs=100,
    patience=20,
    batch=16,        # Reduce to 8 if memory issues
    imgsz=640,
    optimizer='AdamW',
    lr0=0.001,
    val=True,
    save=True,
    device=0 if torch.cuda.is_available() else 'cpu',
    project='models/experiments',
    name='yolov8n-indonesian-plates'
)

print("Training completed!")
```

## 🐛 Common Issues & Solutions

### Issue 1: Out of Memory Error
```
RuntimeError: CUDA out of memory
```
**Solution:**
```python
# In notebook 04, reduce batch size:
training_config["batch"] = 8  # Or even 4
```

### Issue 2: Dataset Not Found
```
Dataset not found error
```
**Solution:**
```python
# Check if .env file exists and has correct API key
import os
from dotenv import load_dotenv
load_dotenv()
print(f"API Key loaded: {'Yes' if os.getenv('ROBOFLOW_API_KEY') else 'No'}")
```

### Issue 3: Slow Training
**Solutions:**
- Reduce epochs for testing: `epochs=10`
- Use smaller image size: `imgsz=416` 
- Enable mixed precision: `amp=True`

### Issue 4: Package Installation Issues
```bash
# If ultralytics installation fails:
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics --no-deps
pip install pyyaml opencv-python pillow numpy pandas matplotlib seaborn
```

## ⚡ Performance Optimization for Remote Training

### GPU Memory Optimization:
```python
# Add to training config
training_config.update({
    "batch": 8,          # Smaller batch size
    "cache": False,      # Don't cache images in memory
    "amp": True,         # Mixed precision training
    "workers": 2,        # Fewer data loading workers
})
```

### Speed Up Dataset Download:
```python
# In notebook 01, add parallel download
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace("your-workspace").project("your-project")
dataset = project.version(1).download("yolov8", 
                                     parallel_workers=4)  # Faster download
```

## 📊 Monitoring Training Progress

### Option 1: TensorBoard (Built-in)
```bash
# Training automatically creates logs
tensorboard --logdir models/experiments
```

### Option 2: Weights & Biases (Optional)
```python
# Add to training config in notebook 04
import wandb
wandb.login()  # Login with your W&B key

# Then training will automatically log to W&B
```

## 💾 Saving and Downloading Results

### Essential Files to Download After Training:
```
models/final/best_model.pt              # Main trained model
results/metrics/final_evaluation.json   # Performance metrics
results/plots/                          # Training plots
results/integration_guide.txt          # Production deployment guide
```

### Download Commands:
```bash
# Compress results for download
tar -czf training_results.tar.gz models/final/ results/

# Or zip format
zip -r training_results.zip models/final/ results/
```

## 🔄 Transfer Back to Local System

After training completes:

1. **Download the trained model:**
   ```bash
   # From your local machine:
   scp user@remote-server:/path/to/models/final/best_model.pt ./models/final/
   ```

2. **Copy to production:**
   ```bash
   cp models/final/best_model.pt ../license-plate/cached_models/yolov8_indonesian_plates.pt
   ```

## 📈 Expected Training Timeline

- **Dataset Download:** 5-10 minutes
- **Data Preparation:** 2-5 minutes  
- **Model Training:** 2-6 hours (depends on GPU)
- **Evaluation:** 10-20 minutes
- **Total:** 3-7 hours

## 🎯 Success Indicators

Training is successful when you see:
- ✅ mAP@0.5 > 0.85 (target from CLAUDE.md)
- ✅ Model size < 50MB
- ✅ Inference time < 100ms
- ✅ No critical errors in evaluation

## 🆘 Emergency Debugging

If anything goes wrong, check these files:
```bash
# Check logs
ls models/experiments/*/
cat models/experiments/*/results.csv

# Check dataset
ls dataset/raw/plat-kendaraan/
ls dataset/raw/plat-kendaraan/train/images/ | wc -l

# Check GPU status
nvidia-smi
```

## 📞 Support

If you encounter issues:
1. Check the error messages against this guide
2. Verify all required files are uploaded
3. Ensure GPU memory is sufficient
4. Try reducing batch size as first debugging step

The training pipeline is designed to be robust - most issues are environment-related and can be solved by adjusting batch sizes or ensuring proper file uploads.

---

**Generated for Indonesian License Plate Detection Project**  
**Compatible with: Google Colab, Paperspace, AWS, Azure ML, etc.**