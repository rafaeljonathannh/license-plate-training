# Indonesian License Plate Training Tutorial

## Complete Step-by-Step Guide

This tutorial will guide you through training a YOLOv8 model for Indonesian license plate detection from scratch.

---

## 📋 Prerequisites

### System Requirements
- **OS**: Windows, macOS, or Linux
- **Python**: 3.8+ (recommended: 3.10+)  
- **GPU**: NVIDIA GPU with CUDA support (recommended for faster training)
- **RAM**: 8GB+ recommended
- **Storage**: 5GB+ free space

### Required Software
- **Git**: For cloning the repository
- **Python**: Latest version from python.org
- **VS Code**: Recommended IDE with Python extension

---

## 🚀 Step 1: Initial Setup

### 1.1 Clone the Repository
```bash
git clone <your-repo-url>
cd license-plate-training
```

### 1.2 Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux  
python -m venv venv
source venv/bin/activate
```

### 1.3 Install Dependencies
```bash
pip install -r requirements_training.txt
```

**Expected installation time**: 5-10 minutes

### 1.4 Verify Installation
```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## 📊 Step 2: Dataset Setup

### 2.1 Get Roboflow API Key
1. Go to [Roboflow.com](https://roboflow.com)
2. Create free account or login
3. Go to Account Settings → API
4. Copy your API key

### 2.2 Create Environment File
Create `.env` file in project root:
```bash
# Create .env file
echo "ROBOFLOW_API_KEY=your_api_key_here" > .env
```

**⚠️ Important**: Replace `your_api_key_here` with your actual API key

### 2.3 Open Jupyter Environment
```bash
# Start Jupyter
jupyter notebook

# Or if using VS Code
code .
```

---

## 📓 Step 3: Run Notebooks Sequentially

### Notebook 01: Setup and Dataset Download

**File**: `notebooks/01_setup_and_dataset.ipynb`

**What it does**:
- Verifies your Python environment
- Checks GPU/CUDA availability  
- Downloads Indonesian license plate dataset (1,607 images)
- Sets up portable project structure

**Runtime**: 5-15 minutes (depending on internet speed)

**Success indicators**:
- ✅ CUDA available (if you have GPU)
- ✅ Dataset downloaded to `../dataset/plat-kendaraan/`
- ✅ 1,607 images with labels downloaded

**Troubleshooting**:
- If Roboflow login fails: Check your API key in `.env` file
- If CUDA not available: Training will use CPU (slower but works)

---

### Notebook 02: Data Exploration

**File**: `notebooks/02_data_exploration.ipynb`

**What it does**:
- Counts images in train/valid/test splits
- Verifies dataset structure and quality
- Generates basic statistics and readiness report

**Runtime**: 2-5 minutes

**Success indicators**:
- ✅ File structure verified
- ✅ Train: ~1120 images, Valid: ~323 images, Test: ~161 images
- ✅ "READY TO TRAIN" status displayed

**Expected output**:
```
TRAINING READINESS: 4.0/4.0
🚀 READY TO TRAIN! Proceed to notebook 04_model_training.ipynb
```

---

### Notebook 04: Model Training

**File**: `notebooks/04_model_training.ipynb`

**What it does**:
- Loads pre-trained YOLOv8n model
- Configures optimal training parameters
- Trains model on Indonesian license plate data
- Monitors training progress and metrics
- Exports best model for production

**Runtime**: 30-120 minutes (depending on hardware)

**Training Configuration**:
- **Model**: YOLOv8n (optimized for speed/size)
- **Epochs**: 100 (with early stopping)
- **Batch Size**: 16 (GPU) / 8 (low memory) / 4 (CPU)
- **Target**: mAP@0.5 > 0.85

**Success indicators**:
- ✅ Training completes without errors
- ✅ Model achieves target mAP@0.5 > 0.85
- ✅ Best model exported to `models/final/best_model.pt`

**Monitoring Training**:
- Watch loss curves (should decrease over time)
- Monitor mAP@0.5 validation scores
- Training will stop early if no improvement for 15 epochs

**If Target Not Met**:
- Try larger model: YOLOv8s or YOLOv8m
- Increase epochs to 150-200
- Adjust learning rate or batch size

---

### Notebook 05: Evaluation and Export

**File**: `notebooks/05_evaluation_export.ipynb`

**What it does**:
- Loads trained model
- Evaluates performance on test set
- Tests inference on sample images
- Validates output format for production
- Creates integration guide

**Runtime**: 5-10 minutes

**Success indicators**:
- ✅ Model loads successfully
- ✅ Test set evaluation completed
- ✅ Sample predictions look accurate
- ✅ Production-ready model exported

---

## 🎯 Step 4: Verify Training Results

### 4.1 Check Model Performance
After training, you should see:

```
🎯 Final Performance:
    mAP@0.5: 0.892  ← Should be > 0.85
  mAP@0.5:0.95: 0.654
     Precision: 0.901
        Recall: 0.847
```

### 4.2 Verify File Structure
```
models/
├── experiments/
│   └── yolov8n_baseline_YYYYMMDD_HHMMSS/  ← Training results
│       ├── weights/
│       │   ├── best.pt                    ← Best model
│       │   └── last.pt
│       ├── results.csv                    ← Training metrics
│       └── plots/                         ← Training curves
└── final/
    └── best_model.pt                      ← Production model
```

### 4.3 Test Model Manually
```python
from ultralytics import YOLO
model = YOLO('models/final/best_model.pt')
results = model('path/to/test/image.jpg')
```

---

## 🔧 Troubleshooting Guide

### Common Issues

**Issue**: `CUDA out of memory`
**Solution**: Reduce batch size in notebook 04
```python
training_config["batch"] = 8  # or 4 for very limited GPU
```

**Issue**: `FileNotFoundError: data.yaml`
**Solution**: Re-run notebook 01 to download dataset properly

**Issue**: Low mAP@0.5 score (< 0.85)
**Solutions**:
1. Try YOLOv8s model: `YOLO('yolov8s.pt')`
2. Increase epochs: `epochs=150`
3. Reduce learning rate: `lr0=0.0005`

**Issue**: Training very slow
**Check**: 
- GPU utilization: `nvidia-smi` (Windows/Linux)
- Reduce batch size if GPU memory is full
- Close other applications using GPU

**Issue**: Roboflow API errors
**Solutions**:
1. Verify API key in `.env` file
2. Check internet connection
3. Try manual download from Roboflow website

---

## 📈 Expected Timeline

| Step | Task | Time | 
|------|------|------|
| 1 | Setup & Installation | 10-15 min |
| 2 | Dataset Download | 5-15 min |
| 3 | Data Exploration | 2-5 min |
| 4 | Model Training | 30-120 min |
| 5 | Evaluation | 5-10 min |
| **Total** | **Complete Workflow** | **1-3 hours** |

---

## ✅ Success Checklist

### After Notebook 01:
- [ ] Environment verification passed
- [ ] CUDA available (if GPU present)
- [ ] Dataset downloaded (1,607 images)
- [ ] Folder structure created

### After Notebook 02:
- [ ] Dataset structure verified
- [ ] Image/label counts match
- [ ] "READY TO TRAIN" status achieved

### After Notebook 04:
- [ ] Training completed without errors  
- [ ] mAP@0.5 > 0.85 achieved
- [ ] Best model exported
- [ ] Model size < 50MB

### After Notebook 05:
- [ ] Model evaluation completed
- [ ] Production format validated
- [ ] Integration guide generated

---

## 🚀 Next Steps (Production Integration)

After completing this tutorial, your trained model will be ready for production use:

1. **Model Location**: `models/final/best_model.pt`
2. **Copy to Production**: Move to your production repository
3. **Test Integration**: Verify with production pipeline
4. **Deploy**: Use in your license plate recognition system

**Production Output Format**: The model will output JSON in this format:
```json
{
  "success": true,
  "detections": [
    {
      "license_plate_number": "B 1234 ABC",
      "confidence_score": 0.92,
      "bbox": [450, 310, 680, 375],
      "processing_time_ms": 85,
      "detection_index": 0
    }
  ],
  "total_detections": 1,
  "total_processing_time_ms": 95,
  "error": null
}
```

---

## 📞 Support

If you encounter issues:

1. **Check CLAUDE.md**: Detailed technical documentation
2. **Review Error Messages**: Most errors have clear solutions
3. **GPU Issues**: Verify CUDA installation and drivers
4. **Dataset Issues**: Re-run notebook 01 for fresh download

**Happy Training! 🎉**