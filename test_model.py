#!/usr/bin/env python3
import sys
from pathlib import Path
import torch
from ultralytics import YOLO
import time
import numpy as np

print("=== MANUAL MODEL EVALUATION ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Load model
model_path = "models/final/best_model.pt"
print(f"\n📦 Loading model: {model_path}")

try:
    model = YOLO(model_path)
    print("✅ Model loaded successfully")
    
    # Model info
    print(f"Model device: {model.device}")
    print(f"Model classes: {model.names}")
    
    # Check model size
    model_size = Path(model_path).stat().st_size / (1024 * 1024)
    print(f"Model size: {model_size:.1f} MB")
    
    # Speed test
    print("\n⚡ Speed Test (20 iterations):")
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(3):
        _ = model(test_image, verbose=False)
    
    # Measure speed
    times = []
    for i in range(20):
        start = time.time()
        results = model(test_image, verbose=False)
        end = time.time()
        times.append((end - start) * 1000)
    
    avg_time = np.mean(times)
    print(f"Average inference time: {avg_time:.1f} ms")
    print(f"Speed target (<100ms): {'✅ PASS' if avg_time < 100 else '❌ FAIL'}")
    print(f"Size target (<50MB): {'✅ PASS' if model_size < 50 else '❌ FAIL'}")
    
    # Dataset evaluation if available
    data_yaml = "dataset/plat-kendaraan/data.yaml"
    if Path(data_yaml).exists():
        print(f"\n📊 Running dataset evaluation...")
        try:
            val_results = model.val(data=data_yaml, verbose=False)
            if hasattr(val_results, 'box'):
                box_results = val_results.box
                map50 = float(box_results.map50) if hasattr(box_results, 'map50') else 0.0
                map50_95 = float(box_results.map) if hasattr(box_results, 'map') else 0.0
                precision = float(box_results.mp) if hasattr(box_results, 'mp') else 0.0
                recall = float(box_results.mr) if hasattr(box_results, 'mr') else 0.0
                
                print(f"mAP@0.5: {map50:.3f}")
                print(f"mAP@0.5:0.95: {map50_95:.3f}")
                print(f"Precision: {precision:.3f}")
                print(f"Recall: {recall:.3f}")
                print(f"Performance target (>0.85): {'✅ PASS' if map50 > 0.85 else '❌ FAIL'}")
                
                # Summary
                print(f"\n🎯 FINAL SUMMARY:")
                print(f"  Model Size: {model_size:.1f} MB ({'✅' if model_size < 50 else '❌'})")
                print(f"  Speed: {avg_time:.1f} ms ({'✅' if avg_time < 100 else '❌'})")
                print(f"  Accuracy: {map50:.3f} ({'✅' if map50 > 0.85 else '❌'})")
                
                all_pass = model_size < 50 and avg_time < 100 and map50 > 0.85
                print(f"  Overall: {'🎉 PRODUCTION READY!' if all_pass else '⚠️ NEEDS IMPROVEMENT'}")
            else:
                print("❌ Could not extract metrics from validation")
        except Exception as e:
            print(f"❌ Dataset evaluation failed: {e}")
    else:
        print(f"⚠️ Dataset not found: {data_yaml}")
    
except Exception as e:
    print(f"❌ Model loading failed: {e}")

print("\n=== TEST COMPLETED ===")
