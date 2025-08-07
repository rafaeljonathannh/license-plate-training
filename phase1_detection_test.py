# File: phase1_detection_test.py
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
import random
import os

print("=== PHASE 1: REAL IMAGE DETECTION TESTING ===")

# Create results directories
os.makedirs("results/phase1_testing/sample_images", exist_ok=True)
os.makedirs("results/phase1_testing/annotated_results", exist_ok=True)
os.makedirs("results/phase1_testing/cropped_plates", exist_ok=True)

# Load model
model_path = "models/final/best_model.pt"
print(f"Loading model: {model_path}")
model = YOLO(model_path)
print(f"✅ Model loaded - Device: {model.device}")

# Find test images - using available license plate images from production folder
test_folders = [
    "../license-plate/media/uploaded_images/"
]

sample_images = []
for folder in test_folders:
    if Path(folder).exists():
        # Look for license plate images
        images = list(Path(folder).glob("*.jpg")) + list(Path(folder).glob("*.png")) + list(Path(folder).glob("*.jpeg"))
        if images:
            # Select all available images, max 10 total
            selected = random.sample(images, min(10, len(images)))
            sample_images.extend(selected)
            print(f"✅ Found {len(images)} images in {folder}, selected {len(selected)}")

# If no production images, try other locations
if not sample_images:
    print("⚠️ No images found in production folder, searching for dataset...")
    dataset_folders = [
        "../dataset/plat-kendaraan/test/images/",
        "../dataset/plat-kendaraan/valid/images/", 
        "../dataset/plat-kendaraan/train/images/"
    ]
    
    for folder in dataset_folders:
        if Path(folder).exists():
            images = list(Path(folder).glob("*.jpg")) + list(Path(folder).glob("*.png"))
            if images:
                selected = random.sample(images, min(3, len(images)))
                sample_images.extend(selected)
                print(f"✅ Found {len(images)} images in {folder}, selected {len(selected)}")

# Limit to 10 images max
sample_images = sample_images[:10]
print(f"\n🎯 Testing with {len(sample_images)} sample images")

if not sample_images:
    print("❌ No test images found!")
    exit(1)

# Detection testing loop
detection_results = []
processing_times = []

print(f"\n🔍 Running detection tests...")

for i, img_path in enumerate(sample_images):
    print(f"\nProcessing {i+1}/{len(sample_images)}: {img_path.name}")
    
    # Copy to sample_images folder
    import shutil
    sample_copy = f"results/phase1_testing/sample_images/{i+1:02d}_{img_path.name}"
    shutil.copy2(str(img_path), sample_copy)
    
    # Load image
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"❌ Failed to load image: {img_path}")
        continue
    
    # Run detection
    start_time = time.time()
    results = model(str(img_path), verbose=False)
    end_time = time.time()
    
    processing_time = (end_time - start_time) * 1000  # ms
    processing_times.append(processing_time)
    
    result = results[0]
    
    # Analyze detection results
    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        for j, box in enumerate(result.boxes):
            bbox = box.xyxy[0].tolist()
            conf = float(box.conf.item())
            
            detection = {
                "bbox": bbox,  # [x1, y1, x2, y2]
                "confidence": conf,
                "detection_id": j
            }
            detections.append(detection)
            
            print(f"  Detection {j}: conf={conf:.3f}, bbox={[int(x) for x in bbox]}")
    else:
        print(f"  ❌ No detections found")
    
    # Store results
    image_result = {
        "image_path": str(img_path),
        "image_name": img_path.name,
        "sample_copy": sample_copy,
        "detections": detections,
        "num_detections": len(detections),
        "processing_time_ms": processing_time,
        "image_shape": image.shape[:2]  # [height, width]
    }
    detection_results.append(image_result)

# Create annotated images with bounding boxes
print(f"\n🎨 Creating visualizations...")

for result in detection_results:
    if result["num_detections"] == 0:
        continue
        
    # Load original image
    image = cv2.imread(result["sample_copy"])
    annotated = image.copy()
    
    # Draw bounding boxes
    for detection in result["detections"]:
        bbox = detection["bbox"]
        conf = detection["confidence"]
        
        # Only show detections with confidence > 0.3
        if conf < 0.3:
            continue
            
        x1, y1, x2, y2 = [int(x) for x in bbox]
        
        # Draw rectangle
        color = (0, 255, 0) if conf > 0.7 else (0, 255, 255)  # Green if high conf, yellow if medium
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Add confidence label
        label = f"Plate {conf:.2f}"
        cv2.putText(annotated, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save annotated image
    output_path = f"results/phase1_testing/annotated_results/annotated_{Path(result['image_name']).stem}.jpg"
    cv2.imwrite(output_path, annotated)
    result["annotated_path"] = output_path
    print(f"  ✅ Saved: {output_path}")

# Extract license plate regions for OCR preparation
print(f"\n✂️ Extracting cropped regions...")

cropped_count = 0
for result in detection_results:
    if result["num_detections"] == 0:
        continue
        
    image = cv2.imread(result["sample_copy"])
    
    for i, detection in enumerate(result["detections"]):
        conf = detection["confidence"]
        
        # Only crop high-confidence detections
        if conf < 0.5:
            continue
            
        bbox = detection["bbox"]
        x1, y1, x2, y2 = [int(x) for x in bbox]
        
        # Add padding
        padding = 5
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)
        
        # Crop region
        cropped = image[y1:y2, x1:x2]
        
        if cropped.size > 0:
            crop_filename = f"crop_{cropped_count:02d}_{Path(result['image_name']).stem}_det{i}.jpg"
            crop_path = f"results/phase1_testing/cropped_plates/{crop_filename}"
            cv2.imwrite(crop_path, cropped)
            
            detection["cropped_path"] = crop_path
            detection["crop_size"] = cropped.shape[:2]
            cropped_count += 1
            
            print(f"  ✅ Cropped: {crop_filename} (size: {cropped.shape[1]}x{cropped.shape[0]})")

print(f"\n📊 Total cropped regions: {cropped_count}")

# Calculate performance metrics
print(f"\n📈 Calculating performance metrics...")

total_images = len(detection_results)
images_with_detections = sum(1 for r in detection_results if r["num_detections"] > 0)
total_detections = sum(r["num_detections"] for r in detection_results)
high_conf_detections = sum(len([d for d in r["detections"] if d["confidence"] > 0.7]) for r in detection_results)

detection_rate = (images_with_detections / total_images) * 100 if total_images > 0 else 0
avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0

# Confidence score analysis
all_confidences = [d["confidence"] for r in detection_results for d in r["detections"]]
avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
max_confidence = max(all_confidences) if all_confidences else 0
min_confidence = min(all_confidences) if all_confidences else 0

performance_metrics = {
    "timestamp": datetime.now().isoformat(),
    "total_test_images": total_images,
    "images_with_detections": images_with_detections,
    "detection_rate_percent": round(detection_rate, 2),
    "total_detections": total_detections,
    "high_confidence_detections": high_conf_detections,
    "average_processing_time_ms": round(avg_processing_time, 2),
    "confidence_scores": {
        "average": round(avg_confidence, 3),
        "maximum": round(max_confidence, 3),
        "minimum": round(min_confidence, 3),
        "count": len(all_confidences)
    },
    "cropped_regions_count": cropped_count
}

# Save detailed results
detailed_results = {
    "performance_metrics": performance_metrics,
    "detection_results": detection_results
}

with open("results/phase1_testing/detection_report.json", "w") as f:
    json.dump(detailed_results, f, indent=2)

print(f"💾 Detailed results saved to: detection_report.json")

# Final assessment and recommendations
print(f"\n📋 PHASE 1 ASSESSMENT REPORT")
print("="*60)

print(f"🎯 DETECTION PERFORMANCE:")
print(f"  Images tested: {total_images}")
print(f"  Detection rate: {detection_rate:.1f}%")
print(f"  Total detections: {total_detections}")
print(f"  High confidence (>0.7): {high_conf_detections}")
print(f"  Average confidence: {avg_confidence:.3f}")
print(f"  Processing speed: {avg_processing_time:.1f} ms/image")

print(f"\n🔍 QUALITY ANALYSIS:")
success_criteria = {
    "detection_rate": detection_rate >= 80,
    "avg_confidence": avg_confidence >= 0.7,
    "processing_speed": avg_processing_time <= 200,
    "cropped_regions": cropped_count >= 3
}

for criteria, passed in success_criteria.items():
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {criteria}: {status}")

overall_pass = all(success_criteria.values())
print(f"\n🎉 OVERALL ASSESSMENT: {'✅ READY FOR PHASE 2' if overall_pass else '⚠️ NEEDS IMPROVEMENT'}")

# Generate recommendations
recommendations = []
if detection_rate < 80:
    recommendations.append("- Consider lowering confidence threshold")
    recommendations.append("- Test with more diverse images")
if avg_confidence < 0.7:
    recommendations.append("- Model may need more training data")
    recommendations.append("- Consider fine-tuning parameters")
if avg_processing_time > 200:
    recommendations.append("- Consider GPU acceleration")
    recommendations.append("- Optimize image preprocessing")

if recommendations:
    print(f"\n📝 RECOMMENDATIONS:")
    for rec in recommendations:
        print(rec)

# Create summary report
assessment_summary = {
    "phase": "Phase 1 - Detection Testing",
    "status": "READY FOR PHASE 2" if overall_pass else "NEEDS IMPROVEMENT",
    "performance_metrics": performance_metrics,
    "success_criteria": success_criteria,
    "overall_pass": overall_pass,
    "recommendations": recommendations,
    "next_steps": [
        "Proceed to Phase 2: OCR Integration" if overall_pass else "Address performance issues",
        "Implement PaddleOCR pipeline",
        "Create detection → crop → OCR workflow"
    ]
}

with open("results/phase1_testing/phase1_assessment.json", "w") as f:
    json.dump(assessment_summary, f, indent=2)

print(f"\n💾 Assessment saved to: phase1_assessment.json")
print("="*60)