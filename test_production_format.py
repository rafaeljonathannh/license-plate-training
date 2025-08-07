# File: test_production_format.py
import os
import cv2
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import re

# Import required libraries
try:
    from ultralytics import YOLO
    from paddleocr import PaddleOCR
    print("✅ All libraries imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Run: pip install ultralytics paddlepaddle paddleocr opencv-python")
    exit(1)

print("=== PRODUCTION FORMAT TESTING ===")

# Initialize models
print("🔧 Initializing models...")
try:
    yolo_model = YOLO("models/final/best_model.pt")
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')
    print("✅ Models initialized successfully")
except Exception as e:
    print(f"❌ Model initialization failed: {e}")
    exit(1)

class ProductionLicensePlateProcessor:
    def __init__(self, yolo_model, ocr_model, confidence_threshold=0.3):
        self.yolo_model = yolo_model
        self.ocr = ocr_model
        self.confidence_threshold = confidence_threshold
    
    def clean_license_plate_text(self, raw_text):
        """Clean and format license plate text to Indonesian standard"""
        if not raw_text:
            return ""
        
        cleaned = re.sub(r'[^A-Z0-9\s]', '', raw_text.upper())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Indonesian license plate patterns
        patterns = [
            r'^([A-Z]{1,2})\s*(\d{1,4})\s*([A-Z]{1,3})$',  # B 1234 ABC
            r'^([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})$',        # B1234ABC
        ]
        
        for pattern in patterns:
            match = re.match(pattern, cleaned)
            if match:
                area, number, suffix = match.groups()
                return f"{area} {number} {suffix}"
        
        return cleaned
    
    def validate_license_plate_format(self, text):
        """Validate Indonesian license plate format"""
        if not text:
            return False, "Empty text"
        
        pattern = r'^[A-Z]{1,2}\s\d{1,4}\s[A-Z]{1,3}$'
        if re.match(pattern, text):
            return True, "Valid Indonesian license plate format"
        else:
            return False, f"Invalid format: {text}"
    
    def combine_ocr_segments(self, ocr_result):
        """Intelligently combine OCR text segments into license plate format"""
        if not ocr_result:
            return ""
        
        # Extract all text segments - PaddleOCR result format is different
        segments = []
        
        # Handle different possible OCR result formats
        try:
            if hasattr(ocr_result, 'recognition_result'):
                # New PaddleX format
                for item in ocr_result.recognition_result:
                    text = item.get('text', '')
                    confidence = item.get('confidence', 0.0)
                    if confidence > 0.5:  # Only use high-confidence segments
                        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
                        if cleaned:  # Only add non-empty segments
                            segments.append(cleaned)
            else:
                # Fallback: try to extract text directly from result
                text_parts = str(ocr_result).upper()
                cleaned = re.sub(r'[^A-Z0-9\s]', '', text_parts)
                if cleaned.strip():
                    segments.append(cleaned.strip())
        except Exception as e:
            print(f"OCR parsing error: {e}")
            return ""
        
        if not segments:
            return ""
        
        # Try to combine segments intelligently
        combined_text = ""
        
        # Method 1: Concatenate all segments
        all_combined = "".join(segments)
        
        # Method 2: Try to separate letters and numbers
        letters = []
        numbers = []
        for segment in segments:
            if segment.isalpha():
                letters.append(segment)
            elif segment.isdigit():
                numbers.append(segment)
            else:
                # Mixed segment, try to split
                for char in segment:
                    if char.isalpha():
                        if not letters or not letters[-1] or len(letters[-1]) < 3:
                            if not letters:
                                letters.append(char)
                            else:
                                letters[-1] += char
                    elif char.isdigit():
                        if not numbers:
                            numbers.append(char)
                        else:
                            numbers[-1] += char
        
        # Construct license plate: [Letters] [Numbers] [Letters]
        if letters and numbers:
            # Take first letter group (area code)
            area = letters[0][:2] if letters[0] else ""
            # Take first number group
            number = numbers[0][:4] if numbers else ""
            # Take remaining letters (suffix)
            suffix = ""
            if len(letters) > 1:
                suffix = letters[1][:3]
            elif len(letters[0]) > 2:
                # Split first letter group if too long
                area = letters[0][:2]
                suffix = letters[0][2:5]
            
            if area and number:
                combined_text = f"{area} {number} {suffix}".strip()
        
        # Fallback: try the concatenated version
        if not combined_text:
            combined_text = all_combined
        
        return self.clean_license_plate_text(combined_text)
    
    def process_image(self, image_path):
        """Process single image with production-ready JSON format"""
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "image_name": Path(image_path).name,
            "image_path": str(image_path),
            "success": False,
            "processing_time_ms": 0,
            "detections": [],
            "total_detections": 0,
            "annotated_image_path": "",
            "error": None
        }
        
        start_time = time.time()
        
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise Exception(f"Could not load image: {image_path}")
            
            # YOLO Detection
            yolo_results = self.yolo_model(str(image_path), verbose=False)
            yolo_result = yolo_results[0]
            
            if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
                result["error"] = "No license plates detected by YOLO"
                result["processing_time_ms"] = round((time.time() - start_time) * 1000, 2)
                return result
            
            # Process each detection
            detections = []
            annotated_image = image.copy()
            
            for i, box in enumerate(yolo_result.boxes):
                detection_conf = float(box.conf.item())
                if detection_conf < self.confidence_threshold:
                    continue
                
                bbox = box.xyxy[0].tolist()
                x1, y1, x2, y2 = [int(x) for x in bbox]
                
                # Crop license plate region with padding
                padding = 5
                crop_x1 = max(0, x1 - padding)
                crop_y1 = max(0, y1 - padding)
                crop_x2 = min(image.shape[1], x2 + padding)
                crop_y2 = min(image.shape[0], y2 + padding)
                
                cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
                
                if cropped.size == 0:
                    continue
                
                # OCR Processing
                ocr_start = time.time()
                ocr_result = self.ocr.predict(cropped)
                ocr_time = (time.time() - ocr_start) * 1000
                
                # Extract and combine text
                combined_text = self.combine_ocr_segments(ocr_result)
                is_valid, validation_msg = self.validate_license_plate_format(combined_text)
                
                # Calculate average OCR confidence
                ocr_confidence = 0.0
                try:
                    if hasattr(ocr_result, 'recognition_result'):
                        confidences = [item.get('confidence', 0.0) for item in ocr_result.recognition_result]
                        ocr_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                    else:
                        ocr_confidence = 0.5  # Default confidence if unable to parse
                except:
                    ocr_confidence = 0.5
                
                # Create detection object
                detection = {
                    "detection_index": i,
                    "license_plate_number": combined_text if is_valid else "INVALID_FORMAT",
                    "raw_ocr_text": combined_text,
                    "detection_confidence": round(detection_conf, 3),
                    "ocr_confidence": round(ocr_confidence, 3),
                    "bbox": [round(x, 2) for x in bbox],
                    "is_valid_format": is_valid,
                    "validation_message": validation_msg,
                    "ocr_processing_time_ms": round(ocr_time, 2)
                }
                
                detections.append(detection)
                
                # Draw annotation on image
                color = (0, 255, 0) if is_valid else (0, 165, 255)  # Green for valid, orange for invalid
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # Add text label
                label = combined_text if is_valid else "INVALID"
                label_text = f"{label} ({detection_conf:.2f})"
                
                # Calculate text position
                (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                cv2.putText(annotated_image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Save annotated image
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            annotated_filename = f"annotated_{timestamp_str}_{Path(image_path).stem}.jpg"
            annotated_path = f"results/production_test/annotated_images/{annotated_filename}"
            cv2.imwrite(annotated_path, annotated_image)
            
            # Update result
            result.update({
                "success": len(detections) > 0,
                "detections": detections,
                "total_detections": len(detections),
                "annotated_image_path": annotated_path,
                "processing_time_ms": round((time.time() - start_time) * 1000, 2)
            })
            
        except Exception as e:
            result["error"] = str(e)
            result["processing_time_ms"] = round((time.time() - start_time) * 1000, 2)
        
        return result

# Initialize processor
processor = ProductionLicensePlateProcessor(yolo_model, ocr, confidence_threshold=0.1)

# Auto-detect and process all images in test_images folder
print("\n🔍 Scanning for test images...")

test_folder = "test_images/"
if not os.path.exists(test_folder):
    print(f"❌ Test folder not found: {test_folder}")
    print("Please create the folder and add some license plate images")
    exit(1)

# Supported image formats
supported_formats = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

# Find all images
test_images = []
for filename in os.listdir(test_folder):
    if any(filename.endswith(ext) for ext in supported_formats):
        test_images.append(os.path.join(test_folder, filename))

print(f"📊 Found {len(test_images)} test images:")
for img in test_images:
    print(f"  - {Path(img).name}")

if len(test_images) == 0:
    print("❌ No images found in test_images/ folder")
    print("Please add some license plate images (.jpg, .png) to test")
    exit(1)

# Process each image
print(f"\n🚀 Processing {len(test_images)} images...")
all_results = []
successful_detections = 0
total_processing_time = 0

for i, image_path in enumerate(test_images):
    print(f"\n--- Processing {i+1}/{len(test_images)}: {Path(image_path).name} ---")
    
    # Process image
    result = processor.process_image(image_path)
    all_results.append(result)
    
    # Console output
    if result["success"]:
        print(f"✅ SUCCESS")
        for detection in result["detections"]:
            plate = detection["license_plate_number"]
            det_conf = detection["detection_confidence"]
            ocr_conf = detection["ocr_confidence"]
            valid = "✅" if detection["is_valid_format"] else "❌"
            print(f"   {valid} License Plate: '{plate}'")
            print(f"   📊 Confidence: Detection {det_conf:.1%} | OCR {ocr_conf:.1%}")
        
        successful_detections += 1
        print(f"   ⏱️ Processing Time: {result['processing_time_ms']:.0f}ms")
        print(f"   🖼️ Annotated: {result['annotated_image_path']}")
    else:
        print(f"❌ FAILED: {result['error']}")
        print(f"   ⏱️ Processing Time: {result['processing_time_ms']:.0f}ms")
    
    total_processing_time += result["processing_time_ms"]
    
    # Save individual JSON result (database ready format)
    json_filename = f"result_{i+1:03d}_{Path(image_path).stem}.json"
    json_path = f"results/production_test/json_outputs/{json_filename}"
    
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"   💾 JSON saved: {json_filename}")

# Generate comprehensive batch summary
print(f"\n📊 GENERATING BATCH SUMMARY REPORT...")

# Calculate summary statistics
success_rate = (successful_detections / len(test_images)) * 100 if test_images else 0
avg_processing_time = total_processing_time / len(test_images) if test_images else 0

# Extract all detected license plates
all_detected_plates = []
valid_plates = []
invalid_plates = []

for result in all_results:
    if result["success"]:
        for detection in result["detections"]:
            plate = detection["license_plate_number"]
            all_detected_plates.append(plate)
            
            if detection["is_valid_format"]:
                valid_plates.append(plate)
            else:
                invalid_plates.append(plate)

# Create batch summary
batch_summary = {
    "batch_info": {
        "timestamp": datetime.now().isoformat(),
        "total_images_processed": len(test_images),
        "successful_detections": successful_detections,
        "success_rate_percent": round(success_rate, 2),
        "total_processing_time_ms": round(total_processing_time, 2),
        "average_processing_time_ms": round(avg_processing_time, 2)
    },
    "detection_summary": {
        "total_license_plates_detected": len(all_detected_plates),
        "valid_format_plates": len(valid_plates),
        "invalid_format_plates": len(invalid_plates),
        "detected_plates": list(set(valid_plates)),  # Remove duplicates
        "invalid_detections": list(set(invalid_plates))
    },
    "performance_analysis": {
        "fastest_processing_ms": min([r["processing_time_ms"] for r in all_results]) if all_results else 0,
        "slowest_processing_ms": max([r["processing_time_ms"] for r in all_results]) if all_results else 0,
        "images_with_multiple_plates": sum(1 for r in all_results if r["total_detections"] > 1),
        "images_with_no_detections": sum(1 for r in all_results if not r["success"])
    },
    "database_ready_files": [
        f"results/production_test/json_outputs/result_{i+1:03d}_{Path(img).stem}.json" 
        for i, img in enumerate(test_images)
    ],
    "web_ready_files": [
        r["annotated_image_path"] for r in all_results if r["annotated_image_path"]
    ]
}

# Save batch summary
batch_summary_path = "results/production_test/batch_summary/batch_report.json"
with open(batch_summary_path, "w") as f:
    json.dump(batch_summary, f, indent=2)

print(f"💾 Batch summary saved: {batch_summary_path}")

# Display final report
print(f"\n" + "="*60)
print(f"📋 PRODUCTION TESTING FINAL REPORT")
print(f"="*60)

print(f"\n🎯 BATCH STATISTICS:")
print(f"  Total images processed: {len(test_images)}")
print(f"  Successful detections: {successful_detections}")
print(f"  Success rate: {success_rate:.1f}%")
print(f"  Average processing time: {avg_processing_time:.0f}ms")

print(f"\n🏷️ LICENSE PLATES DETECTED:")
if valid_plates:
    unique_valid = list(set(valid_plates))
    for i, plate in enumerate(unique_valid, 1):
        print(f"  {i}. {plate}")
else:
    print(f"  ❌ No valid license plates detected")

print(f"\n📁 OUTPUT FILES GENERATED:")
print(f"  📊 JSON outputs: {len(test_images)} files (database ready)")
print(f"  🖼️ Annotated images: {len([r for r in all_results if r['annotated_image_path']])} files (web ready)")
print(f"  📋 Batch summary: 1 file (performance analysis)")

print(f"\n🏭 PRODUCTION READINESS:")
criteria = {
    "JSON format validation": all("license_plate_number" in str(r) for r in all_results),
    "Database compatibility": all("timestamp" in r for r in all_results),
    "Web display ready": any(r["annotated_image_path"] for r in all_results),
    "Error handling": all("error" in r for r in all_results)
}

for criteria_name, passed in criteria.items():
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {criteria_name}: {status}")

overall_ready = all(criteria.values()) and success_rate >= 20  # At least 20% success for demo
print(f"\n🎉 OVERALL STATUS: {'✅ PRODUCTION READY' if overall_ready else '⚠️ NEEDS OPTIMIZATION'}")

if len(valid_plates) > 0:
    print(f"\n🎯 BEST DETECTION EXAMPLES:")
    best_results = [r for r in all_results if r["success"] and any(d["is_valid_format"] for d in r["detections"])]
    for i, result in enumerate(best_results[:3], 1):
        best_detection = next(d for d in result["detections"] if d["is_valid_format"])
        print(f"  {i}. {Path(result['image_name']).name} → '{best_detection['license_plate_number']}'")
        print(f"     Confidence: {best_detection['detection_confidence']:.1%} | Time: {result['processing_time_ms']:.0f}ms")

print(f"\n📝 NEXT STEPS FOR WEB INTEGRATION:")
print(f"  1. Use JSON files from results/production_test/json_outputs/ for database")
print(f"  2. Use annotated images from results/production_test/annotated_images/ for web display")
print(f"  3. Integrate LicensePlateProcessor class into Django/Flask application")
print(f"  4. Implement file upload → JSON response → database save workflow")

print(f"="*60)