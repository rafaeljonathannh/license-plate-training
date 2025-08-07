# File: phase2_ocr_integration.py
import os
import cv2
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import re

# Import libraries
try:
    from paddleocr import PaddleOCR
    print("PaddleOCR imported successfully")
except ImportError as e:
    print(f"PaddleOCR import failed: {e}")
    exit(1)

try:
    from ultralytics import YOLO
    print("YOLO imported successfully")
except ImportError as e:
    print(f"YOLO import failed: {e}")
    exit(1)

print("=== PHASE 2: OCR INTEGRATION TESTING ===")

# Create results directories
os.makedirs("results/phase2_ocr/extracted_text", exist_ok=True)
os.makedirs("results/phase2_ocr/processed_results", exist_ok=True)
os.makedirs("results/phase2_ocr/end_to_end_tests", exist_ok=True)

# Initialize PaddleOCR
print(f"\nInitializing PaddleOCR...")
try:
    # Initialize with CPU, English recognition
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')
    print("PaddleOCR initialized successfully (CPU mode)")
except Exception as e:
    print(f"PaddleOCR initialization failed: {e}")
    exit(1)

# Test OCR on existing cropped license plates
print(f"\nTesting OCR on cropped images from Phase 1...")

cropped_folder = "results/phase1_testing/cropped_plates/"
if not Path(cropped_folder).exists():
    print(f"Cropped images folder not found: {cropped_folder}")
    print("Please run Phase 1 testing first!")
    exit(1)

# Get all cropped images
cropped_images = list(Path(cropped_folder).glob("*.jpg"))
print(f"Found {len(cropped_images)} cropped images to test")

if len(cropped_images) == 0:
    print("No cropped images found for OCR testing!")
    exit(1)

ocr_results = []

for i, crop_path in enumerate(cropped_images):
    print(f"\nProcessing crop {i+1}/{len(cropped_images)}: {crop_path.name}")
    
    # Load image
    image = cv2.imread(str(crop_path))
    if image is None:
        print(f"Failed to load: {crop_path}")
        continue
    
    # Run OCR
    start_time = time.time()
    try:
        result = ocr.predict(str(crop_path))
        end_time = time.time()
        
        processing_time = (end_time - start_time) * 1000  # ms
        
        # Extract text from result
        extracted_texts = []
        confidence_scores = []
        
        if result and len(result) > 0:  # Check if OCR found anything
            res = result[0]  # Get first result
            if 'rec_texts' in res and len(res['rec_texts']) > 0:
                for i, text in enumerate(res['rec_texts']):
                    confidence = res['rec_scores'][i] if i < len(res['rec_scores']) else 0.0
                    extracted_texts.append(text)
                    confidence_scores.append(confidence)
                    print(f"  Detected: '{text}' (confidence: {confidence:.3f})")
            else:
                print(f"  No text detected")
        else:
            print(f"  No text detected")
        
        # Store result
        ocr_result = {
            "image_name": crop_path.name,
            "image_path": str(crop_path),
            "processing_time_ms": round(processing_time, 2),
            "texts_detected": extracted_texts,
            "confidences": confidence_scores,
            "num_texts": len(extracted_texts),
            "raw_result": "success" if result and len(result) > 0 else None
        }
        ocr_results.append(ocr_result)
        
    except Exception as e:
        print(f"OCR failed for {crop_path.name}: {e}")
        ocr_result = {
            "image_name": crop_path.name,
            "image_path": str(crop_path),
            "error": str(e),
            "processing_time_ms": 0,
            "texts_detected": [],
            "confidences": [],
            "num_texts": 0
        }
        ocr_results.append(ocr_result)

print(f"\n📊 OCR testing completed on {len(ocr_results)} images")

# Create text cleaning and formatting functions
def clean_license_plate_text(raw_text):
    """Clean and format license plate text to Indonesian standard"""
    if not raw_text:
        return ""
    
    # Remove special characters and extra spaces
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
            # Format to standard: "B 1234 ABC"
            return f"{area} {number} {suffix}"
    
    # If no pattern matches, return cleaned text
    return cleaned

def combine_ocr_texts(text_list):
    """Combine multiple OCR text detections into a single license plate"""
    if not text_list:
        return ""
    
    # Join all detected texts with spaces
    combined = ' '.join(text_list)
    
    # Clean the combined text
    cleaned = clean_license_plate_text(combined)
    
    # If still not valid, try different combinations
    if not validate_license_plate_format(cleaned)[0]:
        # Try to separate letters and numbers more intelligently
        letters = []
        numbers = []
        
        for text in text_list:
            clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
            if clean_text.isalpha():
                letters.append(clean_text)
            elif clean_text.isdigit():
                numbers.append(clean_text)
            else:
                # Mixed - separate letters and numbers
                letter_part = re.findall(r'[A-Z]+', clean_text)
                number_part = re.findall(r'\d+', clean_text)
                letters.extend(letter_part)
                numbers.extend(number_part)
        
        # Try to construct B 1234 ABC format
        if letters and numbers:
            area_code = letters[0] if letters else ""
            main_number = numbers[0] if numbers else ""
            suffix = letters[1] if len(letters) > 1 else (letters[0] if len(letters) == 1 and not area_code else "")
            
            # Construct potential license plate
            if area_code and main_number:
                if suffix:
                    potential = f"{area_code} {main_number} {suffix}"
                else:
                    potential = f"{area_code} {main_number}"
                
                return clean_license_plate_text(potential)
    
    return cleaned

def validate_license_plate_format(text):
    """Validate if text follows Indonesian license plate format"""
    if not text:
        return False, "Empty text"
    
    # Indonesian license plate pattern: 1-2 letters, 1-4 numbers, 1-3 letters
    pattern = r'^[A-Z]{1,2}\s\d{1,4}\s[A-Z]{1,3}$'
    
    if re.match(pattern, text):
        return True, "Valid Indonesian license plate format"
    else:
        return False, f"Invalid format: {text}"

# Process OCR results with cleaning
print(f"\nPost-processing OCR results...")

processed_results = []
for result in ocr_results:
    if result.get("error"):
        processed_results.append(result)
        continue
    
    processed_texts = []
    for raw_text in result["texts_detected"]:
        cleaned = clean_license_plate_text(raw_text)
        is_valid, validation_msg = validate_license_plate_format(cleaned)
        
        processed_text = {
            "raw_text": raw_text,
            "cleaned_text": cleaned,
            "is_valid_format": is_valid,
            "validation_message": validation_msg
        }
        processed_texts.append(processed_text)
        
        print(f"  {result['image_name']}: '{raw_text}' -> '{cleaned}' ({'VALID' if is_valid else 'INVALID'})")
    
    result["processed_texts"] = processed_texts
    result["best_text"] = ""
    
    # Try combining all texts first
    combined_text = combine_ocr_texts(result["texts_detected"])
    is_combined_valid, _ = validate_license_plate_format(combined_text)
    
    if is_combined_valid:
        result["best_text"] = combined_text
        print(f"  {result['image_name']}: COMBINED -> '{combined_text}' (VALID)")
    else:
        # Find best valid text (highest confidence + valid format)
        valid_texts = [pt for pt in processed_texts if pt["is_valid_format"]]
        if valid_texts and result["confidences"]:
            # Get index of best confidence that corresponds to valid text
            best_idx = 0
            best_conf = 0
            for i, (pt, conf) in enumerate(zip(processed_texts, result["confidences"])):
                if pt["is_valid_format"] and conf > best_conf:
                    best_conf = conf
                    best_idx = i
                    result["best_text"] = pt["cleaned_text"]
    
    processed_results.append(result)

# Create integrated YOLO + OCR pipeline
print(f"\nCreating integrated YOLO + OCR pipeline...")

# Load YOLO model
yolo_model_path = "models/final/best_model.pt"
try:
    yolo_model = YOLO(yolo_model_path)
    print(f"YOLO model loaded: {yolo_model_path}")
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    exit(1)

def integrated_license_plate_recognition(image_path, confidence_threshold=0.3):
    """Complete pipeline: Detection + OCR"""
    
    result = {
        "image_path": str(image_path),
        "success": False,
        "detections": [],
        "total_processing_time_ms": 0,
        "error": None
    }
    
    start_total = time.time()
    
    try:
        # Step 1: YOLO Detection
        start_detection = time.time()
        yolo_results = yolo_model(str(image_path), verbose=False)
        end_detection = time.time()
        
        detection_time = (end_detection - start_detection) * 1000
        
        yolo_result = yolo_results[0]
        if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
            result["error"] = "No license plates detected"
            result["total_processing_time_ms"] = detection_time
            return result
        
        # Step 2: Process each detection
        image = cv2.imread(str(image_path))
        detections = []
        
        for i, box in enumerate(yolo_result.boxes):
            conf = float(box.conf.item())
            if conf < confidence_threshold:
                continue
                
            bbox = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(x) for x in bbox]
            
            # Crop license plate region
            padding = 5
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)
            
            cropped = image[y1:y2, x1:x2]
            
            if cropped.size == 0:
                continue
            
            # Step 3: OCR on cropped region
            start_ocr = time.time()
            ocr_result = ocr.predict(cropped)
            end_ocr = time.time()
            
            ocr_time = (end_ocr - start_ocr) * 1000
            
            # Process OCR result
            extracted_text = ""
            ocr_confidence = 0
            
            if ocr_result and len(ocr_result) > 0:
                res = ocr_result[0]
                if 'rec_texts' in res and len(res['rec_texts']) > 0:
                    # Try combining all detected texts
                    extracted_text = combine_ocr_texts(res['rec_texts'])
                    ocr_confidence = max(res['rec_scores']) if res['rec_scores'] else 0.0
            
            # Validate format
            is_valid, validation_msg = validate_license_plate_format(extracted_text)
            
            detection = {
                "detection_index": i,
                "bbox": bbox,
                "detection_confidence": conf,
                "license_plate_number": extracted_text if is_valid else "INVALID_FORMAT",
                "ocr_confidence": ocr_confidence,
                "is_valid_format": is_valid,
                "validation_message": validation_msg,
                "detection_time_ms": detection_time / len(yolo_result.boxes),
                "ocr_time_ms": ocr_time
            }
            detections.append(detection)
        
        end_total = time.time()
        total_time = (end_total - start_total) * 1000
        
        result.update({
            "success": True,
            "detections": detections,
            "total_detections": len(detections),
            "total_processing_time_ms": round(total_time, 2)
        })
        
    except Exception as e:
        result["error"] = str(e)
        result["total_processing_time_ms"] = (time.time() - start_total) * 1000
    
    return result

# Test integrated pipeline on original images from Phase 1
print(f"\nTesting integrated pipeline...")

sample_images_folder = "results/phase1_testing/sample_images/"
if Path(sample_images_folder).exists():
    sample_images = list(Path(sample_images_folder).glob("*.jpg")) + list(Path(sample_images_folder).glob("*.jpeg"))
    
    integration_results = []
    
    for i, img_path in enumerate(sample_images[:5]):  # Test first 5 images
        print(f"\nIntegrated test {i+1}/5: {img_path.name}")
        
        result = integrated_license_plate_recognition(img_path, confidence_threshold=0.3)
        integration_results.append(result)
        
        if result["success"]:
            for detection in result["detections"]:
                plate_text = detection["license_plate_number"]
                det_conf = detection["detection_confidence"]
                ocr_conf = detection["ocr_confidence"]
                print(f"  Detected: '{plate_text}' (YOLO: {det_conf:.3f}, OCR: {ocr_conf:.3f})")
        else:
            print(f"  Failed: {result['error']}")
else:
    print(f"Sample images folder not found: {sample_images_folder}")
    integration_results = []

# Generate comprehensive performance report
print(f"\nGenerating performance analysis...")

# OCR Performance Metrics
ocr_success_count = sum(1 for r in processed_results if r.get("best_text", ""))
ocr_success_rate = (ocr_success_count / len(processed_results)) * 100 if processed_results else 0

ocr_times = [r["processing_time_ms"] for r in processed_results if "processing_time_ms" in r and r["processing_time_ms"] > 0]
avg_ocr_time = sum(ocr_times) / len(ocr_times) if ocr_times else 0

# Integration Performance Metrics
integration_success_count = sum(1 for r in integration_results if r["success"])
integration_success_rate = (integration_success_count / len(integration_results)) * 100 if integration_results else 0

total_times = [r["total_processing_time_ms"] for r in integration_results if r.get("total_processing_time_ms", 0) > 0]
avg_total_time = sum(total_times) / len(total_times) if total_times else 0

# Valid license plates detected
valid_plates = []
for result in integration_results:
    if result["success"]:
        for detection in result["detections"]:
            if detection["is_valid_format"]:
                valid_plates.append(detection["license_plate_number"])

performance_report = {
    "timestamp": datetime.now().isoformat(),
    "ocr_testing": {
        "total_cropped_images": len(processed_results),
        "successful_ocr": ocr_success_count,
        "ocr_success_rate_percent": round(ocr_success_rate, 2),
        "average_ocr_time_ms": round(avg_ocr_time, 2),
        "extracted_texts": [r.get("best_text", "") for r in processed_results if r.get("best_text", "")]
    },
    "integration_testing": {
        "total_test_images": len(integration_results),
        "successful_integrations": integration_success_count,
        "integration_success_rate_percent": round(integration_success_rate, 2),
        "average_total_time_ms": round(avg_total_time, 2),
        "valid_license_plates_detected": len(valid_plates),
        "detected_plates": valid_plates
    },
    "performance_targets": {
        "ocr_success_rate": {"target": 70, "actual": ocr_success_rate, "pass": ocr_success_rate >= 70},
        "total_processing_time": {"target": 1000, "actual": avg_total_time, "pass": avg_total_time <= 1000},
        "valid_plates": {"target": 2, "actual": len(valid_plates), "pass": len(valid_plates) >= 2}
    }
}

# Save detailed results
detailed_ocr_results = {
    "performance_report": performance_report,
    "ocr_test_results": processed_results,
    "integration_test_results": integration_results
}

with open("results/phase2_ocr/ocr_integration_report.json", "w") as f:
    json.dump(detailed_ocr_results, f, indent=2)

print(f"Detailed results saved to: ocr_integration_report.json")

# Create production-ready code template
print(f"\nGenerating production code template...")

production_code = '''
# File: license_plate_recognition.py
"""
Production-ready License Plate Recognition Pipeline
Combines YOLOv8 detection + PaddleOCR text extraction
"""

import cv2
import time
import re
from ultralytics import YOLO
from paddleocr import PaddleOCR

class LicensePlateRecognizer:
    def __init__(self, yolo_model_path, confidence_threshold=0.3, use_gpu=False):
        """Initialize the license plate recognition pipeline"""
        self.yolo_model = YOLO(yolo_model_path)
        self.ocr = PaddleOCR(use_textline_orientation=True, lang='en')
        self.confidence_threshold = confidence_threshold
    
    def clean_license_plate_text(self, raw_text):
        """Clean and format license plate text to Indonesian standard"""
        if not raw_text:
            return ""
        
        cleaned = re.sub(r'[^A-Z0-9\\s]', '', raw_text.upper())
        cleaned = re.sub(r'\\s+', ' ', cleaned).strip()
        
        patterns = [
            r'^([A-Z]{1,2})\\s*(\\d{1,4})\\s*([A-Z]{1,3})$',
            r'^([A-Z]{1,2})(\\d{1,4})([A-Z]{1,3})$',
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
            return False
        pattern = r'^[A-Z]{1,2}\\s\\d{1,4}\\s[A-Z]{1,3}$'
        return bool(re.match(pattern, text))
    
    def recognize(self, image_path):
        """
        Complete license plate recognition pipeline
        Returns: {
            "success": bool,
            "detections": [
                {
                    "license_plate_number": str,
                    "confidence_score": float,
                    "bbox": [x1, y1, x2, y2],
                    "processing_time_ms": float
                }
            ],
            "total_processing_time_ms": float,
            "error": str or None
        }
        """
        start_time = time.time()
        
        try:
            # YOLO Detection
            yolo_results = self.yolo_model(image_path, verbose=False)
            yolo_result = yolo_results[0]
            
            if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
                return {
                    "success": False,
                    "detections": [],
                    "total_processing_time_ms": (time.time() - start_time) * 1000,
                    "error": "No license plates detected"
                }
            
            # Process detections
            image = cv2.imread(image_path)
            detections = []
            
            for box in yolo_result.boxes:
                conf = float(box.conf.item())
                if conf < self.confidence_threshold:
                    continue
                
                bbox = box.xyxy[0].tolist()
                x1, y1, x2, y2 = [int(x) for x in bbox]
                
                # Crop with padding
                padding = 5
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(image.shape[1], x2 + padding)
                y2 = min(image.shape[0], y2 + padding)
                
                cropped = image[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue
                
                # OCR
                ocr_result = self.ocr.predict(cropped)
                
                license_text = ""
                if ocr_result and len(ocr_result) > 0:
                    res = ocr_result[0]
                    if 'rec_texts' in res and len(res['rec_texts']) > 0:
                        # Try combining all texts first
                        combined = combine_ocr_texts(res['rec_texts'])
                        if self.validate_license_plate_format(combined):
                            license_text = combined
                        else:
                            # Try individual texts
                            for text in res['rec_texts']:
                                cleaned = self.clean_license_plate_text(text)
                                if self.validate_license_plate_format(cleaned):
                                    license_text = cleaned
                                    break
                
                if license_text:
                    detections.append({
                        "license_plate_number": license_text,
                        "confidence_score": conf,
                        "bbox": bbox,
                        "processing_time_ms": (time.time() - start_time) * 1000
                    })
            
            return {
                "success": len(detections) > 0,
                "detections": detections,
                "total_detections": len(detections),
                "total_processing_time_ms": (time.time() - start_time) * 1000,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "detections": [],
                "total_processing_time_ms": (time.time() - start_time) * 1000,
                "error": str(e)
            }

# Usage example:
if __name__ == "__main__":
    recognizer = LicensePlateRecognizer("models/final/best_model.pt")
    result = recognizer.recognize("test_image.jpg")
    print(result)
'''

with open("results/phase2_ocr/license_plate_recognition_production.py", "w") as f:
    f.write(production_code)

print(f"Production code saved to: license_plate_recognition_production.py")

# Generate final assessment
print(f"\nPHASE 2 FINAL ASSESSMENT")
print("="*60)

print(f"OCR PERFORMANCE:")
print(f"  Cropped images processed: {len(processed_results)}")
print(f"  OCR success rate: {ocr_success_rate:.1f}%")
print(f"  Average OCR time: {avg_ocr_time:.1f} ms")

print(f"\nINTEGRATION PERFORMANCE:")
print(f"  End-to-end tests: {len(integration_results)}")
print(f"  Integration success rate: {integration_success_rate:.1f}%")
print(f"  Average total time: {avg_total_time:.1f} ms")
print(f"  Valid license plates: {len(valid_plates)}")

print(f"\nSUCCESS CRITERIA:")
criteria_results = performance_report["performance_targets"]
for criteria, data in criteria_results.items():
    status = "PASS" if data["pass"] else "FAIL"
    print(f"  {criteria}: {status} ({data['actual']} vs target {data['target']})")

overall_pass = all(data["pass"] for data in criteria_results.values())
print(f"\nOVERALL STATUS: {'PRODUCTION READY' if overall_pass else 'NEEDS IMPROVEMENT'}")

if len(valid_plates) > 0:
    print(f"\nDETECTED LICENSE PLATES:")
    for i, plate in enumerate(valid_plates):
        print(f"  {i+1}. {plate}")

print("="*60)