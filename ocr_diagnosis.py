# OCR Diagnosis and Comparison Testing Script
# Based on DIAGNOSISOCR.md specifications

import os
import cv2
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import re

# Import OCR libraries
try:
    from ultralytics import YOLO
    from paddleocr import PaddleOCR
    import easyocr
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    print("✅ All OCR libraries imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Missing libraries. Install with:")
    print("pip install easyocr pytesseract pillow")
    exit(1)

print("=== OCR DIAGNOSIS & COMPARISON TESTING ===")

class OCRDiagnoser:
    def __init__(self, yolo_model_path):
        # Initialize models
        print("🔧 Initializing models...")
        self.yolo_model = YOLO(yolo_model_path)
        
        # Initialize OCR engines
        self.paddle_ocr = PaddleOCR(lang='en')
        self.easy_ocr = easyocr.Reader(['en'], gpu=False)
        
        # Tesseract config for license plates
        self.tesseract_config = '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        print("✅ All OCR models initialized")
    
    def enhance_image_quality(self, image, method='basic'):
        """Apply various image enhancement techniques"""
        enhanced_images = {}
        
        # Convert to PIL for easier manipulation
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 1. Original (as baseline)
        enhanced_images['original'] = image
        
        # 2. Upscaling (2x and 4x)
        height, width = image.shape[:2]
        enhanced_images['upscale_2x'] = cv2.resize(image, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
        enhanced_images['upscale_4x'] = cv2.resize(image, (width*4, height*4), interpolation=cv2.INTER_CUBIC)
        
        # 3. Contrast enhancement (CLAHE)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        clahe_enhanced = clahe.apply(gray)
        enhanced_images['clahe'] = cv2.cvtColor(clahe_enhanced, cv2.COLOR_GRAY2BGR)
        
        # 4. Brightness and contrast adjustment
        enhanced_pil = ImageEnhance.Contrast(pil_image).enhance(2.0)
        enhanced_pil = ImageEnhance.Brightness(enhanced_pil).enhance(1.2)
        enhanced_images['bright_contrast'] = cv2.cvtColor(np.array(enhanced_pil), cv2.COLOR_RGB2BGR)
        
        # 5. Sharpening
        sharpened_pil = pil_image.filter(ImageFilter.SHARPEN)
        enhanced_images['sharpened'] = cv2.cvtColor(np.array(sharpened_pil), cv2.COLOR_RGB2BGR)
        
        # 6. Binary threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        enhanced_images['binary'] = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        # 7. Gaussian blur + sharpen combination
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(blurred, -1, kernel)
        enhanced_images['blur_sharpen'] = sharpened
        
        return enhanced_images
    
    def run_paddle_ocr(self, image):
        """Run PaddleOCR on image"""
        try:
            start_time = time.time()
            result = self.paddle_ocr.ocr(image, cls=True)
            processing_time = (time.time() - start_time) * 1000
            
            texts = []
            confidences = []
            
            if result and result[0]:
                for line in result[0]:
                    bbox, (text, conf) = line
                    texts.append(text)
                    confidences.append(conf)
            
            return {
                "engine": "PaddleOCR",
                "texts": texts,
                "confidences": confidences,
                "combined_text": " ".join(texts),
                "processing_time_ms": round(processing_time, 2),
                "success": len(texts) > 0
            }
        except Exception as e:
            return {
                "engine": "PaddleOCR",
                "error": str(e),
                "success": False,
                "processing_time_ms": 0
            }
    
    def run_easy_ocr(self, image):
        """Run EasyOCR on image"""
        try:
            start_time = time.time()
            results = self.easy_ocr.readtext(image)
            processing_time = (time.time() - start_time) * 1000
            
            texts = []
            confidences = []
            
            for (bbox, text, conf) in results:
                if conf > 0.1:  # Filter very low confidence
                    texts.append(text)
                    confidences.append(conf)
            
            return {
                "engine": "EasyOCR",
                "texts": texts,
                "confidences": confidences,
                "combined_text": " ".join(texts),
                "processing_time_ms": round(processing_time, 2),
                "success": len(texts) > 0
            }
        except Exception as e:
            return {
                "engine": "EasyOCR",
                "error": str(e),
                "success": False,
                "processing_time_ms": 0
            }
    
    def run_tesseract_ocr(self, image):
        """Run Tesseract OCR on image"""
        try:
            start_time = time.time()
            
            # Convert to PIL Image for Tesseract
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Extract text
            text = pytesseract.image_to_string(pil_image, config=self.tesseract_config).strip()
            
            # Get confidence data
            data = pytesseract.image_to_data(pil_image, config=self.tesseract_config, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "engine": "Tesseract",
                "texts": [text] if text else [],
                "confidences": [sum(confidences)/len(confidences)] if confidences else [0],
                "combined_text": text,
                "processing_time_ms": round(processing_time, 2),
                "success": bool(text)
            }
        except Exception as e:
            return {
                "engine": "Tesseract",
                "error": str(e),
                "success": False,
                "processing_time_ms": 0
            }
    
    def clean_and_validate_text(self, raw_text):
        """Clean and validate license plate text"""
        if not raw_text:
            return "", False, "Empty text"
        
        # Clean text
        cleaned = re.sub(r'[^A-Z0-9\s]', '', raw_text.upper())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Indonesian license plate pattern
        pattern = r'^[A-Z]{1,2}\s\d{1,4}\s[A-Z]{1,3}$'
        is_valid = bool(re.match(pattern, cleaned))
        
        return cleaned, is_valid, "Valid Indonesian format" if is_valid else "Invalid format"
    
    def process_test_images(self):
        """Process all test images for comprehensive OCR diagnosis"""
        
        # Find test images
        test_folder = "test_images/"
        if not os.path.exists(test_folder):
            print(f"❌ Test folder not found: {test_folder}")
            return []
        
        supported_formats = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
        test_images = [
            os.path.join(test_folder, f) for f in os.listdir(test_folder)
            if any(f.endswith(ext) for ext in supported_formats)
        ]
        
        if not test_images:
            print("❌ No test images found")
            return []
        
        print(f"📊 Processing {len(test_images)} test images...")
        
        all_results = []
        
        for img_idx, image_path in enumerate(test_images):
            print(f"\n--- Processing Image {img_idx + 1}/{len(test_images)}: {Path(image_path).name} ---")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Could not load image: {image_path}")
                continue
            
            # YOLO Detection to get license plate regions
            yolo_results = self.yolo_model(image_path, verbose=False)
            yolo_result = yolo_results[0]
            
            if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
                print(f"❌ No license plates detected by YOLO")
                continue
            
            # Process each detected license plate
            for det_idx, box in enumerate(yolo_result.boxes):
                conf = float(box.conf.item())
                if conf < 0.3:
                    continue
                
                print(f"\n🔍 Processing Detection {det_idx + 1} (confidence: {conf:.3f})")
                
                # Extract bounding box
                bbox = box.xyxy[0].tolist()
                x1, y1, x2, y2 = [int(x) for x in bbox]
                
                # Crop with generous padding for better OCR
                padding = 10
                crop_x1 = max(0, x1 - padding)
                crop_y1 = max(0, y1 - padding)
                crop_x2 = min(image.shape[1], x2 + padding)
                crop_y2 = min(image.shape[0], y2 + padding)
                
                cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
                
                if cropped.size == 0:
                    continue
                
                # Save original crop
                crop_filename = f"crop_{img_idx+1:02d}_det{det_idx+1}_original.jpg"
                crop_path = f"results/ocr_diagnosis/cropped_images/{crop_filename}"
                cv2.imwrite(crop_path, cropped)
                
                print(f"💾 Saved crop: {crop_filename} (size: {cropped.shape[1]}x{cropped.shape[0]})")
                
                # Generate enhanced versions
                enhanced_images = self.enhance_image_quality(cropped)
                
                # Test all OCR engines on all enhanced versions
                ocr_results = {}
                
                for enhancement_name, enhanced_img in enhanced_images.items():
                    print(f"  📝 Testing OCR on {enhancement_name}...")
                    
                    # Save enhanced image
                    enhanced_filename = f"enhanced_{img_idx+1:02d}_det{det_idx+1}_{enhancement_name}.jpg"
                    enhanced_path = f"results/ocr_diagnosis/enhanced_images/{enhanced_filename}"
                    cv2.imwrite(enhanced_path, enhanced_img)
                    
                    # Test all OCR engines
                    paddle_result = self.run_paddle_ocr(enhanced_img)
                    easy_result = self.run_easy_ocr(enhanced_img)
                    tesseract_result = self.run_tesseract_ocr(enhanced_img)
                    
                    # Clean and validate results
                    for result in [paddle_result, easy_result, tesseract_result]:
                        if result.get("success"):
                            cleaned, is_valid, msg = self.clean_and_validate_text(result["combined_text"])
                            result["cleaned_text"] = cleaned
                            result["is_valid_format"] = is_valid
                            result["validation_message"] = msg
                    
                    ocr_results[enhancement_name] = {
                        "enhancement": enhancement_name,
                        "enhanced_image_path": enhanced_path,
                        "paddle_ocr": paddle_result,
                        "easy_ocr": easy_result,
                        "tesseract_ocr": tesseract_result
                    }
                    
                    # Print results summary
                    for engine_result in [paddle_result, easy_result, tesseract_result]:
                        if engine_result.get("success"):
                            engine = engine_result["engine"]
                            text = engine_result.get("cleaned_text", engine_result["combined_text"])
                            valid = "✅" if engine_result.get("is_valid_format", False) else "❌"
                            print(f"    {engine}: '{text}' {valid}")
                        else:
                            print(f"    {engine_result['engine']}: ❌ Failed")
                
                # Store complete result
                detection_result = {
                    "image_name": Path(image_path).name,
                    "image_path": image_path,
                    "detection_index": det_idx,
                    "detection_confidence": conf,
                    "bbox": bbox,
                    "original_crop_path": crop_path,
                    "crop_size": cropped.shape[:2],
                    "ocr_results": ocr_results
                }
                
                all_results.append(detection_result)
        
        return all_results

def main():
    # Initialize diagnoser
    diagnoser = OCRDiagnoser("models/final/best_model.pt")
    
    # Run diagnosis
    print("\n🚀 Starting comprehensive OCR diagnosis...")
    diagnosis_results = diagnoser.process_test_images()

    if not diagnosis_results:
        print("❌ No license plate detections found for OCR testing")
        return

    print(f"\n📊 Completed diagnosis on {len(diagnosis_results)} license plate detections")

    # Analyze results and find best performing combinations
    print("\n📈 Analyzing OCR performance across all combinations...")

    performance_analysis = {
        "total_detections": len(diagnosis_results),
        "enhancement_performance": {},
        "engine_performance": {},
        "best_combinations": [],
        "success_rates": {}
    }

    # Analyze by enhancement method
    enhancement_methods = set()
    engine_names = ["paddle_ocr", "easy_ocr", "tesseract_ocr"]

    for result in diagnosis_results:
        for enhancement_name in result["ocr_results"].keys():
            enhancement_methods.add(enhancement_name)
            
            if enhancement_name not in performance_analysis["enhancement_performance"]:
                performance_analysis["enhancement_performance"][enhancement_name] = {
                    "total_tests": 0,
                    "successful_extractions": 0,
                    "valid_formats": 0,
                    "average_processing_time": 0,
                    "best_results": []
                }

    # Calculate performance metrics
    for result in diagnosis_results:
        for enhancement_name, ocr_data in result["ocr_results"].items():
            perf = performance_analysis["enhancement_performance"][enhancement_name]
            
            for engine_name in engine_names:
                engine_result = ocr_data[engine_name]
                perf["total_tests"] += 1
                
                if engine_result.get("success"):
                    perf["successful_extractions"] += 1
                    
                    if engine_result.get("is_valid_format"):
                        perf["valid_formats"] += 1
                        perf["best_results"].append({
                            "image": result["image_name"],
                            "engine": engine_result["engine"],
                            "text": engine_result.get("cleaned_text", ""),
                            "confidence": engine_result.get("confidences", [0])[0] if engine_result.get("confidences") else 0
                        })

    # Calculate success rates
    for enhancement_name, perf in performance_analysis["enhancement_performance"].items():
        if perf["total_tests"] > 0:
            success_rate = (perf["valid_formats"] / perf["total_tests"]) * 100
            performance_analysis["success_rates"][enhancement_name] = round(success_rate, 2)

    # Find best performing combinations
    best_combinations = []
    for enhancement_name, perf in performance_analysis["enhancement_performance"].items():
        if perf["valid_formats"] > 0:
            for result in perf["best_results"]:
                best_combinations.append({
                    "enhancement": enhancement_name,
                    "engine": result["engine"],
                    "extracted_text": result["text"],
                    "confidence": result["confidence"],
                    "image": result["image"]
                })

    # Sort by confidence
    best_combinations.sort(key=lambda x: x["confidence"], reverse=True)
    performance_analysis["best_combinations"] = best_combinations[:10]  # Top 10

    print(f"📊 Performance Analysis Summary:")
    print(f"  Total detections tested: {performance_analysis['total_detections']}")
    print(f"  Enhancement methods: {len(enhancement_methods)}")
    print(f"  Best valid extractions: {len(best_combinations)}")

    # Create detailed comparison report
    print(f"\n📋 Generating detailed comparison report...")

    # Generate comprehensive report
    detailed_report = {
        "diagnosis_timestamp": datetime.now().isoformat(),
        "summary": performance_analysis,
        "detailed_results": diagnosis_results,
        "recommendations": [],
        "next_steps": []
    }

    # Generate recommendations based on results
    if performance_analysis["success_rates"]:
        best_enhancement = max(performance_analysis["success_rates"], key=performance_analysis["success_rates"].get)
        best_rate = performance_analysis["success_rates"][best_enhancement]
        
        detailed_report["recommendations"].append(f"Best enhancement method: {best_enhancement} ({best_rate}% success rate)")
        
        # Engine analysis
        engine_success = {"PaddleOCR": 0, "EasyOCR": 0, "Tesseract": 0}
        for combo in best_combinations:
            engine_success[combo["engine"]] += 1
        
        best_engine = max(engine_success, key=engine_success.get)
        detailed_report["recommendations"].append(f"Best performing OCR engine: {best_engine}")
        
        if best_rate > 70:
            detailed_report["next_steps"].append("Implement best enhancement + engine combination")
            detailed_report["next_steps"].append("Test on larger dataset for validation")
        else:
            detailed_report["next_steps"].append("Consider custom OCR training for Indonesian license plates")
            detailed_report["next_steps"].append("Collect more diverse training data")

    # Save detailed report
    report_path = "results/ocr_diagnosis/diagnosis_report.json"
    with open(report_path, "w") as f:
        json.dump(detailed_report, f, indent=2)

    print(f"💾 Detailed diagnosis report saved: {report_path}")

    # Print final diagnosis summary
    print(f"\n" + "="*70)
    print(f"📋 OCR DIAGNOSIS FINAL REPORT")
    print(f"="*70)

    print(f"\n🎯 EXECUTIVE SUMMARY:")
    print(f"  Total license plates tested: {performance_analysis['total_detections']}")
    print(f"  Enhancement methods tested: {len(enhancement_methods)}")
    print(f"  OCR engines tested: 3 (PaddleOCR, EasyOCR, Tesseract)")

    if performance_analysis["success_rates"]:
        best_enhancement = max(performance_analysis["success_rates"], key=performance_analysis["success_rates"].get)
        best_rate = performance_analysis["success_rates"][best_enhancement]
        
        print(f"\n📊 PERFORMANCE RESULTS:")
        print(f"  Best enhancement method: {best_enhancement}")
        print(f"  Best success rate: {best_rate}%")
        
        print(f"\n🏆 TOP PERFORMING COMBINATIONS:")
        for i, combo in enumerate(performance_analysis["best_combinations"][:5], 1):
            print(f"  {i}. {combo['enhancement']} + {combo['engine']}")
            print(f"     Text: '{combo['extracted_text']}' (conf: {combo['confidence']:.3f})")
        
        print(f"\n📈 SUCCESS RATES BY ENHANCEMENT:")
        for enhancement, rate in sorted(performance_analysis["success_rates"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {enhancement}: {rate}%")
    else:
        print(f"\n❌ NO SUCCESSFUL TEXT EXTRACTIONS FOUND")
        print(f"   This indicates a fundamental OCR problem that requires:")
        print(f"   1. Custom OCR model training")
        print(f"   2. Different approach to license plate recognition")
        print(f"   3. Commercial OCR API testing")

    print(f"\n📁 GENERATED FILES:")
    print(f"  📸 Cropped images: results/ocr_diagnosis/cropped_images/")
    print(f"  🖼️ Enhanced images: results/ocr_diagnosis/enhanced_images/")
    print(f"  📊 Diagnosis report: results/ocr_diagnosis/diagnosis_report.json")

    print(f"\n🎯 RECOMMENDATIONS:")
    for rec in detailed_report["recommendations"]:
        print(f"  • {rec}")

    print(f"\n📝 NEXT STEPS:")
    for step in detailed_report["next_steps"]:
        print(f"  • {step}")

    print(f"="*70)

if __name__ == "__main__":
    main()