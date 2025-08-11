# File: test_new_images_ocr.py
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
    from ultralytics import YOLO
    import easyocr
    from PIL import Image, ImageEnhance
    print("✅ Required libraries imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Install with: pip install ultralytics easyocr pillow")
    exit(1)

print("=== TESTING NEW IMAGES WITH OCR DIAGNOSIS ===")

class EnhancedOCRTester:
    def __init__(self):
        # Initialize models
        print("🔧 Initializing models...")
        self.yolo_model = YOLO("models/final/best_model.pt")
        self.easy_ocr = easyocr.Reader(['en'], gpu=False)
        print("✅ Models initialized")
    
    def enhance_image_brightness_contrast(self, image):
        """Apply brightness and contrast enhancement (best performing method)"""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhanced = ImageEnhance.Contrast(pil_image).enhance(2.0)
        enhanced = ImageEnhance.Brightness(enhanced).enhance(1.2)
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    
    def run_easyocr(self, image):
        """Run EasyOCR with timing and confidence"""
        try:
            start_time = time.time()
            results = self.easy_ocr.readtext(image)
            processing_time = (time.time() - start_time) * 1000
            
            texts = []
            confidences = []
            
            for (bbox, text, conf) in results:
                if conf > 0.3:  # Filter low confidence
                    texts.append(text)
                    confidences.append(conf)
            
            combined_text = " ".join(texts)
            
            return {
                "success": len(texts) > 0,
                "texts": texts,
                "confidences": confidences,
                "combined_text": combined_text,
                "processing_time_ms": round(processing_time, 2)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": 0
            }
    
    def clean_license_plate_text(self, raw_text):
        """Clean and format Indonesian license plate text"""
        if not raw_text:
            return "", False
        
        # Clean text
        cleaned = re.sub(r'[^A-Z0-9\s]', '', raw_text.upper())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Try Indonesian patterns
        patterns = [
            r'^([A-Z]{1,2})\s*(\d{1,4})\s*([A-Z]{1,3})$',  # B 1234 ABC
            r'^([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})$',        # B1234ABC
        ]
        
        for pattern in patterns:
            match = re.match(pattern, cleaned)
            if match:
                area, number, suffix = match.groups()
                formatted = f"{area} {number} {suffix}"
                return formatted, True
        
        return cleaned, False
    
    def process_all_test_images(self):
        """Process all images in test_images folder"""
        
        # Find all test images
        test_folder = "test_images/"
        if not os.path.exists(test_folder):
            print(f"❌ Test folder not found: {test_folder}")
            return []
        
        # Get all image files
        supported_formats = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
        all_images = []
        
        for filename in os.listdir(test_folder):
            if any(filename.endswith(ext) for ext in supported_formats):
                all_images.append(os.path.join(test_folder, filename))
        
        if not all_images:
            print("❌ No images found in test_images folder")
            return []
        
        print(f"📊 Found {len(all_images)} total images:")
        
        # Categorize images
        old_images = [img for img in all_images if not "your_new_car" in img]
        new_images = [img for img in all_images if "your_new_car" in img]
        
        print(f"  📁 Original images: {len(old_images)}")
        for img in old_images:
            print(f"    - {Path(img).name}")
        
        print(f"  🚗 New car images: {len(new_images)}")
        for img in new_images:
            print(f"    - {Path(img).name}")
        
        # Process all images
        all_results = []
        successful_detections = 0
        total_processing_time = 0
        
        for i, image_path in enumerate(all_images):
            print(f"\n--- Processing {i+1}/{len(all_images)}: {Path(image_path).name} ---")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Could not load image")
                continue
            
            # YOLO Detection
            yolo_start = time.time()
            yolo_results = self.yolo_model(image_path, verbose=False)
            yolo_time = (time.time() - yolo_start) * 1000
            
            yolo_result = yolo_results[0]
            
            if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
                print(f"❌ No license plates detected by YOLO")
                result = {
                    "image_name": Path(image_path).name,
                    "image_path": image_path,
                    "image_category": "new" if "your_new_car" in image_path else "original",
                    "yolo_detections": 0,
                    "yolo_time_ms": round(yolo_time, 2),
                    "success": False,
                    "error": "No YOLO detections"
                }
                all_results.append(result)
                continue
            
            # Process each detection
            detections = []
            image_successful = False
            
            for det_idx, box in enumerate(yolo_result.boxes):
                detection_conf = float(box.conf.item())
                if detection_conf < 0.3:
                    continue
                
                print(f"🔍 Processing detection {det_idx+1} (confidence: {detection_conf:.3f})")
                
                # Extract and crop license plate
                bbox = box.xyxy[0].tolist()
                x1, y1, x2, y2 = [int(x) for x in bbox]
                
                # Crop with padding
                padding = 10
                crop_x1 = max(0, x1 - padding)
                crop_y1 = max(0, y1 - padding)
                crop_x2 = min(image.shape[1], x2 + padding)
                crop_y2 = min(image.shape[0], y2 + padding)
                
                cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
                
                if cropped.size == 0:
                    continue
                
                # Save crop for inspection
                crop_filename = f"crop_{i+1:02d}_det{det_idx+1}_{Path(image_path).stem}.jpg"
                crop_folder = "results/new_image_test/crops/"
                os.makedirs(crop_folder, exist_ok=True)
                crop_path = f"{crop_folder}{crop_filename}"
                cv2.imwrite(crop_path, cropped)
                
                print(f"💾 Saved crop: {crop_filename} ({cropped.shape[1]}x{cropped.shape[0]})")
                
                # Test both original and enhanced image
                test_results = {}
                
                # 1. Original crop
                original_result = self.run_easyocr(cropped)
                test_results["original"] = original_result
                
                # 2. Enhanced crop (brightness/contrast)
                enhanced = self.enhance_image_brightness_contrast(cropped)
                enhanced_result = self.run_easyocr(enhanced)
                test_results["enhanced"] = enhanced_result
                
                # Save enhanced crop
                enhanced_filename = f"enhanced_{i+1:02d}_det{det_idx+1}_{Path(image_path).stem}.jpg"
                enhanced_path = f"{crop_folder}{enhanced_filename}"
                cv2.imwrite(enhanced_path, enhanced)
                
                # Analyze results
                best_result = None
                best_confidence = 0
                
                for method, result in test_results.items():
                    if result["success"]:
                        cleaned_text, is_valid = self.clean_license_plate_text(result["combined_text"])
                        result["cleaned_text"] = cleaned_text
                        result["is_valid_format"] = is_valid
                        
                        if is_valid:
                            avg_conf = sum(result["confidences"]) / len(result["confidences"])
                            if avg_conf > best_confidence:
                                best_confidence = avg_conf
                                best_result = result
                                best_result["method"] = method
                        
                        status = "✅" if is_valid else "❌"
                        avg_conf = sum(result["confidences"]) / len(result["confidences"]) if result["confidences"] else 0
                        print(f"  {method}: '{cleaned_text}' {status} (conf: {avg_conf:.3f})")
                
                # Store detection result
                detection_data = {
                    "detection_index": det_idx,
                    "detection_confidence": detection_conf,
                    "bbox": bbox,
                    "crop_size": cropped.shape[:2],
                    "crop_path": crop_path,
                    "enhanced_path": enhanced_path,
                    "ocr_results": test_results,
                    "best_result": best_result,
                    "success": best_result is not None
                }
                
                detections.append(detection_data)
                
                if best_result:
                    image_successful = True
                    print(f"✅ BEST: {best_result['method']} → '{best_result['cleaned_text']}'")
            
            # Store image result
            image_result = {
                "image_name": Path(image_path).name,
                "image_path": image_path,
                "image_category": "new" if "your_new_car" in image_path else "original",
                "yolo_detections": len(yolo_result.boxes),
                "yolo_time_ms": round(yolo_time, 2),
                "processed_detections": len(detections),
                "successful_detections": len([d for d in detections if d["success"]]),
                "detections": detections,
                "success": image_successful,
                "total_processing_time_ms": round(yolo_time + sum([d.get("best_result", {}).get("processing_time_ms", 0) if d.get("best_result") else 0 for d in detections]), 2)
            }
            
            all_results.append(image_result)
            
            if image_successful:
                successful_detections += 1
            
            total_processing_time += image_result["total_processing_time_ms"]
        
        # Generate summary statistics
        self.generate_summary_report(all_results, successful_detections, total_processing_time)
        
        return all_results
    
    def generate_summary_report(self, all_results, successful_detections, total_processing_time):
        """Generate comprehensive summary report"""
        
        print(f"\n" + "="*70)
        print(f"📊 COMPREHENSIVE OCR TESTING REPORT")
        print(f"="*70)
        
        # Overall statistics
        total_images = len(all_results)
        total_detections = sum(r.get("processed_detections", 0) for r in all_results)
        successful_extractions = sum(r.get("successful_detections", 0) for r in all_results)
        
        # Category breakdown
        original_images = [r for r in all_results if r["image_category"] == "original"]
        new_images = [r for r in all_results if r["image_category"] == "new"]
        
        original_success = sum(1 for r in original_images if r["success"])
        new_success = sum(1 for r in new_images if r["success"])
        
        print(f"\n🎯 OVERALL STATISTICS:")
        print(f"  Total images processed: {total_images}")
        print(f"  Successful image processing: {successful_detections} ({(successful_detections/total_images)*100:.1f}%)")
        print(f"  Total license plate detections: {total_detections}")
        print(f"  Successful text extractions: {successful_extractions}")
        print(f"  Average processing time: {total_processing_time/total_images:.0f}ms per image")
        
        print(f"\n📊 PERFORMANCE BY IMAGE CATEGORY:")
        print(f"  Original images: {len(original_images)} images, {original_success} successful ({(original_success/len(original_images))*100:.1f}% if len(original_images) > 0 else 0)")
        print(f"  New car images: {len(new_images)} images, {new_success} successful ({(new_success/len(new_images))*100:.1f}% if len(new_images) > 0 else 0)")
        
        # Extract all successful license plates
        all_plates = []
        for result in all_results:
            for detection in result.get("detections", []):
                if detection.get("success") and detection.get("best_result"):
                    best = detection["best_result"]
                    all_plates.append({
                        "image": result["image_name"],
                        "category": result["image_category"],
                        "text": best["cleaned_text"],
                        "confidence": sum(best["confidences"]) / len(best["confidences"]),
                        "method": best["method"]
                    })
        
        print(f"\n🏷️ SUCCESSFULLY EXTRACTED LICENSE PLATES:")
        if all_plates:
            for i, plate in enumerate(all_plates, 1):
                category_icon = "🚗" if plate["category"] == "new" else "📁"
                print(f"  {i}. {category_icon} {plate['image']} → '{plate['text']}' (conf: {plate['confidence']:.3f}, method: {plate['method']})")
        else:
            print(f"  ❌ No valid license plates extracted")
        
        # Performance analysis
        if len(new_images) > 0 and len(original_images) > 0:
            print(f"\n📈 IMPROVEMENT ANALYSIS:")
            old_rate = (original_success / len(original_images)) * 100
            new_rate = (new_success / len(new_images)) * 100
            improvement = new_rate - old_rate
            
            if improvement > 0:
                print(f"  ✅ New images performing better: {new_rate:.1f}% vs {old_rate:.1f}% (+{improvement:.1f}%)")
            elif improvement < 0:
                print(f"  ⚠️ New images performing worse: {new_rate:.1f}% vs {old_rate:.1f}% ({improvement:.1f}%)")
            else:
                print(f"  ➡️ Similar performance: {new_rate:.1f}% vs {old_rate:.1f}%")
        
        # Save detailed report
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_images": total_images,
                "successful_images": successful_detections,
                "success_rate": (successful_detections/total_images)*100 if total_images > 0 else 0,
                "total_detections": total_detections,
                "successful_extractions": successful_extractions,
                "average_processing_time_ms": total_processing_time/total_images if total_images > 0 else 0
            },
            "category_breakdown": {
                "original_images": {
                    "count": len(original_images),
                    "successful": original_success,
                    "success_rate": (original_success/len(original_images))*100 if original_images else 0
                },
                "new_images": {
                    "count": len(new_images),
                    "successful": new_success,
                    "success_rate": (new_success/len(new_images))*100 if new_images else 0
                }
            },
            "extracted_plates": all_plates,
            "detailed_results": all_results
        }
        
        # Save report
        os.makedirs("results/new_image_test/", exist_ok=True)
        report_path = "results/new_image_test/comprehensive_report.json"
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Detailed report saved: {report_path}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if new_success > original_success:
            print(f"  ✅ New images show improved detection - good image selection!")
        
        if successful_extractions > 0:
            best_method_count = {}
            for plate in all_plates:
                method = plate["method"]
                best_method_count[method] = best_method_count.get(method, 0) + 1
            
            best_method = max(best_method_count, key=best_method_count.get)
            print(f"  🏆 Best enhancement method: {best_method} ({best_method_count[best_method]}/{len(all_plates)} successes)")
        
        if (successful_detections/total_images)*100 >= 70:
            print(f"  🎉 Ready for production deployment with EasyOCR!")
        else:
            print(f"  ⚠️ Consider adding more diverse test images or custom OCR training")
        
        print(f"="*70)

# Initialize and run testing
tester = EnhancedOCRTester()
results = tester.process_all_test_images()

print("\n🎯 Testing completed! Check results in:")
print("📁 results/new_image_test/crops/ - Cropped license plate images")
print("📁 results/new_image_test/comprehensive_report.json - Detailed analysis")