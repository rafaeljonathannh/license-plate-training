
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
        
        cleaned = re.sub(r'[^A-Z0-9\s]', '', raw_text.upper())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        patterns = [
            r'^([A-Z]{1,2})\s*(\d{1,4})\s*([A-Z]{1,3})$',
            r'^([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})$',
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
        pattern = r'^[A-Z]{1,2}\s\d{1,4}\s[A-Z]{1,3}$'
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
