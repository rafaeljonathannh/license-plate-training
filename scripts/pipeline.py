#!/usr/bin/env python3
"""
Indonesian License Plate Detection Pipeline
Core pipeline functions as specified in CLAUDE.md

This module implements the core pipeline functions for Indonesian license plate detection
and recognition, designed to be compatible with the existing production system.
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np
from PIL import Image
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Run: pip install ultralytics")
    YOLO = None

try:
    import paddleocr
except ImportError:
    print("Warning: PaddleOCR not installed. OCR functionality will be limited.")
    paddleocr = None


class IndonesianLicensePlateError(Exception):
    """Custom exception for license plate detection errors"""
    pass


def load_model(model_path: str) -> YOLO:
    """
    Loads the YOLOv8 model from a file path. Raises FileNotFoundError if missing.
    
    Args:
        model_path: Path to the YOLOv8 model file (.pt)
        
    Returns:
        YOLO: Loaded YOLOv8 model instance
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        IndonesianLicensePlateError: If model loading fails
    """
    try:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if YOLO is None:
            raise IndonesianLicensePlateError("YOLO not available. Please install ultralytics.")
        
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully: {model_path}")
        return model
        
    except Exception as e:
        raise IndonesianLicensePlateError(f"Failed to load model from {model_path}: {str(e)}")


def perform_detection(model: YOLO, image: Image.Image, confidence_threshold: float = 0.3) -> list:
    """
    Runs object detection on the image and returns YOLO's results object.
    
    Args:
        model: Loaded YOLOv8 model
        image: PIL Image object
        confidence_threshold: Minimum confidence threshold for detections
        
    Returns:
        list: YOLO detection results
        
    Raises:
        IndonesianLicensePlateError: If detection fails
    """
    try:
        if model is None:
            raise IndonesianLicensePlateError("Model not loaded")
        
        if not isinstance(image, Image.Image):
            raise IndonesianLicensePlateError("Input must be a PIL Image object")
        
        # Convert PIL Image to numpy array for YOLO
        image_array = np.array(image)
        
        # Run detection
        results = model(image_array, conf=confidence_threshold, verbose=False)
        
        return results
        
    except Exception as e:
        raise IndonesianLicensePlateError(f"Detection failed: {str(e)}")


def crop_from_bbox(image: Image.Image, bbox: List[Union[int, float]]) -> Image.Image:
    """
    Crops the original image using the provided bounding box coordinates.
    
    Args:
        image: PIL Image object to crop
        bbox: Bounding box coordinates [x1, y1, x2, y2] in pixel coordinates
        
    Returns:
        Image.Image: Cropped image containing the license plate region
        
    Raises:
        IndonesianLicensePlateError: If cropping fails
    """
    try:
        if not isinstance(image, Image.Image):
            raise IndonesianLicensePlateError("Input must be a PIL Image object")
        
        if len(bbox) != 4:
            raise IndonesianLicensePlateError("Bounding box must have 4 coordinates [x1, y1, x2, y2]")
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Validate bounding box coordinates
        img_width, img_height = image.size
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        x2 = max(x1, min(x2, img_width))
        y2 = max(y1, min(y2, img_height))
        
        if x2 <= x1 or y2 <= y1:
            raise IndonesianLicensePlateError("Invalid bounding box coordinates")
        
        # Crop the image
        cropped_image = image.crop((x1, y1, x2, y2))
        
        return cropped_image
        
    except Exception as e:
        raise IndonesianLicensePlateError(f"Failed to crop image: {str(e)}")


def read_plate_with_ocr(plate_image: Image.Image) -> str:
    """
    Takes a cropped plate image and returns the recognized text string using PaddleOCR.
    Returns empty string on failure.
    
    Args:
        plate_image: PIL Image object containing the license plate
        
    Returns:
        str: Recognized license plate text, empty string if recognition fails
    """
    try:
        if paddleocr is None:
            print("Warning: PaddleOCR not available, returning empty string")
            return ""
        
        if not isinstance(plate_image, Image.Image):
            print("Warning: Invalid input image, returning empty string")
            return ""
        
        # Initialize PaddleOCR (cached after first use)
        if not hasattr(read_plate_with_ocr, '_ocr'):
            read_plate_with_ocr._ocr = paddleocr.PaddleOCR(
                use_angle_cls=True, 
                lang='en',
                show_log=False
            )
        
        # Convert PIL Image to numpy array
        plate_array = np.array(plate_image)
        
        # Run OCR
        results = read_plate_with_ocr._ocr.ocr(plate_array, cls=True)
        
        # Extract text from results
        text_parts = []
        if results and results[0]:
            for line in results[0]:
                if line and len(line) >= 2 and line[1] and line[1][0]:
                    text_parts.append(line[1][0])
        
        # Join and clean text
        full_text = " ".join(text_parts).strip()
        
        # Clean and format Indonesian license plate text
        cleaned_text = _clean_indonesian_plate_text(full_text)
        
        return cleaned_text
        
    except Exception as e:
        print(f"OCR failed: {str(e)}")
        return ""


def _clean_indonesian_plate_text(text: str) -> str:
    """
    Clean and format Indonesian license plate text
    
    Args:
        text: Raw OCR text
        
    Returns:
        str: Cleaned license plate text
    """
    if not text:
        return ""
    
    # Convert to uppercase
    text = text.upper()
    
    # Common OCR error corrections
    text = text.replace("O", "0")  # Letter O to digit 0
    text = text.replace("I", "1")  # Letter I to digit 1
    text = text.replace("S", "5")  # Sometimes S is misread as 5
    text = text.replace("G", "6")  # Sometimes G is misread as 6
    
    # Remove special characters except spaces
    import re
    text = re.sub(r'[^A-Z0-9\s]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Indonesian license plate format validation and correction
    # Expected format: [Area Code] [Number] [Suffix]
    # Examples: "B 1234 ABC", "D 5678 XY", "AA 1234 BB"
    
    parts = text.split()
    if len(parts) >= 3:
        # Try to format as Indonesian license plate
        area_code = parts[0]
        number_part = ""
        suffix_part = ""
        
        # Find the numeric part
        for part in parts[1:]:
            if part.isdigit():
                number_part = part
                break
        
        # Find the suffix (letters after number)
        suffix_parts = []
        found_number = False
        for part in parts[1:]:
            if part.isdigit():
                found_number = True
            elif found_number and part.isalpha():
                suffix_parts.append(part)
        
        if suffix_parts:
            suffix_part = "".join(suffix_parts)
        
        # Reconstruct if we have valid parts
        if area_code.isalpha() and number_part.isdigit() and suffix_part.isalpha():
            return f"{area_code} {number_part} {suffix_part}"
    
    # If formatting fails, return cleaned text
    return text


def create_detection_response(
    detections: List[Dict[str, Any]], 
    total_time_ms: int, 
    error: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create standardized detection response format as per CLAUDE.md specifications
    
    Args:
        detections: List of detection dictionaries
        total_time_ms: Total processing time in milliseconds
        error: Error message if any
        
    Returns:
        Dict: Standardized response format
    """
    return {
        "success": error is None,
        "detections": detections,
        "total_detections": len(detections),
        "total_processing_time_ms": total_time_ms,
        "error": error
    }


def detect_license_plate(image_path: str, model_path: str = None, confidence_threshold: float = 0.3) -> Dict[str, Any]:
    """
    Complete pipeline for Indonesian license plate detection and recognition
    
    This function implements the core pipeline logic as specified in CLAUDE.md:
    1. Load the main input image
    2. Pass the image to perform_detection() using the trained model
    3. Loop through detection results from YOLO
    4. For each license plate detection with confidence > threshold:
       a. Get bbox coordinates
       b. Crop the license plate region
       c. Extract text using OCR
       d. Store detection information
    5. Format results into production JSON structure
    
    Args:
        image_path: Path to input image file
        model_path: Path to YOLO model (optional, uses default if not provided)
        confidence_threshold: Minimum confidence for detections
        
    Returns:
        Dict: Detection results in production format
    """
    start_time = time.time()
    
    try:
        # Validate input file
        if not Path(image_path).exists():
            return create_detection_response(
                [], 
                int((time.time() - start_time) * 1000),
                f"Error: Input file not found at {image_path}"
            )
        
        # Load model
        if model_path is None:
            # Try default locations
            possible_paths = [
                "cached_models/yolov8_indonesian_plates.pt",
                "../cached_models/yolov8_indonesian_plates.pt",
                "models/final/best_model.pt",
                "../models/final/best_model.pt"
            ]
            
            model_path = None
            for path in possible_paths:
                if Path(path).exists():
                    model_path = path
                    break
            
            if model_path is None:
                return create_detection_response(
                    [],
                    int((time.time() - start_time) * 1000),
                    "Error: No model found in default locations"
                )
        
        try:
            model = load_model(model_path)
        except Exception as e:
            return create_detection_response(
                [],
                int((time.time() - start_time) * 1000),
                f"Error: Failed to load model - {str(e)}"
            )
        
        # Load input image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            return create_detection_response(
                [],
                int((time.time() - start_time) * 1000),
                f"Error: Failed to load image - {str(e)}"
            )
        
        # Perform detection
        try:
            results = perform_detection(model, image, confidence_threshold)
        except Exception as e:
            return create_detection_response(
                [],
                int((time.time() - start_time) * 1000),
                f"Error: Detection failed - {str(e)}"
            )
        
        # Process detection results
        detections = []
        
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i, box in enumerate(boxes):
                try:
                    # Get detection information
                    bbox_xyxy = box.xyxy[0].cpu().numpy().astype(int)  # [x1, y1, x2, y2]
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Only process license plate detections (assuming class 0 is license-plate)
                    if class_id == 0 and confidence >= confidence_threshold:
                        detection_start = time.time()
                        
                        # Crop license plate region
                        try:
                            plate_image = crop_from_bbox(image, bbox_xyxy)
                        except Exception as e:
                            print(f"Failed to crop detection {i}: {e}")
                            continue
                        
                        # Extract text using OCR
                        license_plate_text = read_plate_with_ocr(plate_image)
                        
                        detection_time = int((time.time() - detection_start) * 1000)
                        
                        # Create detection object in production format
                        detection = {
                            "license_plate_number": license_plate_text,
                            "confidence_score": confidence,
                            "bbox": bbox_xyxy.tolist(),
                            "processing_time_ms": detection_time,
                            "detection_index": i
                        }
                        
                        detections.append(detection)
                        
                except Exception as e:
                    print(f"Error processing detection {i}: {e}")
                    continue
        
        # Calculate total processing time
        total_time = int((time.time() - start_time) * 1000)
        
        # Return results in production format
        return create_detection_response(detections, total_time)
        
    except Exception as e:
        total_time = int((time.time() - start_time) * 1000)
        return create_detection_response(
            [],
            total_time,
            f"Error: Unexpected error - {str(e)}"
        )


def main():
    """Example usage of the pipeline functions"""
    
    print("Indonesian License Plate Detection Pipeline")
    print("=" * 50)
    
    # Example usage
    model_path = "models/final/best_model.pt"
    image_path = "dataset/test/images/sample.jpg"
    
    if Path(model_path).exists() and Path(image_path).exists():
        print(f"Running detection on: {image_path}")
        
        result = detect_license_plate(image_path, model_path)
        
        print("\nResults:")
        print(json.dumps(result, indent=2))
        
    else:
        print("Model or test image not found. Testing individual functions...")
        
        # Test model loading
        try:
            print("\nTesting model loading...")
            if Path(model_path).exists():
                model = load_model(model_path)
                print("✅ Model loading successful")
            else:
                print("⚠️  Model file not found")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
        
        # Test image operations
        try:
            print("\nTesting image operations...")
            if Path(image_path).exists():
                image = Image.open(image_path)
                
                # Test cropping
                bbox = [100, 100, 300, 200]  # Sample bbox
                cropped = crop_from_bbox(image, bbox)
                print(f"✅ Image cropping successful: {cropped.size}")
                
                # Test OCR
                text = read_plate_with_ocr(cropped)
                print(f"✅ OCR test completed: '{text}'")
                
            else:
                print("⚠️  Test image not found")
                
        except Exception as e:
            print(f"❌ Image operations failed: {e}")


if __name__ == "__main__":
    main()