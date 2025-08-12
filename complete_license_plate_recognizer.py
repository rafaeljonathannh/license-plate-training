#!/usr/bin/env python3
"""
COMPLETE LICENSE PLATE RECOGNITION SYSTEM
=========================================
Integration: YOLO Detection + OCR Optimization + Indonesian Parsing

Components:
1. YOLO Detection (99.4% mAP)
2. OCR Optimization (EasyOCR + brightness/contrast, 99.6% accuracy)
3. Indonesian License Plate Parsing (area codes, dates, validation)

Output: Comprehensive JSON with detailed parsing breakdown
"""

import os
import json
import time
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import glob

class IndonesianLicensePlateParser:
    """Indonesian License Plate Parser with comprehensive area code mapping"""
    
    def __init__(self):
        # Indonesian area code mapping
        self.area_codes = {
            # Jakarta & Sekitarnya
            "B": {"name": "Jakarta", "region": "DKI Jakarta"},
            
            # Jawa Barat  
            "D": {"name": "Bandung", "region": "Jawa Barat"},
            "F": {"name": "Bogor", "region": "Jawa Barat"},
            "T": {"name": "Purwakarta", "region": "Jawa Barat"},
            "Z": {"name": "Garut", "region": "Jawa Barat"},
            
            # Jawa Tengah
            "G": {"name": "Brebes", "region": "Jawa Tengah"},
            "H": {"name": "Semarang", "region": "Jawa Tengah"},
            "K": {"name": "Pekalongan", "region": "Jawa Tengah"},
            "R": {"name": "Banjarnegara", "region": "Jawa Tengah"},
            
            # Jawa Timur
            "L": {"name": "Surabaya", "region": "Jawa Timur"},
            "M": {"name": "Madura", "region": "Jawa Timur"},
            "N": {"name": "Malang", "region": "Jawa Timur"},
            "P": {"name": "Besuki", "region": "Jawa Timur"},
            "S": {"name": "Bojonegoro", "region": "Jawa Timur"},
            "W": {"name": "Sidoarjo", "region": "Jawa Timur"},
            
            # Banten
            "A": {"name": "Banten", "region": "Banten"},
            
            # Multi-letter codes
            "AA": {"name": "Magelang", "region": "Jawa Tengah"},
            "AB": {"name": "Yogyakarta", "region": "DI Yogyakarta"},
            "AD": {"name": "Solo", "region": "Jawa Tengah"},
            "BA": {"name": "Sumatra Barat", "region": "Sumatra Barat"},
            "BB": {"name": "Tapanuli", "region": "Sumatra Utara"},
            "BG": {"name": "Sumatra Selatan", "region": "Sumatra Selatan"},
            "DK": {"name": "Bali", "region": "Bali"},
            "KB": {"name": "Kalimantan Barat", "region": "Kalimantan Barat"},
            "KT": {"name": "Kalimantan Timur", "region": "Kalimantan Timur"},
            
            # Additional codes
            "BK": {"name": "Sumatra Utara", "region": "Sumatra Utara"},
            "BL": {"name": "Aceh", "region": "Aceh"},
            "BM": {"name": "Riau", "region": "Riau"},
            "BN": {"name": "Bangka Belitung", "region": "Bangka Belitung"},
            "BP": {"name": "Kepulauan Riau", "region": "Kepulauan Riau"},
            "CC": {"name": "Sulawesi Utara", "region": "Sulawesi Utara"},
            "DD": {"name": "Sulawesi Selatan", "region": "Sulawesi Selatan"},
            "DT": {"name": "Sulawesi Tenggara", "region": "Sulawesi Tenggara"},
            "DN": {"name": "Sulawesi Tengah", "region": "Sulawesi Tengah"},
            "DR": {"name": "Lombok", "region": "Nusa Tenggara Barat"},
            "EB": {"name": "Nusa Tenggara Timur", "region": "Nusa Tenggara Timur"},
            "ED": {"name": "Sumba", "region": "Nusa Tenggara Timur"},
            "KU": {"name": "Kalimantan Utara", "region": "Kalimantan Utara"},
            "KH": {"name": "Kalimantan Tengah", "region": "Kalimantan Tengah"},
            "DA": {"name": "Kalimantan Selatan", "region": "Kalimantan Selatan"},
            "PA": {"name": "Papua", "region": "Papua"},
            "PB": {"name": "Papua Barat", "region": "Papua Barat"}
        }
        
        # Month names in Indonesian
        self.month_names = {
            "01": "Januari", "02": "Februari", "03": "Maret",
            "04": "April", "05": "Mei", "06": "Juni", 
            "07": "Juli", "08": "Agustus", "09": "September",
            "10": "Oktober", "11": "November", "12": "Desember"
        }
        
        # License plate patterns (expanded for better coverage)
        self.patterns = [
            # Format 1: "B 1234 ABC 02.28" - With registration date (flexible separators)
            r'^([A-Z]{1,2})\s+(\d{1,4})\s+([A-Z]{1,4})\s+(\d{2})[\.;:](\d{2})$',
            # Format 2: "B 1234 ABC" - Without registration date (expanded series)
            r'^([A-Z]{1,2})\s+(\d{1,4})\s+([A-Z]{1,4})$',
            # Additional patterns for variations (no spaces after area/number)
            r'^([A-Z]{1,2})(\d{1,4})([A-Z]{1,4})\s+(\d{2})[\.;:](\d{2})$',
            r'^([A-Z]{1,2})(\d{1,4})([A-Z]{1,4})$',
            # Extra flexible patterns for OCR noise
            r'^([A-Z]{1,2})\s+(\d{1,4})\s+([A-Z]{1,4})\s*(\d{2})[\.;:](\d{2})$'
        ]
    
    def correct_area_code_ocr(self, raw_text: str) -> str:
        """Correct common OCR mistakes in area codes before regex matching
        
        Args:
            raw_text: Raw OCR text input
            
        Returns:
            str: Text with area code corrections applied
        """
        try:
            # Define area code corrections for common OCR mistakes
            AREA_CODE_CORRECTIONS = {
                "C": "G",   # C → G (Brebes, Jawa Tengah) - Main target fix
                "0": "D",   # Zero → D (Bandung, Jawa Barat) 
                "6": "G",   # Six → G (additional OCR confusion)
                "Q": "G",   # Q → G (similar shape to G)
                "8": "B",   # Eight → B (Jakarta) - shape similarity
                "5": "S",   # Five → S (Bojonegoro) - partial similarity
            }
            
            # Apply corrections only at start of text (area code position)
            corrected_text = raw_text
            
            for incorrect, correct in AREA_CODE_CORRECTIONS.items():
                # Check if text starts with incorrect area code followed by space
                if corrected_text.startswith(f"{incorrect} "):
                    # Replace only the first occurrence (area code position)
                    corrected_text = corrected_text.replace(f"{incorrect} ", f"{correct} ", 1)
                    print(f"  🔧 Area code correction: {incorrect} → {correct}")
                    break  # Only apply one correction per text
            
            return corrected_text
            
        except Exception as e:
            print(f"Area code correction error: {e}")
            return raw_text  # Return original on error
    
    def correct_number_section_ocr(self, raw_text: str) -> str:
        """Correct OCR mistakes in number section only, preserving area codes and series
        
        Args:
            raw_text: Text with area code corrections already applied
            
        Returns:
            str: Text with number section corrections applied
        """
        try:
            # Define number corrections for common OCR mistakes in numeric section
            NUMBER_SECTION_CORRECTIONS = {
                "L": "4",   # L → 4 (main target - shape similarity)
                "I": "1",   # I → 1 (shape similarity)
                "O": "0",   # O → 0 (shape similarity) 
                "Z": "2",   # Z → 2 (partial similarity)
                "S": "5",   # S → 5 (curved shape confusion)
                "G": "6",   # G → 6 (in numeric context)
                "B": "8",   # B → 8 (shape similarity)
                "T": "7",   # T → 7 (partial similarity)
            }
            
            # Split text into parts for section-aware processing
            parts = raw_text.strip().split()
            
            # Need at least 3 parts: [AREA] [NUMBER] [SERIES]
            if len(parts) < 3:
                return raw_text
            
            area_part = parts[0]      # Keep area code unchanged
            number_part = parts[1]    # Apply corrections here
            series_part = parts[2]    # Keep series unchanged  
            date_part = ""           # Handle optional date part
            
            # Handle optional date part (4th element)
            if len(parts) >= 4:
                date_part = " " + " ".join(parts[3:])
            
            # Apply corrections ONLY to number section
            corrected_number = number_part
            corrections_applied = []
            
            for letter, digit in NUMBER_SECTION_CORRECTIONS.items():
                if letter in corrected_number:
                    old_number = corrected_number
                    corrected_number = corrected_number.replace(letter, digit)
                    if old_number != corrected_number:
                        corrections_applied.append(f"{letter}→{digit}")
            
            # Reconstruct text with corrected number section
            if corrections_applied:
                corrected_text = f"{area_part} {corrected_number} {series_part}{date_part}"
                print(f"  🔢 Number section corrections: {', '.join(corrections_applied)} in '{number_part}' → '{corrected_number}'")
                return corrected_text
            
            return raw_text  # No corrections needed
            
        except Exception as e:
            print(f"Number section correction error: {e}")
            return raw_text  # Return original on error

    def validate_number_section(self, number_text: str) -> bool:
        """Validate that number section contains only digits after correction
        
        Args:
            number_text: Number section text to validate
            
        Returns:
            bool: True if valid numeric format
        """
        try:
            # Check if contains only digits
            if not number_text.isdigit():
                return False
                
            # Check reasonable length (1-4 digits for Indonesian plates)
            if len(number_text) < 1 or len(number_text) > 4:
                return False
                
            # Check reasonable numeric range (not just zeros)
            num_value = int(number_text)
            if num_value < 1 or num_value > 9999:
                return False
                
            return True
            
        except (ValueError, TypeError):
            return False

    def advanced_text_cleaning(self, raw_text: str) -> str:
        """Apply advanced section-aware cleaning for complex OCR mistakes
        
        Args:
            raw_text: Raw text needing advanced cleaning
            
        Returns:
            str: Text with advanced corrections applied
        """
        try:
            cleaned_text = raw_text
            
            # Handle common multi-character OCR confusions in number section
            ADVANCED_NUMBER_PATTERNS = {
                # Common OCR mistakes that create non-numeric sequences
                "LL": "44",    # Double L → Double 4
                "II": "11",    # Double I → Double 1  
                "OO": "00",    # Double O → Double 0
                "l0": "10",    # Lowercase l + zero → 1 + 0
                "1O": "10",    # 1 + capital O → 1 + 0
                "L0": "40",    # L + zero → 4 + 0
                "0L": "04",    # Zero + L → 0 + 4
            }
            
            # Split for section-aware replacement
            parts = cleaned_text.strip().split()
            if len(parts) >= 3:
                area_part = parts[0]
                number_part = parts[1]
                series_part = parts[2]
                date_part = " " + " ".join(parts[3:]) if len(parts) > 3 else ""
                
                # Apply advanced patterns only to number section
                corrected_number = number_part
                for pattern, replacement in ADVANCED_NUMBER_PATTERNS.items():
                    if pattern in corrected_number:
                        old_number = corrected_number
                        corrected_number = corrected_number.replace(pattern, replacement)
                        if old_number != corrected_number:
                            print(f"  🔧 Advanced pattern: '{pattern}' → '{replacement}' in number section")
                
                cleaned_text = f"{area_part} {corrected_number} {series_part}{date_part}"
            
            return cleaned_text
            
        except Exception as e:
            print(f"Advanced cleaning error: {e}")
            return raw_text

    def enhanced_number_validation(self, extracted_number: str, original_text: str) -> tuple[str, bool]:
        """Enhanced validation and correction for extracted number section
        
        Args:
            extracted_number: Number extracted from regex groups
            original_text: Original text for context
            
        Returns:
            tuple: (corrected_number, is_valid)
        """
        try:
            # If number section is already valid, return as-is
            if self.validate_number_section(extracted_number):
                return extracted_number, True
            
            # Try one more round of corrections for edge cases
            corrected = extracted_number
            final_corrections = {
                "l": "1",    # Lowercase l missed in previous steps
                "o": "0",    # Lowercase o missed in previous steps
            }
            
            for char, digit in final_corrections.items():
                corrected = corrected.replace(char, digit)
            
            # Validate final result
            is_valid = self.validate_number_section(corrected)
            
            if is_valid and corrected != extracted_number:
                print(f"  🔧 Final number correction: '{extracted_number}' → '{corrected}'")
            
            return corrected, is_valid
            
        except Exception as e:
            print(f"Enhanced validation error: {e}")
            return extracted_number, False
    
    def parse_plate(self, raw_text: str) -> Dict[str, Any]:
        """Parse Indonesian license plate text into structured format"""
        
        # Clean the raw text (more aggressive cleaning)
        cleaned_text = raw_text.strip().upper()
        
        # **EXISTING: Apply area code corrections first**
        cleaned_text = self.correct_area_code_ocr(cleaned_text)
        
        # **NEW: Apply advanced text cleaning patterns**
        cleaned_text = self.advanced_text_cleaning(cleaned_text)
        
        # **NEW: Apply number section corrections**
        cleaned_text = self.correct_number_section_ocr(cleaned_text)
        
        # Fix common OCR mistakes (EXISTING)
        cleaned_text = cleaned_text.replace(';', '.')  # Fix semicolon to dot
        cleaned_text = cleaned_text.replace(':', '.')  # Fix colon to dot
        cleaned_text = cleaned_text.replace('|', '1')  # Fix pipe to 1
        
        # **MODIFY: Remove aggressive O→0 replacement (causes issues in series)**
        # OLD: cleaned_text = cleaned_text.replace('O', '0')  # Fix O to 0 in numbers
        # NEW: Skip this - O→0 now handled section-specifically in number corrections
        
        # Remove extra spaces (EXISTING)
        cleaned_text = ' '.join(cleaned_text.split())
        
        result = {
            "full_plate_number": "",
            "area_code": "",
            "area_name": "",
            "area_region": "",
            "number": "",
            "series": "",
            "registration_date": None
        }
        
        validation = {
            "format_valid": False,
            "area_code_valid": False,
            "number_valid": False,
            "series_valid": False,
            "date_valid": False,
            "validation_message": "Invalid format"
        }
        
        # Try each pattern
        for pattern in self.patterns:
            match = re.match(pattern, cleaned_text)
            if match:
                groups = match.groups()
                
                if len(groups) >= 3:  # At least area, number, series
                    area_code = groups[0]
                    raw_number = groups[1]
                    series = groups[2]
                    
                    # **NEW: Apply enhanced number validation and correction**
                    number, number_is_valid = self.enhanced_number_validation(raw_number, cleaned_text)
                    
                    # Check if date is present
                    registration_date = None
                    if len(groups) >= 5:  # With date
                        month = groups[3]
                        year = groups[4]
                        registration_date = self._parse_date(month, year)
                    
                    # Build result
                    result["area_code"] = area_code
                    result["number"] = number
                    result["series"] = series
                    result["full_plate_number"] = f"{area_code} {number} {series}"
                    
                    if registration_date:
                        result["registration_date"] = registration_date
                    
                    # Area code mapping
                    if area_code in self.area_codes:
                        result["area_name"] = self.area_codes[area_code]["name"]
                        result["area_region"] = self.area_codes[area_code]["region"]
                        validation["area_code_valid"] = True
                    
                    # Validation (updated for longer series)
                    validation["format_valid"] = True
                    validation["number_valid"] = len(number) >= 1 and len(number) <= 4
                    validation["series_valid"] = len(series) >= 1 and len(series) <= 4  # Extended to 4 letters
                    validation["date_valid"] = registration_date is not None if len(groups) >= 5 else True
                    
                    if all([validation["format_valid"], validation["area_code_valid"], 
                           validation["number_valid"], validation["series_valid"], validation["date_valid"]]):
                        validation["validation_message"] = "Valid Indonesian license plate format"
                    else:
                        validation["validation_message"] = "Some validation checks failed"
                    
                    break
        
        return result, validation
    
    def _parse_date(self, month: str, year: str) -> Optional[Dict[str, str]]:
        """Parse MM.YY registration date format"""
        try:
            if len(month) != 2 or len(year) != 2:
                return None
                
            month_int = int(month)
            year_int = int(year)
            
            if month_int < 1 or month_int > 12:
                return None
                
            # Convert YY to YYYY (assume 2000s)
            full_year = f"20{year}"
            
            # Calculate expiry (registration + 5 years)
            expiry_year = int(full_year) + 5
            expiry_month = month
            
            return {
                "month": month,
                "month_name": self.month_names.get(month, "Unknown"),
                "year": year,
                "full_year": full_year,
                "formatted_date": f"{month}.{year}",
                "expiry_estimate": f"{expiry_month}.{expiry_year}"
            }
            
        except (ValueError, KeyError):
            return None


class AngkatanManager:
    """Manage ANGKATAN (batch session) system"""
    
    def __init__(self, base_path: str = "results/complete_recognition"):
        self.base_path = Path(base_path)
        self.current_angkatan = None
        self.angkatan_path = None
        self.master_summary_path = self.base_path / "master_summary.json"
        
        # Ensure base directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def get_next_angkatan_number(self) -> int:
        """Scan existing angkatan folders and return next number"""
        existing_angkatan = []
        
        # Look for angkatan directories
        for folder in self.base_path.iterdir():
            if folder.is_dir() and folder.name.startswith('angkatan'):
                try:
                    number = int(folder.name.replace('angkatan', ''))
                    existing_angkatan.append(number)
                except ValueError:
                    continue
        
        if not existing_angkatan:
            return 1
        
        return max(existing_angkatan) + 1
    
    def create_new_angkatan(self) -> str:
        """Create new angkatan folder and return path"""
        next_number = self.get_next_angkatan_number()
        self.current_angkatan = next_number
        self.angkatan_path = self.base_path / f"angkatan{next_number}"
        
        # Create angkatan directory
        self.angkatan_path.mkdir(parents=True, exist_ok=True)
        
        return str(self.angkatan_path)
    
    def create_foto_folder(self, image_name: str, foto_number: int) -> str:
        """Create individual foto folder within current angkatan"""
        if self.angkatan_path is None:
            raise ValueError("Angkatan not initialized. Call create_new_angkatan() first.")
        
        # Clean filename (remove extension and special characters)
        clean_name = Path(image_name).stem.replace(' ', '_').replace('-', '_')
        
        # Create foto folder name
        foto_folder_name = f"foto{foto_number}_{clean_name}"
        foto_path = self.angkatan_path / foto_folder_name
        
        # Create foto folder and subfolders
        foto_path.mkdir(parents=True, exist_ok=True)
        (foto_path / "parsed_data").mkdir(exist_ok=True)
        (foto_path / "cropped_plates").mkdir(exist_ok=True)
        
        return str(foto_path)
    
    def get_existing_angkatan_summary(self) -> Dict[str, Any]:
        """Get summary of existing angkatan sessions"""
        existing_summary = []
        
        for i in range(1, self.get_next_angkatan_number()):
            angkatan_folder = self.base_path / f"angkatan{i}"
            if angkatan_folder.exists():
                existing_summary.append(f"angkatan{i}")
        
        return {
            "existing_sessions": existing_summary,
            "total_completed": len(existing_summary)
        }
    
    def save_angkatan_summary(self, results: List[Dict[str, Any]], start_time: datetime) -> str:
        """Save summary for current angkatan"""
        if self.angkatan_path is None or self.current_angkatan is None:
            raise ValueError("Angkatan not initialized")
        
        # Calculate statistics
        total_fotos = len(results)
        successful_recognitions = sum(1 for r in results if r.get("detection_success", False))
        success_rate = (successful_recognitions / total_fotos * 100) if total_fotos > 0 else 0
        total_plates = sum(r.get("num_plates_detected", 0) for r in results)
        
        # Area distribution
        area_distribution = {}
        plates_with_dates = 0
        plates_without_dates = 0
        registration_years = {}
        
        for result in results:
            # Handle nested structure
            detection_results = result.get("detection_results", result)
            plates = detection_results.get("license_plates", [])
            
            for plate in plates:
                parsed = plate.get("parsed_plate", {})
                
                # Area distribution
                if parsed.get("area_name") and parsed.get("area_code"):
                    area_key = f"{parsed['area_name']} ({parsed['area_code']})"
                    area_distribution[area_key] = area_distribution.get(area_key, 0) + 1
                
                # Date statistics
                if parsed.get("registration_date"):
                    plates_with_dates += 1
                    year = parsed["registration_date"].get("full_year")
                    if year:
                        registration_years[year] = registration_years.get(year, 0) + 1
                else:
                    plates_without_dates += 1
        
        # Build foto results summary
        foto_results = []
        for i, result in enumerate(results, 1):
            # Handle both old and new data structures
            image_name = result.get("image_name") or result.get("image_info", {}).get("image_name", f"unknown_{i}")
            detection_results = result.get("detection_results", result)  # Handle nested structure
            
            foto_results.append({
                "foto_number": i,
                "foto_name": f"foto{i}_{Path(image_name).stem}",
                "image_name": image_name,
                "plates_found": detection_results.get("num_plates_detected", 0),
                "success": detection_results.get("detection_success", False)
            })
        
        # Create summary object
        summary = {
            "angkatan_info": {
                "angkatan_number": self.current_angkatan,
                "timestamp": start_time.isoformat(),
                "total_fotos_processed": total_fotos,
                "successful_recognitions": successful_recognitions,
                "success_rate_percentage": round(success_rate, 1),
                "total_plates_found": total_plates
            },
            "foto_results": foto_results,
            "angkatan_statistics": {
                "area_distribution": area_distribution,
                "registration_years": registration_years,
                "plates_with_dates": plates_with_dates,
                "plates_without_dates": plates_without_dates
            }
        }
        
        # Save angkatan summary
        summary_path = self.angkatan_path / f"angkatan{self.current_angkatan}_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return str(summary_path)
    
    def update_master_summary(self, angkatan_summary: Dict[str, Any]) -> str:
        """Update master summary with new angkatan data"""
        # Load existing master summary
        master_data = {
            "system_info": {
                "last_updated": datetime.now().isoformat(),
                "total_angkatan_completed": 0,
                "latest_angkatan": f"angkatan{self.current_angkatan}"
            },
            "cumulative_statistics": {
                "total_images_processed": 0,
                "total_plates_found": 0,
                "cumulative_success_rate": 0.0,
                "all_area_distribution": {}
            },
            "angkatan_history": []
        }
        
        # Load existing data if file exists
        if self.master_summary_path.exists():
            try:
                with open(self.master_summary_path, 'r', encoding='utf-8') as f:
                    master_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass  # Use default structure
        
        # Add current angkatan to history
        angkatan_info = angkatan_summary["angkatan_info"]
        new_entry = {
            "angkatan": self.current_angkatan,
            "date": angkatan_info["timestamp"],
            "images": angkatan_info["total_fotos_processed"],
            "plates": angkatan_info["total_plates_found"]
        }
        
        # Remove existing entry for this angkatan (if any) and add new one
        master_data["angkatan_history"] = [
            entry for entry in master_data["angkatan_history"] 
            if entry["angkatan"] != self.current_angkatan
        ]
        master_data["angkatan_history"].append(new_entry)
        master_data["angkatan_history"].sort(key=lambda x: x["angkatan"])
        
        # Update system info
        master_data["system_info"]["total_angkatan_completed"] = len(master_data["angkatan_history"])
        master_data["system_info"]["latest_angkatan"] = f"angkatan{self.current_angkatan}"
        master_data["system_info"]["last_updated"] = datetime.now().isoformat()
        
        # Calculate cumulative statistics
        total_images = sum(entry["images"] for entry in master_data["angkatan_history"])
        total_plates = sum(entry["plates"] for entry in master_data["angkatan_history"])
        
        # Merge area distributions from all angkatan
        all_areas = master_data["cumulative_statistics"].get("all_area_distribution", {})
        current_areas = angkatan_summary["angkatan_statistics"]["area_distribution"]
        
        # Update area distribution (this is simplified - ideally we'd recalculate from all angkatan)
        for area, count in current_areas.items():
            all_areas[area] = all_areas.get(area, 0) + count
        
        # Calculate success rate (simplified)
        success_rate = 0.0
        if master_data["angkatan_history"]:
            # Load all angkatan summaries to calculate accurate success rate
            total_successful = 0
            for entry in master_data["angkatan_history"]:
                angkatan_num = entry["angkatan"]
                summary_file = self.base_path / f"angkatan{angkatan_num}" / f"angkatan{angkatan_num}_summary.json"
                if summary_file.exists():
                    try:
                        with open(summary_file, 'r', encoding='utf-8') as f:
                            ang_data = json.load(f)
                            total_successful += ang_data["angkatan_info"]["successful_recognitions"]
                    except:
                        pass
            
            success_rate = (total_successful / total_images * 100) if total_images > 0 else 0.0
        
        master_data["cumulative_statistics"] = {
            "total_images_processed": total_images,
            "total_plates_found": total_plates,
            "cumulative_success_rate": round(success_rate, 1),
            "all_area_distribution": all_areas
        }
        
        # Save master summary
        with open(self.master_summary_path, 'w', encoding='utf-8') as f:
            json.dump(master_data, f, indent=2, ensure_ascii=False)
        
        return str(self.master_summary_path)


class CompleteLicensePlateRecognizer:
    """Complete License Plate Recognition System"""
    
    def __init__(self, yolo_model_path: str = "models/final/best_model.pt"):
        """Initialize the complete recognition system"""
        
        # Initialize YOLO model
        self.yolo_model = YOLO(yolo_model_path)
        
        # Initialize EasyOCR reader
        self.ocr_reader = easyocr.Reader(['en'], gpu=True)
        
        # Initialize Indonesian parser
        self.parser = IndonesianLicensePlateParser()
        
        # Initialize AngkatanManager
        self.angkatan_manager = AngkatanManager()
        
        # Processing statistics
        self.stats = {
            "total_images": 0,
            "successful_recognitions": 0,
            "total_plates": 0,
            "plates_with_dates": 0,
            "plates_without_dates": 0,
            "area_code_distribution": {},
            "processing_times": []
        }
    
    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order corner points: top-left, top-right, bottom-right, bottom-left"""
        # Reshape to ensure proper format
        pts = pts.reshape(4, 2)
        
        # Initialize ordered points array
        ordered = np.zeros((4, 2), dtype=np.float32)
        
        # Sum and difference for corner identification
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        # Top-left: smallest sum, Top-right: smallest diff
        # Bottom-right: largest sum, Bottom-left: largest diff
        ordered[0] = pts[np.argmin(s)]     # top-left
        ordered[1] = pts[np.argmin(diff)]  # top-right  
        ordered[2] = pts[np.argmax(s)]     # bottom-right
        ordered[3] = pts[np.argmax(diff)]  # bottom-left
        
        return ordered

    def find_dest(self, pts: np.ndarray) -> np.ndarray:
        """Calculate destination rectangle dimensions for perspective correction"""
        (tl, tr, br, bl) = pts
        
        # Calculate width: max of top and bottom edge lengths
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        max_width = max(int(widthA), int(widthB))
        
        # Calculate height: max of left and right edge lengths  
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        max_height = max(int(heightA), int(heightB))
        
        # Ensure minimum dimensions for license plates
        min_width, min_height = 100, 30
        max_width = max(max_width, min_width)
        max_height = max(max_height, min_height)
        
        # Create destination rectangle
        destination = np.array([
            [0, 0],                           # top-left
            [max_width - 1, 0],              # top-right
            [max_width - 1, max_height - 1], # bottom-right
            [0, max_height - 1]              # bottom-left
        ], dtype=np.float32)
        
        return destination

    def detect_license_plate_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect 4 corners of license plate using contour detection"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection with multiple thresholds for robustness
            edges_list = []
            thresholds = [(50, 150), (30, 100), (70, 200)]
            
            for low, high in thresholds:
                edges = cv2.Canny(blurred, low, high)
                edges_list.append(edges)
            
            # Try edge detection results in order of preference
            for edges in edges_list:
                # Find contours
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours:
                    continue
                    
                # Sort contours by area (largest first)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                # Try to find rectangular contour
                for contour in contours[:5]:  # Check top 5 largest contours
                    # Calculate contour perimeter
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # License plate should have 4 corners
                    if len(approx) == 4:
                        # Verify it's a reasonable rectangle
                        area = cv2.contourArea(approx)
                        image_area = image.shape[0] * image.shape[1]
                        
                        # Area should be significant portion of image (license plate crop)
                        if area > image_area * 0.1:  # At least 10% of cropped image
                            return approx.reshape(4, 2)
            
            return None
            
        except Exception as e:
            print(f"Corner detection error: {e}")
            return None

    def correct_perspective(self, image: np.ndarray) -> np.ndarray:
        """Apply perspective correction to license plate crop
        
        Args:
            image: Cropped license plate image (np.ndarray BGR format)
            
        Returns:
            np.ndarray: Perspective-corrected image, or original if correction fails
        """
        try:
            # Validate input
            if image is None or image.size == 0:
                return image
                
            # Get image dimensions
            h, w = image.shape[:2]
            
            # Skip correction for very small images
            if w < 50 or h < 20:
                return image
                
            # Detect license plate corners
            corners = self.detect_license_plate_corners(image)
            
            if corners is None:
                # Fallback: try different approach or return original
                return self._fallback_perspective_correction(image)
                
            # Order corners properly
            ordered_corners = self.order_points(corners)
            
            # Calculate destination rectangle
            destination_corners = self.find_dest(ordered_corners)
            
            # Get perspective transformation matrix
            matrix = cv2.getPerspectiveTransform(ordered_corners, destination_corners)
            
            # Apply perspective correction
            dest_width = int(destination_corners[2][0])
            dest_height = int(destination_corners[2][1])
            
            corrected = cv2.warpPerspective(image, matrix, (dest_width, dest_height))
            
            # Validate result
            if corrected is None or corrected.size == 0:
                return image
                
            # Ensure result has reasonable dimensions
            corr_h, corr_w = corrected.shape[:2]
            if corr_w < 30 or corr_h < 10 or corr_w > w * 3 or corr_h > h * 3:
                return image
                
            return corrected
            
        except Exception as e:
            print(f"Perspective correction error: {e}")
            return image

    def _fallback_perspective_correction(self, image: np.ndarray) -> np.ndarray:
        """Fallback perspective correction when corner detection fails"""
        try:
            # Simple approach: try basic geometric transformations
            h, w = image.shape[:2]
            
            # Create slight perspective correction based on common license plate angles
            # Assume slight perspective distortion
            offset = max(1, min(w // 10, h // 4))  # Small offset based on image size
            
            # Source points (original corners with slight perspective)
            src_pts = np.float32([
                [offset, 0],           # top-left (slightly right)
                [w - 1, 0],           # top-right
                [w - 1, h - 1],       # bottom-right  
                [0, h - 1]            # bottom-left (slightly left)
            ])
            
            # Destination points (perfect rectangle)
            dst_pts = np.float32([
                [0, 0],               # top-left
                [w - offset, 0],      # top-right (compensate)
                [w - offset, h - 1],  # bottom-right
                [0, h - 1]            # bottom-left
            ])
            
            # Apply gentle perspective correction
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            corrected = cv2.warpPerspective(image, matrix, (w - offset, h))
            
            return corrected if corrected is not None and corrected.size > 0 else image
            
        except Exception as e:
            print(f"Fallback perspective correction error: {e}")
            return image

    def enhance_image_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Apply brightness/contrast enhancement (best method from Phase 1-4)"""
        
        # Convert to LAB color space for better brightness control
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Additional brightness/contrast adjustment
        alpha = 1.2  # Contrast control
        beta = 10    # Brightness control
        enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
        
        return enhanced
    
    def extract_text_with_ocr(self, image: np.ndarray) -> Tuple[str, float]:
        """Extract text using EasyOCR with confidence score"""
        
        try:
            # Run OCR
            results = self.ocr_reader.readtext(image)
            
            if not results:
                return "", 0.0
            
            # Combine all detected text with confidence weighting
            combined_text = ""
            total_confidence = 0.0
            
            for (bbox, text, confidence) in results:
                combined_text += text + " "
                total_confidence += confidence
            
            combined_text = combined_text.strip()
            avg_confidence = total_confidence / len(results) if results else 0.0
            
            return combined_text, avg_confidence * 100
            
        except Exception as e:
            print(f"OCR Error: {e}")
            return "", 0.0

    def test_confidence_thresholds(self, image_path: str):
        """Test multiple confidence thresholds untuk compare detection counts"""
        
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        results = {}
        
        image = cv2.imread(image_path)
        
        for conf in thresholds:
            yolo_results = self.yolo_model(image, conf=conf)
            detections = yolo_results[0].boxes
            detection_count = len(detections) if detections is not None else 0
            
            results[conf] = {
                'count': detection_count,
                'confidences': [float(det.conf[0]) for det in detections] if detections else []
            }
            
            print(f"Confidence {conf}: {detection_count} detections")
            if detections:
                conf_list = [f"{conf:.3f}" for conf in results[conf]['confidences']]
                print(f"  Individual confidences: {conf_list}")
        
        return results

    def verify_multiple_detection(self, image_path: str) -> dict:
        """Comprehensive verification of multiple plate detection"""
        
        print(f"\n🔍 MULTIPLE DETECTION VERIFICATION")
        print(f"Image: {os.path.basename(image_path)}")
        print("=" * 50)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not load image"}
        
        h, w = image.shape[:2]
        print(f"Image dimensions: {w} x {h}")
        
        # YOLO Detection with current settings
        yolo_results = self.yolo_model(image, conf=0.3)
        detections = yolo_results[0].boxes
        
        detection_count = len(detections) if detections is not None else 0
        print(f"YOLO detections: {detection_count}")
        
        if detection_count == 0:
            print("❌ No plates detected")
            return {"detection_count": 0, "plates": []}
        
        # Process each detection dengan detailed logging
        processed_plates = []
        successful_ocr = 0
        
        for i, detection in enumerate(detections):
            print(f"\n📋 Processing Detection {i+1}/{detection_count}")
            
            # Extract bbox and confidence
            box = detection.xyxy[0].cpu().numpy()
            confidence = float(detection.conf[0].cpu().numpy())
            x1, y1, x2, y2 = map(int, box)
            
            print(f"  YOLO confidence: {confidence:.3f}")
            print(f"  Bbox: ({x1}, {y1}) to ({x2}, {y2})")
            print(f"  Size: {x2-x1} x {y2-y1} pixels")
            
            # Crop license plate
            plate_crop = image[y1:y2, x1:x2]
            
            if plate_crop.size == 0:
                print(f"  ❌ Empty crop - skipping")
                continue
            
            # Apply Phase 1 + Phase 2 corrections
            corrected_crop = self.correct_perspective(plate_crop)
            enhanced_crop = self.enhance_image_for_ocr(corrected_crop)
            
            # OCR processing
            raw_text, ocr_confidence = self.extract_text_with_ocr(enhanced_crop)
            print(f"  OCR result: \"{raw_text}\" (confidence: {ocr_confidence:.1f}%)")
            
            # Indonesian parsing dengan Phase 1 + 2 corrections
            parsed_plate, validation = self.parser.parse_plate(raw_text)
            
            if validation["format_valid"] and validation["area_code_valid"]:
                successful_ocr += 1
                print(f"  ✅ Parsed: {parsed_plate['area_name']} - {parsed_plate['full_plate_number']}")
            else:
                print(f"  ⚠️  Parsing failed: {validation['validation_message']}")
            
            processed_plates.append({
                "detection_id": i,
                "yolo_confidence": confidence,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "ocr_text": raw_text,
                "ocr_confidence": ocr_confidence,
                "parsed_success": validation["format_valid"] and validation["area_code_valid"],
                "parsed_plate": parsed_plate if validation["format_valid"] else {}
            })
        
        print(f"\n📊 SUMMARY:")
        print(f"Total detections: {detection_count}")
        print(f"Successful OCR+Parsing: {successful_ocr}")
        print(f"Success rate: {(successful_ocr/detection_count*100):.1f}%" if detection_count > 0 else "0%")
        
        return {
            "detection_count": detection_count,
            "successful_parsing": successful_ocr,
            "success_rate": (successful_ocr/detection_count*100) if detection_count > 0 else 0,
            "plates": processed_plates
        }

    def optimize_detection_confidence(self, image: np.ndarray) -> float:
        """Determine optimal confidence threshold based on image analysis"""
        
        try:
            # Test multiple confidence levels
            test_confidences = [0.1, 0.2, 0.3, 0.4, 0.5]
            detection_counts = []
            
            for conf in test_confidences:
                # Multi-detection optimized parameters
                results = self.yolo_model(image, conf=conf, iou=0.3, max_det=50)
                count = len(results[0].boxes) if results[0].boxes is not None else 0
                detection_counts.append(count)
            
            # Find optimal confidence (balances detection count vs quality)
            # Strategy: Use lowest confidence that gives reasonable detections
            for i, count in enumerate(detection_counts):
                if count > 0 and count <= 5:  # Reasonable range: 1-5 plates per image
                    optimal_conf = test_confidences[i]
                    print(f"  🎯 Optimal confidence: {optimal_conf} ({count} detections)")
                    return optimal_conf
            
            # Fallback to default
            return 0.3
            
        except Exception as e:
            print(f"Confidence optimization error: {e}")
            return 0.3  # Default fallback

    def process_single_image_with_adaptive_detection(self, image_path: str, foto_number: int, foto_folder_path: str) -> Dict[str, Any]:
        """Enhanced version with adaptive confidence untuk multiple detection optimization"""
        
        start_time = time.time()
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {
                "angkatan_info": {
                    "angkatan_number": self.angkatan_manager.current_angkatan,
                    "foto_number": foto_number,
                    "angkatan_session": f"angkatan{self.angkatan_manager.current_angkatan}",
                    "foto_folder": Path(foto_folder_path).name
                },
                "image_info": {
                    "image_name": os.path.basename(image_path),
                    "image_path": image_path,
                    "processing_time_ms": 0
                },
                "detection_results": {
                    "detection_success": False,
                    "num_plates_detected": 0,
                    "license_plates": [],
                    "error": f"Could not load image: {image_path}"
                },
                "files_generated": {
                    "cropped_plates": [],
                    "annotated_image": None,
                    "parsed_data": None
                }
            }
        
        image_height, image_width = image.shape[:2]
        
        # **MODIFIED: Use fixed optimal parameters for multi-detection**
        optimal_conf = 0.1  # Use proven optimal confidence
        
        # **UPDATED: Multi-detection optimized parameters**
        yolo_results = self.yolo_model(image, conf=optimal_conf, iou=0.3, max_det=50)
        
        print(f"  🎯 Adaptive detection: conf={optimal_conf}, iou=0.3, max_det=50")
        detections = yolo_results[0].boxes
        
        print(f"  🎯 Using confidence {optimal_conf}: {len(detections) if detections else 0} detections")
        
        license_plates = []
        cropped_files = []
        
        if detections is not None and len(detections) > 0:
            for i, detection in enumerate(detections):
                # Extract bounding box
                box = detection.xyxy[0].cpu().numpy()
                confidence = detection.conf[0].cpu().numpy()
                
                x1, y1, x2, y2 = map(int, box)
                
                # Crop license plate region
                plate_crop = image[y1:y2, x1:x2]
                
                if plate_crop.size == 0:
                    continue
                
                # Apply perspective correction
                corrected_crop = self.correct_perspective(plate_crop)
                
                # Save corrected plate in foto-specific folder
                crop_filename = f"crop_det{i}.jpg"
                crop_path = Path(foto_folder_path) / "cropped_plates" / crop_filename
                cv2.imwrite(str(crop_path), corrected_crop)
                
                # Optional: Save original crop for comparison
                if corrected_crop is not plate_crop:  # Only if correction was applied
                    original_filename = f"original_det{i}.jpg"
                    original_path = Path(foto_folder_path) / "cropped_plates" / original_filename
                    cv2.imwrite(str(original_path), plate_crop)
                
                cropped_files.append(crop_filename)
                
                # Enhance image for OCR
                enhanced_crop = self.enhance_image_for_ocr(corrected_crop)
                
                # Extract text with OCR
                raw_text, ocr_confidence = self.extract_text_with_ocr(enhanced_crop)
                
                # Parse Indonesian license plate
                parsed_plate, validation = self.parser.parse_plate(raw_text)
                
                # Build detection result
                plate_result = {
                    "detection_id": i,
                    "raw_ocr_text": raw_text,
                    "ocr_confidence": round(ocr_confidence, 1),
                    "parsed_plate": parsed_plate,
                    "validation": validation,
                    "bbox": {
                        "x1": float(x1), "y1": float(y1),
                        "x2": float(x2), "y2": float(y2)
                    },
                    "yolo_confidence": float(confidence),
                    "adaptive_confidence_used": optimal_conf
                }
                
                license_plates.append(plate_result)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Save individual plate parsing data
        parsing_data = {
            "plates": license_plates,
            "processing_metadata": {
                "yolo_model": "models/final/best_model.pt",
                "ocr_engine": "EasyOCR + brightness_contrast",
                "adaptive_confidence": optimal_conf,
                "processing_time_ms": round(processing_time, 0)
            }
        }
        
        parsing_file = "plate_parsing_adaptive.json"
        parsing_path = Path(foto_folder_path) / "parsed_data" / parsing_file
        with open(parsing_path, 'w', encoding='utf-8') as f:
            json.dump(parsing_data, f, indent=2, ensure_ascii=False)
        
        # Create annotated image in foto folder
        annotated_file = "annotated_image_adaptive.jpg"
        self._create_annotated_image_angkatan(image, license_plates, foto_folder_path, annotated_file)
        
        # Build result matching the specified JSON structure
        result = {
            "angkatan_info": {
                "angkatan_number": self.angkatan_manager.current_angkatan,
                "foto_number": foto_number,
                "angkatan_session": f"angkatan{self.angkatan_manager.current_angkatan}",
                "foto_folder": Path(foto_folder_path).name
            },
            "image_info": {
                "image_name": os.path.basename(image_path),
                "image_path": image_path,
                "processing_time_ms": round(processing_time, 0)
            },
            "detection_results": {
                "detection_success": len(license_plates) > 0,
                "num_plates_detected": len(license_plates),
                "license_plates": license_plates,
                "adaptive_confidence_used": optimal_conf
            },
            "files_generated": {
                "cropped_plates": cropped_files,
                "annotated_image": annotated_file,
                "parsed_data": parsing_file
            }
        }
        
        # Save individual foto recognition result
        result_path = Path(foto_folder_path) / "recognition_result_adaptive.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result
    
    def process_single_image_with_angkatan(self, image_path: str, foto_number: int, foto_folder_path: str) -> Dict[str, Any]:
        """Process single image with ANGKATAN folder organization"""
        
        start_time = time.time()
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {
                "angkatan_info": {
                    "angkatan_number": self.angkatan_manager.current_angkatan,
                    "foto_number": foto_number,
                    "angkatan_session": f"angkatan{self.angkatan_manager.current_angkatan}",
                    "foto_folder": Path(foto_folder_path).name
                },
                "image_info": {
                    "image_name": os.path.basename(image_path),
                    "image_path": image_path,
                    "processing_time_ms": 0
                },
                "detection_results": {
                    "detection_success": False,
                    "num_plates_detected": 0,
                    "license_plates": [],
                    "error": f"Could not load image: {image_path}"
                },
                "files_generated": {
                    "cropped_plates": [],
                    "annotated_image": None,
                    "parsed_data": None
                }
            }
        
        image_height, image_width = image.shape[:2]
        
        # YOLO Detection - Multi-detection optimized
        yolo_results = self.yolo_model(image, conf=0.1, iou=0.3, max_det=50)
        detections = yolo_results[0].boxes
        
        # Enhanced logging for multi-detection
        detection_count = len(detections) if detections is not None else 0
        print(f"  🎯 Multi-detection: {detection_count} plates found (conf=0.1, iou=0.3)")
        
        license_plates = []
        cropped_files = []
        
        if detections is not None and len(detections) > 0:
            for i, detection in enumerate(detections):
                # Extract bounding box
                box = detection.xyxy[0].cpu().numpy()
                confidence = detection.conf[0].cpu().numpy()
                
                x1, y1, x2, y2 = map(int, box)
                
                # Crop license plate region
                plate_crop = image[y1:y2, x1:x2]
                
                if plate_crop.size == 0:
                    continue
                
                # Apply perspective correction
                corrected_crop = self.correct_perspective(plate_crop)
                
                # Save corrected plate in foto-specific folder
                crop_filename = f"crop_det{i}.jpg"
                crop_path = Path(foto_folder_path) / "cropped_plates" / crop_filename
                cv2.imwrite(str(crop_path), corrected_crop)
                
                # Optional: Save original crop for comparison
                if corrected_crop is not plate_crop:  # Only if correction was applied
                    original_filename = f"original_det{i}.jpg"
                    original_path = Path(foto_folder_path) / "cropped_plates" / original_filename
                    cv2.imwrite(str(original_path), plate_crop)
                
                cropped_files.append(crop_filename)
                
                # Enhance image for OCR
                enhanced_crop = self.enhance_image_for_ocr(corrected_crop)
                
                # Extract text with OCR
                raw_text, ocr_confidence = self.extract_text_with_ocr(enhanced_crop)
                
                # Parse Indonesian license plate
                parsed_plate, validation = self.parser.parse_plate(raw_text)
                
                # Build detection result
                plate_result = {
                    "detection_id": i,
                    "raw_ocr_text": raw_text,
                    "ocr_confidence": round(ocr_confidence, 1),
                    "parsed_plate": parsed_plate,
                    "validation": validation,
                    "bbox": {
                        "x1": float(x1), "y1": float(y1),
                        "x2": float(x2), "y2": float(y2)
                    },
                    "yolo_confidence": float(confidence)
                }
                
                license_plates.append(plate_result)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Save individual plate parsing data
        parsing_data = {
            "plates": license_plates,
            "processing_metadata": {
                "yolo_model": "models/final/best_model.pt",
                "ocr_engine": "EasyOCR + brightness_contrast",
                "processing_time_ms": round(processing_time, 0)
            }
        }
        
        parsing_file = "plate_parsing.json"
        parsing_path = Path(foto_folder_path) / "parsed_data" / parsing_file
        with open(parsing_path, 'w', encoding='utf-8') as f:
            json.dump(parsing_data, f, indent=2, ensure_ascii=False)
        
        # Create annotated image in foto folder
        annotated_file = "annotated_image.jpg"
        self._create_annotated_image_angkatan(image, license_plates, foto_folder_path, annotated_file)
        
        # Build result matching the specified JSON structure
        result = {
            "angkatan_info": {
                "angkatan_number": self.angkatan_manager.current_angkatan,
                "foto_number": foto_number,
                "angkatan_session": f"angkatan{self.angkatan_manager.current_angkatan}",
                "foto_folder": Path(foto_folder_path).name
            },
            "image_info": {
                "image_name": os.path.basename(image_path),
                "image_path": image_path,
                "processing_time_ms": round(processing_time, 0)
            },
            "detection_results": {
                "detection_success": len(license_plates) > 0,
                "num_plates_detected": len(license_plates),
                "license_plates": license_plates
            },
            "files_generated": {
                "cropped_plates": cropped_files,
                "annotated_image": annotated_file,
                "parsed_data": parsing_file
            }
        }
        
        # Save individual foto recognition result
        result_path = Path(foto_folder_path) / "recognition_result.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result
    
    def process_single_image(self, image_path: str) -> Dict[str, Any]:
        """Process a single image through the complete pipeline"""
        
        start_time = time.time()
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {
                "image_name": os.path.basename(image_path),
                "image_path": image_path,
                "processing_time_ms": 0,
                "detection_success": False,
                "num_plates_detected": 0,
                "license_plates": [],
                "error": f"Could not load image: {image_path}"
            }
        
        image_height, image_width = image.shape[:2]
        
        # YOLO Detection - Multi-detection optimized
        yolo_results = self.yolo_model(image, conf=0.1, iou=0.3, max_det=50)
        detections = yolo_results[0].boxes
        
        # Enhanced logging for multi-detection
        detection_count = len(detections) if detections is not None else 0
        print(f"🎯 Multi-detection: {detection_count} plates found (conf=0.1, iou=0.3)")
        
        license_plates = []
        
        if detections is not None and len(detections) > 0:
            for i, detection in enumerate(detections):
                # Extract bounding box
                box = detection.xyxy[0].cpu().numpy()
                confidence = detection.conf[0].cpu().numpy()
                
                x1, y1, x2, y2 = map(int, box)
                
                # Crop license plate region
                plate_crop = image[y1:y2, x1:x2]
                
                if plate_crop.size == 0:
                    continue
                
                # Apply perspective correction
                corrected_crop = self.correct_perspective(plate_crop)
                
                # Save cropped plate
                crop_filename = f"crop_{i:02d}_{os.path.splitext(os.path.basename(image_path))[0]}_det{i}.jpg"
                crop_path = os.path.join("results/complete_recognition/cropped_plates", crop_filename)
                cv2.imwrite(crop_path, corrected_crop)  # Save corrected version
                
                # Enhance image for OCR
                enhanced_crop = self.enhance_image_for_ocr(corrected_crop)
                
                # Extract text with OCR
                raw_text, ocr_confidence = self.extract_text_with_ocr(enhanced_crop)
                
                # Parse Indonesian license plate
                parsed_plate, validation = self.parser.parse_plate(raw_text)
                
                # Build detection result
                plate_result = {
                    "detection_id": i,
                    "raw_ocr_text": raw_text,
                    "ocr_confidence": round(ocr_confidence, 1),
                    "parsed_plate": parsed_plate,
                    "validation": validation,
                    "detection_info": {
                        "yolo_confidence": round(float(confidence), 3),
                        "bbox": {
                            "x1": round(float(x1), 2),
                            "y1": round(float(y1), 2), 
                            "x2": round(float(x2), 2),
                            "y2": round(float(y2), 2)
                        },
                        "ocr_method": "brightness_contrast + EasyOCR",
                        "enhancement_applied": "brightness_contrast"
                    }
                }
                
                license_plates.append(plate_result)
                
                # Save individual plate parsing
                plate_json_path = os.path.join("results/complete_recognition/parsed_data", 
                                             f"plate_{i:02d}_{os.path.splitext(os.path.basename(image_path))[0]}_det{i}.json")
                with open(plate_json_path, 'w', encoding='utf-8') as f:
                    json.dump(plate_result, f, indent=2, ensure_ascii=False)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Build result
        result = {
            "image_name": os.path.basename(image_path),
            "image_path": image_path,
            "image_dimensions": [image_height, image_width],
            "processing_time_ms": round(processing_time, 0),
            "detection_success": len(license_plates) > 0,
            "num_plates_detected": len(license_plates),
            "license_plates": license_plates
        }
        
        # Create annotated image
        self._create_annotated_image(image, license_plates, image_path)
        
        return result
    
    def verify_multi_detection_fix(self, image_path: str) -> dict:
        """Verify multiple detection fix with before/after comparison"""
        
        print(f"\n🔍 MULTI-DETECTION FIX VERIFICATION")
        print(f"Image: {os.path.basename(image_path)}")
        print("=" * 50)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not load image"}
        
        # Test OLD parameters (simulated)
        print("📊 OLD Parameters (simulated):")
        old_results = self.yolo_model(image, conf=0.3)  # Missing iou, defaults to 0.7
        old_detections = old_results[0].boxes
        old_count = len(old_detections) if old_detections is not None else 0
        print(f"  conf=0.3, iou=0.7 (default): {old_count} detections")
        
        # Test NEW parameters
        print("\n🎯 NEW Parameters:")
        new_results = self.yolo_model(image, conf=0.1, iou=0.3, max_det=50)
        new_detections = new_results[0].boxes
        new_count = len(new_detections) if new_detections is not None else 0
        print(f"  conf=0.1, iou=0.3, max_det=50: {new_count} detections")
        
        # Show improvement
        improvement = new_count - old_count
        print(f"\n📈 IMPROVEMENT: +{improvement} additional plates detected")
        
        if new_detections:
            confidences = [f"{float(det.conf[0]):.3f}" for det in new_detections]
            print(f"🎯 Detection confidences: {confidences}")
        
        return {
            "old_detection_count": old_count,
            "new_detection_count": new_count,
            "improvement": improvement,
            "success": new_count > old_count
        }
    
    def _create_annotated_image(self, image: np.ndarray, license_plates: List[Dict], image_path: str):
        """Create annotated image with bounding boxes and parsed text"""
        
        annotated = image.copy()
        
        for plate in license_plates:
            bbox = plate["detection_info"]["bbox"]
            x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare text
            parsed = plate["parsed_plate"]
            if parsed["full_plate_number"]:
                display_text = parsed["full_plate_number"]
                if parsed["registration_date"]:
                    display_text += f" {parsed['registration_date']['formatted_date']}"
            else:
                display_text = plate["raw_ocr_text"]
            
            # Add confidence
            confidence = plate["detection_info"]["yolo_confidence"]
            label = f"{display_text} ({confidence:.2f})"
            
            # Draw text background
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1-text_height-10), (x1+text_width, y1), (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(annotated, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Save annotated image
        annotated_filename = f"annotated_{os.path.basename(image_path)}"
        annotated_path = os.path.join("results/complete_recognition/annotated_images", annotated_filename)
        cv2.imwrite(annotated_path, annotated)
    
    def _create_annotated_image_angkatan(self, image: np.ndarray, license_plates: List[Dict], foto_folder_path: str, filename: str):
        """Create annotated image with bounding boxes for angkatan system"""
        
        annotated = image.copy()
        
        for plate in license_plates:
            bbox = plate["bbox"]
            x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare text
            parsed = plate["parsed_plate"]
            if parsed["full_plate_number"]:
                display_text = parsed["full_plate_number"]
                if parsed["registration_date"]:
                    display_text += f" {parsed['registration_date']['formatted_date']}"
            else:
                display_text = plate["raw_ocr_text"]
            
            # Add confidence
            confidence = plate["yolo_confidence"]
            label = f"{display_text} ({confidence:.2f})"
            
            # Draw text background
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1-text_height-10), (x1+text_width, y1), (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(annotated, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Save annotated image in foto folder
        annotated_path = Path(foto_folder_path) / filename
        cv2.imwrite(str(annotated_path), annotated)
    
    def process_batch_with_angkatan(self, image_folder: str = "test_images") -> Dict[str, Any]:
        """Process batch of images with ANGKATAN system organization"""
        
        start_time = datetime.now()
        
        print("ANGKATAN LICENSE PLATE RECOGNITION SYSTEM")
        print("==========================================")
        
        # Check existing angkatan
        existing_summary = self.angkatan_manager.get_existing_angkatan_summary()
        if existing_summary["existing_sessions"]:
            print(f"Scanning existing angkatan...")
            print(f"Found existing: {', '.join(existing_summary['existing_sessions'])} ({existing_summary['total_completed']} completed sessions)")
        else:
            print("No previous angkatan sessions found.")
        
        # Create new angkatan
        angkatan_path = self.angkatan_manager.create_new_angkatan()
        print(f"Starting new session: ANGKATAN {self.angkatan_manager.current_angkatan}")
        print()
        
        # Find images (prevent duplicates)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files_set = set()
        
        if os.path.exists(image_folder):
            for ext in image_extensions:
                # Use case-insensitive matching to prevent duplicates
                image_files_set.update(Path(image_folder).glob(f"*{ext}"))
                image_files_set.update(Path(image_folder).glob(f"*{ext.upper()}"))
        
        # Convert to sorted list for consistent processing order
        image_files = sorted(list(image_files_set), key=lambda x: x.name.lower())
        
        if not image_files:
            print(f"No images found in {image_folder}")
            return {}
        
        print(f"Found {len(image_files)} images in {image_folder}/")
        print(f"Creating: {Path(angkatan_path).relative_to('results/complete_recognition')}/")
        print()
        
        # Process images with angkatan system
        recognition_results = []
        
        for i, image_file in enumerate(image_files, 1):
            print(f"[ANGKATAN {self.angkatan_manager.current_angkatan} - FOTO {i}/{len(image_files)}] Processing: {image_file.name}...")
            
            # Create foto folder
            foto_folder_path = self.angkatan_manager.create_foto_folder(image_file.name, i)
            print(f"  📁 Creating: {Path(foto_folder_path).relative_to('results/complete_recognition')}/")
            
            # Process image with angkatan system
            result = self.process_single_image_with_angkatan(str(image_file), i, foto_folder_path)
            recognition_results.append(result)
            
            # Print processing results
            detection_results = result["detection_results"]
            if detection_results["detection_success"]:
                plates = detection_results["license_plates"]
                
                if len(plates) == 1:
                    plate = plates[0]
                    yolo_conf = plate["yolo_confidence"]
                    ocr_conf = plate["ocr_confidence"]
                    print(f"  ✓ YOLO: 1 plate detected ({yolo_conf:.3f} confidence)")
                    print(f"  ✓ OCR: \"{plate['raw_ocr_text']}\" ({ocr_conf:.1f}% confidence)")
                    
                    parsed = plate["parsed_plate"]
                    if parsed["area_name"]:
                        date_info = ""
                        if parsed["registration_date"]:
                            date_info = f", Date={parsed['registration_date']['month_name']}-{parsed['registration_date']['full_year']}"
                        print(f"  ✓ Parsed: Area={parsed['area_name']}, Number={parsed['number']}, Series={parsed['series']}{date_info}")
                    else:
                        print(f"  ⚠ Parsing incomplete: {plate['validation']['validation_message']}")
                        
                else:  # Multiple plates
                    confidences = [f'{p["yolo_confidence"]:.3f}' for p in plates]
                    print(f"  ✓ YOLO: {len(plates)} plates detected ({', '.join(confidences)} confidence)")
                    for j, plate in enumerate(plates, 1):
                        ocr_conf = plate["ocr_confidence"]
                        print(f"  ✓ OCR Plate {j}: \"{plate['raw_ocr_text']}\" ({ocr_conf:.1f}% confidence)")
                        
                        parsed = plate["parsed_plate"]
                        if parsed["area_name"]:
                            date_info = ""
                            if parsed["registration_date"]:
                                date_info = f", Date={parsed['registration_date']['month_name']}-{parsed['registration_date']['full_year']}"
                            print(f"  ✓ Parsed Plate {j}: Area={parsed['area_name']}, Number={parsed['number']}, Series={parsed['series']}{date_info}")
                        else:
                            print(f"  ⚠ Parsing Plate {j} incomplete: {plate['validation']['validation_message']}")
                
                print(f"  💾 Saved to: {Path(foto_folder_path).relative_to('results/complete_recognition')}/")
            else:
                print(f"  ✗ No license plates detected")
            
            print()
        
        # Generate angkatan summary
        summary_path = self.angkatan_manager.save_angkatan_summary(recognition_results, start_time)
        
        # Load and display angkatan summary
        with open(summary_path, 'r', encoding='utf-8') as f:
            angkatan_summary = json.load(f)
        
        # Update master summary
        master_path = self.angkatan_manager.update_master_summary(angkatan_summary)
        
        # Print final summary
        print(f"ANGKATAN {self.angkatan_manager.current_angkatan} COMPLETE!")
        print("============================================")
        
        angkatan_info = angkatan_summary["angkatan_info"]
        print(f"Angkatan: {angkatan_info['angkatan_number']}")
        print(f"Total images processed: {angkatan_info['total_fotos_processed']}")
        print(f"Successful recognitions: {angkatan_info['successful_recognitions']} ({angkatan_info['success_rate_percentage']:.1f}%)")
        print(f"Total license plates found: {angkatan_info['total_plates_found']}")
        
        stats = angkatan_summary["angkatan_statistics"]
        print(f"- Plates with registration dates: {stats['plates_with_dates']}")
        print(f"- Plates without dates: {stats['plates_without_dates']}")
        print()
        
        if stats["area_distribution"]:
            print(f"ANGKATAN {self.angkatan_manager.current_angkatan} STATISTICS:")
            for area, count in sorted(stats["area_distribution"].items()):
                print(f"{area}: {count} plate{'s' if count > 1 else ''}")
            
            if stats["registration_years"]:
                print(f"Registration years: {', '.join([f'{year}({count})' for year, count in sorted(stats['registration_years'].items())])}")
        
        print()
        print("FILES CREATED:")
        print(f"📁 angkatan{self.angkatan_manager.current_angkatan}/ - Main angkatan folder")
        print(f"📄 angkatan{self.angkatan_manager.current_angkatan}_summary.json - Angkatan summary")
        
        for result in recognition_results:
            foto_folder = result["angkatan_info"]["foto_folder"]
            print(f"📁 {foto_folder}/ - Individual foto results")
        
        print(f"📄 master_summary.json - Updated master summary")
        print()
        
        # Load and display cumulative statistics
        with open(master_path, 'r', encoding='utf-8') as f:
            master_data = json.load(f)
        
        cumulative = master_data["cumulative_statistics"]
        print("CUMULATIVE STATISTICS:")
        print(f"Total angkatan completed: {master_data['system_info']['total_angkatan_completed']}")
        print(f"Total images processed: {cumulative['total_images_processed']}")
        print(f"Total license plates found: {cumulative['total_plates_found']}")
        
        if cumulative["all_area_distribution"]:
            most_common = max(cumulative["all_area_distribution"].items(), key=lambda x: x[1])
            print(f"Most common area: {most_common[0]} ({most_common[1]} plates)")
        
        print()
        print(f"Next run will be: ANGKATAN {self.angkatan_manager.current_angkatan + 1}")
        
        return {
            "angkatan_summary": angkatan_summary,
            "master_summary": master_data,
            "recognition_results": recognition_results
        }
    
    def process_batch(self, image_folder: str = "test_images") -> Dict[str, Any]:
        """Process batch of images and generate comprehensive results"""
        
        print("COMPLETE LICENSE PLATE RECOGNITION - YOLO + OCR + PARSING")
        print("==========================================================")
        print("Pipeline: YOLO Detection → OCR Optimization → Indonesian Parsing")
        print("Models: best_model.pt (99.4% mAP) + EasyOCR (99.6% accuracy)")
        print()
        
        # Find images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        if os.path.exists(image_folder):
            for ext in image_extensions:
                image_files.extend(Path(image_folder).glob(f"*{ext}"))
                image_files.extend(Path(image_folder).glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No images found in {image_folder}")
            return {}
        
        print(f"Scanning {image_folder}/ folder...")
        print(f"Found {len(image_files)} image files")
        print()
        
        # Process images
        recognition_results = []
        total_plates = 0
        successful_recognitions = 0
        plates_with_dates = 0
        area_distribution = {}
        
        batch_start_time = time.time()
        
        for i, image_file in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] Processing: {image_file.name}...")
            
            result = self.process_single_image(str(image_file))
            recognition_results.append(result)
            
            # Update statistics
            if result["detection_success"]:
                successful_recognitions += 1
                total_plates += result["num_plates_detected"]
                
                for plate in result["license_plates"]:
                    parsed = plate["parsed_plate"]
                    
                    # Count dates
                    if parsed["registration_date"]:
                        plates_with_dates += 1
                    
                    # Area distribution
                    if parsed["area_name"]:
                        area_key = f"{parsed['area_name']} ({parsed['area_code']})"
                        area_distribution[area_key] = area_distribution.get(area_key, 0) + 1
                    
                    # Print summary
                    yolo_conf = plate["detection_info"]["yolo_confidence"]
                    ocr_conf = plate["ocr_confidence"]
                    
                    print(f"  ✓ YOLO: 1 plate detected ({yolo_conf:.3f} confidence)")
                    print(f"  ✓ OCR: \"{plate['raw_ocr_text']}\" ({ocr_conf:.1f}% confidence)")
                    
                    if parsed["area_name"]:
                        date_info = ""
                        if parsed["registration_date"]:
                            date_info = f", Date={parsed['registration_date']['month_name']}-{parsed['registration_date']['full_year']}"
                        print(f"  ✓ Parsed: Area={parsed['area_name']}, Number={parsed['number']}, Series={parsed['series']}{date_info}")
                    else:
                        print(f"  ⚠ Parsing incomplete: {plate['validation']['validation_message']}")
            else:
                print(f"  ✗ No license plates detected")
            
            print()
        
        total_processing_time = (time.time() - batch_start_time) * 1000
        avg_processing_time = total_processing_time / len(image_files) if image_files else 0
        
        # Build comprehensive results
        batch_info = {
            "timestamp": datetime.now().isoformat(),
            "total_images_processed": len(image_files),
            "successful_recognitions": successful_recognitions,
            "success_rate_percentage": round((successful_recognitions / len(image_files)) * 100, 1) if image_files else 0,
            "pipeline_components": ["YOLO_Detection", "OCR_Optimization", "Indonesian_Parsing"],
            "average_processing_time_ms": round(avg_processing_time, 0),
            "model_info": {
                "yolo_model": "models/final/best_model.pt",
                "yolo_performance": "99.4% mAP",
                "ocr_engine": "EasyOCR + brightness_contrast",
                "ocr_accuracy": "99.6%"
            }
        }
        
        results = {
            "batch_info": batch_info,
            "recognition_results": recognition_results
        }
        
        # Save results
        results_path = "results/complete_recognition/complete_recognition_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("BATCH COMPLETE!")
        print("==========================================================")
        print(f"Total images processed: {len(image_files)}")
        print(f"Successful recognitions: {successful_recognitions} ({batch_info['success_rate_percentage']:.1f}%)")
        print(f"Total license plates found: {total_plates}")
        print(f"- Plates with registration dates: {plates_with_dates}")
        print(f"- Plates without dates: {total_plates - plates_with_dates}")
        print(f"Average processing time: {avg_processing_time/1000:.1f} seconds per image")
        print(f"Results saved to: {results_path}")
        print()
        
        if area_distribution:
            print("PARSING STATISTICS:")
            for area, count in sorted(area_distribution.items()):
                print(f"{area}: {count} plate{'s' if count > 1 else ''}")
        
        return results


def main():
    """Main execution function"""
    
    # Initialize recognizer
    recognizer = CompleteLicensePlateRecognizer()
    
    # Process batch with ANGKATAN system
    results = recognizer.process_batch_with_angkatan("test_images")
    
    return results


if __name__ == "__main__":
    main()