#!/usr/bin/env python3
"""
SISTEM PENGENALAN PLAT NOMOR KENDARAAN
=====================================
Pipeline: YOLO Detection → Kontras Maksimal → OCR → Koreksi Manual
"""

import os
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr


class AngkatanManager:
    """manajemen folder angkatan untuk debugging"""
    
    def __init__(self, base_path: str = "results"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
    def get_next_angkatan_number(self) -> int:
        """cari nomor angkatan berikutnya"""
        existing_nums = []
        
        for folder in self.base_path.iterdir():
            if folder.is_dir() and folder.name.startswith('angkatan'):
                try:
                    num = int(folder.name.replace('angkatan', ''))
                    existing_nums.append(num)
                except ValueError:
                    continue
        
        return max(existing_nums) + 1 if existing_nums else 1
    
    def create_angkatan_folder(self) -> Path:
        """buat folder angkatan baru"""
        angkatan_num = self.get_next_angkatan_number()
        angkatan_path = self.base_path / f"angkatan{angkatan_num}"
        angkatan_path.mkdir(exist_ok=True)
        return angkatan_path
    
    def create_foto_folder(self, angkatan_path: Path, image_name: str, foto_num: int) -> Path:
        """buat folder foto individual dalam angkatan"""
        clean_name = Path(image_name).stem.replace(' ', '_')
        foto_folder = angkatan_path / f"foto_{foto_num:02d}_{clean_name}"
        foto_folder.mkdir(exist_ok=True)
        return foto_folder


class MinimalPlateRecognizer:
    """sistem pengenalan plat nomor minimal - fokus akurasi maksimal"""
    
    def __init__(self, yolo_model_path: str = "models/final/best_model.pt"):
        """inisialisasi recognizer dengan yolo dan ocr"""
        self.yolo_model = YOLO(yolo_model_path)
        self.ocr_reader = easyocr.Reader(['en'], gpu=True)
        self.angkatan_manager = AngkatanManager()
        
        # mapping koreksi ocr untuk kesalahan umum pada bagian nomor
        self.number_corrections = {
            'L': '4',    # l -> 4 (kesalahan paling umum)
            'I': '1',    # i -> 1  
            'O': '0',    # o -> 0 (pada nomor saja)
            'S': '5',    # s -> 5
            'G': '6',    # g -> 6 (pada nomor)
            'Z': '2',    # z -> 2
            'B': '8',    # b -> 8 (pada nomor)
            'T': '7',    # t -> 7
        }
        
        # kode area valid indonesia (yang umum)
        self.valid_area_codes = {
            'B', 'D', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'W', 'Z',
            'AA', 'AB', 'AD', 'AE', 'AG', 'BA', 'BB', 'BE', 'BG', 'BH', 'BK', 'BL', 'BM', 
            'BN', 'BP', 'BR', 'BS', 'BT', 'BZ', 'DA', 'DB', 'DD', 'DE', 'DG', 'DH', 'DK', 
            'DL', 'DM', 'DN', 'DP', 'DR', 'DS', 'DT', 'DZ'
        }
    
    def maximize_contrast_for_ocr(self, crop: np.ndarray) -> np.ndarray:
        """terapkan peningkatan kontras maksimal untuk akurasi ocr optimal"""
        try:
            # konversi ke grayscale
            if len(crop.shape) == 3:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = crop.copy()
            
            # terapkan clahe (contrast limited adaptive histogram equalization)
            clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(gray)
            
            # binary threshold untuk plat putih dengan teks hitam
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # operasi morfologi untuk pembersihan noise
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # denoising tambahan
            cleaned = cv2.medianBlur(cleaned, 3)
            
            return cleaned
            
        except Exception as e:
            print(f"error kontras enhancement: {e}")
            return crop
    
    def validate_indonesian_area_code(self, area_code: str, confidence: float) -> str:
        """validasi dan perbaiki kesalahan ocr kode area umum"""
        try:
            # handle confusion g/6 - g adalah kode jakarta yang valid, 6 bukan
            if area_code == "6" and confidence < 70:
                if "G" in self.valid_area_codes:
                    print(f"  koreksi kode area: 6 -> g (confidence rendah: {confidence:.1f}%)")
                    return "G"
            
            # handle confusion 8/b - b lebih umum daripada 8 sebagai kode area
            if area_code == "8" and confidence < 70:
                if "B" in self.valid_area_codes:
                    print(f"  koreksi kode area: 8 -> b (confidence rendah: {confidence:.1f}%)")
                    return "B"
            
            return area_code
            
        except Exception as e:
            print(f"error validasi kode area: {e}")
            return area_code
    
    def extract_text_with_ocr(self, enhanced_crop: np.ndarray) -> Tuple[str, float]:
        """ekstrak teks menggunakan easyocr pada gambar kontras maksimal"""
        try:
            # jalankan ocr pada gambar binary yang ditingkatkan
            results = self.ocr_reader.readtext(enhanced_crop)
            
            if not results:
                return "", 0.0
            
            # gabungkan semua teks yang terdeteksi dengan pembobotan confidence
            combined_text = ""
            total_confidence = 0.0
            
            for (bbox, text, confidence) in results:
                combined_text += text + " "
                total_confidence += confidence
            
            combined_text = combined_text.strip()
            avg_confidence = total_confidence / len(results) if results else 0.0
            
            return combined_text, avg_confidence * 100
            
        except Exception as e:
            print(f"error ocr: {e}")
            return "", 0.0
    
    def manual_text_correction(self, raw_text: str, ocr_confidence: float = 100.0) -> str:
        """terapkan koreksi manual untuk kesalahan ocr umum pada plat indonesia"""
        try:
            corrected_text = raw_text.strip().upper()
            
            # pisahkan ke bagian: [area] [nomor] [seri]
            parts = corrected_text.split()
            
            if len(parts) < 3:
                return corrected_text
            
            area_part = parts[0]      # terapkan validasi kode area
            number_part = parts[1]    # terapkan koreksi nomor
            series_part = parts[2]    # biarkan seri tidak berubah (abc, coc, dll)
            
            # validasi kode area - perbaiki kesalahan ocr umum
            corrected_area = self.validate_indonesian_area_code(area_part, ocr_confidence)
            
            # terapkan koreksi hanya pada bagian nomor
            corrected_number = number_part
            corrections_applied = []
            
            for wrong_char, correct_char in self.number_corrections.items():
                if wrong_char in corrected_number:
                    old_number = corrected_number
                    corrected_number = corrected_number.replace(wrong_char, correct_char)
                    if old_number != corrected_number:
                        corrections_applied.append(f"{wrong_char}->{correct_char}")
            
            # rekonstruksi teks
            result = f"{corrected_area} {corrected_number} {series_part}"
            
            # print koreksi jika ada yang diterapkan
            if corrections_applied:
                print(f"  koreksi nomor: {', '.join(corrections_applied)} pada '{number_part}' -> '{corrected_number}'")
            
            return result
            
        except Exception as e:
            print(f"error koreksi manual: {e}")
            return raw_text
    
    def simple_parse(self, corrected_text: str) -> Dict:
        """parsing sederhana tanpa mapping kota - fokus validasi format saja"""
        try:
            # bersihkan teks lebih agresif sebelum parsing
            cleaned = corrected_text.strip().upper()
            
            # hapus pola tanggal terlebih dulu (jj.28, 05.28, 12,28, dll)
            cleaned = re.sub(r'\d{2}[\.\,:]\d{2}', '', cleaned)
            
            # koreksi kode area: digit tunggal "8" di awal -> "b"
            if re.match(r'^8\s+', cleaned):
                cleaned = re.sub(r'^8\s+', 'B ', cleaned)
                print(f"  koreksi kode area: 8 -> b")
            
            # penyisipan spasi pintar untuk kasus seperti "3877eue" -> "3877 eue"
            cleaned = re.sub(r'(\d{3,4})([A-Z]{2,3})', r'\1 \2', cleaned)
            
            # hapus karakter noise ocr umum
            noise_chars = ['*', '@', '#', '$', '%', '&', '(', ')', '+', '=', '{', '}', '[', ']', '|', '\\', '/', '?', '<', '>', '~', '`']
            for char in noise_chars:
                cleaned = cleaned.replace(char, '')
            
            # hapus tanda baca yang tersisa kecuali spasi
            cleaned = re.sub(r'[^\w\s]', '', cleaned)
            
            # normalisasi beberapa spasi menjadi spasi tunggal
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            # pisahkan berdasarkan spasi dan coba identifikasi komponen plat nomor
            parts = cleaned.split()
            
            if len(parts) >= 3:
                # coba cari kode area (1-2 huruf di awal)
                area_candidate = parts[0]
                
                # cari bagian nomor (harus digit)
                number_candidate = None
                series_candidate = None
                
                for i, part in enumerate(parts[1:], 1):
                    # cari bagian numerik
                    if re.match(r'^\d{1,4}$', part):
                        number_candidate = part
                        # bagian selanjutnya harus seri (huruf)
                        if i + 1 < len(parts):
                            next_part = parts[i + 1]
                            if re.match(r'^[A-Z]{1,4}$', next_part):
                                series_candidate = next_part
                                break
                
                # jika kita menemukan semua komponen, validasi
                if (area_candidate and number_candidate and series_candidate and
                    re.match(r'^[A-Z\d]{1,2}$', area_candidate) and
                    re.match(r'^\d{1,4}$', number_candidate) and
                    re.match(r'^[A-Z]{1,4}$', series_candidate)):
                    
                    reconstructed = f"{area_candidate} {number_candidate} {series_candidate}"
                    
                    return {
                        "area_code": area_candidate,
                        "number": number_candidate,
                        "series": series_candidate,
                        "full_plate": reconstructed,
                        "valid": True,
                        "format": "Indonesian license plate",
                        "original_text": corrected_text
                    }
            
            # jika parsing gagal, coba ekstraksi yang lebih sederhana
            simple_pattern = r'([A-Z\d]{1,2})\s*(\d{1,4})\s*([A-Z]{1,4})'
            match = re.search(simple_pattern, cleaned)
            
            if match:
                area_code = match.group(1)
                number = match.group(2)
                series = match.group(3)
                reconstructed = f"{area_code} {number} {series}"
                
                return {
                    "area_code": area_code,
                    "number": number,
                    "series": series,
                    "full_plate": reconstructed,
                    "valid": True,
                    "format": "Indonesian license plate (extracted)",
                    "original_text": corrected_text
                }
            
            # upaya terakhir: coba ekstrak kombinasi yang masuk akal
            fallback_pattern = r'([A-Z]{1,3})\s*(\d{1,4})\s*([A-Z]{1,3})'
            fallback_match = re.search(fallback_pattern, cleaned)
            
            if fallback_match:
                area_code = fallback_match.group(1)
                number = fallback_match.group(2)
                series = fallback_match.group(3)
                
                # hanya terima jika terlihat masuk akal (tidak terlalu panjang)
                if len(area_code) <= 2 and len(series) <= 3:
                    reconstructed = f"{area_code} {number} {series}"
                    
                    return {
                        "area_code": area_code,
                        "number": number,
                        "series": series,
                        "full_plate": reconstructed,
                        "valid": True,
                        "format": "Indonesian license plate (fallback)",
                        "original_text": corrected_text,
                        "confidence": "low"
                    }
            
            return {
                "valid": False,
                "raw_text": corrected_text,
                "cleaned_text": cleaned,
                "error": "Could not extract valid license plate format"
            }
            
        except Exception as e:
            return {
                "valid": False,
                "raw_text": corrected_text,
                "error": f"Parsing error: {e}"
            }
    
    def process_single_image(self, image_path: str) -> Dict:
        """proses gambar tunggal melalui pipeline minimal"""
        start_time = time.time()
        
        # load gambar
        image = cv2.imread(image_path)
        if image is None:
            return {
                "image_name": os.path.basename(image_path),
                "error": f"Could not load image: {image_path}",
                "plates": []
            }
        
        print(f"Memproses: {os.path.basename(image_path)}")
        
        # deteksi yolo dengan parameter optimal
        yolo_results = self.yolo_model(image, conf=0.1, iou=0.3, max_det=50)
        detections = yolo_results[0].boxes
        
        detection_count = len(detections) if detections is not None else 0
        print(f"  YOLO mendeteksi: {detection_count} plat")
        
        plates = []
        
        if detections is not None and len(detections) > 0:
            for i, detection in enumerate(detections):
                # ekstrak bounding box
                box = detection.xyxy[0].cpu().numpy()
                confidence = float(detection.conf[0].cpu().numpy())
                
                x1, y1, x2, y2 = map(int, box)
                
                # crop area plat nomor
                plate_crop = image[y1:y2, x1:x2]
                
                if plate_crop.size == 0:
                    continue
                
                print(f"  Memproses deteksi {i+1} (confidence: {confidence:.3f})")
                
                # peningkatan kontras maksimal
                enhanced_crop = self.maximize_contrast_for_ocr(plate_crop)
                
                # ekstraksi ocr
                raw_text, ocr_confidence = self.extract_text_with_ocr(enhanced_crop)
                print(f"  Hasil OCR: \"{raw_text}\" (confidence: {ocr_confidence:.1f}%)")
                
                if not raw_text:
                    continue
                
                # koreksi teks manual
                corrected_text = self.manual_text_correction(raw_text, ocr_confidence)
                
                # parsing sederhana
                parsed_result = self.simple_parse(corrected_text)
                
                # bangun hasil plat
                plate_result = {
                    "detection_id": i,
                    "yolo_confidence": round(confidence, 3),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "raw_ocr_text": raw_text,
                    "corrected_text": corrected_text,
                    "ocr_confidence": round(ocr_confidence, 1),
                    "parsed": parsed_result
                }
                
                plates.append(plate_result)
                
                # print hasil
                if parsed_result.get("valid"):
                    print(f"  berhasil diparsing: {parsed_result['full_plate']}")
                else:
                    print(f"  parsing gagal: {parsed_result.get('error', 'format tidak valid')}")
        
        processing_time = (time.time() - start_time) * 1000
        
        result = {
            "image_name": os.path.basename(image_path),
            "processing_time_ms": round(processing_time, 0),
            "detections_found": detection_count,
            "plates": plates
        }
        
        print(f"  waktu pemrosesan: {processing_time/1000:.2f}s")
        print()
        
        return result
    
    def process_with_angkatan(self, image_path: str, foto_folder: Path) -> Dict:
        """proses gambar tunggal dengan sistem debug angkatan"""
        start_time = time.time()
        
        # load gambar asli
        original_image = cv2.imread(image_path)
        if original_image is None:
            return {"error": f"Could not load image: {image_path}"}
        
        # simpan gambar asli
        original_file = foto_folder / "original.jpg"
        cv2.imwrite(str(original_file), original_image)
        
        print(f"Memproses: {os.path.basename(image_path)}")
        
        # deteksi yolo
        yolo_results = self.yolo_model(original_image, conf=0.1, iou=0.3, max_det=50)
        detections = yolo_results[0].boxes
        
        detection_count = len(detections) if detections is not None else 0
        print(f"  YOLO mendeteksi: {detection_count} plat")
        
        if detection_count == 0:
            # simpan gambar teranotasi (tidak ada deteksi)
            annotated_file = foto_folder / "annotated_image.jpg"
            cv2.imwrite(str(annotated_file), original_image)
            
            result = {
                "image_name": os.path.basename(image_path),
                "processing_time_ms": round((time.time() - start_time) * 1000, 0),
                "yolo_detections": 0,
                "plates": []
            }
            
            # simpan results.json
            results_file = foto_folder / "results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            return result
        
        # proses deteksi
        annotated_image = original_image.copy()
        plates_results = []
        
        for i, detection in enumerate(detections):
            # Ekstrak bounding box
            box = detection.xyxy[0].cpu().numpy()
            confidence = float(detection.conf[0].cpu().numpy())
            
            x1, y1, x2, y2 = map(int, box)
            
            # Crop area plat nomor
            plate_crop = original_image[y1:y2, x1:x2]
            
            if plate_crop.size == 0:
                continue
            
            print(f"  Memproses deteksi {i+1} (confidence: {confidence:.3f})")
            
            # simpan crop yolo
            crop_file = foto_folder / f"crop_yolo_{i+1}.jpg"
            cv2.imwrite(str(crop_file), plate_crop)
            
            # keputusan enhancement pintar berdasarkan confidence yolo
            if confidence >= 0.9:
                print(f"  confidence tinggi ({confidence:.3f}) - gunakan crop asli")
                ocr_crop = plate_crop  # gunakan crop asli
            else:
                print(f"  confidence rendah ({confidence:.3f}) - terapkan enhancement")
                ocr_crop = self.maximize_contrast_for_ocr(plate_crop)  # gunakan crop yang ditingkatkan
            
            # Ekstraksi OCR
            raw_text, ocr_confidence = self.extract_text_with_ocr(ocr_crop)
            print(f"  Hasil OCR: \"{raw_text}\" (confidence: {ocr_confidence:.1f}%)")
            
            if not raw_text:
                continue
            
            # Koreksi teks manual
            corrected_text = self.manual_text_correction(raw_text, ocr_confidence)
            
            # Parsing sederhana
            parsed_result = self.simple_parse(corrected_text)
            
            # Bangun hasil plat
            plate_result = {
                "detection_id": i + 1,
                "yolo_confidence": round(confidence, 3),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "raw_ocr_text": raw_text,
                "corrected_text": corrected_text,
                "ocr_confidence": round(ocr_confidence, 1),
                "parsed_result": parsed_result
            }
            
            plates_results.append(plate_result)
            
            # gambar pada gambar teranotasi
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # tambah label teks
            if parsed_result.get("valid"):
                label = f"{parsed_result['full_plate']} ({confidence:.2f})"
                print(f"  Berhasil diparsing: {parsed_result['full_plate']}")
            else:
                label = f"{raw_text} ({confidence:.2f})"
                print(f"  teks mentah: {raw_text}")
            
            # gambar background teks dan teks
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_image, (x1, y1-text_height-10), (x1+text_width, y1), (0, 255, 0), -1)
            cv2.putText(annotated_image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # simpan gambar teranotasi
        annotated_file = foto_folder / "annotated_image.jpg"
        cv2.imwrite(str(annotated_file), annotated_image)
        
        processing_time = (time.time() - start_time) * 1000
        
        # bangun hasil akhir
        result = {
            "image_name": os.path.basename(image_path),
            "processing_time_ms": round(processing_time, 0),
            "yolo_detections": detection_count,
            "plates": plates_results
        }
        
        # Simpan results.json
        results_file = foto_folder / "results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"  waktu pemrosesan: {processing_time/1000:.2f}s")
        print(f"  disimpan ke: {foto_folder.name}/")
        print()
        
        return result
    
    def process_batch_with_angkatan(self, image_folder: str = "test_images") -> Dict:
        """proses batch dengan sistem angkatan"""
        print("SISTEM PENGENALAN PLAT NOMOR MINIMAL - SISTEM DEBUG ANGKATAN")
        print("=" * 60)
        
        # buat folder angkatan
        angkatan_path = self.angkatan_manager.create_angkatan_folder()
        print(f"dibuat: {angkatan_path}")
        print()
        
        # cari gambar (cegah duplikasi dengan set)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files_set = set()
        
        if os.path.exists(image_folder):
            for ext in image_extensions:
                image_files_set.update(Path(image_folder).glob(f"*{ext}"))
                image_files_set.update(Path(image_folder).glob(f"*{ext.upper()}"))
        
        if not image_files_set:
            print(f"tidak ada gambar ditemukan di {image_folder}")
            return {}
        
        # konversi ke list terurut untuk konsistensi
        image_files = sorted(list(image_files_set), key=lambda x: x.name.lower())
        print(f"ditemukan {len(image_files)} gambar")
        print()
        
        # proses setiap gambar
        batch_results = []
        
        for i, image_file in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}]", end=" ")
            
            # buat folder foto
            foto_folder = self.angkatan_manager.create_foto_folder(
                angkatan_path, image_file.name, i
            )
            
            # proses dengan sistem debug
            result = self.process_with_angkatan(str(image_file), foto_folder)
            batch_results.append(result)
        
        print(f"angkatan selesai! hasil disimpan di: {angkatan_path}")
        
        return {
            "angkatan_path": str(angkatan_path),
            "total_images": len(image_files),
            "results": batch_results
        }
    
    def process_batch(self, image_folder: str = "test_images") -> Dict:
        """proses batch gambar dengan pipeline minimal"""
        print("SISTEM PENGENALAN PLAT NOMOR MINIMAL")
        print("===================================")
        print("Pipeline: YOLO -> Kontras Maksimal -> OCR -> Koreksi Manual")
        print()
        
        # cari gambar
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        if os.path.exists(image_folder):
            for ext in image_extensions:
                image_files.extend(Path(image_folder).glob(f"*{ext}"))
                image_files.extend(Path(image_folder).glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"tidak ada gambar ditemukan di {image_folder}")
            return {}
        
        print(f"ditemukan {len(image_files)} gambar di {image_folder}/")
        print()
        
        # proses gambar
        batch_results = []
        total_plates = 0
        successful_extractions = 0
        
        batch_start_time = time.time()
        
        for i, image_file in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}]", end=" ")
            
            result = self.process_single_image(str(image_file))
            batch_results.append(result)
            
            # stats update
            valid_plates = [p for p in result["plates"] if p["parsed"].get("valid")]
            total_plates += len(result["plates"])
            successful_extractions += len(valid_plates)
        
        total_processing_time = (time.time() - batch_start_time) * 1000
        
        # bangun hasil akhir
        batch_summary = {
            "total_images": len(image_files),
            "total_detections": total_plates,
            "successful_extractions": successful_extractions,
            "success_rate": round((successful_extractions / total_plates * 100), 1) if total_plates > 0 else 0,
            "total_processing_time_ms": round(total_processing_time, 0),
            "avg_time_per_image_ms": round(total_processing_time / len(image_files), 0)
        }
        
        results = {
            "summary": batch_summary,
            "results": batch_results
        }
        
        # simpan hasil
        output_path = "results/minimal_recognition_results.json"
        os.makedirs("results", exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # print ringkasan akhir
        print("pemrosesan batch selesai!")
        print("=" * 40)
        print(f"gambar diproses: {batch_summary['total_images']}")
        print(f"plat terdeteksi: {batch_summary['total_detections']}")
        print(f"ekstraksi berhasil: {batch_summary['successful_extractions']}")
        print(f"tingkat keberhasilan: {batch_summary['success_rate']:.1f}%")
        print(f"rata-rata waktu per gambar: {batch_summary['avg_time_per_image_ms']/1000:.2f}s")
        print(f"hasil disimpan ke: {output_path}")
        
        return results


def main():
    """fungsi eksekusi utama dengan sistem angkatan"""
    recognizer = MinimalPlateRecognizer()
    results = recognizer.process_batch_with_angkatan("test_images")
    return results


if __name__ == "__main__":
    main()