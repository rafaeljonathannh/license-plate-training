  Analisis Images 15-20

  15_images.jpg - NO DETECTION

  - YOLO Result: 0 plates detected
  - Issue: YOLO tidak mendeteksi license plate sama sekali
  - Analysis Required: Image quality, plate visibility, atau model limitation

  16_images.jpg - PARSING ERROR

  - YOLO: 1 plate (confidence: 0.845)
  - OCR Result: "3689 SKJ 8 28 R"
  - Parsed: "89 5 KJ" ❌ Wrong parsing
  - Issue: Complex text dengan noise, parsing gagal extract proper format

  17_images.jpg - MINIMAL OCR

  - YOLO: 1 plate (confidence: 0.199) - Very low!
  - OCR Result: "7" only
  - Issue: YOLO detection sangat buruk, crop mungkin tidak complete

  18_images.jpg - MULTIPLE DETECTIONS

  - YOLO: 2 plates detected
  - Detection 1: "B Z713SMM" → "Z7 13 SMM" ❌ Wrong
  - Detection 2: "(06.26" → Raw text (date fragment)
  - Issue: Multiple detection dengan quality buruk

  19_images.jpg - PARSING ERROR

  - YOLO: 1 plate (confidence: 0.871)
  - OCR Result: "4155 SME 2"
  - Parsed: "55 5 ME" ❌ Wrong parsing
  - Issue: OCR good tapi parsing logic error

  20_images.jpg - NOISY OCR

  - YOLO: 1 plate (confidence: 0.752)
  - OCR Result: "{ 5109 EhJL YLII 593F" (very noisy)
  - Parsed: "51 09 EHJL" ❌ Wrong
  - Issue: Very noisy OCR dengan banyak artifacts