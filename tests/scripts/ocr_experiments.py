import sys, os, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backend.ai.tesseract_ocr import TesseractOCREngine
from backend.vision.frame import OCRInput
from backend.ai.errors import AIVisionError

def has_tesseract():
    import pytesseract
    try:
        _ = TesseractOCREngine()
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

def generate_fixtures():
    fixtures = {}
    
    # 1. Short uppercase word
    img1 = Image.new('L', (200, 50), color=255)
    d = ImageDraw.Draw(img1)
    d.text((10, 10), "HELLO", fill=0)
    fixtures["uppercase"] = np.array(img1)
    
    # 2. Short lowercase word
    img2 = Image.new('L', (200, 50), color=255)
    d = ImageDraw.Draw(img2)
    d.text((10, 10), "world", fill=0)
    fixtures["lowercase"] = np.array(img2)
    
    # 3. Two-word phrase
    img3 = Image.new('L', (300, 50), color=255)
    d = ImageDraw.Draw(img3)
    d.text((10, 10), "AI GLASSES", fill=0)
    fixtures["two_word"] = np.array(img3)
    
    # 4. Numeric string
    img4 = Image.new('L', (200, 50), color=255)
    d = ImageDraw.Draw(img4)
    d.text((10, 10), "12345", fill=0)
    fixtures["numeric"] = np.array(img4)
    
    # 5. Blank image
    img5 = Image.new('L', (200, 50), color=255)
    fixtures["blank"] = np.array(img5)
    
    # 6. Grayscale text image
    img6 = Image.new('L', (200, 50), color=128)
    d = ImageDraw.Draw(img6)
    d.text((10, 10), "GRAY", fill=30)
    fixtures["gray"] = np.array(img6)
    
    # 7. RGB text image
    img7 = Image.new('RGB', (200, 50), color=(255, 200, 200))
    d = ImageDraw.Draw(img7)
    d.text((10, 10), "COLOR", fill=(0, 0, 255))
    fixtures["rgb"] = np.array(img7)
    
    return fixtures

def run_experiments():
    import os
    if not has_tesseract():
        print("Tesseract not available. Skipping experiments and latency measurements.")
        return
        
    engine = TesseractOCREngine()
    
    # 1. Base test image (simulated original capture)
    # We will draw a greyish text on a slightly noisy background
    img = Image.new('RGB', (300, 60), color=(180, 180, 180))
    d = ImageDraw.Draw(img)
    d.text((20, 15), "AI GLASSES TEST", fill=(50, 50, 50))
    arr_rgb = np.array(img)
    
    fixtures = generate_fixtures()
    
    print("--- FIXTURES RECOGNITION TEST ---")
    for fix_name, fix_arr in fixtures.items():
        h, w = fix_arr.shape[:2]
        c = 1 if len(fix_arr.shape) == 2 else fix_arr.shape[2]
        inp = OCRInput(fix_arr, w, h, c, (0, 255), 100, 1)
        res = engine.process(inp)
        print(f"Fixture: {fix_name.ljust(12)} | Output: '{res.full_text}' | Regions: {len(res.regions)}")

    print("\n--- PREPROCESSING EXPERIMENTS ---")
    expected_text = "AI GLASSES TEST"
    
    from backend.vision.pipeline import VisionPipeline
    pipeline = VisionPipeline()
    from backend.vision.mock_camera import CameraFrame
    raw_frame = CameraFrame(300, 60, "RGB", arr_rgb.tobytes(), 100, 1)
    
    # A. Original RGB
    vframe_rgb = pipeline.process(raw_frame)
    # B. Converted to Grayscale using existing Phase 4B pipeline
    vframe_gray = pipeline.convert_format(vframe_rgb, "GRAYSCALE")
    # C. Existing contrast normalized
    vframe_contrast = pipeline.contrast_normalize(vframe_gray)
    # D. OCR-local thresholding on grayscale data
    arr_thresh = np.where(vframe_gray.data < 128, 0, 255).astype(np.uint8)
    
    variations = {
        "A. RGB": vframe_rgb.data,
        "B. Grayscale": vframe_gray.data,
        "C. Contrast Norm": vframe_contrast.data,
        "D. Thresholding": arr_thresh
    }
    
    for name, arr in variations.items():
        h, w = arr.shape[:2]
        c = 1 if len(arr.shape) == 2 else arr.shape[2]
        inp = OCRInput(arr, w, h, c, (0, 255), 0, 0)
        
        try:
            res = engine.process(inp)
            success = expected_text in res.full_text.upper()
            conf = res.regions[0].confidence if res.regions else 0.0
            print(f"{name.ljust(18)} | Expected: '{expected_text}' | Recog: '{res.full_text}' | Recovered: {success} | Conf: {conf:.2f}")
        except AIVisionError as e:
            print(f"{name.ljust(18)} | Error: {e}")
            
    # Latency checks
    print("\n--- PERFORMANCE OBSERVATIONS ---")
    latencies = []
    
    arr = fixtures["two_word"]
    h, w = arr.shape[:2]
    c = 1
    inp = OCRInput(arr, w, h, c, (0, 255), 0, 0)
    
    # Warmup
    engine.process(inp)
    
    for i in range(5):
        t0 = time.perf_counter()
        engine.process(inp)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        
    print(f"Samples: 5")
    print(f"Median latency: {np.median(latencies):.2f} ms")
    print(f"Max latency:    {np.max(latencies):.2f} ms")

if __name__ == "__main__":
    run_experiments()
