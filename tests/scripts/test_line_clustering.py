import os, sys, json
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath('.'))
from backend.ai.rapid_ocr import RapidOCREngine
from backend.ai.tesseract_ocr import TesseractOCREngine
from backend.vision.frame import OCRInput
from backend.ai.ocr_benchmark import calculate_cer, calculate_wer, normalize_text

def cluster_and_order_regions(regions, line_overlap_ratio=0.55):
    if not regions:
        return ""
    
    boxes = []
    for r in regions:
        xmin = r.box.x
        ymin = r.box.y
        xmax = r.box.x + r.box.width
        ymax = r.box.y + r.box.height
        yc = ymin + r.box.height / 2.0
        h = max(r.box.height, 0.001)
        boxes.append({
            'text': r.text.strip(),
            'xmin': xmin,
            'xmax': xmax,
            'ymin': ymin,
            'ymax': ymax,
            'yc': yc,
            'h': h,
            'conf': r.confidence
        })
    
    # Filter out empty or whitespace tokens
    boxes = [b for b in boxes if len(b['text']) > 0]
    if not boxes:
        return ""

    # Sort primarily by vertical midpoint
    boxes.sort(key=lambda b: b['yc'])
    
    lines = []
    curr_line = [boxes[0]]
    curr_yc = boxes[0]['yc']
    curr_h = boxes[0]['h']
    
    for b in boxes[1:]:
        # If the vertical center difference is within ratio * line_height, group as same line
        if abs(b['yc'] - curr_yc) < (curr_h * line_overlap_ratio):
            curr_line.append(b)
            curr_yc = np.mean([x['yc'] for x in curr_line])
            curr_h = np.mean([x['h'] for x in curr_line])
        else:
            # Sort left to right within line
            curr_line.sort(key=lambda x: x['xmin'])
            lines.append(" ".join([x['text'] for x in curr_line]))
            curr_line = [b]
            curr_yc = b['yc']
            curr_h = b['h']
            
    if curr_line:
        curr_line.sort(key=lambda x: x['xmin'])
        lines.append(" ".join([x['text'] for x in curr_line]))
        
    return "\n".join(lines)

def run_dev_test():
    with open('data/external/notebooks/metadata.jsonl', 'r') as f:
        records = [json.loads(line) for line in f if line.strip()][:15] # 15 Dev samples

    rapid = RapidOCREngine()

    print("=== DEV SET EVALUATION: RAW RAPIDOCR VS LINE-CLUSTERED SEGMENTATION ===")
    raw_cers = []
    clust_cers = []

    for r in records:
        img_path = os.path.join('data/external/notebooks', r['file_name'])
        with Image.open(img_path) as img:
            arr = np.array(img.convert('RGB'))
        
        inp = OCRInput(image=arr, width=arr.shape[1], height=arr.shape[0], channels=3, numerical_range=(0, 255), timestamp=1, seq_num=1)
        res = rapid.process(inp)
        
        clustered_text = cluster_and_order_regions(res.regions)
        raw_cer = calculate_cer(r['transcription'], res.full_text)
        clust_cer = calculate_cer(r['transcription'], clustered_text)
        
        raw_cers.append(raw_cer)
        clust_cers.append(clust_cer)
        
        print(f"Sample {r['id']:<6} | Raw CER: {raw_cer*100:>6.2f}% | Clustered CER: {clust_cer*100:>6.2f}% | Gain: {(raw_cer - clust_cer)*100:>+6.2f}%")

    print("----------------------------------------------------------------------")
    print(f"Dev Macro CER: Raw = {np.mean(raw_cers)*100:.2f}% | Clustered = {np.mean(clust_cers)*100:.2f}% (Gain: {(np.mean(raw_cers) - np.mean(clust_cers))*100:+.2f}%)")

if __name__ == '__main__':
    run_dev_test()
