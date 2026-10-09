import json
import numpy as np

def analyze_whiteboard_comparison():
    with open('tests/results/phase4c14-rapidocr-comparison.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    full_21 = data["datasets"]["Whiteboard_Full_21"]
    tess_evals = {e["sample_id"]: e for e in full_21["tesseract"]["sample_evaluations"]}
    rapid_evals = {e["sample_id"]: e for e in full_21["rapidocr"]["sample_evaluations"]}
    
    dev_ids = ["01", "02", "03", "04", "05", "07", "08", "09", "10", "11", "17"]
    heldout_ids = ["06", "12", "13", "14", "15", "16", "18", "19", "20", "21"]
    
    print("=" * 110)
    print(f"{'ID':<4} | {'Split':<7} | {'Category':<10} | {'Tess CER':<10} | {'Rapid CER':<10} | {'Diff (T-R)':<12} | {'Tess Lat':<10} | {'Rapid Lat':<10} | Better Engine")
    print("=" * 110)
    
    dev_tess_cers, dev_rapid_cers = [], []
    held_tess_cers, held_rapid_cers = [], []
    
    for s_id in sorted(tess_evals.keys()):
        t = tess_evals[s_id]
        r = rapid_evals[s_id]
        
        split = "DEV" if s_id in dev_ids else "HELDOUT"
        cat = t["category"]
        t_cer = t["cer"]
        r_cer = r["cer"]
        diff = t_cer - r_cer # positive means RapidOCR is better
        
        t_lat = t["latency_ms"]
        r_lat = r["latency_ms"]
        
        better = "RAPID" if diff > 0.05 else ("TESS" if diff < -0.05 else "TIED")
        
        if split == "DEV":
            dev_tess_cers.append(t_cer)
            dev_rapid_cers.append(r_cer)
        else:
            held_tess_cers.append(t_cer)
            held_rapid_cers.append(r_cer)
            
        print(f"{s_id:<4} | {split:<7} | {cat:<10} | {t_cer*100:>8.2f}% | {r_cer*100:>8.2f}% | {diff*100:>10.2f}% | {t_lat:>8.1f}ms | {r_lat:>8.1f}ms | {better}")
        
    print("-" * 110)
    print(f"DEV (11 samples):     Tesseract Mean CER = {np.mean(dev_tess_cers)*100:.2f}% | RapidOCR Mean CER = {np.mean(dev_rapid_cers)*100:.2f}% (Tess is +{np.mean(dev_rapid_cers)*100 - np.mean(dev_tess_cers)*100:.2f}% better)")
    print(f"HELDOUT (10 samples): Tesseract Mean CER = {np.mean(held_tess_cers)*100:.2f}% | RapidOCR Mean CER = {np.mean(held_rapid_cers)*100:.2f}% (Rapid is +{np.mean(held_tess_cers)*100 - np.mean(held_rapid_cers)*100:.2f}% better)")
    print("=" * 110)
    
    # Detailed inspection of specific samples with biggest divergence
    print("\nSAMPLE DEEP-DIVE:")
    for s_id in ["06", "11", "12", "14", "20", "01", "04", "08"]:
        t = tess_evals[s_id]
        r = rapid_evals[s_id]
        print(f"\n--- Sample {s_id} ({t['category']}) | Tess CER: {t['cer']*100:.1f}%, Rapid CER: {r['cer']*100:.1f}% ---")
        print(f"  Reference: {t['reference_text'][:120]}...")
        print(f"  Tesseract: {t['prediction'][:120]}...")
        print(f"  RapidOCR:  {r['prediction'][:120]}...")

if __name__ == '__main__':
    analyze_whiteboard_comparison()
