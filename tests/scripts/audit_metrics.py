import json
import numpy as np

def audit_file(path):
    print(f"==================================================")
    print(f"AUDITING: {path}")
    print(f"==================================================")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if 'sample_evaluations' in data:
        evals = data['sample_evaluations']
        cers = [e['cer'] for e in evals]
        wers = [e['wer'] for e in evals]
        lats = [e['latency_ms'] for e in evals]
        exacts = [e['exact_match'] for e in evals]
        
        print(f"Sample Count:              {len(evals)}")
        print(f"Mean CER (recomputed):     {np.mean(cers):.6f} | Recorded: {data.get('mean_cer')}")
        print(f"Mean WER (recomputed):     {np.mean(wers):.6f} | Recorded: {data.get('mean_wer')}")
        print(f"Exact Matches:             {sum(exacts)}/{len(evals)} ({np.mean(exacts):.4f}) | Recorded: {data.get('exact_match_count')}")
        print(f"Mean Latency (ms):         {np.mean(lats):.2f} | Recorded: {data.get('mean_latency_ms')}")
        print(f"Median Latency (ms):       {np.median(lats):.2f} | Recorded: {data.get('median_latency_ms')}")
        
        # Check if any predictions are empty or skipped
        empty_count = sum(1 for e in evals if not e.get('actual_prediction', '').strip())
        print(f"Empty/Failed predictions:  {empty_count}")
        
    elif 'dev_matrix_results' in data:
        print("Dev Matrix (11 samples):")
        for item in data['dev_matrix_results']:
            k = item.get('config_name', item.get('label', 'unknown'))
            evals = item.get('evaluations', [])
            cers = [e['cer'] for e in evals]
            lats = [e['latency_ms'] for e in evals]
            print(f"  {k:<20}: CER={np.mean(cers):.4f} (Rec: {item['mean_cer']:.4f}), Lat={np.mean(lats):.1f}ms (Rec: {item['mean_latency_ms']:.1f}ms)")
            
        print("\nHeld-out Split (10 samples):")
        held = data.get('heldout_results', [])
        if isinstance(held, list):
            for item in held:
                k = item.get('config_name', item.get('label', 'unknown'))
                evals = item.get('evaluations', [])
                cers = [e['cer'] for e in evals]
                lats = [e['latency_ms'] for e in evals]
                print(f"  {k:<20}: CER={np.mean(cers):.4f} (Rec: {item['mean_cer']:.4f}), Lat={np.mean(lats):.1f}ms (Rec: {item['mean_latency_ms']:.1f}ms)")
        elif isinstance(held, dict):
            for k, item in held.items():
                evals = item.get('evaluations', [])
                cers = [e['cer'] for e in evals]
                lats = [e['latency_ms'] for e in evals]
                print(f"  {k:<20}: CER={np.mean(cers):.4f} (Rec: {item['mean_cer']:.4f}), Lat={np.mean(lats):.1f}ms (Rec: {item['mean_latency_ms']:.1f}ms)")
            
        full = data.get('full_comparison', [])
        if isinstance(full, list):
            print(f"\nFull Dataset Comparison (21 samples):")
            for item in full:
                k = item.get('config_name', item.get('label', 'unknown'))
                evals = item.get('evaluations', [])
                cers = [e['cer'] for e in evals]
                lats = [e['latency_ms'] for e in evals]
                print(f"  {k:<20}: CER={np.mean(cers):.4f} (Rec: {item['mean_cer']:.4f}), Lat={np.mean(lats):.1f}ms (Rec: {item['mean_latency_ms']:.1f}ms)")
        elif isinstance(full, dict):
            print(f"\nFull Dataset Comparison (21 samples):")
            for k, item in full.items():
                evals = item.get('evaluations', [])
                cers = [e['cer'] for e in evals]
                lats = [e['latency_ms'] for e in evals]
                print(f"  {k:<20}: CER={np.mean(cers):.4f} (Rec: {item['mean_cer']:.4f}), Lat={np.mean(lats):.1f}ms (Rec: {item['mean_latency_ms']:.1f}ms)")
            
    print()

if __name__ == '__main__':
    audit_file('tests/results/phase4c6-whiteboard-ocr.json')
    audit_file('tests/results/phase4c7-whiteboard-diagnostics.json')
    audit_file('tests/results/phase4c10-printed-scene-ocr.json')
    audit_file('tests/results/phase4c11-handwriting-ocr.json')
