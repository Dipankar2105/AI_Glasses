import os
import json
import numpy as np

def audit_datasets():
    print("================================================================================")
    print("PROVENANCE & INTEGRITY AUDIT OF ALL 3 EVALUATION DATASETS")
    print("================================================================================")
    
    # 1. Whiteboard Dataset
    wb_meta = "data/external/whiteboards/metadata.jsonl"
    print("\n--- 1. Whiteboard Dataset (Phase 4C.6 / 4C.7) ---")
    if os.path.exists(wb_meta):
        with open(wb_meta, 'r', encoding='utf-8') as f:
            wb_records = [json.loads(line) for line in f if line.strip()]
        print(f"Total entries: {len(wb_records)}")
        print(f"Source: HuggingFace danielrosehill/Whiteboards (CC BY 4.0)")
        for r in wb_records:
            img_path = os.path.join("data/external/whiteboards", r["file_name"])
            exists = os.path.exists(img_path)
            size_bytes = os.path.getsize(img_path) if exists else 0
            print(f"  [{r['id']}] {r['file_name']} | Exists: {exists} ({size_bytes/1024:.1f} KB) | Category: {r.get('category')} | Ref len: {len(r.get('transcription', ''))}")
    else:
        print("Whiteboard metadata not found!")

    # 2. Printed & Scene Dataset
    ps_meta = "data/external/printed_scene/metadata.jsonl"
    print("\n--- 2. Printed & Scene Dataset (Phase 4C.10) ---")
    if os.path.exists(ps_meta):
        with open(ps_meta, 'r', encoding='utf-8') as f:
            ps_records = [json.loads(line) for line in f if line.strip()]
        print(f"Total entries: {len(ps_records)}")
        for r in ps_records:
            img_path = os.path.join("data/external/printed_scene", r["file_name"])
            exists = os.path.exists(img_path)
            size_bytes = os.path.getsize(img_path) if exists else 0
            print(f"  [{r['id']}] {r['file_name']} | Exists: {exists} ({size_bytes/1024:.1f} KB) | Provenance: {r.get('provenance')} | License: {r.get('license')} | Ref len: {len(r.get('transcription', ''))}")
    else:
        print("Printed & Scene metadata not found!")

    # 3. Handwriting Dataset
    hw_meta = "data/external/handwriting/metadata.jsonl"
    print("\n--- 3. Handwriting Dataset (Phase 4C.11) ---")
    if os.path.exists(hw_meta):
        with open(hw_meta, 'r', encoding='utf-8') as f:
            hw_records = [json.loads(line) for line in f if line.strip()]
        print(f"Total entries: {len(hw_records)}")
        for r in hw_records:
            img_path = os.path.join("data/external/handwriting", r["file_name"])
            exists = os.path.exists(img_path)
            size_bytes = os.path.getsize(img_path) if exists else 0
            print(f"  [{r['id']}] {r['file_name']} | Exists: {exists} ({size_bytes/1024:.1f} KB) | Provenance: {r.get('provenance')} | License: {r.get('license')} | Ref len: {len(r.get('transcription', ''))}")
    else:
        print("Handwriting metadata not found!")

if __name__ == '__main__':
    audit_datasets()
