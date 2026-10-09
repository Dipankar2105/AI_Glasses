import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import json
import numpy as np
from backend.ai.ocr_benchmark import calculate_cer, calculate_wer, normalize_text

def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y), dtype=int)
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y
    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x, y] = matrix[x-1, y-1]
            else:
                matrix[x, y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x, y-1] + 1,
                    matrix[x-1, y-1] + 1
                )
    return int(matrix[size_x - 1, size_y - 1])

def audit_handwriting():
    with open('tests/results/phase4c11-handwriting-ocr.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    evals = data['sample_evaluations']
    print(f"Total samples: {len(evals)}")
    print("=" * 100)
    
    total_ref_chars = 0
    total_char_edits = 0
    total_ref_words = 0
    total_word_edits = 0
    
    sample_cers = []
    sample_wers = []
    
    for i, e in enumerate(evals):
        ref = e['reference_transcription']
        pred = e['actual_prediction']
        
        norm_ref = normalize_text(ref)
        norm_pred = normalize_text(pred)
        
        char_dist = levenshtein(list(norm_ref), list(norm_pred))
        ref_char_len = max(len(norm_ref), 1)
        calc_cer = calculate_cer(ref, pred)
        
        ref_words = norm_ref.split()
        pred_words = norm_pred.split()
        word_dist = levenshtein(ref_words, pred_words)
        ref_word_len = max(len(ref_words), 1)
        calc_wer = calculate_wer(ref, pred)
        
        exact = (norm_ref == norm_pred) and (len(norm_ref) > 0)
        
        sample_cers.append(calc_cer)
        sample_wers.append(calc_wer)
        
        total_ref_chars += ref_char_len
        total_char_edits += char_dist
        total_ref_words += ref_word_len
        total_word_edits += word_dist
        
        print(f"Sample {i+1}: [{e['sample_id']}] ({e['category']})")
        print(f"  Ref:       '{ref}' (norm: '{norm_ref}')")
        print(f"  Pred:      '{pred}' (norm: '{norm_pred}')")
        print(f"  Char Dist: {char_dist} / {ref_char_len} => CER: {calc_cer:.4f} (recorded: {e['cer']})")
        print(f"  Word Dist: {word_dist} / {ref_word_len} => WER: {calc_wer:.4f} (recorded: {e['wer']})")
        print(f"  Exact:     {exact} (recorded: {e['exact_match']})")
        print(f"  Ref words:  {ref_words}")
        print(f"  Pred words: {pred_words}")
        print("-" * 100)
        
    print(f"\nUnweighted Mean CER: {np.mean(sample_cers):.6f} (Recorded: {data['mean_cer']})")
    print(f"Unweighted Mean WER: {np.mean(sample_wers):.6f} (Recorded: {data['mean_wer']})")
    print(f"Micro-averaged CER:  {total_char_edits / total_ref_chars:.6f} ({total_char_edits}/{total_ref_chars})")
    print(f"Micro-averaged WER:  {total_word_edits / total_ref_words:.6f} ({total_word_edits}/{total_ref_words})")

if __name__ == '__main__':
    import numpy as np
    audit_handwriting()
