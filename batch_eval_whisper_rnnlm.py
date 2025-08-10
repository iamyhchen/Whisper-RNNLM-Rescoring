#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch evaluate Whisper + RNNLM rescoring over a dataset and compute WER/CER.

Inputs:
  - text file:    lines like "<id> <reference transcription>"
  - audio_paths:  lines like "<id> <path/to/audio.wav>"

Outputs:
  - predict.txt : lines "<id> <prediction>"
  - predict.csv : CSV with header: id,predict
  - metrics.json: overall WER, CER, and counts

Usage example:
python batch_eval_whisper_rnnlm.py \
  --text dataset/data-pinyin-Taipu-modify/test/text \
  --audio_paths dataset/data-pinyin-Taipu-modify/test/audio_paths \
  --asr_model /path/to/your-finetuned-whisper \
  --rnnlm_ckpt ./rnnlm_zh/rnnlm.pt \
  --out_dir ./pred_out \
  --num_beams 10 --num_return_sequences 10 --lm_weight 0.7 --length_beta 0.0

Note:
- Requires whisper_rnnlm_rescore.py in PYTHONPATH (same folder is fine), providing
  `load_rnnlm_from_ckpt` and `transcribe_with_rnnlm`.
- Tokenizer must match Whisper; for pinyin with spaces, WER uses space-split; CER ignores spaces.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple, List

import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Import rescoring utilities
from whisper_rnnlm_rescore import load_rnnlm_from_ckpt, transcribe_with_rnnlm


def read_kv_file(path: str) -> Dict[str, str]:
    """Read lines of the form: <id> <value...> (first space splits)."""
    mp: Dict[str, str] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split(maxsplit=1)
            if len(parts) == 1:
                key, val = parts[0], ""
            else:
                key, val = parts[0], parts[1]
            mp[key] = val
    return mp


def load_wav_16k_mono(path: str):
    wav, sr = torchaudio.load(path)  # [C, T]
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.mean(dim=0).numpy().astype('float32')
    return wav


# -------------------------------
# Metrics
# -------------------------------

def levenshtein(a: List[str], b: List[str]) -> int:
    import numpy as np
    n, m = len(a), len(b)
    dp = np.zeros((n+1, m+1), dtype=int)
    for i in range(n+1):
        dp[i, 0] = i
    for j in range(m+1):
        dp[0, j] = j
    for i in range(1, n+1):
        ai = a[i-1]
        for j in range(1, m+1):
            cost = 0 if ai == b[j-1] else 1
            dp[i, j] = min(
                dp[i-1, j] + 1,
                dp[i, j-1] + 1,
                dp[i-1, j-1] + cost,
            )
    return int(dp[n, m])


def wer(ref: str, hyp: str) -> float:
    ref_toks = ref.strip().split()
    hyp_toks = hyp.strip().split()
    if len(ref_toks) == 0:
        return 0.0 if len(hyp_toks) == 0 else 1.0
    d = levenshtein(ref_toks, hyp_toks)
    return d / len(ref_toks)


def cer(ref: str, hyp: str) -> float:
    # Character-level CER; ignore spaces to suit pinyin-with-spaces datasets
    ref_chars = [c for c in ref if not c.isspace()]
    hyp_chars = [c for c in hyp if not c.isspace()]
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0
    d = levenshtein(ref_chars, hyp_chars)
    return d / len(ref_chars)


# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--text', type=str, required=True, help='Path to text file: <id> <reference>')
    ap.add_argument('--audio_paths', type=str, required=True, help='Path to audio_paths file: <id> <wav_path>')
    ap.add_argument('--asr_model', type=str, required=True, help='HF id or local path to Whisper (finetuned)')
    ap.add_argument('--rnnlm_ckpt', type=str, required=True, help='Path to rnnlm.pt')
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--num_beams', type=int, default=10)
    ap.add_argument('--num_return_sequences', type=int, default=10)
    ap.add_argument('--lm_weight', type=float, default=0.7)
    ap.add_argument('--length_beta', type=float, default=0.0)
    ap.add_argument('--language', type=str, default='zh')
    ap.add_argument('--task', type=str, default='transcribe')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    device = torch.device(args.device)
    print('Loading ASR model & processor...')
    processor = WhisperProcessor.from_pretrained(args.asr_model, language=args.language, task=args.task)
    tokenizer = processor.tokenizer
    asr_model = WhisperForConditionalGeneration.from_pretrained(args.asr_model).to(device).eval()

    print('Loading RNNLM...')
    rnnlm, _ = load_rnnlm_from_ckpt(args.rnnlm_ckpt, device=device)

    # Read dataset index
    print('Reading dataset index...')
    id2ref = read_kv_file(args.text)
    id2path = read_kv_file(args.audio_paths)

    # Iterate and predict
    rows = []  # for CSV
    pred_lines = []  # for txt

    n = 0
    wer_sum, cer_sum = 0.0, 0.0

    # Sort by id for determinism
    for utt_id in sorted(id2path.keys()):
        wav_path = f'../{id2path[utt_id]}'
        if utt_id not in id2ref:
            print(f"[WARN] id {utt_id} has audio but no reference text; skipping.")
            continue
        ref = id2ref[utt_id]

        wav = load_wav_16k_mono(wav_path)
        best, nbest = transcribe_with_rnnlm(
            audio=wav,
            asr_model=asr_model,
            processor=processor,
            tokenizer=tokenizer,
            rnnlm=rnnlm,
            lm_weight=args.lm_weight,
            length_beta=args.length_beta,
            num_beams=args.num_beams,
            num_return_sequences=args.num_return_sequences,
            language=args.language,
            task=args.task,
            device=device,
            no_timestamp=True,
        )

        # Save predictions
        rows.append({"id": utt_id, "predict": best})
        pred_lines.append(f"{utt_id} {best}")

        # Metrics
        w = wer(ref, best)
        c = cer(ref, best)
        wer_sum += w
        cer_sum += c
        n += 1
        if n % 50 == 0:
            print(f"Processed {n} utts... current avg WER={wer_sum/n:.4f}, CER={cer_sum/n:.4f}")

    avg_wer = wer_sum / max(1, n)
    avg_cer = cer_sum / max(1, n)

    # Write outputs
    # Write predict.txt with both GT and PRED per utterance
    with open(out_dir / 'predict.txt', 'w', encoding='utf-8') as f:
        for utt_id in sorted(id2path.keys()):
            if utt_id not in id2ref:
                continue
            ref = id2ref[utt_id]
            # Find prediction for this utt_id
            pred = None
            for r in rows:
                if r["id"] == utt_id:
                    pred = r["predict"]
                    break
            if pred is None:
                continue
            f.write(f"{utt_id}\n")
            f.write(f"[GT] {ref}\n")
            f.write(f"[PRED] {pred}\n\n")
    with open(out_dir / 'predict.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'predict'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump({
            'num_utts': n,
            'WER': f'{avg_wer*100:.2f}%',
            'CER': f'{avg_cer*100:.2f}%',
        }, f, ensure_ascii=False, indent=2)

    print(f"\nDone. num_utts={n}")
    print(f"WER={avg_wer*100:.2f}% | CER={avg_cer*100:.2f}%")
    print(f"Outputs -> {out_dir / 'predict.txt'}, {out_dir / 'predict.csv'}, {out_dir / 'metrics.json'}")


if __name__ == '__main__':
    main()
