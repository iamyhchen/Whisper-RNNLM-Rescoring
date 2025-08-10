#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Whisper (Transformers) N-best ➜ RNNLM rescoring ➜ best text

This module provides:
  - transcribe_with_rnnlm(...): run Whisper beam search, score with RNNLM, return best text + N-best details
  - load_rnnlm_from_ckpt(...): load the RNNLM checkpoint produced by train_rnnlm_whisper.py

Requirements:
  pip install torch torchaudio transformers
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import math

import torchaudio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperTokenizerFast,
)

# -------------------------------
# RNNLM (match your training script)
# -------------------------------
class RNNLM(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 512, hid: int = 1024, layers: int = 2,
                 dropout: float = 0.3, rnn_type: str = "lstm", tie_weights: bool = False):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        self.rnn_type = rnn_type.lower()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[self.rnn_type]
        self.rnn = rnn_cls(
            emb, hid, num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True
        )
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hid, vocab_size)
        if tie_weights:
            if emb != hid:
                raise ValueError("For weight tying, emb must equal hid.")
            self.out.weight = self.emb.weight

    def forward(self, x, h=None):
        e = self.emb(x)       # [B, T, E]
        y, h = self.rnn(e, h) # [B, T, H]
        y = self.drop(y)
        logits = self.out(y)  # [B, T, V]
        return logits, h


@dataclass
class Hypo:
    text: str
    token_ids: List[int]
    asr_logp: float
    lm_logp: float
    score: float


# -------------------------------
# Helpers
# -------------------------------
def strip_prefix(seq: List[int], prefix: List[int]) -> List[int]:
    """Remove the forced decoder prompt (language/task) prefix from a generated sequence."""
    if not prefix:
        return seq
    i = 0
    while i < len(prefix) and i < len(seq) and seq[i] == prefix[i]:
        i += 1
    return seq[i:]


def filter_content_tokens(ids: List[int], tokenizer: WhisperTokenizerFast) -> List[int]:
    """Remove special/control tokens for LM scoring."""
    special = set(tokenizer.all_special_ids)
    return [t for t in ids if t not in special]


@torch.no_grad()
def sequence_logprob(model: RNNLM, ids: List[int], device: torch.device) -> float:
    """
    Sum of log probabilities log P(y) over the whole sequence.
    ids: content tokens only (no control/special), length >= 2
    """
    if len(ids) < 2:
        return 0.0
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)  # [1, T-1]
    y = torch.tensor(ids[1:],  dtype=torch.long, device=device).unsqueeze(0)  # [1, T-1]
    logits, _ = model(x)
    logp = torch.log_softmax(logits, dim=-1)
    tgt_lp = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)  # [1, T-1]
    return float(tgt_lp.sum().item())


# -------------------------------
# Main API
# -------------------------------
@torch.no_grad()
def transcribe_with_rnnlm(
    audio,  # numpy array float32 mono 16k, or anything WhisperProcessor accepts
    asr_model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    tokenizer: WhisperTokenizerFast,
    rnnlm: RNNLM,
    lm_weight: float = 0.7,
    length_beta: float = 0.0,
    num_beams: int = 10,
    num_return_sequences: int = 10,
    language: str = "zh",
    task: str = "transcribe",
    device: Optional[torch.device] = None,
    no_timestamp: bool = True,
) -> Tuple[str, List[Hypo]]:
    """
    Run Whisper, get N-best, rescore with RNNLM, return (best_text, nbest_list).
    """
    if device is None:
        device = asr_model.device
    asr_model.eval()
    rnnlm.eval()

    # 1) Features
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    # 2) Forced decoder prompt (language/task)
    forced_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    prefix_ids = [tok_id for _, tok_id in forced_ids] if isinstance(forced_ids, list) else []

    gen_kwargs = dict(
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        output_scores=True,
        return_dict_in_generate=True,
    )
    if no_timestamp:
        gen_kwargs["return_timestamps"] = False

    generated = asr_model.generate(
        inputs=input_features,
        forced_decoder_ids=forced_ids,
        **gen_kwargs,
    )
    sequences = generated.sequences              # [N, T]
    seq_scores = generated.sequences_scores      # [N] (HF length-normalized log prob-like score)

    # 3) Prepare N-best & LM scores
    nbest: List[Hypo] = []
    for i in range(sequences.size(0)):
        ids_full: List[int] = sequences[i].tolist()
        # For display/debug
        text = tokenizer.decode(ids_full, skip_special_tokens=True).strip()

        # Strip prompt & special tokens for LM scoring
        ids_wo_prefix = strip_prefix(ids_full, prefix_ids)
        ids_content = filter_content_tokens(ids_wo_prefix, tokenizer)

        lm_lp = sequence_logprob(rnnlm, ids_content, device=device)
        asr_lp = float(seq_scores[i].item())  # treat as ASR per-sequence log score
        length_pen = length_beta * float(len(ids_content))
        final_score = asr_lp + lm_weight * lm_lp + length_pen

        nbest.append(Hypo(text=text, token_ids=ids_content, asr_logp=asr_lp, lm_logp=lm_lp, score=final_score))

    # 4) Select best
    nbest.sort(key=lambda h: h.score, reverse=True)
    best_text = nbest[0].text if nbest else ""
    return best_text, nbest


# -------------------------------
# Loader for saved RNNLM checkpoint
# -------------------------------
def load_rnnlm_from_ckpt(ckpt_path: str, device: Optional[torch.device] = None) -> Tuple[RNNLM, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device or "cpu")
    hp = ckpt["model_hparams"]
    model = RNNLM(
        vocab_size=ckpt["vocab_size"],
        emb=hp["emb"], hid=hp["hid"], layers=hp["layers"],
        dropout=hp["dropout"], rnn_type=hp["rnn_type"], tie_weights=hp["tie_weights"],
    )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device).eval()
    meta = {
        "vocab_size": ckpt["vocab_size"],
        "tok_name": ckpt.get("tok_name"),
        "language": ckpt.get("language"),
        "task": ckpt.get("task"),
        "model_hparams": hp,
    }
    return model, meta


# -------------------------------
# CLI example
# -------------------------------
if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--asr_model", type=str, default="openai/whisper-small")
    parser.add_argument("--rnnlm_ckpt", type=str, required=True)
    parser.add_argument("--lm_weight", type=float, default=0.7)
    parser.add_argument("--length_beta", type=float, default=0.0)
    parser.add_argument("--num_beams", type=int, default=10)
    parser.add_argument("--num_return_sequences", type=int, default=10)
    parser.add_argument("--language", type=str, default="zh")
    parser.add_argument("--task", type=str, default="transcribe")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading ASR model & processor...")
    processor = WhisperProcessor.from_pretrained(args.asr_model, language=args.language, task=args.task)
    tokenizer = processor.tokenizer
    asr_model = WhisperForConditionalGeneration.from_pretrained(args.asr_model).to(device).eval()

    print("Loading RNNLM...")
    rnnlm, meta = load_rnnlm_from_ckpt(args.rnnlm_ckpt, device=device)

    print("Loading audio...")
    wav, sr = torchaudio.load(args.audio)  # [C, T]
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
    wav = wav.mean(dim=0).numpy().astype("float32")  # mono float32

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

    print("\nBest:", best)
    print("\nTop-10:")
    for h in nbest[:10]:
        print(f"score={h.score:.3f} | asr={h.asr_logp:.3f} | lm={h.lm_logp:.3f} | text={h.text}")
