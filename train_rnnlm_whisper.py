#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train a BPE-level RNN language model (RNNLM) using the SAME tokenizer as
Hugging Face Transformers' Whisper. Use this LM for N-best rescoring or
shallow fusion with your Whisper ASR.

Example:
python train/train_rnnlm_whisper.py \
  --train_txt data/lm/train.txt \
  --valid_txt data/lm/dev.txt \
  --whisper_tokenizer openai/whisper-small \
  --language zh --task transcribe \
  --rnn_type lstm --layers 2 --emb 512 --hid 1024 --dropout 0.3 \
  --seq_len 256 --batch_size 64 --epochs 10 --lr 2e-3 \
  --out_dir ./rnnlm_zh_whisper_small
"""
import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperTokenizerFast

# -------------------------------
# Dataset: pack text into one long token stream, TBPTT chunks
# -------------------------------
class PackedTextDataset(Dataset):
    """Tokenize lines with Whisper tokenizer (no special tokens), concatenate,
    then expose fixed-length (seq_len+1) windows for next-token LM training."""
    def __init__(self, txt_path: str, tokenizer, seq_len: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        ids: List[int] = []
        for ln in lines:
            enc = tokenizer(
                ln,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False
            )
            ids.extend(enc["input_ids"])
        if len(ids) < (seq_len + 1):
            raise ValueError("Not enough tokens; provide more data or reduce --seq_len.")
        self.tokens = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        # Number of available (seq_len+1) windows
        return (len(self.tokens) - 1) // self.seq_len - 1

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]
        x = chunk[:-1]  # inputs
        y = chunk[1:]   # targets
        return x, y

def collate_pad(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)

# -------------------------------
# Model
# -------------------------------
class RNNLM(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 512, hid: int = 1024, layers: int = 2,
                 dropout: float = 0.3, rnn_type: str = "lstm", tie_weights: bool = False):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        rnn_type = rnn_type.lower()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn_type]
        self.rnn = rnn_cls(emb, hid, num_layers=layers,
                           dropout=dropout if layers > 1 else 0.0,
                           batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hid, vocab_size)
        if tie_weights:
            if emb != hid:
                raise ValueError("For weight tying, emb must equal hid.")
            self.out.weight = self.emb.weight

    def forward(self, x, h=None):
        e = self.emb(x)           # [B, T, E]
        y, h = self.rnn(e, h)     # [B, T, H]
        y = self.drop(y)
        logits = self.out(y)      # [B, T, V]
        return logits, h

# -------------------------------
# Train / Eval
# -------------------------------
@dataclass
class TrainConfig:
    train_txt: str
    valid_txt: str
    whisper_tokenizer: str
    language: str
    task: str
    rnn_type: str
    layers: int
    emb: int
    hid: int
    dropout: float
    tie_weights: bool
    seq_len: int
    batch_size: int
    epochs: int
    lr: float
    betas: Tuple[float, float]
    weight_decay: float
    clip_grad: float
    out_dir: str
    seed: int
    device: str

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_whisper_tokenizer(name: str, language: str, task: str):
    # Same tokenizer Whisper uses; we WON'T add special tokens during encoding
    return WhisperTokenizerFast.from_pretrained(name, language=language, task=task)

@torch.no_grad()
def evaluate(model, loader, device, vocab_size):
    model.eval()
    nll_sum, tok_cnt = 0.0, 0
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits, _ = model(x)
        loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
        nll_sum += loss.item()
        tok_cnt += y.numel()
    avg_nll = nll_sum / max(1, tok_cnt)
    ppl = math.exp(avg_nll)
    return avg_nll, ppl

def train(cfg: TrainConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    tokenizer = get_whisper_tokenizer(cfg.whisper_tokenizer, cfg.language, cfg.task)
    vocab_size = len(tokenizer)

    train_ds = PackedTextDataset(cfg.train_txt, tokenizer, cfg.seq_len)
    valid_ds = PackedTextDataset(cfg.valid_txt, tokenizer, cfg.seq_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_pad)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=0, collate_fn=collate_pad)

    model = RNNLM(vocab_size, emb=cfg.emb, hid=cfg.hid, layers=cfg.layers,
                  dropout=cfg.dropout, rnn_type=cfg.rnn_type,
                  tie_weights=cfg.tie_weights).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            betas=cfg.betas, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_val = float('inf')
    log_path = os.path.join(cfg.out_dir, 'train_log.jsonl')
    open(log_path, 'w', encoding='utf-8').close()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tok_loss_sum, tok_cnt = 0.0, 0

        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device); y = y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                logits, _ = model(x)
                loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
            scaler.scale(loss).backward()
            if cfg.clip_grad > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            scaler.step(opt)
            scaler.update()

            tok_loss_sum += loss.item() * y.numel()
            tok_cnt += y.numel()

            if step % 100 == 0:
                avg_nll = tok_loss_sum / max(1, tok_cnt)
                print(f"Epoch {epoch} Step {step}: train NLL/token={avg_nll:.4f} PPL={math.exp(avg_nll):.2f}")

        train_avg_nll = tok_loss_sum / max(1, tok_cnt)
        train_ppl = math.exp(train_avg_nll)
        val_avg_nll, val_ppl = evaluate(model, valid_loader, device, vocab_size)

        rec = {"epoch": epoch, "train_nll": train_avg_nll, "train_ppl": train_ppl,
               "valid_nll": val_avg_nll, "valid_ppl": val_ppl}
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[Epoch {epoch}] train PPL={train_ppl:.2f} | valid PPL={val_ppl:.2f}")

        if val_avg_nll < best_val:
            best_val = val_avg_nll
            save_path = os.path.join(cfg.out_dir, 'rnnlm.pt')
            torch.save({
                'model_state': model.state_dict(),
                'vocab_size': vocab_size,
                'tok_name': cfg.whisper_tokenizer,
                'language': cfg.language,
                'task': cfg.task,
                'model_hparams': {
                    'emb': cfg.emb, 'hid': cfg.hid, 'layers': cfg.layers,
                    'dropout': cfg.dropout, 'rnn_type': cfg.rnn_type,
                    'tie_weights': cfg.tie_weights,
                }
            }, save_path)
            with open(os.path.join(cfg.out_dir, 'train_config.json'), 'w', encoding='utf-8') as f:
                json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
            print(f"Saved best model to {save_path}")

# -------------------------------
# CLI
# -------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_txt', type=str, required=True)
    p.add_argument('--valid_txt', type=str, required=True)
    p.add_argument('--whisper_tokenizer', type=str, default='openai/whisper-small')
    p.add_argument('--language', type=str, default='zh')
    p.add_argument('--task', type=str, default='transcribe')
    p.add_argument('--rnn_type', type=str, choices=['lstm', 'gru'], default='lstm')
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--emb', type=int, default=512)
    p.add_argument('--hid', type=int, default=1024)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--tie_weights', action='store_true')
    p.add_argument('--seq_len', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.98))
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--clip_grad', type=float, default=1.0)
    p.add_argument('--out_dir', type=str, required=True)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    return TrainConfig(
        train_txt=args.train_txt,
        valid_txt=args.valid_txt,
        whisper_tokenizer=args.whisper_tokenizer,
        language=args.language,
        task=args.task,
        rnn_type=args.rnn_type,
        layers=args.layers,
        emb=args.emb,
        hid=args.hid,
        dropout=args.dropout,
        tie_weights=bool(args.tie_weights),
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        betas=tuple(args.betas),
        weight_decay=args.weight_decay,
        clip_grad=args.clip_grad,
        out_dir=args.out_dir,
        seed=args.seed,
        device=args.device,
    )

if __name__ == '__main__':
    cfg = parse_args()
    train(cfg)
