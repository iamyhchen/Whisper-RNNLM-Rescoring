# Whisper + RNNLM
This script is used to train an RNN language model and apply it for Whisper rescoring.

## 三步驟執行
step1. 訓練語言模型
```
python3 train_rnnlm_whisper.py \
  --train_txt dataset/Taipu-char/train.txt \
  --valid_txt dataset/Taipu-char/eval.txt \
  --whisper_tokenizer openai/whisper-medium \
  --language en \
  --task transcribe \
  --rnn_type gru \
  --layers 2 \
  --emb 512 \
  --hid 1024 \
  --dropout 0.3 \
  --seq_len 256 \
  --batch_size 64 \
  --epochs 80 \
  --lr 2e-3 \
  --out_dir model/RNNLM/rnnlm_Taipu_char_whisper_medium_gru
```
> **參數設置**
> - `whisper_tokenizer` 參數要與準備做rescoring的whisper模型相同
> - `rnn_type` 可以指定 lstm 或 gru

step2. 把微調好的 whisper 模型放入`model/Whisper`資料夾，模型要有以下檔案
```
.
├── added_tokens.json
├── config.json
├── generation_config.json
├── merges.txt
├── model.safetensors
├── normalizer.json
├── preprocessor_config.json
├── special_tokens_map.json
├── tokenizer_config.json
└── vocab.json
```
step3. whisper rescoring
```
python3 batch_eval_whisper_rnnlm.py \
  --text ./dataset/data-pinyin-Zhaoan/test/text \
  --audio_paths ./dataset/data-pinyin-Zhaoan/test/audio_paths \
  --asr_model ./model/Whisper/YOUR-WHISPER-MODEL \
  --rnnlm_ckpt ./model/RNNLM/rnnlm_Taipu_char_whisper_medium_gru/rnnlm.pt \
  --out_dir ./pred
  --num_beams 10 \
  --num_return_sequences 10 \
  --lm_weight 0.0 \
  --length_beta 0.0
```
> **參數設置**
> - `num_beams`: whisper beam size，產生幾條候選
> - `num_return_sequences`: 從`num_beams`條候選選幾條給LM計算加權分數
> - `lm_weight`: 語言模型權重
> - `length_beta`: 長度正規化權重
