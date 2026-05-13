# Qwen3-TTS STT 実装プラン

## 目的

Qwen3-TTS-12Hz の speech tokenizer と Talker を流用し、音声からテキストを生成する STT/ASR モデルを実装する。

狙いは、音声を Qwen3-TTS tokenizer の codebook 列へ変換し、特に semantic である codebook0 を Talker に入力して、Qwen3 tokenizer の text token を自己回帰生成すること。フルスクラッチの ASR ではなく、Qwen3-0.6B 由来の Transformer body と text head を初期値として再利用する。

## 前提

- 対象はまず Qwen3-TTS-12Hz-0.6B-Base 相当。
- speech tokenizer は 12.5 Hz、16 codebook 構成。
- codebook0 は semantic、codebook1-15 は acoustic/RVQ residual。
- Qwen3-TTS Talker は text -> codebook0 を主目的として訓練され、MTP が codebook1-15 を生成する。
- STT では MTP は必須ではない。まず Talker body と text head で semantic -> text を学習する。

## 調査結果

### Qwen3-0.6B と Talker の互換性

`config_files/qwen3_config.json` と `config_files/qwen3_tts_config.json` の比較では、Qwen3-0.6B と Qwen3-TTS 0.6B Talker の主要 Transformer 寸法は一致している。

| 項目 | Qwen3-0.6B | Qwen3-TTS Talker |
| --- | --- | --- |
| hidden_size | 1024 | 1024 |
| num_hidden_layers | 28 | 28 |
| intermediate_size | 3072 | 3072 |
| num_attention_heads | 16 | 16 |
| num_key_value_heads | 8 | 8 |
| head_dim | 128 | 128 |
| rope_theta | 1000000 | 1000000 |

このため、Talker body は Qwen3-0.6B の Transformer block を内包できる構造と見てよい。

### text head の再利用

Qwen3 の `Qwen3ForCausalLM` は以下の head を持つ。

```python
self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
```

Qwen3-0.6B では `hidden_size=1024`, `vocab_size=151936` なので、`lm_head.weight` の shape は `[151936, 1024]`。

Talker の hidden state も 1024 次元なので、STT 用に `text_head = nn.Linear(1024, 151936, bias=False)` を追加すれば、Qwen3 の `lm_head.weight` をそのまま初期値として読み込める。

Qwen3 config は `tie_word_embeddings=true` なので、checkpoint に `lm_head.weight` が明示的に無い場合は `model.embed_tokens.weight` を `text_head.weight` に使う。

### 注意点

Qwen3-TTS の既存 `codec_head` は `hidden_size -> codec vocab` の head であり、text 出力には使えない。

```python
self.codec_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
```

また、Qwen3-TTS の `text_embedding` は `text_hidden_size=2048` で、Qwen3-0.6B の `embed_tokens.weight` `[151936, 1024]` とは shape が一致しない。STT の text decoder 入力には、Qwen3 由来の 1024 次元 text embedding を別に持つ方が自然。

## 推奨アーキテクチャ

まずは TTS inference path とは別に、ASR 用の thin wrapper を追加する。

```text
audio waveform
  -> Qwen3-TTS-Tokenizer-12Hz.encode()
  -> audio_codes: [T_audio, 16]
  -> speech prefix embeddings
       codebook0 embedding
       optionally sum codebook1-15 embeddings
  -> Talker Transformer body
  -> ASR text autoregressive region
       Qwen3 text embedding for previous text tokens
       Qwen3 text head for next text token
  -> transcript text ids
```

### 入力表現

最初の実験は codebook0 のみで行う。

```text
[ASR_BOS] + codec0_1 + codec0_2 + ... + codec0_T + [TEXT_BOS] + text_1 + ... + text_N
```

loss は text region のみで計算する。

```text
labels = -100 for ASR_BOS and codec prefix
labels = text ids for text generation positions
```

codebook1-15 は ablation として後から追加する。

```text
speech_embed_t =
    codec_embedding(codebook0_t)
    + sum_i code_predictor.codec_embedding[i - 1](codebook_i_t)
```

この形は既存 `finetuning/sft_12hz.py` の TTS fine-tuning に近い。

## 実装方針

### 1. ASR config の追加

候補:

- `Qwen3TTSASRConfig`
- 既存 `Qwen3TTSConfig` に ASR 用フィールドを追加

最小実装では独立 config よりも、既存 config を読み込んで wrapper 側で ASR 固有設定を持つ方が安全。

必要な ASR 固有設定:

- `asr_bos_token_id`
- `asr_eos_token_id`
- `asr_pad_token_id`
- `use_acoustic_codebooks`
- `freeze_talker`
- `freeze_text_head`
- `init_text_head_from_qwen3`

### 2. ASR model class の追加

候補ファイル:

- `qwen_tts/core/models/modeling_qwen3_tts_asr.py`

候補クラス:

- `Qwen3TTSForSpeechRecognition`

責務:

- 既存 `Qwen3TTSForConditionalGeneration` または `Qwen3TTSTalkerForConditionalGeneration` を保持する。
- `asr_text_embedding = nn.Embedding(text_vocab_size, hidden_size)` を持つ。
- `text_head = nn.Linear(hidden_size, text_vocab_size, bias=False)` を持つ。
- speech code prefix と text prefix を 1024 次元 embedding 列に変換する。
- Talker body に `inputs_embeds` を渡す。
- text region の logits/loss を返す。

既存 Talker の `text_embedding` と `text_projection` は TTS 用として残し、ASR path では原則使わない。

### 3. Qwen3 weight のロード

Qwen3-0.6B checkpoint から以下を読む。

| Qwen3 key | ASR key | 用途 |
| --- | --- | --- |
| `model.embed_tokens.weight` | `asr_text_embedding.weight` | text decoder 入力 |
| `lm_head.weight` | `text_head.weight` | text decoder 出力 |
| `model.embed_tokens.weight` | `text_head.weight` | `lm_head.weight` が無い場合の fallback |

Talker body には Qwen3-TTS checkpoint の重みを使う。Qwen3-TTS checkpoint の Talker body がすでに Qwen3 由来なら、STT ではそちらを初期値にする方が TTS semantic token 空間との接続が残る。

### 4. Dataset の追加

候補ファイル:

- `finetuning/asr_dataset.py`

既に Qwen3-TTS-Tokenizer-12Hz でエンコード済みの WebDataset 形式データセットを使う。形式は `docs/extract_audio_tokens_hf_output_format.md` を前提にする。

入力ディレクトリ:

```text
output_dir/
├── audios/
│   ├── shard-000000.tar
│   └── ...
├── txts/
│   ├── shard-000000.jsonl
│   └── ...
├── data.lst
└── errors.jsonl
```

`data.lst` は各 shard の tar と JSONL metadata を対応付ける manifest として読む。

```text
<tar_path> <jsonl_path> <sample_count> <total_duration>
```

各 tar shard 内の `{sample_id}.npy` は以下の形式。

| 項目 | 値 |
| --- | --- |
| dtype | `int16` |
| shape | `(16, T)` |
| value range | `0..2047` |
| frame rate | Qwen3-TTS-Tokenizer-12Hz |

各 JSONL 行には `id`, `text`, `language_id`, `audio_duration`, `speaker`, `dnsmos`, `num_tokens` などが含まれる。STT 学習の必須フィールドは `id`, `text`, `num_tokens`。`language_id`, `speaker`, `dnsmos`, `audio_duration` は filtering または analysis に使える。

重要な shape 変換:

```python
# stored npy: [16, T]
codes = np.load(...)
codes = torch.from_numpy(codes.astype(np.int64)).transpose(0, 1)
# model/collator internal: [T, 16]
```

`metadata["num_tokens"]` は `.npy` の `T` と一致するはずなので、dataset load 時に検証する。

この形式では音声波形は不要なので、基本 training path では `prepare_asr_data.py` は不要。必要になった場合のみ、既存 `finetuning/prepare_data.py` 相当の変換スクリプトを追加する。

collator 出力:

- `attention_mask`
- `labels`
- `text_region_mask`
- `audio_codes`
- `text_ids`
- optional metadata: `id`, `language_id`, `speaker`, `audio_duration`, `dnsmos`

最初は collator 内で `input_ids` を作るより、model forward 内で embedding 化する方が保守しやすい。

### 5. Training script の追加

候補ファイル:

- `finetuning/sft_asr_12hz.py`

初期実験:

- Talker body: LoRA または一部 freeze
- `asr_text_embedding`: Qwen3 初期値、trainable
- `text_head`: Qwen3 初期値、trainable
- speech tokenizer: freeze
- MTP/code predictor: freeze または未使用

loss:

```text
cross_entropy(text_logits, text_labels, ignore_index=-100)
```

最初は通常の teacher forcing で十分。

## 実験計画

### Phase 0: shape/loading 検証

- Qwen3-TTS 0.6B config で ASR wrapper を instantiate。
- Qwen3 checkpoint の `model.embed_tokens.weight` を `asr_text_embedding` にロード。
- `lm_head.weight` または `model.embed_tokens.weight` を `text_head` にロード。
- dummy `audio_codes` と `text_ids` で forward/loss が通ることを確認。

### Phase 1: codebook0 only

目的:

- semantic token だけで ASR が成立するか確認する。

設定:

- input: codebook0 のみ
- output: text ids
- Talker body: LoRA または最終数 layer のみ trainable
- text embedding/head: trainable

評価:

- CER/WER
- 言語別 CER/WER
- hallucination/repetition
- 数字/固有名詞の誤り

### Phase 2: all codebooks

目的:

- codebook1-15 の acoustic 情報が STT に有効か確認する。

比較:

- codebook0 only
- codebook0 + codebook1-15 sum
- codebook0 + small adapter over all codebook embeddings

期待:

- 多くの内容認識は codebook0 で十分な可能性がある。
- 固有名詞、曖昧音、短い発話、ノイズ条件では acoustic codebook が改善する可能性がある。

### Phase 3: streaming/online 化

目的:

- 12.5 Hz semantic token prefix から incremental transcription できるか確認する。

方式候補:

- chunk ごとに speech prefix を伸ばして text continuation。
- CTC/RNNT ではなく AR text generation のまま、遅延を許容して逐次 decode。

この phase は精度検証後でよい。

## リスク

### hidden distribution mismatch

Qwen3 text head は Qwen3 body の hidden state 分布に最適化されている。Qwen3-TTS Talker は TTS fine-tuning 済みなので、hidden distribution がずれている可能性がある。

対策:

- text head は固定せず fine-tune する。
- 低 LR で Talker body も LoRA fine-tune する。
- text head のみ学習、LoRA 学習、full fine-tune を比較する。

### semantic token の情報落ち

codebook0 は semantic だが、完全な transcript 復元を保証しない。

対策:

- codebook1-15 を入力に含める ablation を行う。
- 数字、固有名詞、句読点、言語切替を重点評価する。

### TTS ordering と STT ordering の違い

TTS は text -> speech の順序で学習されている。STT は speech prefix -> text generation なので、既存 Talker の causal ordering とは異なる。

対策:

- ASR 専用 SFT を必須とする。
- zero-shot 性能には期待しない。
- speech prefix + text suffix の標準 causal LM 形式で安定させる。

### text embedding dimension mismatch

Qwen3-TTS 既存 `text_embedding` は 2048 次元で、Qwen3-0.6B embedding は 1024 次元。

対策:

- ASR 用に `asr_text_embedding: [151936, 1024]` を追加する。
- TTS 用 `text_embedding + text_projection` は触らない。

## 成功基準

最小成功:

- Qwen3-TTS checkpoint と Qwen3 checkpoint から ASR wrapper を初期化できる。
- dummy batch で loss が計算できる。
- 小規模データで overfit できる。

実験成功:

- codebook0 only で meaningful な transcription が出る。
- Qwen3 text head 初期化が random head より収束を改善する。
- codebook1-15 追加の有無について定量比較できる。

実用化判断:

- 対象言語ごとの WER/CER が既存 ASR baseline と比較可能な水準に近づく。
- 長尺音声で repetition/hallucination が許容範囲。
- streaming chunk 処理でも破綻しない。

## 実装順序

1. `Qwen3TTSForSpeechRecognition` の skeleton を追加。
2. Qwen3 text embedding/head のロード関数を追加。
3. dummy forward test を追加。
4. `prepare_asr_data.py` で audio_codes 抽出。
5. `asr_dataset.py` と collator を追加。
6. `sft_asr_12hz.py` を追加。
7. codebook0 only で overfit test。
8. all-codebook ablation。
9. generation API と簡易 CLI を追加。

## 現在の実装状況

初期実装として以下を追加済み。

- `qwen_tts/core/models/modeling_qwen3_tts_asr.py`
  - `Qwen3TTSForSpeechRecognition`
  - Qwen3 text embedding/head ロード helper
  - codebook0 only / all-codebook sum の speech embedding builder
  - teacher-forcing forward と greedy generation
- `finetuning/asr_dataset.py`
  - `data.lst` reader
  - tar shard + JSONL metadata を読む `IterableDataset`
  - `.npy` shape `[16, T]` から internal shape `[T, 16]` への変換
  - ASR teacher-forcing 用 collator
- `finetuning/sft_asr_12hz.py`
  - WebDataset shard から直接 SFT する実験用 training entrypoint

初期 training 例:

```bash
python finetuning/sft_asr_12hz.py \
  --init_tts_model_path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --qwen3_model_path Qwen/Qwen3-0.6B \
  --data_lst /path/to/output_dir/data.lst \
  --output_dir asr_output \
  --batch_size 2 \
  --num_epochs 1
```

## メモ

STT 化の本質は「TTS を逆再生する」ことではなく、「Qwen3-TTS が持つ text/semantic token alignment と Qwen3 の text prior を ASR の初期値として使う」こと。したがって、最初から既存 TTS generation path を無理に反転させるより、ASR 専用 wrapper と training script を追加する方が実装と評価が明確になる。
