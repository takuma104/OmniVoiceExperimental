# OmniVoice STT 実装プラン

## 目的

OmniVoice の既存 TTS checkpoint を土台に、音声 token から transcript text token を自己回帰生成する STT/ASR モデルを実装する。

Qwen3-TTS で確認できた「speech prefix を与え、LLM body のみを学習して STT 化する」方針を OmniVoice に移植する。ただし OmniVoice は Higgs Audio の 8 codebook を直接扱う single-stage NAR TTS であり、8 codebook は acoustic token 群として扱うのが自然である。そのため、STT 入力も既存 OmniVoice と同じく all-codebook embedding sum を標準方針にする。

## 前提

- 元の training loop は `omnivoice/training/trainer.py` を流用する。
- Dataset / manifest / WebDataset reader / codec token は既存 TTS training と同じものを使う。
- Audio tokenizer は OmniVoice 既定の Higgs Audio tokenizer を使う。
- OmniVoice の backbone は Qwen3-0.6B 由来の LLM body で初期化されている。
- 現状 OmniVoice には text generation 用 head がないため、Qwen3-0.6B から `lm_head` を持ってくる。
- STT の text generation は NAR ではなく AR にする。
- loss は生成 text 部分のみで計算する。
- 初期実験では LLM body のみを trainable とし、embedding/head は freeze する。

## 基本方針

既存 `OmniVoice` を直接大きく変更するのではなく、ASR 専用の thin wrapper を追加する。

```text
audio tokens [C=8, T]
  -> OmniVoice audio_embeddings, codebook sum
  -> OmniVoice / Qwen3 LLM body
  -> Qwen3-0.6B text_head
  -> AR transcript tokens
```

Qwen3-TTS STT 実装では ASR 用 text embedding を新規追加する必要があったが、OmniVoice では `self.llm.get_input_embeddings()` が Qwen3 text embedding として既に存在する。このため、まずは既存 LLM embedding をそのまま使い、追加するのは text output head のみにする。

## 推奨アーキテクチャ

### 追加モデル

候補ファイル:

- `omnivoice/models/omnivoice_asr.py`

候補クラス:

- `OmniVoiceForSpeechRecognition`

責務:

- 既存 `OmniVoice` checkpoint を読み込む。
- `audio_embeddings` と `llm` body を再利用する。
- `audio_heads` は ASR では使わず freeze する。
- `text_head = nn.Linear(hidden_size, text_vocab_size, bias=False)` を追加する。
- Qwen3-0.6B の `lm_head.weight` を `text_head.weight` にロードする。
- checkpoint に `lm_head.weight` が無い場合は `model.embed_tokens.weight` を fallback として使う。
- teacher forcing forward と greedy / sampling generation を提供する。

初期 freeze 方針:

| Module | 初期状態 | 理由 |
| --- | --- | --- |
| `llm` body | trainable | speech prefix から text へ写像する主学習対象 |
| `llm.embed_tokens` | frozen | Qwen text token 空間を保つ |
| `audio_embeddings` | frozen | OmniVoice の codec token 空間を保つ |
| `text_head` | frozen | Qwen の text prior を保つ |
| `audio_heads` | frozen / unused | ASR loss では使わない |

うまく収束しない場合のみ、`text_head`、`audio_embeddings`、LoRA adapter の順に trainable 範囲を広げる。

## 入力形式

最初は task token + optional language token + audio prefix + text suffix の causal LM 形式にする。

```text
<|asr|>
<|lang_start|>ja<|lang_end|>
audio_1 audio_2 ... audio_T
<|text_start|>
text_1 text_2 ... text_N <|text_end|>
```

実装上は audio token 位置だけ `audio_mask=True` にし、それ以外は text token embedding を使う。

### Audio embedding

既存 OmniVoice と同じく、各時刻の 8 codebook embedding を sum する。

```python
shifted_ids = audio_tokens + codebook_layer_offsets
audio_embeds = audio_embeddings(shifted_ids).sum(dim=1)
```

STT でも all 8 codebooks を常に入力する。Qwen3-TTS でも単一 semantic codebook 入力より全 codebook embedding の sum の方が性能が出たため、OmniVoice では all-codebook 入力のみを標準実験にする。

必要になった場合の比較候補:

- all-codebook embedding sum
- all-codebook embedding sum + small adapter
- codebook weighted sum

## Attention mask

単純な causal mask でも学習は可能だが、OmniVoice の NAR 的な音声表現を活かすため、最初から prefix-LM mask を入れるのが望ましい。

```text
audio prefix 内:
  bidirectional

text suffix:
  all audio prefix + previous text tokens のみ参照

packed batch:
  document_ids が異なる sample 間は参照禁止
```

つまり text token 生成時は音声全体を条件として見られるが、未来の text token は見ない。

既存 `document_ids` packing を利用しつつ、ASR 用に以下の追加 mask を collator から渡す。

- `audio_mask`: audio token 位置
- `text_region_mask`: text generation 位置
- `document_ids`: packed sample 境界
- `position_ids`: sample 内 position

## Loss

loss は transcript 部分だけで計算する。

```text
logits at <|text_start|> -> text_1
logits at text_1         -> text_2
...
logits at text_N         -> <|text_end|>
```

Audio prefix、task token、language token、padding は `-100`。

`labels` は ASR では `[B, L]` でよい。既存 TTS の audio loss は `[B, C, L]` なので、ASR wrapper 側で text CE を独立して計算する。

```python
loss = F.cross_entropy(
    text_logits[:, :-1].reshape(-1, vocab_size),
    labels[:, 1:].reshape(-1),
    ignore_index=-100,
)
```

実装時は `<|text_start|>` 位置から `text_1` を予測できるよう、collator 側か model 側のどちらか一方で shift を明確に管理する。

## Dataset / Processor

既存 dataset をそのまま利用する。

候補追加:

- `OmniVoiceASRSampleProcessor`
- `ASRPackingDataCollator`

既存流用:

- `omnivoice/data/dataset.py`
  - `prepare_data_manifests_from_json`
  - `WebDatasetReader`
  - `SampleDecoder`
- `omnivoice/data/batching.py`
  - `PackingIterableDataset`

### Processor 出力

```python
{
    "input_ids": ...,          # text/token placeholder sequence
    "audio_tokens": ...,      # [C, T_audio] or packed into input_ids format
    "audio_mask": ...,        # [L]
    "text_region_mask": ...,  # [L]
    "labels": ...,            # [L], text-only loss
    "position_ids": ...,      # [L]
    "length": ...,
}
```

既存 `PackingDataCollator` は `input_ids` と `labels` が `[C, L]` であることを前提にしているため、ASR 用 collator を分ける方が安全。

## Training loop

`OmniTrainer` は基本的にそのまま使う。

変更が必要な可能性がある箇所:

- logging で `batch["document_ids"]` と `batch["audio_mask"]` を前提にしている。
- ASR では text loss token 数を `labels != -100` から数える方が自然。
- `evaluate()` は `outputs.loss` を返せばそのまま使える。

最小対応:

- ASR collator でも `document_ids` と `audio_mask` を返す。
- `trainer.py` の token logging は audio/text mask が無い batch でも落ちないようにする。

## Config

既存 `TrainingConfig` に ASR 用フィールドを足すか、ASR 専用 config を追加する。

最初は既存 config 拡張でよい。

候補フィールド:

```python
task: str = "tts"  # "tts" | "asr"
asr_qwen3_model_path: str = "Qwen/Qwen3-0.6B"
asr_freeze_text_embedding: bool = True
asr_freeze_text_head: bool = True
asr_freeze_audio_embeddings: bool = True
asr_train_llm_body: bool = True
asr_codebook_mode: str = "all_sum"
asr_attention_mode: str = "prefix_lm"
```

ASR 専用 entrypoint に分ける場合:

- `omnivoice/cli/train_asr.py`
- `omnivoice/training/asr_builder.py`

既存 TTS training への影響を避けるなら、こちらの分離案を推奨する。

## 初期実験設定

```json
{
  "task": "asr",
  "init_from_checkpoint": "k2-fsa/OmniVoice",
  "llm_name_or_path": "Qwen/Qwen3-0.6B",
  "asr_qwen3_model_path": "Qwen/Qwen3-0.6B",
  "asr_codebook_mode": "all_sum",
  "asr_attention_mode": "prefix_lm",
  "asr_freeze_text_embedding": true,
  "asr_freeze_text_head": true,
  "asr_freeze_audio_embeddings": true,
  "asr_train_llm_body": true,
  "learning_rate": 1e-5,
  "warmup_ratio": 0.03,
  "batch_tokens": 8192,
  "mixed_precision": "bf16"
}
```

LR はまず `1e-5` から始め、loss が動きにくい場合に `2e-5` から `5e-5` を試す。

## 実装順序

1. `OmniVoiceForSpeechRecognition` skeleton を追加する。
2. Qwen3 text head loader を追加する。
3. dummy `audio_tokens` + `text_ids` で forward/loss が通ることを確認する。
4. ASR processor / collator を追加する。
5. prefix-LM attention mask を追加する。
6. `train_asr.py` または `task="asr"` branch を追加する。
7. 100 sample 程度で overfit test を行う。
8. greedy generation API を追加する。
9. text_head frozen / trainable、audio_embeddings frozen / trainable を比較する。
10. 必要に応じて all-codebook sum + adapter などの入力表現を比較する。

## 評価

最小評価:

- train loss
- dev loss
- 100 sample overfit で transcript が再現できるか
- greedy decode の repetition / hallucination

定量評価:

- CER
- WER
- language 別 CER/WER
- duration bucket 別 CER/WER
- 数字・固有名詞・句読点の誤り
- all-codebook 入力表現別比較

推奨比較:

| 実験 | 入力 | trainable | head |
| --- | --- | --- | --- |
| A | all-codebook sum | LLM body only | frozen |
| B | all-codebook sum | LLM body + text_head | trainable |
| C | all-codebook sum | LoRA only | frozen |
| D | all-codebook sum | LLM body + audio_embeddings | frozen head |
| E | all-codebook sum + adapter | LLM body only | frozen |

## STS を見据えた設計

将来的に Speech-to-Speech 化するなら、ASR 実装時点で task token と adapter 境界を入れておくとよい。

### Task token

```text
<|asr|>
<|tts|>
<|sts|>
<|src_lang_start|>...<|src_lang_end|>
<|tgt_lang_start|>...<|tgt_lang_end|>
```

ASR では:

```text
<|asr|> source_audio -> source_text
```

STS では将来的に:

```text
<|sts|> source_audio + optional target speaker prompt -> target_audio
```

または中間 text を挟む multi-task:

```text
source_audio -> source_text -> target_audio
```

### LoRA / Adapter

ASR fine-tuning で TTS 能力を壊さないよう、full body fine-tune とは別に LoRA / adapter training path を用意するのが望ましい。

推奨:

- ASR adapter
- TTS adapter
- STS adapter

初期実験は full body trainable でよいが、STS 統合を考えると adapter 版も早めに作る価値がある。

### Shared token layout

ASR と STS で processor を分けすぎると後で統合が難しくなるため、内部表現は以下に寄せる。

- mixed sequence of text/audio regions
- `audio_mask`
- `text_region_mask`
- `loss_region_mask`
- `task_id` or task special token
- `document_ids`

## リスク

### Acoustic codebook 入力の情報量

OmniVoice の Higgs Audio 8 codebook は acoustic token 群として扱う。STT では transcript に直接関係する情報だけでなく、音素・発音・曖昧音・固有名詞などの復元に acoustic residual 情報が効く可能性が高い。

対策:

- 既存 OmniVoice と同じ all-codebook embedding sum を標準入力にする。
- 入力側の自由度を増やしたい場合は codebook 削減ではなく、all-codebook sum 後の adapter や codebook weighting を検討する。

### NAR body を AR text generation に使う mismatch

OmniVoice は bidirectional / diffusion-style objective で TTS 学習されている。STT では text suffix を causal に生成するため、attention pattern と objective が変わる。

対策:

- prefix-LM mask にして audio prefix は bidirectional のまま活かす。
- text suffix のみ causal にする。
- zero-shot には期待せず SFT 前提にする。

### text_head distribution mismatch

Qwen3 text head は Qwen3 causal LM hidden state に最適化されている。OmniVoice fine-tuned body の hidden distribution とはズレがある可能性がある。

対策:

- 初期は frozen head で Qwen prior を保つ。
- 収束が悪い場合は text_head のみ低 LR で unfreeze する。
- head trainable / frozen の ablation を必ず行う。

### TTS 能力の破壊

LLM body を ASR に full fine-tune すると、既存 TTS 能力が落ちる可能性が高い。

対策:

- ASR 専用 checkpoint として割り切る。
- STS 統合を見据える場合は LoRA / adapter を併用する。
- 必要なら TTS loss との multi-task training を後段で検討する。

## 成功基準

最小成功:

- OmniVoice checkpoint と Qwen3 text head から ASR wrapper を初期化できる。
- dummy batch で loss が計算できる。
- 小規模 sample で overfit できる。

実験成功:

- all-codebook 入力で meaningful な transcription が出る。
- LLM body-only training で dev CER/WER が継続的に改善する。
- all-codebook sum で安定して学習できる。

実用化判断:

- 対象言語ごとの CER/WER が既存 ASR baseline と比較できる水準に近づく。
- 長尺 audio で repetition / hallucination が許容範囲。
- STS へ拡張可能な task-token layout と checkpoint 分離が維持できている。

## メモ

OmniVoice STT 化の本質は、TTS の NAR decoder をそのまま逆向きに使うことではない。OmniVoice が持つ codec token embedding と Qwen3-initialized body を speech-conditioned text LM として再利用し、ASR 専用 objective で再配線することにある。

したがって最初の勝ち筋は以下。

```text
OmniVoice checkpoint
+ Qwen3-0.6B text_head
+ all 8 codebook speech prefix
+ prefix-LM attention
+ text-only AR loss
+ LLM body-only training
```
