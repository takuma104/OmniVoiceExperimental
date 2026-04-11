# `OmniVoice.generate()` — `ref_text` / `ref_audio` 指定時の処理フロー

本ドキュメントでは、`OmniVoice.generate()` において `ref_text` と `ref_audio` が渡された場合（**Voice Clone モード**）の処理を、
LLM への入出力と、後段の反復マスク拡散デコード（masked diffusion decoding）に焦点を当てて説明します。

---

## 全体フロー概観

```
ref_audio (waveform)          ref_text (transcript)
      │                                │
      ▼                                │
 [前処理・正規化]                        │
      │                                │
      ▼                                │
 Qwen3TTSTokenizer.encode()            │
  → ref_audio_tokens (C=16, T_ref)     │
      │                                │
      └──────────────┬─────────────────┘
                     │
                     ▼
          [入力シーケンス構築]
   ┌──────────────────────────────────────────────────┐
   │ style_tokens │ text_tokens │ ref_audio │ masked  │
   │  (テキスト)   │  (テキスト)  │ tokens   │ target  │
   └──────────────────────────────────────────────────┘
                     │
                     ▼
          [埋め込み変換 _prepare_embed_inputs]
          テキスト位置 → LLM テキスト埋め込み
          音声位置    → 16 コードブック埋め込みの総和
                     │
                     ▼
             ┌──────────────┐
             │  LLM forward │  (Qwen3 ベーストランスフォーマー)
             └──────────────┘
                     │  hidden_states (B, S, H)
                     ▼
             ┌──────────────┐
             │ audio_heads  │  Linear: H → C × V (C=16 コードブック, V=2049)
             └──────────────┘
                     │  audio_logits (B, C, S, V)
                     ▼
         [反復マスク拡散デコード × 48 ステップ]
           + Classifier-Free Guidance
           + 層別ペナルティ (layer_penalty_factor)
           + Gumbel ノイズによる位置選択
                     │
                     │  generated tokens (C=16, T_target)
                     ▼
        Qwen3TTSTokenizer.decode()
                     │
                     ▼
          [後処理: 無音除去・音量正規化・フェード]
                     │
                     ▼
          output waveform (1, T_out) @ 24 kHz
```

---

## ステップ 1: 前処理 — `_preprocess_all()` → `create_voice_clone_prompt()`

`generate()` は最初に `_preprocess_all()` を呼び、その中で `create_voice_clone_prompt()` を実行します。

### 1-1. 参照音声の正規化

| 処理 | 詳細 |
|------|------|
| ロード | `load_audio(ref_audio, sampling_rate=24000)` でファイルまたは波形タプルを読み込み、モデルのサンプリングレートへリサンプリング |
| モノラル化 | `torch.mean(waveform, dim=0)` で多チャンネルをモノラルへ変換 |
| RMS 計算 | `ref_rms = sqrt(mean(wav^2))` — 後の音量正規化に使用 |
| 音量正規化 | RMS が低い場合 (`0 < ref_rms < 0.1`) に音量を 0.1 レベルへ正規化 |

### 1-2. 参照音声の前処理 (`preprocess_prompt=True` の場合)

```
trim_long_audio()      20 秒超の音声を最大沈黙区間で分割してトリム
                       ※ ref_text が明示指定された場合はスキップ
      ↓
remove_silence()       中間沈黙 200 ms 超を除去
                       先頭沈黙 100 ms 超・末尾沈黙 200 ms 超をトリム
```

### 1-3. ref_text の自動生成 (ref_text が None の場合)

```python
# Whisper ASR モデルが自動ロードされ書き起こしを実行
ref_text = self.transcribe((ref_wav, self.sampling_rate))
```

### 1-4. 参照音声のトークナイズ — Qwen3-TTS-Tokenizer-12Hz

```python
enc_out = self.audio_tokenizer.encode(
    [ref_wav.squeeze().numpy()], sr=self.sampling_rate
)
ref_audio_tokens = enc_out.audio_codes[0].T  # (C, T_ref)
```

- **エンコーダ**: `Qwen3TTSTokenizer`（Qwen3-TTS-Tokenizer-12Hz）
- **フレームレート**: 12.5 Hz（1 秒 = 12.5 フレーム）
- **コードブック数 C**: 16
- **語彙サイズ V**: 2048（有効トークン）+ 1（マスクトークン `audio_mask_id=2048`）= 2049
- **出力形状**: `(C=16, T_ref)` — C コードブックそれぞれに T_ref フレーム分の離散コード

### 1-5. ref_text の後処理

`add_punctuation(ref_text)` — 文末に句読点がない場合は追加（例: "Hello world" → "Hello world."）

---

## ステップ 2: 目標トークン数の推定 — `_estimate_target_tokens()`

```python
est = self.duration_estimator.estimate_duration(
    target_text, ref_text, num_ref_audio_tokens
)
num_target_tokens = max(1, int(est / speed))
```

`RuleDurationEstimator` は文字種別（ラテン=1.0、CJK=3.0、アラビア語=1.5 など 600+ 言語対応）の重みで速度を推定し、
目標テキストの発話フレーム数を計算します。

---

## ステップ 3: 短尺 / 長尺のルーティング

| 条件 | 処理 |
|------|------|
| `target_len ≤ audio_chunk_threshold × 12.5` (デフォルト 30s = 375 フレーム) | `_generate_iterative()` を直接呼び出し |
| 超過 | `_generate_chunked()` でテキストを約 15 秒チャンクに分割し、チャンクごとに `_generate_iterative()` を呼び出してバッチ処理 |

長尺チャンク処理では、2チャンク目以降も同じ `ref_audio_tokens` / `ref_texts` を再利用することで声質の一貫性を保ちます。

---

## ステップ 4: LLM 入力シーケンスの構築 — `_prepare_inference_inputs()`

### 4-1. シーケンス構造

```
cond_input_ids (shape: [1, C=16, N1 + N2 + T_ref + T_target])
┌──────────────────┬──────────────────┬────────────────────┬────────────────────┐
│  style_tokens    │   text_tokens    │  ref_audio_tokens  │  target (all MASK) │
│  (テキストトークン) │  (テキストトークン) │  (音声コード)        │  (マスクトークン)    │
│  [1, C, N1]      │  [1, C, N2]      │  [1, C, T_ref]     │  [1, C, T_target]  │
└──────────────────┴──────────────────┴────────────────────┴────────────────────┘
←────────── audio_mask = False ──────────────┬──────────── audio_mask = True ──────────→
                                             ^ cond_audio_start_idx
```

**注意**: テキスト位置（`audio_mask=False`）のトークン ID は dim=1 全コードブックで同一（`.repeat(C, 1)`）ですが、
埋め込み時にはテキスト埋め込みのみが使われます。

### 4-2. style_tokens (N1 トークン)

```python
style_text = ""
if denoise and ref_audio_tokens is not None:
    style_text += "<|denoise|>"       # 拡散デノイズモードを有効化
style_text += f"<|lang_start|>{lang}<|lang_end|>"
style_text += f"<|instruct_start|>{instruct}<|instruct_end|>"
```

| トークン | 内容 | 例 |
|----------|------|----|
| `<|denoise|>` | Voice Clone 時に参照音声のノイズ除去を示す制御トークン | — |
| `<|lang_start|>...<|lang_end|>` | 言語 ID | `<|lang_start|>en<|lang_end|>` |
| `<|instruct_start|>...<|instruct_end|>` | スタイル指示 or `None` | `<|instruct_start|>None<|instruct_end|>` |

### 4-3. text_tokens (N2 トークン)

```python
full_text = ref_text.strip() + " " + text.strip()   # 参照テキスト + 目標テキスト連結
wrapped_text = f"<|text_start|>{full_text}<|text_end|>"
text_tokens = _tokenize_with_nonverbal_tags(wrapped_text, self.text_tokenizer)
```

- `ref_text` と `text` は空白 1 つで結合
- `[laughter]` / `[sigh]` などの非言語タグは前後文脈から独立してトークナイズ（一貫した ID を保証）
- 日中文字前後の空白は自動除去

### 4-4. ref_audio_tokens (T_ref フレーム)

`VoiceClonePrompt.ref_audio_tokens` をそのまま連結。形状 `[1, C, T_ref]`。

### 4-5. target_audio_tokens (T_target フレーム) — マスクトークンで初期化

```python
target_audio_tokens = torch.full(
    (1, C, T_target), fill_value=audio_mask_id,  # 2048
    dtype=torch.long
)
```

全位置がマスクトークン (`2048`) で埋められた状態から拡散デコードが始まります。

---

## ステップ 5: 埋め込み変換 — `_prepare_embed_inputs()`

### テキスト位置 (`audio_mask=False`)

LLM の標準テキスト埋め込みテーブルを使用:

```python
text_embeds = self.get_input_embeddings()(input_ids[:, 0, :])  # [B, S, H]
```

dim=1 (コードブック次元) の最初の要素 (layer 0) のトークン ID を使用。

### 音声位置 (`audio_mask=True`)

16 コードブックの埋め込みを合算して1ベクトルに集約:

```python
shifted_ids = input_ids + codebook_layer_offsets.view(1, -1, 1)
# Layer k のトークン ID を k × audio_vocab_size だけシフト
# → [Batch, 16, Seq] の各要素がグローバル埋め込み ID へ

audio_embeds = self.audio_embeddings(shifted_ids).sum(dim=1)
# [Batch, 16, Seq, H] → sum over layers → [Batch, Seq, H]
```

`audio_embeddings` のサイズは `(num_audio_codebook × audio_vocab_size, hidden_size)` = `(16 × 2049, H)`。
各レイヤーのトークンに対応する埋め込みを取得して合算します。

### 合成

```python
return torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)
# 音声位置 → audio_embeds、テキスト位置 → text_embeds
```

---

## ステップ 6: LLM フォワードパス — `forward()`

```python
llm_outputs = self.llm(
    inputs_embeds=inputs_embeds,    # (B, S, H)
    attention_mask=attention_mask,  # (B, 1, S, S) または None
    return_dict=True,
)
hidden_states = llm_outputs[0]     # (B, S, H)

logits_flat = self.audio_heads(hidden_states)  # (B, S, C × V)
audio_logits = logits_flat.view(B, S, C, V).permute(0, 2, 1, 3)
#                                              → (B, C=16, S, V=2049)
```

- **LLM**: Qwen3 ベーストランスフォーマー（`hidden_size = H`）
- **attention_mask**: `_generate_iterative` ではシーケンス全域を True にした因果的マスクを使用
- **audio_heads**: 独立した線形レイヤー。LLM の language head とは別物で、16 コードブック分の音声ロジットを一括出力

---

## ステップ 7: Classifier-Free Guidance (CFG)

`_generate_iterative()` では各バッチ項目に対し **Conditional / Unconditional の 2 系列** を構築してまとめて forward します:

| スロット | 内容 | 長さ |
|----------|------|------|
| Conditional (0 〜 B-1) | style + text + ref_audio + masked_target | `N1+N2+T_ref+T_target` |
| Unconditional (B 〜 2B-1) | masked_target のみ (コンテキストなし) | `T_target` |

CFG 合成:

```python
c_log_probs = F.log_softmax(c_logits, dim=-1)   # Conditional
u_log_probs = F.log_softmax(u_logits, dim=-1)   # Unconditional

log_probs = log_softmax(
    c_log_probs + guidance_scale * (c_log_probs - u_log_probs)
)
# guidance_scale = 0.5 (デフォルト)
```

`guidance_scale` を大きくするほどコンテキスト（参照音声・テキスト）の影響が強まります。

---

## ステップ 8: 反復マスク拡散デコード — `_generate_iterative()`

Voice Clone のコアです。**MaskGIT 型の離散マスク拡散** を採用しています。

### 8-1. タイムステップと解除スケジュール

```python
timesteps = _get_time_steps(t_start=0.0, t_end=1.0, num_step=49, t_shift=0.1)
# t_shift 補正: t' = t_shift × t / (1 + (t_shift - 1) × t)
# → 低 SNR 領域 (t 小) を強調、スケジュールを前倒し
```

各ステップで解除するトークン数:

```python
num_to_unmask[step] = ceil(total_mask_tokens × (timesteps[step+1] - timesteps[step]))
```

`total_mask_tokens = T_target × C`（最初はすべてマスク）

### 8-2. 各ステップの処理

```
┌─────────────────────────────────────────────────────────────────────┐
│ for step in range(48):                                              │
│                                                                     │
│  1. LLM forward (batch_input_ids, batch_audio_mask, attention_mask) │
│     → audio_logits (2B, C, S, V)                                   │
│                                                                     │
│  2. CFG ログ確率合成                                                  │
│     c_logits[i] + guidance_scale × (c_logits[i] - u_logits[i])     │
│     → log_probs (1, C, T_target, V)                                 │
│                                                                     │
│  3. マスクトークンの確率を -inf に設定                                   │
│     log_probs[..., audio_mask_id] = -inf                            │
│                                                                     │
│  4. 予測トークンと信頼スコアの計算                                       │
│     pred_tokens = argmax(log_probs)    (class_temp=0 のとき)         │
│     scores = max(log_probs, dim=-1)   (各位置の最大対数確率)           │
│                                                                     │
│  5. 層別ペナルティ適用                                                  │
│     scores -= layer_id × layer_penalty_factor  (layer_penalty=2.3) │
│     → 低 ID コードブックほどスコアが高くなり先に解除される               │
│                                                                     │
│  6. Gumbel ノイズ付加 (position_temperature=4.0)                     │
│     scores += Gumbel(0, 1) / temperature                           │
│     → 同スコアの位置への確率的選択、多様性確保                           │
│                                                                     │
│  7. 既に解除済みの位置をマスク (score = -inf)                           │
│                                                                     │
│  8. top-k (k = num_to_unmask[step]) の位置を選択して解除              │
│     flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]        │
│                                                                     │
│  9. Conditional / Unconditional 両スロットの input_ids を更新         │
└─────────────────────────────────────────────────────────────────────┘
```

### 8-3. 層別ペナルティの意味

コードブック階層はピラミッド構造を持ちます（粗い特徴 → 細かい詳細）。

```
layer_penalty_factor = 2.3
 Layer 0: penalty = 0   → 最も先に解除 (基本音韻・韻律)
 Layer 1: penalty = 2.3
 Layer 2: penalty = 4.6
 ...
 Layer 15: penalty = 34.5 → 最後に解除 (細粒度の音質)
```

この設計により、全コードブック位置が混在した状態でも自然な解除順序が保たれます。

### 8-4. 収束後の出力

```python
return [tokens[i, :, :task.target_lens[i]] for i in range(B)]
# → [(C=16, T_i), ...] のリスト
```

---

## ステップ 9: 音声デコード — `_decode_and_post_process()`

```python
# tokens: (C=16, T_target)
wavs, _ = self.audio_tokenizer.decode(
    Qwen3TTSTokenizerV2EncoderOutput(
        tokens.to(tokenizer_device).T.unsqueeze(0)
        # (C, T) → (T, C) → (1, T, C)  ← decoder 期待フォーマット
    )
)
audio_waveform = torch.Tensor(wavs[0]).unsqueeze(0).cpu()
# → (1, T_out) @ 24 kHz
```

- **デコーダ**: `Qwen3TTSTokenizerV2`（Qwen3-TTS-Tokenizer-12Hz の逆変換）
- 入力フォーマット: `(Batch, Time, Codebooks)` = `(1, T_target, 16)`
- 出力: 24 kHz 波形

長尺チャンク処理の場合は `cross_fade_chunks()` で各チャンクを結合（境界に 0.3 秒の無音バッファ + フェード）。

---

## ステップ 10: 後処理 — `_post_process_audio()`

```
remove_silence()
  中間沈黙 500 ms 超を除去
  先頭沈黙 100 ms 超・末尾沈黙 100 ms 超をトリム

音量正規化
  ref_rms < 0.1 の場合: generated_audio × ref_rms / 0.1
  ref_rms が None (Voice Design モード) の場合: ピーク正規化 → 0.5

fade_and_pad_audio()
  100 ms フェードイン / フェードアウト
  両端に 100 ms 無音パディング
```

---

## テンソル形状サマリー

| フェーズ | テンソル | 形状 | 補足 |
|----------|----------|------|------|
| 入力参照音声 | `ref_wav` | `(1, T_wav)` | 24 kHz モノラル |
| 参照音声トークン | `ref_audio_tokens` | `(C=16, T_ref)` | 12.5 Hz |
| LLM 入力 `input_ids` | — | `(1, C=16, N1+N2+T_ref+T_target)` | 全コードブック同一 ID (テキスト位置) |
| LLM 入力 `audio_mask` | — | `(1, N1+N2+T_ref+T_target)` | `True` = 音声位置 |
| 埋め込み `inputs_embeds` | — | `(2B, S, H)` | Cond + Uncond |
| LLM 出力 `hidden_states` | — | `(2B, S, H)` | — |
| 音声ロジット `audio_logits` | — | `(2B, C=16, S, V=2049)` | — |
| 生成トークン `tokens` | — | `(B, C=16, T_target)` | 48 ステップ反復後 |
| 出力音声 | `audio_waveform` | `(1, T_out)` | 24 kHz |

---

## 主要ハイパーパラメータ (`OmniVoiceGenerationConfig`)

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| `num_step` | 48 | 反復マスク解除のステップ数 |
| `guidance_scale` | 0.5 | CFG スケール。大きいほど参照音声への追従が強まる |
| `t_shift` | 0.1 | タイムステップシフト。小さいほど低 SNR 領域を強調 |
| `layer_penalty_factor` | 2.3 | コードブック層別ペナルティ。低層を先に解除するための係数 |
| `position_temperature` | 4.0 | Gumbel ノイズ温度。高いほど解除位置の選択が確率的になる |
| `class_temperature` | 0.0 | トークンサンプリング温度 (`0` = greedy argmax) |
| `audio_chunk_duration` | 15.0 | 長尺チャンクの分割単位 (秒) |
| `audio_chunk_threshold` | 30.0 | この秒数を超えたらチャンク分割を適用 |

---

## 関連ソースコード

| ファイル | 内容 |
|----------|------|
| `omnivoice/models/omnivoice.py` | `OmniVoice` クラス全体、`generate()`、`_generate_iterative()`、`_prepare_inference_inputs()` |
| `omnivoice/utils/audio.py` | `remove_silence()`、`cross_fade_chunks()`、`fade_and_pad_audio()` |
| `omnivoice/utils/duration.py` | `RuleDurationEstimator` — テキストから音声長推定 |
| `omnivoice/utils/text.py` | `add_punctuation()`、`chunk_text_punctuation()` |
