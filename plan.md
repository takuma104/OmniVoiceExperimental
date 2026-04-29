# Plan: MALLE-style Continuous Mel Tokens for OmniVoice

## Context

OmniVoice (`malle-style` ブランチ) は現在、Qwen3-TTS audio tokenizer (8 codebook, 12Hz) で音声を**離散トークン化**し、Qwen3-0.6B LM をバックボーンに **iterative masked decoding (MaskGCT風)** で並列生成している ([omnivoice/models/omnivoice.py:209-222, 421-443](omnivoice/models/omnivoice.py#L209-L443))。

本実験では MALLE 論文 (arXiv:2407.08551) に倣い、ニューラルコーデックを介さず **Mel spectrogram そのものを連続トークン**として扱い、バックボーンを学習させる。Mel→Wav 変換は既に動作確認済の **BigVGAN 44kHz** (`bigvgan/`、[vocoder.ipynb](vocoder.ipynb)) を用いる。

ただしユーザー選択により、生成パラダイムは MALLE の純粋 AR ではなく、**OmniVoice 既存の iterative-mask 機構を維持しつつトークンだけを連続値化** する **MAR (Masked Autoregressive) 風** にする。これにより OV の packing/processor/generation 配管をほぼそのまま流用しつつ、損失と embedding/head のみ差し替える最小変更で実験可能。

実装方針の合意事項:
- **Generation paradigm**: Iterative masked decoding (現状維持) + 連続 mel
- **Mel config**: 44.1kHz / 128 mels / hop=512 / n_fft=2048 (frame rate ≈ 86.13Hz)、`bigvgan/weights/44k/` を vocoder として使用
- **Feature scope (v1)**: Pre-net + 連続出力 head + L1+L2 regression loss のみ。Latent sampling / flux loss / post-net は後続に回す
- **Stop prediction**: Iterative-mask は target length 既知の前提のため、stop head は**入れない**。長さは既存 `RuleDurationEstimator` を流用 (frame_rate のみ 12 → 86.13 に置換)

---

## アーキテクチャ変更 ([omnivoice/models/omnivoice.py](omnivoice/models/omnivoice.py))

`OmniVoiceConfig` ([L157-L182](omnivoice/models/omnivoice.py#L157-L182)) に下記を追加し、離散用 field は MALLE モードで未使用にする:
- `mel_mode: bool = False` (新フラグ。`True` で MALLE-style)
- `num_mels: int = 128`
- `mel_pad_value: float = 0.0` (mask 位置の代入値、実際は学習可能ベクトルで置換)

`OmniVoice.__init__` を `mel_mode` で分岐:

```python
if self.config.mel_mode:
    # Pre-net: 3-layer MLP (Tacotron風)
    hidden = self.config.llm_config.hidden_size
    self.mel_prenet = nn.Sequential(
        nn.Linear(config.num_mels, hidden), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(hidden, hidden),         nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(hidden, hidden),
    )
    # Mask 用学習可能 embedding (shape [hidden])
    self.mel_mask_embed = nn.Parameter(torch.zeros(hidden))
    # Output head: hidden -> num_mels (regression)
    self.mel_head = nn.Linear(hidden, config.num_mels)
else:
    # 既存の audio_embeddings / audio_heads を維持
```

`_prepare_embed_inputs` ([L353-L373](omnivoice/models/omnivoice.py#L353-L373)) を分岐:
- mel_mode で呼ばれる場合、`input_ids` ではなく `mel_inputs: [B, T, num_mels]` と `mel_mask: [B, T]` (frame-level mask) を受け取る別経路を作る
- audio 領域: `inputs_embeds = mel_prenet(mel_inputs)`、masked frame は `mel_mask_embed` で上書き
- text 領域: 従来通り `text_embeds = self.get_input_embeddings()(text_ids)`
- `audio_mask`: 従来通り audio フレームを示す bool マスクを渡し、上記 2 系統を `torch.where` で結合

`forward` ([L375-L448](omnivoice/models/omnivoice.py#L375-L448)) を分岐:
- mel_mode: 出力 `hidden_states` から `mel_pred = self.mel_head(hidden_states)` ([B, T, num_mels])
- 損失: regression のみ。`mel_loss_mask: [B, T]` (masked & not-prompt な位置のみ True) で
  - `loss = F.l1_loss(mel_pred[mel_loss_mask], mel_target[mel_loss_mask]) + F.mse_loss(...)`
- 既存の `OmniVoiceModelOutput.logits` には `mel_pred` を入れて返す (互換のため)

---

## データパイプライン変更

### parquet からの on-the-fly mel 計算

新クラス `ParquetDatasetReader` を [omnivoice/data/dataset.py](omnivoice/data/dataset.py) に追加 (既存の `JsonlDatasetReader` を雛形とする):

- 入力: `encode_audio_tokens_to_parquet.py` ([omnivoice/scripts/encode_audio_tokens_to_parquet.py:86-97](omnivoice/scripts/encode_audio_tokens_to_parquet.py#L86-L97)) と同形式の parquet (`row_id`, `transcribe`, `audio_data` bytes)
- pyarrow で行単位読み出し、shard を rank/worker で分割 (既存 `JsonlDatasetReader` の分散ロジックを移植)
- 各行で:
  1. `load_audio_bytes(audio_data, sample_rate=44100)` ([omnivoice/utils/audio.py:91-121](omnivoice/utils/audio.py#L91-L121)) で waveform `(1, T)` を取得
  2. RMS 正規化 (既存 `SampleDecoder` 同様)
  3. yield `{"audio": waveform, "label": {"text": transcribe, ...}}` (mel 計算は processor 側に遅延、collate worker での GPU 化を可能にするため)

### 新 processor: `MelSampleProcessor`

[omnivoice/data/processor.py](omnivoice/data/processor.py) に `OmniVoiceSampleProcessor` を雛形にして追加:

- BigVGAN の mel hyper-params を保持 (`from bigvgan.meldataset import get_mel_spectrogram`、`h = AttrDict(bigvgan/weights/44k/config.json)`)
- `__call__(sample)`:
  1. `audio = sample["audio"]` (waveform `(1, T)`)
  2. `mel = get_mel_spectrogram(audio, h)` → `[1, num_mels=128, T_frame]` → squeeze + transpose → `[T_frame, num_mels]`
  3. text/style tokens は既存ロジックを再利用 ([processor.py:92-127](omnivoice/data/processor.py#L92-L127))
  4. **Frame-level masking** (現状の per-token masking と相似):
     - `prompt_length = int(T_frame * prompt_ratio)`
     - `mask_ratio ~ U(0, 1)`
     - `frame_mask = torch.rand(T_frame - prompt_length) < mask_ratio`
     - `mel_input = mel.clone()`、`mel_input[prompt_length:][frame_mask] = NaN` (NaN を mask placeholder にして collate でマスク embed に置換)
     - `mel_loss_mask = frame_mask` (masked 位置のみ loss)
  5. text token 部分には dummy mel フレーム (zeros) を挿入し、`audio_mask` で識別
- 戻り値:
  ```
  {
    "text_input_ids": [N_text],         # text+style トークン id
    "mel_input": [T_frame, num_mels],   # NaN が masked
    "mel_target": [T_frame, num_mels],  # ground truth
    "mel_loss_mask": [T_frame],         # bool, loss 計算位置
    "audio_mask": [N_text + T_frame],   # mel フレーム位置を示す bool
    "length": N_text + T_frame
  }
  ```

### Collator / Packing 調整

- [omnivoice/data/batching.py:108-167](omnivoice/data/batching.py#L108-L167) `PackingIterableDataset` は length ベースで packing しているため、mel フレーム長で組み直すだけで動く想定
- [omnivoice/data/collator.py:30-93](omnivoice/data/collator.py#L30-L93) `PackingDataCollator` は `[C, L]` 連結を行うので、mel-mode 用に **新クラス `MelPackingDataCollator`** を作る:
  - text 部分: `[N_text]` を id で連結
  - mel 部分: `[T_frame, num_mels]` を frame 軸で連結
  - 最終 batch は `{text_input_ids, mel_input, mel_target, mel_loss_mask, audio_mask, document_ids, position_ids}`

### Builder 更新 ([omnivoice/training/builder.py](omnivoice/training/builder.py))

`build_dataloaders` ([L125-L182](omnivoice/training/builder.py#L125-L182)) で `config.mel_mode` フラグ参照:
- True なら `ParquetDatasetReader` + `MelSampleProcessor` + `MelPackingDataCollator` を構築
- データ config JSON に parquet パスのリストを記述する形に拡張 (`prepare_data_manifests_from_json` も parquet パス対応の関数を追加)

### Training config ([omnivoice/training/config.py](omnivoice/training/config.py)) に追加

```python
mel_mode: bool = False
num_mels: int = 128
mel_sample_rate: int = 44100
mel_n_fft: int = 2048
mel_hop_size: int = 512
mel_win_size: int = 2048
mel_fmin: int = 0
mel_fmax: Optional[int] = None
bigvgan_config_path: str = "bigvgan/weights/44k/config.json"
parquet_data_paths: List[str] = field(default_factory=list)  # train/dev parquet path 群
```

既存の `audio_vocab_size`/`num_audio_codebook`/`audio_codebook_weights` は `mel_mode=True` 時には未使用となる (config 自体には残す)。

---

## Generation (推論) 変更

`_prepare_inference_inputs` ([omnivoice/models/omnivoice.py:1051-1130](omnivoice/models/omnivoice.py#L1051-L1130)) を mel_mode 用に新メソッド `_prepare_mel_inference_inputs` に分岐:
- target length: 現状の `_estimate_target_tokens` を流用 (frame_rate を `audio_tokenizer.config.frame_rate` ではなく `mel_sample_rate / mel_hop_size = 86.13` に置換)
- mel_input: 全位置 NaN 埋め (= mask)、ref audio がある場合は ref mel を prompt 領域に prepend
- text/style は従来通り

`_generate_iterative` ([L1132-L1284](omnivoice/models/omnivoice.py#L1132-L1284)) を mel_mode 用に新メソッド `_generate_iterative_mel` に分岐:
- 各 step:
  1. forward pass で `mel_pred: [B, T_frame, num_mels]` を取得
  2. 連続値なので **confidence score 不要**、unmask 順は **ランダム** (各 step で k フレームを残 mask からランダム選択)
  3. 選択フレームの値を `mel_pred` で確定し、`mel_input` の該当位置を `mel_pred` 値 (NaN→値) で更新
- N step 完了後、`mel_final: [B, T_frame, num_mels]` を返す
- 既存の CFG (`guidance_scale`) は `c_logits / u_logits` を `c_pred / u_pred` に読み替え、`pred_cfg = c_pred + s * (c_pred - u_pred)` で実装

`_decode_and_post_process` ([L697-L735](omnivoice/models/omnivoice.py#L697-L735)) を mel_mode 用に分岐:
- `audio_tokenizer.decode` の代わりに **BigVGAN** を呼ぶ:
  ```python
  with torch.inference_mode():
      wav = self.vocoder(mel_final.transpose(1, 2))  # [B, num_mels, T] -> [B, 1, T_audio]
  ```
- vocoder は `OmniVoice.from_pretrained` で `mel_mode=True` 時に `BigVGAN.from_pretrained('bigvgan/weights/44k/')` をロードし `model.vocoder` に保持

---

## 修正対象ファイルまとめ

| ファイル | 変更内容 |
|---|---|
| [omnivoice/models/omnivoice.py](omnivoice/models/omnivoice.py) | `OmniVoiceConfig` 拡張 / `OmniVoice.__init__` `_prepare_embed_inputs` `forward` を mel_mode 分岐 / 新 `_prepare_mel_inference_inputs` `_generate_iterative_mel` / `from_pretrained` で BigVGAN ロード / `_decode_and_post_process` で vocoder 切替 |
| [omnivoice/data/dataset.py](omnivoice/data/dataset.py) | `ParquetDatasetReader` 追加 (pyarrow ベース、`audio_data` bytes → waveform) |
| [omnivoice/data/processor.py](omnivoice/data/processor.py) | `MelSampleProcessor` 追加 (on-the-fly mel + frame-level random masking) |
| [omnivoice/data/collator.py](omnivoice/data/collator.py) | `MelPackingDataCollator` 追加 |
| [omnivoice/training/config.py](omnivoice/training/config.py) | `mel_mode` 関連フィールド追加 |
| [omnivoice/training/builder.py](omnivoice/training/builder.py) | `mel_mode` で reader/processor/collator を分岐 |
| [examples/](examples/) (新規) | `mel_train_config.json` / `mel_data_config.json` サンプル追加 |

`bigvgan/` 配下は変更しない (`get_mel_spectrogram`, `BigVGAN.from_pretrained` を import するのみ)。

既存の Qwen3-TTS tokenizer 関連コード ([omnivoice/scripts/encode_audio_tokens_to_parquet.py](omnivoice/scripts/encode_audio_tokens_to_parquet.py) 等) は `mel_mode=False` 時の動作維持のため触らない。

---

## 既存実装の再利用ポイント

- [bigvgan/meldataset.py:51-143](bigvgan/meldataset.py#L51-L143): `mel_spectrogram()` / `get_mel_spectrogram()` を training/inference 両方で使用
- [bigvgan/bigvgan.py:414-498](bigvgan/bigvgan.py#L414-L498): `BigVGAN.from_pretrained()` を vocoder ロードに使用
- [omnivoice/utils/audio.py:91-121](omnivoice/utils/audio.py#L91-L121): `load_audio_bytes()` で parquet bytes → waveform
- [omnivoice/utils/duration.py](omnivoice/utils/duration.py) `RuleDurationEstimator`: target frame 長推定 (frame_rate 切替のみ必要)
- [omnivoice/data/batching.py](omnivoice/data/batching.py) `PackingIterableDataset`: length ベース packing はそのまま流用
- [omnivoice/scripts/encode_audio_tokens_to_parquet.py:86-97](omnivoice/scripts/encode_audio_tokens_to_parquet.py#L86-L97): pyarrow 読み出しパターンを `ParquetDatasetReader` で参考

---

## 検証手順 (Verification)

実装完了後、以下の順で動作確認する:

1. **Unit-level smoke test** ([vocoder.ipynb](vocoder.ipynb) と同じセットアップで):
   - 新規ノートブック `mel_pipeline.ipynb` を作り、parquet の 1 行を `ParquetDatasetReader` 経由で読み出し、`MelSampleProcessor` を通して shape (`mel_input`, `mel_target`, `mel_loss_mask`) を確認
   - `get_mel_spectrogram` の出力が BigVGAN config と一致しているか (T_frame ≈ T_audio / 512) を assert
   - 計算した mel を直接 BigVGAN に通して、元音声が再構築できることを確認 (vocoder.ipynb と同等)

2. **Forward/backward smoke test**:
   - `OmniVoiceConfig(mel_mode=True, num_mels=128, ...)` で `OmniVoice` を構築
   - 1 batch を `forward()` に通し、loss が finite かつ backward が通ることを確認
   - 形状: `mel_pred.shape == (B, T, 128)`、`loss.requires_grad == True`

3. **小規模 overfit テスト**:
   - parquet の 8 サンプル程度を 1000 step over-fit させて train loss が単調減少 (例: 5.0 → 0.1 以下) することを確認
   - 推論 → BigVGAN → wav で、入力音声に近い波形が出ることを聴感確認

4. **本訓練の小規模ラン**:
   - 既存 [train.sh](train.sh) を雛形に `mel_train.sh` を作り、parquet config と `mel_mode=True` を渡して数千 step 訓練
   - wandb に train_loss, eval_loss, GPU memory, throughput を記録
   - 1000 step ごとに評価サンプルを wav 出力し、聴感品質を確認

5. **比較**:
   - 同じバックボーン (Qwen3-0.6B) / 同データ量で、現行 discrete-token 版との train loss / 学習速度 / sample 品質を比較し、wandb report にまとめる

---

## v1 完了後の拡張候補 (今回はやらない)

- Latent Sampling Module + KL loss (MALLE 3.2.2)
- Spectrogram flux loss (MALLE 3.3)
- Convolutional post-net (MALLE 3.2.3)
- Reduction factor r > 1 (複数フレーム同時予測でシーケンス短縮)
- Pure AR モード (causal mask + teacher forcing) を別 mode として並走
