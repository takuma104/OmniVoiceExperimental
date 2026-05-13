# OmniVoice の純粋AR版 実装プラン

## 設計方針: 「Parallel-codebook AR」を選ぶ

複数の選択肢があります:

| 選択肢                             | 説明                                 | 既存コード流用度   | 比較の妥当性     |
| ------------------------------- | ---------------------------------- | ---------- | ---------- |
| **A. Parallel-codebook AR**     | 各時刻 t で 8 codebook を並列予測。時間軸方向にAR。 | ★★★ ほぼそのまま | ◎ 同条件比較に適切 |
| B. Delay-pattern AR (MusicGen式) | codebook k を k ステップ遅延              | ★★ 多少改造    | ◎          |
| C. Flatten AR                   | 時間×codebook を1次元に展開                | ★ 大改造      | △ 系列長 8倍   |
| D. RQ-Transformer (CosyVoice2式) | depth transformerを足す               | ☆ 別物       | × 比較不能     |

→ **論文との純粋比較**という目的、かつ「既存コードをできるだけ流用」という制約から **A. Parallel-codebook AR** を採用します。これは現行 NAR と以下だけが違います:

- attention が**causal**(NARはbidirectional)
- 損失が**1ステップシフト next-token CE**(NARはmasked CE)
- 推論が**逐次サンプリング+KVキャッシュ**(NARは32-step iterative unmasking)

入出力テンソル形状 `(B, C, L)`、codebook 別 embedding/head、 codebook weighted loss、データパイプライン、packing は**完全に維持**できます。

---

## 変更マップ(ファイル単位)

### 1. モデル本体 — [omnivoice/models/omnivoice.py](vscode-webview://1cgucbmtq4mq62usavvdp4b7hanckuvl2m0uhl9bph9fbhthg8f0/omnivoice/models/omnivoice.py)

`OmniVoice.forward` の **attention マスク構築だけ** を AR モードで分岐:

- 現状 [omnivoice.py:387-398](vscode-webview://1cgucbmtq4mq62usavvdp4b7hanckuvl2m0uhl9bph9fbhthg8f0/omnivoice/models/omnivoice.py#L387-L398) は `_mask_mod_packed` (`document_ids[q] == document_ids[kv]`) で双方向ブロックマスクを作っている
- AR 用に `_mask_mod_packed_causal` を追加し、`same_doc & (q_idx >= kv_idx)` にする
- フラグは `config.ar_mode` (新設) または別クラス `OmniVoiceAR(OmniVoice)` をサブクラス化

`forward` 内のロス計算は**現状のままで動く** — labels に -100 を入れる位置を processor 側で1ステップずらすだけで、既存の `cross_entropy(..., ignore_index=-100)` が自然に next-token loss になる。

### 2. EOS トークン

audio側に **`audio_eos_id`** を追加(現状 vocab=1025: 1024 valid + 1 mask 。ARでは mask 不要なので index 1024 を EOS に転用、もしくは 1026 化)。

- 学習時: 各サンプルの音声末尾に1ステップ追加(全 codebook に EOS を入れる、または codebook 0 のみ EOS)
- 推論時: codebook 0 が EOS を出したら停止

### 3. データプロセッサ — [omnivoice/data/processor.py](vscode-webview://1cgucbmtq4mq62usavvdp4b7hanckuvl2m0uhl9bph9fbhthg8f0/omnivoice/data/processor.py)

`OmniVoiceARSampleProcessor` を追加。基本ロジックは [omnivoice/data/processor.py:66-174](vscode-webview://1cgucbmtq4mq62usavvdp4b7hanckuvl2m0uhl9bph9fbhthg8f0/omnivoice/data/processor.py#L66-L174) を踏襲:

- style / text の組み立ては**完全に同じ**(`<|denoise|>`, `<|lang_start|>`, instruct, `<|text_start|>` 等もそのまま — voice cloning では prompt audio を `audio_inputs` の前半として concat、これも同じ)
- 違うのはマスキング処理:
    - `audio_inputs = audio_tokens` (マスクなし、teacher forcing)
    - `audio_labels[:, :-1] = audio_tokens[:, 1:]` (1ステップ左シフト)
    - `audio_labels[:, -1] = EOS_ID` (最後のラベル位置で EOS を予測)
    - prompt 区間と style/text 区間は `-100` で loss を除外
- `mask_ratio_range`, `audio_mask_id` は AR では未使用

### 4. トレーニング構成 — [omnivoice/training/builder.py](vscode-webview://1cgucbmtq4mq62usavvdp4b7hanckuvl2m0uhl9bph9fbhthg8f0/omnivoice/training/builder.py), [omnivoice/training/config.py](vscode-webview://1cgucbmtq4mq62usavvdp4b7hanckuvl2m0uhl9bph9fbhthg8f0/omnivoice/training/config.py)

- `TrainingConfig` に `ar_mode: bool = False`、 `audio_eos_id: int = 1024` を追加
- [builder.py:131-142](vscode-webview://1cgucbmtq4mq62usavvdp4b7hanckuvl2m0uhl9bph9fbhthg8f0/omnivoice/training/builder.py#L131-L142) で `ar_mode` のとき `OmniVoiceARSampleProcessor` をインスタンス化
- [builder.py:103-110](vscode-webview://1cgucbmtq4mq62usavvdp4b7hanckuvl2m0uhl9bph9fbhthg8f0/omnivoice/training/builder.py#L103-L110) で AR 時はモデル側にもフラグを渡す

trainer / collator / batching / dataset は**改変不要**(packing と document_ids は AR でも causal-within-document として使える)。

### 5. 推論 — `generate_ar` の追加

[omnivoice.py:1132-1284](vscode-webview://1cgucbmtq4mq62usavvdp4b7hanckuvl2m0uhl9bph9fbhthg8f0/omnivoice/models/omnivoice.py#L1132-L1284) の `_generate_iterative` を **使わず**、新規 `_generate_ar` を追加:

- Qwen3 ベースモデルの `past_key_values` (KVキャッシュ) を使う逐次デコード
- 各ステップで 8 codebook を並列サンプル(temperature, top-k, top-p)
- CFG は AR でも標準的に使える(conditional/unconditional をバッチに並べる)
- 停止条件: codebook 0 の EOS or `target_lens` 上限
- `_prepare_inference_inputs` は流用(マスクテンソルだけ AR ではダミー)
- 既存の `_decode_and_post_process`, `_generate_chunked` (長文チャンキング) はそのまま再利用可能

attention 実装は学習時の `flex_attention` から、推論時は `sdpa` (Qwen3 デフォルト) に切替えるとKVキャッシュが素直に効く。

### 6. 設定ファイル / 起動スクリプト

- `examples/config/train_config_emilia_ar.json` を作成 — 元の `train_config_emilia.json` をベースに `"ar_mode": true` 追加 + 不要なマスキング系を削除
- `train.sh` / `run_emilia.sh` の AR 版

---

## ステップバイステップ実装順

1. **Config 拡張**: `TrainingConfig.ar_mode`, `OmniVoiceConfig.ar_mode`, `audio_eos_id` を追加
2. **Processor**: `OmniVoiceARSampleProcessor` を実装し、単体で 1サンプル流して shape/labels を確認
3. **モデルforward**: `_mask_mod_packed_causal` を追加、`forward` で `ar_mode` 分岐(変更は数行)
4. **smoke test**: 1〜2サンプルで loss が下がることだけ確認(数百ステップ)
5. **AR 推論**: `_generate_ar` を実装。論文評価不要なら最小限の greedy / sampling から
6. **学習設定 + 起動スクリプト**: `train_config_emilia_ar.json` を作成、Emilia で論文と同じ 300k step 学習
7. **比較評価**: [examples/run_eval.sh](vscode-webview://1cgucbmtq4mq62usavvdp4b7hanckuvl2m0uhl9bph9fbhthg8f0/examples/run_eval.sh) で SIM-o / WER / UTMOS を NAR(本家)と AR(本実装) で比較

---

## 公平比較のための注意点

論文 Table 1 で AR baselines (CosyVoice3, VoxCPM 等) は**各社独自データ + 独自トークナイザ**で訓練されており直接比較になっていません。本実装で論文に欠けている純粋比較を埋めるには以下を**揃える**:

- 同じ Higgs-audio v2 トークナイザ、同じ Qwen3-0.6B 初期化
- 同じ Emilia 100k h データ、同じ 300k step
- 同じバッチトークン数、同じ学習率スケジュール
- 評価データ・メトリクス・推論温度等の設定

**Parallel-codebook AR** は基本的に Higgs-Audio や DistAR の素朴版に相当する設計で、過去文献では NAR より intelligibility は高いが speaker similarity は低めという傾向が報告されています。期待される結果はそれに近いはず。

---

## リスク / 未確定事項

- **EOS 設計**: codebook 0 にだけ置くか全 codebook に置くか — codebook 0 だけで十分なはず(MusicGen と同流儀)。実装時に確認
- **CFG**: AR 時は時刻ごとの logits に対して適用する設計に。NAR と同じ `guidance_scale=2` で動くか要検証
- **長文生成**: `_generate_chunked` は流用可能だが、チャンク間の韻律連続性は AR の方が担保しやすい(prefix continuation が自然)
- **学習速度**: AR は1サンプルあたり全位置にロスがかかる(NARは平均50%)ので **per-step gradient signal は約2倍**。総 step 数を減らすか、同条件比較なら同 step で良い

