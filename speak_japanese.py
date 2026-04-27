#!/usr/bin/env python3
"""
日本語テキストを「。」で文に分割し、生成と再生をパイプライン処理するスクリプト。
次の文の生成と現在の文の再生を並行して行い、連続再生を実現します。

使い方:
    python speak_japanese.py                          # サンプルテキストを読み上げ
    python speak_japanese.py --text "文章。次の文章。"   # テキストを直接指定
    python speak_japanese.py --file input.txt         # ファイルから読み込み
    python speak_japanese.py --ref_audio ref.wav      # 音声クローニング

依存: sounddevice をインストールするとより安定します
    uv add sounddevice  または  pip install sounddevice
"""

import argparse
import io
import queue
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SAMPLE_TEXT = """
吾輩は猫である。名前はまだ無い。
どこで生れたかとんと見当がつかぬ。
何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。
吾輩はここで始めて人間というものを見た。
"""


def split_sentences(text: str) -> list[str]:
    sentences = []
    for s in text.replace("\n", "").split("。"):
        s = s.strip()
        if s:
            sentences.append(s + "。")
    return sentences


def play_audio_numpy(audio: np.ndarray, sample_rate: int = 24000) -> None:
    try:
        import sounddevice as sd
        sd.play(audio, samplerate=sample_rate)
        sd.wait()
        return
    except ImportError:
        pass

    # macOS フォールバック: 一時ファイルに書き込んで afplay で再生
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    sf.write(tmp_path, audio, sample_rate)
    subprocess.run(["afplay", tmp_path], check=True)
    Path(tmp_path).unlink(missing_ok=True)


def generation_worker(
    model,
    sentences: list[str],
    audio_queue: queue.Queue,
    generate_kwargs: dict,
) -> None:
    for i, sentence in enumerate(sentences):
        # print(f"[{i + 1}/{len(sentences)}] 生成中: {sentence}", flush=True)
        try:
            audio_list = model.generate(text=sentence, **generate_kwargs)
            audio_queue.put((i, sentence, audio_list[0]))
        except Exception as e:
            print(f"  生成エラー: {e}", file=sys.stderr)
            audio_queue.put((i, sentence, None))
    audio_queue.put(None)  # 終了シグナル


def playback_worker(audio_queue: queue.Queue, sample_rate: int = 24000) -> None:
    while True:
        item = audio_queue.get()
        if item is None:
            break
        i, sentence, audio = item
        if audio is None:
            continue
        print(f"{sentence}", flush=True)
        play_audio_numpy(audio, sample_rate)


def main():
    parser = argparse.ArgumentParser(description="日本語テキストをOmniVoiceで連続読み上げ")
    parser.add_argument("--text", type=str, help="読み上げるテキスト")
    parser.add_argument("--file", type=str, help="テキストファイルのパス")
    parser.add_argument("--ref_audio", type=str, help="音声クローニング用の参照音声ファイル")
    parser.add_argument("--ref_text", type=str, help="参照音声のトランスクリプト（省略時はWhisperで自動認識）")
    parser.add_argument("--instruct", type=str, default="female, middle age", help="音声デザイン指示（ref_audioがない場合に使用）")
    parser.add_argument("--model", type=str, default="k2-fsa/OmniVoice", help="モデルID")
    parser.add_argument("--device", type=str, default="mps", help="デバイス (mps / cuda:0 / cpu)")
    parser.add_argument("--speed", type=float, default=1.0, help="読み上げ速度 (1.0が標準)")
    parser.add_argument("--buffer", type=int, default=4, help="先行生成バッファ数")
    args = parser.parse_args()

    # テキスト取得
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        text = SAMPLE_TEXT

    sentences = split_sentences(text)
    if not sentences:
        print("読み上げる文が見つかりませんでした。", file=sys.stderr)
        sys.exit(1)

    print(f"文の数: {len(sentences)}")
    for i, s in enumerate(sentences):
        print(f"  [{i + 1}] {s}")
    print()

    # モデルロード
    print(f"モデルをロード中: {args.model} ({args.device})")
    from omnivoice import OmniVoice
    model = OmniVoice.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch.float16,
    )

    # 生成パラメータ
    generate_kwargs: dict = {"speed": args.speed}
    if args.ref_audio:
        generate_kwargs["ref_audio"] = args.ref_audio
        if args.ref_text:
            generate_kwargs["ref_text"] = args.ref_text
    else:
        generate_kwargs["instruct"] = args.instruct

    # キューを使った生成・再生パイプライン
    audio_queue: queue.Queue = queue.Queue(maxsize=args.buffer)

    gen_thread = threading.Thread(
        target=generation_worker,
        args=(model, sentences, audio_queue, generate_kwargs),
        daemon=True,
    )
    gen_thread.start()

    playback_worker(audio_queue)

    gen_thread.join()


if __name__ == "__main__":
    main()
