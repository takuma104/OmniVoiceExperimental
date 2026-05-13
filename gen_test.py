from omnivoice import OmniVoice
import torch
import soundfile as sf
import IPython.display as ipd
from IPython.display import display
import logging
logging.basicConfig(level=logging.DEBUG)

"""

{"id": "248dadb35e7efcbd3d8f0515b43d8bc2_000000032", "language_id": "ja", "text": "だいじょぶ？なんか、うなされてたけど", "num_tokens": 92, "audio_duration": 3.68}
{"id": "248dadb35e7efcbd3d8f0515b43d8bc2_000000047", "language_id": "ja", "text": "都会なら、甘ーいお菓子とかきっと沢山あるよね？何奢ってもらおうかな～", "num_tokens": 184, "audio_duration": 7.36}
{"id": "248dadb35e7efcbd3d8f0515b43d8bc2_000000041", "language_id": "ja", "text": "そだっけ？ヤエカ～どうしよぉ～って、べそかいてたの、昨日のことみたいに思い出せちゃう", "num_tokens": 267, "audio_duration": 10.68}
{"id": "248dadb35e7efcbd3d8f0515b43d8bc2_000000009", "language_id": "ja", "text": "わ、わかってます", "num_tokens": 42, "audio_duration": 1.68}
{"id": "248dadb35e7efcbd3d8f0515b43d8bc2_000000087", "language_id": "ja", "text": "ど、どれも古い建物だと思うから……", "num_tokens": 91, "audio_duration": 3.64}
{"id": "248dadb35e7efcbd3d8f0515b43d8bc2_000000040", "language_id": "ja", "text": "もう、バカにして……！最後にやったのは何年も前だよ！", "num_tokens": 153, "audio_duration": 6.12}
{"id": "248dadb35e7efcbd3d8f0515b43d8bc2_000000058", "language_id": "ja", "text": "まあ……ね。世も末だなぁ", "num_tokens": 123, "audio_duration": 4.92}
{"id": "248dadb35e7efcbd3d8f0515b43d8bc2_000000079", "language_id": "ja", "text": "ふふ……そうだね", "num_tokens": 68, "audio_duration": 2.72}
{"id": "248dadb35e7efcbd3d8f0515b43d8bc2_000000062", "language_id": "ja", "text": "まだ間に合うって。僕の方は召集だから仕方ないけど、ヤエカは違うんだし……何も志願なんてしなくても", "num_tokens": 224, "audio_duration": 8.96}
{"id": "248dadb35e7efcbd3d8f0515b43d8bc2_000000068", "language_id": "ja", "text": "そう、上手くいくかなぁ……", "num_tokens": 68, "audio_duration": 2.72}
{"id": "248dadb35e7efcbd3d8f0515b43d8bc2_000000013", "language_id": "ja", "text": "見りゃ分かる。初弾装填！", "num_tokens": 64, "audio_duration": 2.56}

"""

if __name__ == "__main__":
    # torch.manual_seed(42)

    model = OmniVoice.from_pretrained(
        "output_ar/run0/checkpoint-1000",
        device_map="cuda",
        dtype=torch.bfloat16
    )

    audio = model.generate(
        text="だいじょぶ？なんか、うなされてたけど",
        language="ja",
        # class_temperature=1.0,
    )
    assert len(audio[0]) != 0
    sf.write("gen_test_output.wav", audio[0], samplerate=24000)

