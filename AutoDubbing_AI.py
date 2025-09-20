import os
import torch
from PIL import Image
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS
torch.cuda.empty_cache()
print(torch.__version__)
print(torch.cuda.is_available())
print(f"Allocated VRAM: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
print(f"Reserved VRAM: {torch.cuda.memory_reserved()/1024**2:.2f} MB")

import whisper
import soundfile as sf
import tempfile
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip
from bark import SAMPLE_RATE, generate_audio
from transformers import AutoModelForSeq2SeqLM
from transformers.models.nllb import NllbTokenizer
import numpy
import torch.serialization

torch.serialization.add_safe_globals({
    numpy.core.multiarray.scalar,
    numpy.dtype,
    numpy.dtypes.Float64DType,
})

input_path = "C:/Users/ADMIN/input_movie.mp4"
output_path = "C:/Users/ADMIN/final_dubbed_video.mp4"

def clean_text(text):
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore").strip()

print("Loading Whisper model...")
clip = VideoFileClip(input_path)
clip_resized = clip.resize(height=480)
clip_resized.audio.write_audiofile("audio.wav")

whisper_model = whisper.load_model("medium")
result = whisper_model.transcribe("audio.wav", verbose=False, task="translate")
segments = result["segments"]
video_duration_ms = int(clip_resized.duration * 1000)

print("Loading NLLB translator on CPU...")
device = torch.device("cpu")
tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_fast=False)
translator = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(device)
tokenizer.src_lang = "hin_Deva"
tgt_lang_code = "eng_Latn"
tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang_code)

def translate_hi_to_en(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    output = translator.generate(**inputs, forced_bos_token_id=tgt_lang_id)
    return tokenizer.batch_decode(output, skip_special_tokens=True)[0]

print("Translating and generating TTS...")
final_audio = AudioSegment.silent(duration=video_duration_ms)
chosen_voice = "v2/en_speaker_2"

for i, seg in enumerate(segments):
    hindi = clean_text(seg["text"])
    if not hindi:
        continue
    try:
        print(f"[{i+1}/{len(segments)}] Translating: {hindi}")
        english = translate_hi_to_en(hindi)
        print(f"Translated: {english}")
        bark_audio = generate_audio(english, history_prompt=chosen_voice)
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        sf.write(temp_wav, bark_audio, SAMPLE_RATE)
        tts_audio = AudioSegment.from_wav(temp_wav)
        os.remove(temp_wav)
        start_time_ms = int(seg["start"] * 1000)
        final_audio = final_audio.overlay(tts_audio, position=start_time_ms)
    except Exception as e:
        print(f"Segment {i+1} failed: {e}")
        continue

print("Exporting final dubbed audio...")
final_audio.export("dubbed.wav", format="wav")

print("Combining with original video...")
dubbed_audio = AudioFileClip("dubbed.wav")
final_video = clip_resized.set_audio(dubbed_audio)
final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

print("Dubbed video created at:", output_path)
os.startfile(output_path)
