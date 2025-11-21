
import os
import librosa
import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

# ===========================================================
# 1. AUDIO PROCESSING
# ===========================================================

def load_audio(path, sr=22050, mono=True, duration=10):
    y, sr = librosa.load(path, sr=sr, mono=mono, duration=duration)
    return y, sr


# ===========================================================
# 2. FEATURE EXTRACTION
# ===========================================================

def extract_basic_features(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = float(np.mean(librosa.feature.rms(y=y)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    return {
        "tempo": float(tempo),
        "rms": rms,
        "centroid": centroid,
        "zcr": zcr
    }


# ===========================================================
# 3. MOOD / SEMANTIC DESCRIPTORS
# ===========================================================

def generate_descriptors(features):
    tempo = features["tempo"]
    centroid = features["centroid"]

    mood = "energetic" if tempo > 120 else "calm"
    brightness = "bright" if centroid > 2000 else "warm"

    return {
        "mood": mood,
        "brightness": brightness,
        "tempo": tempo
    }


# ===========================================================
# 4. PROMPT GENERATION
# ===========================================================

def compose_prompt(desc):
    return (
        f"{desc['mood']} mood, {desc['brightness']} colors, "
        f"art based on sound. Tempo: {desc['tempo']:.1f}. "
        "Highly detailed, artistically composed."
    )


# ===========================================================
# 5. SDXL MODEL WRAPPER
# ===========================================================

def load_sd_model(device="cuda"):
    model = "stabilityai/stable-diffusion-xl-base-1.0"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)
    return pipe


# ===========================================================
# 6. GENERATE IMAGE
# ===========================================================

def generate_image(prompt, pipe, num_images=1):
    images = []
    for _ in range(num_images):
        result = pipe(prompt)
        images.append(result.images[0])
    return images


# ===========================================================
# 7. SAVE IMAGE
# ===========================================================

def save_image(img: Image.Image, path="output.png"):
    img.save(path)
    return path


# ===========================================================
# 8. FULL PIPELINE
# ===========================================================

def run_pipeline(audio_path, num_images=1, device="cuda"):

    print("🔊 Loading audio...")
    y, sr = load_audio(audio_path)

    print("🎛 Extracting features...")
    features = extract_basic_features(y, sr)

    print("🎨 Generating descriptors...")
    desc = generate_descriptors(features)

    print("📝 Creating prompt...")
    prompt = compose_prompt(desc)
    print("\nGenerated Prompt:\n", prompt)

    print("\n🧨 Loading SDXL model (first time = slow)...")
    pipe = load_sd_model(device=device)

    print("\n🖼 Generating images with SDXL...")
    images = generate_image(prompt, pipe, num_images)

    os.makedirs("outputs", exist_ok=True)
    paths = []

    for i, img in enumerate(images):
        out_path = f"outputs/generated_{i}.png"
        save_image(img, out_path)
        print("Saved:", out_path)
        paths.append(out_path)

    return paths


# ===========================================================
# 9. COLAB FILE UPLOAD SUPPORT
# ===========================================================

try:
    from google.colab import files
    from IPython.display import display
    COLAB = True
except:
    COLAB = False

if COLAB:
    print("\n📁 Upload an audio file (.wav/.mp3)")
    uploaded = files.upload()

    audio_path = list(uploaded.keys())[0]

    print("\n🚀 Running pipeline...")
    output_paths = run_pipeline(audio_path, num_images=1)

    print("\n🎉 Completed!")
    display(Image.open(output_paths[0]))
