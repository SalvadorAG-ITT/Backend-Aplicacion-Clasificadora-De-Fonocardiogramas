cimport os
from pydub import AudioSegment
import librosa
SUPPORTED_FORMATS = (".wav", ".mp3", ".flac", ".ogg", " .aac ")
def is_audio_file(filename):
return filename.lower().endswith(SUPPORTED_FORMATS)
def optimize_audio(input_path, output_path, bitrate=
"192k"):

# Cargar audio con pydub
audio = AudioSegment.from_file(input_path)

# Análisis con librosa (para saber si es voz o música)
y, sr = librosa.load(input_path, mono=False)
duration = librosa.get_duration(y=y, sr=sr)

# Heurística simple: audios cortos suelen ser voz
if duration < 600: # menos de 10 minutos → voz
audio = audio.set_channels(1)
audio = audio.set_frame_rate(44100)
audio.export(output_path, format="mp3", bitrate=bitrate)
def process_path(input_path, output_dir):
os.makedirs(output_dir, exist_ok=True)
if os.path.isfile(input_path):
filename = os.path.basename(input_path)
name, _ = os.path.splitext(filename)
output_file = os.path.join(output_dir, f"{name}_optimized.mp3")
optimize_audio(input_path, output_file)
print(f"✔ Optimizado: {filename}")
elif os.path.isdir(input_path):
for root, _, files in os.walk(input_path):
for file in files:
if is_audio_file(file):
full_input = os.path.join(root, file)
name, _ = os.path.splitext(file)
output_file = os.path.join(output_dir, f"{name}_optimized.mp3")
optimize_audio(full_input, output_file)
print(f"✔ Optimizado: {file}")
else:
print("❌ Ruta no válida")
if __name__ == "__main__" : input_path = input("Ruta del archivo o carpeta de audios: ")
output_dir = input("Carpeta de salida:")
process_path(input_path, output_dir)