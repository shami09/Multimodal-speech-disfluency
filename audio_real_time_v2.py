import pyaudio
import wave
import os

RATE = 16000
CHUNK = 256
RECORD_SECONDS = 3.001

p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

frames = []

print("Recording audio...")

for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):  # Record for RECORD_SECONDS seconds
    frames.append(stream.read(CHUNK))

print("Finished recording")

stream.stop_stream()
stream.close()
p.terminate()

# Save the recorded audio to a WAV file
filename = "recorded_audio_1_.wav"
with wave.open(filename, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))

print("Audio saved as:", os.path.abspath(filename))
