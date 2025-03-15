import whisper
import os
import sounddevice as sd
import numpy as np
import wave


def record_audio(filename, duration=5, sample_rate=44100):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print(f"Recording saved as {filename}")
    
    
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16 bits
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    
    
    sd.play(audio, sample_rate)
    sd.wait()  # Wait for playback
    print("Audio playback finished.")

# Record audio
audio_file = "recorded_audio.wav"
record_audio(audio_file)

# Load the Whisper model
model = whisper.load_model("base")  # Use "small", "medium", or "large" 
print("Model loaded successfully.")

# Transcribe audio
result = model.transcribe(audio_file, language='en')  

# Output transcription
print("Transcription:", result["text"])  # This SHOULD display the actual transcription text
