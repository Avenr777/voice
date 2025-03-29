import torch

from model import ECAPA_gender
import pyaudio
import wave
import time
import keyboard
chunk = 1024
format = pyaudio.paInt16
channels = 2
rate = 44100
Output_Filename = "Recorded.wav"
audio=pyaudio.PyAudio()

stream = audio.open(format=format,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=chunk)

frames = []
print("Press SPACE to start recording")
keyboard.wait('space')
print("Recording... Press SPACE to stop.")
time.sleep(0.2)
while True:
    try:
        data = stream.read(chunk)  
        frames.append(data)
    except KeyboardInterrupt:
        break
    if keyboard.is_pressed('space'):
        print("stopping recording") 
        time.sleep(0.2)
        break   

stream.stop_stream()
stream.close()
audio.terminate()
wf = wave.open(Output_Filename, 'wb')  
wf.setnchannels(channels)
wf.setsampwidth(audio.get_sample_size(format))
wf.setframerate(rate)
wf.writeframes(b''.join(frames))
wf.close()

# You could directly download the model from the huggingface model hub
model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier")
model.eval()

# If you are using gpu or not.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the example file and use predict function to directly get the output
example_file = "Recorded.wav"
with torch.no_grad():
    output = model.predict(example_file, device=device)
    print("Gender : ", output)
