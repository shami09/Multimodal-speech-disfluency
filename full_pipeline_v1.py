###### Libraries for audio recording#######################
import pyaudio
import wave
import os

##### Libraries for audio inference########################
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsampler import ImbalancedDatasetSampler
import sys
import math
import matplotlib.pyplot as plt
import IPython.display as ipd
from tqdm import tqdm


######## Import basic libraries##################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, balanced_accuracy_score
import glob
from torch.utils.tensorboard import SummaryWriter
import random
import argparse

######### Importing helper functions ##################
sys.path.append('/Users/shamikalikhite/Documents/Multimodal_speech_disfluency/Realtime_audio')
# Custom functions
from helper_functions import set_seed, __shuffle_pick_quarter_data__, save_checkpoint
from helper_functions import AverageMeter, ProgressMeter
from audio_helper_functions import _resample_if_necessary, _cut_if_necessary, _right_pad_if_necessary, _mix_down_if_necessary

# Audio recording parameters
RATE = 16000
CHUNK = 256
RECORD_SECONDS = 3

# Folder to save audio recordings
output_folder = "/Users/shamikalikhite/Documents/Multimodal_speech_disfluency/Realtime_audio/Audio_recordings/"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Function to save recorded audio to a WAV file
def save_audio(frames, filename):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

# Function to process audio data
def process_audio(audio_data):
    # Load audio
    signal = torch.from_numpy(np.frombuffer(audio_data, dtype=np.int16)).float()
    
    # Preprocess audio
    signal = _resample_if_necessary(signal, RATE)
    signal = _mix_down_if_necessary(signal)
    signal = _cut_if_necessary(signal)
    signal = _right_pad_if_necessary(signal)
    signal = signal.unsqueeze(1).unsqueeze(2)
    
    return signal

# Function to perform inference
def perform_inference(model, audio_data):
    # Convert audio data to tensor
    # Reshape the audio data to have batch and channel dimensions
    audio_tensor = audio_data.unsqueeze(0).unsqueeze(0)
    #print(audio_data)
    #audio_tensor=audio_data
    # Move tensor to device
    audio_tensor = audio_tensor.to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(audio_tensor)
    
    return output

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open audio stream
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

##################################################################################################
# Model Definition
##################################################################################################
## Testing transformer model 
class PositionalEncoding(nn.Module):
    # def __init__(self, d_model, dropout, max_len):
    def __init__(self, device, d_model, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        max_len = 149
        # max_len = 376 # FIXME :: UPdeate in the class definitions
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        # (L, N, F)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x, device):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)].to(device)
        return self.dropout(x)

  
class encoder(nn.Module):
    def __init__(self, d_model, device): #FIXME
        super(encoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=4) ## README: d_model is the "f" in forward function of class network
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1) ## num_layers is same as N in the transformer figure in the transformer paper
        self.positional_encoding = PositionalEncoding(device,d_model)
    def forward(self, tgt):
        tgt = self.positional_encoding(tgt, device) ##for positional encoding
        out = self.transformer_encoder(tgt) ##when masking not required, just remove mask=tgt_mask
        return out
### TF model
feat_channel = 3
class AudioOnlyStutterNet(nn.Module):
    def __init__(self):
        super(AudioOnlyStutterNet, self).__init__()
        self.fc_dim = 384

        self.bn0 = nn.BatchNorm2d(1)
        self.layer1 = nn.Sequential(
            torch.nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.5)
        )
        self.layer1_bn = nn.BatchNorm2d(1)

        self.tf_encoder = encoder(self.fc_dim, device)
        
        self.clf_head = nn.Sequential(
            nn.Linear(self.fc_dim, self.fc_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(self.fc_dim // 2, self.fc_dim // 4),
            nn.LeakyReLU(),
            nn.Linear(self.fc_dim // 4, 2),
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        # print('Input data shape:', x.shape)
        x =self.bn0(x)
        out = self.layer1(x)
        # print('Input data shape after layer1:', out.shape)
        out = self.layer1_bn(out)
        # print('Input data shape after bn1:', out.shape)


        if (len(out.shape) == 4):
            # print('Here')
            out = out.squeeze(1)
        # x = x.unsqueeze(1)
        # print('Input to tf data shape:', out.shape)
        # Current input -- B,T,F
        # Expected input -- T,B,F
        out = out.permute(1, 0, 2)
        # x = self.bn0(x)
        # print('Input data shape after permute:', out.shape)
        out = self.tf_encoder(out)   
        # print('Input data shape after tf_encoder is:', out.shape) # T,B,F
        
        out = out.mean(0, keepdim=True)  
        # x = nn.Flatten()(x)
        # print('Input data shape after mean pooling:', out.shape) # 1, B, F
        # unsqueeze the feature dimension
        out = out.squeeze(0)
        # print('Input data shape after view:', out.shape)
        out = self.clf_head(out)
        # print('Input data shape after clf_head:', out.shape)
        # print('Output:', out)
        # breakpoint()
        
        return out
print("Model Loaded")
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")


# Load model
model_path = "/Users/shamikalikhite/Documents/Multimodal_speech_disfluency/Realtime_audio/Realtime_audiobaseline_0149.pth.tar"
model = AudioOnlyStutterNet().to(device)
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Main loop for streaming and inference
print("Streaming audio... Press Ctrl+C to stop.")
try:
    frames = []
    while True:
        # Read audio data from stream
        data = stream.read(CHUNK)
        frames.append(data)
        
        # Check if enough data is collected for processing
        if len(frames) >= RATE * RECORD_SECONDS / CHUNK:
            # Save audio recording
            filename = os.path.join(output_folder, f"recorded_audio_{len(os.listdir(output_folder)) + 1}.wav")
            save_audio(frames, filename)
            
            # Perform inference
            audio_data = process_audio(b"".join(frames))
            output = perform_inference(model, audio_data)
            
            # Process inference output
            _, predicted_class = torch.max(output, 1)
            if predicted_class == 0:
                print("No stuttering")
            else:
                print("Blocks")
            
            # Reset frames list for next recording
            frames = []
except KeyboardInterrupt:
    print("Recording stopped.")
finally:
    # Close the audio stream and terminate PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()