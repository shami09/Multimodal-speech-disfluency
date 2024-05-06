import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os
from torch.profiler import profile, record_function, ProfilerActivity

  # Import your model class
import sys

import torch
import torchaudio
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torchsampler import ImbalancedDatasetSampler
import sys
import math


import matplotlib.pyplot as plt
import IPython.display as ipd

from tqdm import tqdm

#from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import glob
from torch.utils.tensorboard import SummaryWriter
import random
from tqdm import tqdm
import argparse
from sklearn.metrics import f1_score, balanced_accuracy_score


sys.path.append('/home/payal/multimodal_speech/Visual_Speech_Recognition_for_Multiple_Languages/')
# Custom functions
from helper_functions import set_seed, __shuffle_pick_quarter_data__, save_checkpoint
from helper_functions import AverageMeter, ProgressMeter
from audio_helper_functions import _resample_if_necessary, _cut_if_necessary, _right_pad_if_necessary, _mix_down_if_necessary

from pipelines.model import AVSR
from pipelines.data.data_module import AVSRDataLoader
from pipelines.detectors.mediapipe.detector import LandmarksDetector

print("Code running")

target_sample_rate = 16000 ## We need all audio to be of 16kHz sampling rate
num_samples = target_sample_rate * 3
root_dir = '/home/payal/multimodal_speech/main_database/FB_audio/'
root_dir_video_feat = '/home/payal/multimodal_speech/main_database/FB_video_feat/'


#set_seed(seed_num)
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
print(device)
## If the checkpoint path is not present create it
#if not os.path.exists(args.chkpt_pth):
#    os.makedirs(args.chkpt_pth)

class InferencePipeline(torch.nn.Module):
    def __init__(self, modality, model_path, model_conf, detector="mediapipe", face_track=False, device="cuda:0"):
        super(InferencePipeline, self).__init__()
        self.device = device
        # modality configuration
        self.modality = modality
        self.dataloader = AVSRDataLoader(modality, detector=detector)
        self.model = AVSR(modality, model_path, model_conf, rnnlm=None, rnnlm_conf=None, penalty=0.0, ctc_weight=0.1, lm_weight=0.0, beam_size=40, device=device)
        if face_track and self.modality in ["video", "audiovisual"]:
            self.landmarks_detector = LandmarksDetector()
        else:
            self.landmarks_detector = None


    def process_landmarks(self, data_filename, landmarks_filename):
        if self.modality == "audio":
            return None
        if self.modality in ["video", "audiovisual"]:
            landmarks = self.landmarks_detector(data_filename)
            return landmarks


    def forward(self, data_filename, landmarks_filename=None):
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."
        landmarks = self.process_landmarks(data_filename, landmarks_filename)
        data = self.dataloader.load_data(data_filename, landmarks)
        transcript = self.model.infer(data)
        return transcript

    def extract_features(self, data_filename, landmarks_filename=None, extract_resnet_feats=False):
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."
        landmarks = self.process_landmarks(data_filename, landmarks_filename)
        data = self.dataloader.load_data(data_filename, landmarks)
        with torch.no_grad():
            if isinstance(data, tuple):
                enc_feats = self.model.model.encode(data[0].to(self.device), data[1].to(self.device), extract_resnet_feats)
            else:
                enc_feats = self.model.model.encode(data.to(self.device), extract_resnet_feats)
        return enc_feats
         
#modality = "video"
#model_conf = "/home/payal/multimodal_speech/data_new/LRS3_V_WER19.1/model.json"  
#model_path = "/home/payal/multimodal_speech/data_new/LRS3_V_WER19.1/model.pth"
#pipeline = InferencePipeline(modality, model_path, model_conf, face_track=True)

#features = pipeline.extract_features("/home/payal/multimodal_speech/Database/Fluency_Bank/Video/Block/output28.mp4")
#print(features.size())

class VideoInferencePipeline(Dataset):
    def __init__(self, target_sample_rate, num_samples, video_file_path):
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.video_file_path = video_file_path
    
    def __len__(self):
        return 1

    def __getitem__(self, index):
        # Load audio
        #signal, sr = torchaudio.load(self.video_file_path)
        #video_sample_path=self._get_video_sample_path()
        modality = "video"
        model_conf = "/home/payal/multimodal_speech/data_new/LRS3_V_WER19.1/model.json"  
        model_path = "/home/payal/multimodal_speech/data_new/LRS3_V_WER19.1/model.pth"
        pipeline = InferencePipeline(modality, model_path, model_conf, face_track=True)
        feature = pipeline.extract_features(self.video_file_path)
        #print('Feature shape:', feature.shape)

        return feature

# Initialize the dataset and dataloader
model_path = "/home/payal/multimodal_speech/Realtime_audiobaseline_0094.pth.tar"
video_file_path = "/home/payal/multimodal_speech/Database/Fluency_Bank/Video/Block/output28.mp4"
target_sample_rate = 16000  # Assuming the target sample rate is 16 kHz
num_samples = 48000  # Assuming you want 10 seconds of audio

pipeline = VideoInferencePipeline(target_sample_rate, num_samples, video_file_path)
BATCH_SIZE = 1
test_dataloader = DataLoader(dataset=pipeline, batch_size=BATCH_SIZE, shuffle=True)

print("Dataloader Processed")

# Iterate over the dataloader and print the shape of each feature
#for feature in test_dataloader:
 #   print("Feature shape:", feature.shape)



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
class VideoOnlyStutterNet(nn.Module):
    def __init__(self):
        super(VideoOnlyStutterNet, self).__init__()
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
        #print('Input data shape:', x.shape)
        x =self.bn0(x)
        #print('bn0 shape:', x.shape )
        out = self.layer1(x)
        #print('Input data shape after layer1:', out.shape)
        out = self.layer1_bn(out)
        #print('Input data shape after bn1:', out.shape)


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
    
##############################################################################

print("Model Loaded")
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(model_path)  # Load the checkpoint

model=VideoOnlyStutterNet().to(device)
model.load_state_dict(checkpoint['state_dict'])  # Load the model's state dictionary
print("State dict correct")
breakpoint()
n_total_steps=len(test_dataloader)


with torch.no_grad():
    n_correct=0
    n_samples=0

    for sr in test_dataloader:
        print("Inside testloader")
        sr = sr.unsqueeze(1)
        sr=sr.to(device)
        
        #print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

        model.eval()
        print("Model Loaded in loop")
        
    
        outputs=model(sr)

        #value, index

        _,predictions=torch.max(outputs,1)
        #print(predictions)
        if predictions.item() == 0:
            print("No stuttering")
        else:
            print("Blocks")


