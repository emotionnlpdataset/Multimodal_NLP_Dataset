import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from barbar import Bar
import os
import sys
import csv
from natsort import natsorted
from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import h5py
import librosa
from itertools import chain


def get_corresponding_data(video_number):
    audio_folder = "C:/Users/User/OneDrive/Documents/Research Project Audio Files/"
    audio_filename = "Video" + str(video_number) + ".mp3"
    audio_file = os.path.join(audio_folder, audio_filename)

    labels_file = "C:/Users/User/PycharmProjects/Research Project/New_Labels_By_Classification_Emotions_Threshold15.npy"
    labels_data = np.load(labels_file)
    label_clip = labels_data[video_number - 1]
    label_clip = label_clip.astype(float)

    return audio_file, label_clip


def make_whole_dataset():
    whole_audio_file_list = []
    whole_label_list = []
    whole_condition_list = []
    for i in range(1000):
        audio_file, label_clip = get_corresponding_data(i + 1)
        whole_audio_file_list.append(audio_file)
        whole_label_list.append(label_clip)
    return whole_audio_file_list, whole_label_list


num_epochs = 20
weights_file = f"audio_weights_epoch{num_epochs}_emotions_mlc_wav2vec2.pth"
checkpoint = torch.load(weights_file)
model.load_state_dict(checkpoint['model_state_dict'])
classifier.load_state_dict(checkpoint['classifier_state_dict'])

whole_audio_file_list, whole_label_list = make_whole_dataset()
whole_dataset = AudioDataset(whole_audio_file_list, whole_label_list)

whole_loader = DataLoader(whole_dataset, batch_size=4, collate_fn=collate_fn)

audio_embedding = []
model.eval()
model.to(device)

# Testing
with torch.no_grad():
    for audio_input, label, cond_label in Bar(whole_loader):
        inputs = {k: v.to(device) for k, v in audio_input.items()}
        label = label.to(device)
        output = model(**inputs)
        hidden_states = output.last_hidden_state
        embeddings = hidden_states.mean(dim=1)
        audio_embedding.append(embeddings.squeeze().tolist())

# Save Audio Embedding
audio_embedding_file_npy = "C:/Users/User/PycharmProjects/Research Project/audio_embeddings_pretrained_emotions_mlc_wav2vec2.npy"
audio_embedding = list(chain.from_iterable(audio_embedding))
audio_embedding = np.asarray(audio_embedding, dtype=np.float32)
audio_embedding = np.squeeze(audio_embedding)
np.save(audio_embedding_file_npy, audio_embedding)
print("Audio Embedding for Emotions Saved")






