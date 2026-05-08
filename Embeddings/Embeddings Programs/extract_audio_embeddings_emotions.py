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

num_epochs = 20
weights_file = f"audio_weights_epoch{num_epochs}_emotions_mlc_wav2vec2.pth"
checkpoint = torch.load(weights_file)
model.load_state_dict(checkpoint['model_state_dict'])
classifier.load_state_dict(checkpoint['classifier_state_dict'])

whole_audio_file_list, whole_label_list, whole_condition_list = make_whole_dataset()
whole_dataset = AudioDataset(whole_audio_file_list, whole_label_list, whole_condition_list)

whole_loader = DataLoader(whole_dataset, batch_size=4, collate_fn=collate_fn)

audio_embedding = []
whole_pred_array = []
whole_label_array = []
neurotypical_pred_array = []
neurotypical_label_array = []
neurodivergent_pred_array = []
neurodivergent_label_array = []
total_loss = 0.0
model.eval()
model.to(device)
classifier.to(device)
# Testing
with torch.no_grad():
    for audio_input, label, cond_label in Bar(whole_loader):
        inputs = {k: v.to(device) for k, v in audio_input.items()}
        label = label.to(device)

        output = model(**inputs)
        hidden_states = output.last_hidden_state
        embeddings = hidden_states.mean(dim=1)
        logits = classifier(embeddings)
        loss = criterion(logits, label)

        prob = torch.sigmoid(logits)
        pred = (prob > 0.35).float()

        audio_embedding.append(embeddings.squeeze().tolist())
        whole_pred_array.append(pred.detach())
        whole_label_array.append(label.detach())
        
        for i in range(cond_label.shape[0]):
            if cond_label[i] == 1:
                neurodivergent_pred_array.append(pred[i].detach())
                neurodivergent_label_array.append(label[i].detach())
            elif cond_label[i] == 0:
                neurotypical_pred_array.append(pred[i].detach())
                neurotypical_label_array.append(label[i].detach())

        total_loss += loss.item()



