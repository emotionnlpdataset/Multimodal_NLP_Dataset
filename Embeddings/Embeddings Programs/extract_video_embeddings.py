import numpy as np
import pandas as pd
import os
import librosa
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
from barbar import Bar
import sys
import csv
from itertools import chain
import torchvision
from transformers import VivitImageProcessor, VivitModel, VivitConfig
from natsort import natsorted


emotions_task = True
if emotions_task is True:
    num_epochs = 10
    weights_file = f"video_weights_epoch{num_epochs}_emotions_mlc.pth"
else:
    num_epochs = 10
    weights_file = f"video_weights_epoch{num_epochs}_attributes_mlc.pth"
checkpoint = torch.load(weights_file)
model.load_state_dict(checkpoint['model_state_dict'])
head.load_state_dict(checkpoint['head_state_dict'])

whole_vid_data_list, whole_label_list, whole_condition_list = make_whole_dataset()
whole_dataset = VideoDataset(whole_vid_data_list, whole_label_list, whole_condition_list)
whole_loader = DataLoader(whole_dataset, batch_size=4)

model.eval()
video_embedding = []
with torch.no_grad():
    for video_data, label, cond_label in Bar(whole_loader):
        video_data = video_data.to(device)
        label = label.to(device)
        outputs = model(video_data)
        cls = outputs.last_hidden_state[:, 0, :]
        video_embedding.append(cls.tolist())
    
# Save Video Embeddings
if emotions_task is True:
    video_embedding_file_npy = "C:/Users/User/PycharmProjects/Research Project/video_embeddings_pretrained_emotions_mlc.npy"
else:
    video_embedding_file_npy = "C:/Users/User/PycharmProjects/Research Project/video_embeddings_pretrained_attributes_mlc.npy"
video_embedding = list(chain.from_iterable(video_embedding))
video_embedding = np.asarray(video_embedding, dtype=np.float32)
video_embedding = np.squeeze(video_embedding)
np.save(video_embedding_file_npy, video_embedding)
print("Video Embeddings Saved")






