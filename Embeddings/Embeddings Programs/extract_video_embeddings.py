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


def get_corresponding_data(video_number, emotions_task):
    hdf5_folder = r"C:/Users/User/OneDrive/Documents/ResearchProjectHDF5Files/"
    hdf5_filename = "output_Video" + str(video_number) + ".h5"
    hdf5_file = os.path.join(hdf5_folder, hdf5_filename)
    with h5py.File(hdf5_file, 'r') as f:
        video_data = f['video_data']
        label_data = f['label']
        video_data = video_data[:]

    if emotions_task is True:
        labels_file = "C:/Users/User/PycharmProjects/Research Project/New_Labels_By_Classification_Emotions_Threshold15.npy"
    else:
        labels_file = "C:/Users/User/PycharmProjects/Research Project/Revised_New_Labels_By_Classification_Attributes.npy"
    labels_data = np.load(labels_file)
    label_clip = labels_data[video_number - 1]
    label_clip = label_clip.astype(float)

    return video_data, label_clip


def make_whole_dataset(emotions_task):
    whole_vid_data_list = []
    whole_label_list = []
    whole_condition_list = []
    for i in range(1000):
        vid_data, label_clip = get_corresponding_data(i + 1, emotions_task)
        whole_vid_data_list.append(vid_data)
        whole_label_list.append(label_clip)
    return whole_vid_data_list, whole_label_list


emotions_task = True
if emotions_task is True:
    num_epochs = 10
    weights_file = f"video_weights_epoch{num_epochs}_emotions_mlc.pth"
else:
    num_epochs = 10
    weights_file = f"video_weights_epoch{num_epochs}_attributes_mlc.pth"

config = VivitConfig.from_pretrained("google/vivit-b-16x2-kinetics400")
config.num_frames = 16
model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400", config=config, ignore_mismatched_sizes=True)

checkpoint = torch.load(weights_file)
model.load_state_dict(checkpoint['model_state_dict'])
head.load_state_dict(checkpoint['head_state_dict'])

whole_vid_data_list, whole_label_list, whole_condition_list = make_whole_dataset(emotions_task)
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






