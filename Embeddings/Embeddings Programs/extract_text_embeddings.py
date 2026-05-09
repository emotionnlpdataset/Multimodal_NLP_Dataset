import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
import os
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import csv
import ast
import h5py
from barbar import Bar
from natsort import natsorted
from itertools import chain


def get_corresponding_data(video_number, input_ids, attention_masks):
    input_id = input_ids[video_number]
    attention_mask = attention_masks[video_number]

    labels_file = "C:/Users/User/PycharmProjects/Research Project/New_Labels_By_Classification_Emotions_Threshold15.npy"
    labels_data = np.load(labels_file)
    label_clip = labels_data[video_number]
    label_clip = label_clip.astype(float)

    condition_label_file = "C:/Users/User/PycharmProjects/Research Project/Condition_Labels.csv"
    condition_label = pd.read_csv(condition_label_file)
    cond_label = condition_label['Neurodivergent'].loc[video_number]
    # Yes (1): Autism/Neurodivergent, No (0): Normal/Neurotypical

    return input_id, attention_mask, label_clip, cond_label


def make_whole_dataset(input_ids, attention_masks):
    whole_text_input_ids_list = []
    whole_attention_masks_list = []
    whole_label_list = []
    whole_condition_list = []
    for i in range(1000):
        text_input_id, attention_mask, new_label_clip, condition_label = get_corresponding_data(i, input_ids, attention_masks)
        whole_text_input_ids_list.append(text_input_id)
        whole_attention_masks_list.append(attention_mask)
        whole_label_list.append(new_label_clip)
        whole_condition_list.append(condition_label)
    return whole_text_input_ids_list, whole_attention_masks_list, whole_label_list, whole_condition_list


emotions_category = True
if emotions_category is True:
    num_epochs = 20
    weights_file = f"text_weights_epoch{num_epochs}_emotions_mlc.pth"
else:
    num_epochs = 20
    weights_file = f"text_weights_epoch{num_epochs}_attributes_mlc.pth"
checkpoint = torch.load(weights_file)
model.load_state_dict(checkpoint['model_state_dict'])

whole_text_input_ids_list, whole_attention_masks_list, whole_label_list, whole_condition_list = make_whole_dataset(text_input_ids, attention_masks)
whole_dataset = TextDataset(whole_text_input_ids_list, whole_attention_masks_list, whole_label_list, whole_condition_list)
whole_loader = DataLoader(whole_dataset, batch_size=16)

text_embedding = []
with torch.no_grad():
    for text_input_id, attention_mask, label, cond_label in Bar(whole_loader):
        output, temp_text_embedding = model(text_input_id, attention_mask, return_embedding=True)
        text_embedding.append(temp_text_embedding.tolist())

# Save Text Embeddings
if emotions_category is True:
    text_embedding_file_npy = "/home/user2/text_embeddings_pretrained_emotions_mlc.npy"
else:
    text_embedding_file_npy = "/home/user2/text_embeddings_pretrained_attributes_mlc.npy"
text_embedding = list(chain.from_iterable(text_embedding))
text_embedding = np.asarray(text_embedding, dtype=np.float32)
text_embedding = np.squeeze(text_embedding)
np.save(text_embedding_file_npy, text_embedding)
print("Text Embeddings Saved")







