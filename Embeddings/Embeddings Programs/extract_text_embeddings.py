import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
import os
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import csv
import ast
from barbar import Bar
from natsort import natsorted


def get_corresponding_data(video_number, input_ids, attention_masks, emotions_task):
    input_id = input_ids[video_number]
    attention_mask = attention_masks[video_number]

    if emotions_task is True:
        labels_file = "C:/Users/User/PycharmProjects/Research Project/New_Labels_By_Classification_Emotions_Threshold15.npy"
    else:
        labels_file = "C:/Users/User/PycharmProjects/Research Project/Revised_New_Labels_By_Classification_Attributes.npy"
    labels_data = np.load(labels_file)
    label_clip = labels_data[video_number]
    label_clip = label_clip.astype(float)

    return input_id, attention_mask, label_clip


def make_whole_dataset(input_ids, attention_masks, emotions_task):
    whole_text_input_ids_list = []
    whole_attention_masks_list = []
    whole_label_list = []
    whole_condition_list = []
    for i in range(1000):
        text_input_id, attention_mask, new_label_clip, condition_label = get_corresponding_data(i, input_ids, attention_masks, emotions_task)
        whole_text_input_ids_list.append(text_input_id)
        whole_attention_masks_list.append(attention_mask)
        whole_label_list.append(new_label_clip)
    return whole_text_input_ids_list, whole_attention_masks_list, whole_label_list


model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name)

class TextModelEmotions(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = base_model
        self.fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 7)
        )

    def forward(self, inputs_ids, attention_mask, return_embedding=False):
        outputs = self.bert(input_ids=inputs_ids, attention_mask=attention_mask)
        x_embedding = outputs.last_hidden_state.mean(dim=1)
        x = self.fc(x_embedding)
        if return_embedding is True:
            return x, x_embedding
        return x


class TextModelEmotionalDimensions(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = base_model
        self.fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )

    def forward(self, inputs_ids, attention_mask, return_embedding=False):
        outputs = self.bert(input_ids=inputs_ids, attention_mask=attention_mask)
        x_embedding = outputs.last_hidden_state.mean(dim=1)
        x = self.fc(x_embedding)
        if return_embedding is True:
            return x, x_embedding
        return x


class TextDataset(Dataset):
    def __init__(self, text_input_ids, attention_masks, labels):
        super().__init__()
        self.text_input_ids = text_input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text_input = self.text_input_ids[idx]
        attention_mask = self.attention_masks[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return text_input, attention_mask, label


emotions_task = True
if emotions_task is True:
    num_epochs = 20
    weights_file = f"text_weights_epoch{num_epochs}_emotions_mlc.pth"
    model = TextModelEmotions()
else:
    num_epochs = 20
    weights_file = f"text_weights_epoch{num_epochs}_attributes_mlc.pth"
    model = TextModelEmotionalDimensions()

checkpoint = torch.load(weights_file)
model.load_state_dict(checkpoint['model_state_dict'])

text_preprocessing_file = "C:/Users/User/Downloads/Text_Preprocessing.txt"
with open(text_preprocessing_file, 'r') as f:
    content = f.read()
    text_preprocessing_data = ast.literal_eval(content)
text_inputs = tokenizer(text_preprocessing_data, padding=True, truncation=True, return_tensors="pt")
text_input_ids = text_inputs['input_ids']
attention_masks = text_inputs['attention_mask']

whole_text_input_ids_list, whole_attention_masks_list, whole_label_list = make_whole_dataset(text_input_ids, attention_masks, emotions_task)
whole_dataset = TextDataset(whole_text_input_ids_list, whole_attention_masks_list, whole_label_list)
whole_loader = DataLoader(whole_dataset, batch_size=16)

text_embedding = []
with torch.no_grad():
    for text_input_id, attention_mask, label, cond_label in Bar(whole_loader):
        output, temp_text_embedding = model(text_input_id, attention_mask, return_embedding=True)
        text_embedding.append(temp_text_embedding.tolist())

# Save Text Embeddings
if emotions_task is True:
    text_embedding_file_npy = "/home/user2/text_embeddings_pretrained_emotions_mlc.npy"
else:
    text_embedding_file_npy = "/home/user2/text_embeddings_pretrained_attributes_mlc.npy"
text_embedding = list(chain.from_iterable(text_embedding))
text_embedding = np.asarray(text_embedding, dtype=np.float32)
text_embedding = np.squeeze(text_embedding)
np.save(text_embedding_file_npy, text_embedding)
print("Text Embeddings Saved")







