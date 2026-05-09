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


weights_file = f"text_weights_epoch{num_epochs}_emotions_mlc.pth"
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

whole_pred_temp = torch.cat(whole_pred_array, dim=0)
whole_label_temp = torch.cat(whole_label_array, dim=0)
whole_preds_array = whole_pred_temp.view(-1).detach().cpu().numpy()
whole_labels_array = whole_label_temp.view(-1).detach().cpu().numpy()

overall_accuracy = accuracy_score(whole_labels_array, whole_preds_array)
f1_micro = f1_score(whole_labels_array, whole_preds_array, average='micro')
f1_macro = f1_score(whole_labels_array, whole_preds_array, average='macro')
f1_weighted = f1_score(whole_labels_array, whole_preds_array, average='weighted')
f1_per_label = f1_score(whole_label_temp.squeeze(1).detach().cpu().numpy(), whole_pred_temp.squeeze(1).detach().cpu().numpy(), average=None)
overall_loss = total_loss / len(whole_loader)

label_emotion_names = ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
label_attribute_names = ["Valence", "Arousal", "Dominance"]

print(f"Predictions: {pred}")
print(f"Labels: {label}")
print(f"Accuracy: {overall_accuracy:.6f}")
print(f"Loss: {overall_loss:.6f}")
print(f"F1_Micro: {f1_micro:.5f}")
print(f"F1_Macro: {f1_macro:.5f}")
print(f"F1_Weighted: {f1_weighted:.5f}")
print(f"F1 Per Emotion:")
for l, f in zip(label_emotion_names, f1_per_label):
    print(f"{l}: {f:.5f}")


# Evaluate Neurotypical
neurotypical_pred_temp = torch.cat(neurotypical_pred_array, dim=0)
neurotypical_label_temp = torch.cat(neurotypical_label_array, dim=0)
neurotypical_preds_array = neurotypical_pred_temp.view(-1).detach().cpu().numpy()
neurotypical_labels_array = neurotypical_label_temp.view(-1).detach().cpu().numpy()

overall_accuracy_neurotypical = accuracy_score(neurotypical_labels_array, neurotypical_preds_array)
f1_micro_neurotypical = f1_score(neurotypical_labels_array, neurotypical_preds_array, average='micro')
f1_macro_neurotypical = f1_score(neurotypical_labels_array, neurotypical_preds_array, average='macro')
f1_weighted_neurotypical = f1_score(neurotypical_labels_array, neurotypical_preds_array, average='weighted')
neurotypical_labels = np.array([x.detach().cpu().numpy() for x in neurotypical_label_array])
neurotypical_preds = np.array([x.detach().cpu().numpy() for x in neurotypical_pred_array])
f1_per_label_neurotypical = f1_score(neurotypical_labels, neurotypical_preds, average=None)

print(f"\nNeurotypical")
print(f"Accuracy: {overall_accuracy_neurotypical:.6f}")
print(f"F1_Micro: {f1_micro_neurotypical:.5f}")
print(f"F1_Macro: {f1_macro_neurotypical:.5f}")
print(f"F1_Weighted: {f1_weighted_neurotypical:.5f}")
print(f"F1 Per Emotion for Neurotypical:")
for l, f in zip(label_emotion_names, f1_per_label_neurotypical):
    print(f"{l}: {f:.5f}")



# Evaluate Neurodivergent
neurodivergent_pred_temp = torch.cat(neurodivergent_pred_array, dim=0)
neurodivergent_label_temp = torch.cat(neurodivergent_label_array, dim=0)
neurodivergent_preds_array = neurodivergent_pred_temp.view(-1).detach().cpu().numpy()
neurodivergent_labels_array = neurodivergent_label_temp.view(-1).detach().cpu().numpy()

overall_accuracy_neurodivergent = accuracy_score(neurodivergent_labels_array, neurodivergent_preds_array)
f1_micro_neurodivergent = f1_score(neurodivergent_labels_array, neurodivergent_preds_array, average='micro')
f1_macro_neurodivergent = f1_score(neurodivergent_labels_array, neurodivergent_preds_array, average='macro')
f1_weighted_neurodivergent = f1_score(neurodivergent_labels_array, neurodivergent_preds_array, average='weighted')
neurodivergent_labels = np.array([x.detach().cpu().numpy() for x in neurodivergent_label_array])
neurodivergent_preds = np.array([x.detach().cpu().numpy() for x in neurodivergent_pred_array])
f1_per_label_neurodivergent = f1_score(neurodivergent_labels, neurodivergent_preds, average=None)

print(f"\nNeurodivergent")
print(f"Accuracy: {overall_accuracy_neurodivergent:.6f}")
print(f"F1_Micro: {f1_micro_neurodivergent:.5f}")
print(f"F1_Macro: {f1_macro_neurodivergent:.5f}")
print(f"F1_Weighted: {f1_weighted_neurodivergent:.5f}")
print(f"F1 Per Emotion for Neurodivergent:")
for l, f in zip(label_emotion_names, f1_per_label_neurodivergent):
    print(f"{l}: {f:.5f}")
    

# Save Text Embeddings
text_embedding_file_npy = "/home/user2/text_embeddings_pretrained_emotions_mlc.npy"
text_embedding = list(chain.from_iterable(text_embedding))
text_embedding = np.asarray(text_embedding, dtype=np.float32)
text_embedding = np.squeeze(text_embedding)
np.save(text_embedding_file_npy, text_embedding)
print("Text Embeddings Saved")







