import numpy as np
import pandas as pd
import os
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
from sklearn.metrics import f1_score, classification_report, accuracy_score, confusion_matrix


def get_video_number(file_path):
    file = os.path.basename(file_path)
    video_number = file.split("eo")[1]
    video_number = int(video_number.split(".")[0])
    return video_number


def get_corresponding_data(video_number, emotions_task):
    hdf5_folder = "C:/Users/User/OneDrive/Documents/ResearchProjectHDF5Files/"
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

    condition_label_file = "C:/Users/User/PycharmProjects/Research Project/Condition_Labels.csv"
    condition_label = pd.read_csv(condition_label_file)
    cond_label = condition_label['Neurodivergent'].loc[video_number - 1]
    # Yes (1): Autism/Neurodivergent, No (0): Normal/Neurotypical

    return video_data, label_clip, cond_label


def get_split_data(phase_split, phase_file_list, emotions_task):
    phase_split_vid_data_list = []
    phase_split_label_list = []
    phase_split_condition_list = []
    for file in phase_file_list:
        video_number = get_video_number(file)
        vid_data, new_label_clip, cond_label = get_corresponding_data(video_number, emotions_task)
        phase_split_vid_data_list.append(vid_data)
        phase_split_label_list.append(new_label_clip)
        phase_split_condition_list.append(cond_label)
    return phase_split_vid_data_list, phase_split_label_list, phase_split_condition_list


def return_train_and_valid_indices(train_validation_split, epoch):
    remainder = epoch % 5
    split_size = int(len(train_validation_split) / 5)
    indices = np.arange(len(train_validation_split))
    valid_indices = indices[(remainder*(split_size)):((remainder+1)*split_size)]
    train_indices = np.concatenate([indices[:(remainder*split_size)], indices[((remainder+1)*split_size):]])
    return train_indices, valid_indices


class VideoDataset(Dataset):
    def __init__(self, video_data, labels, cond_labels):
        self.video_data = video_data
        self.labels = labels
        self.cond_labels = cond_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        video_clip_data = torch.from_numpy(self.video_data[idx])
        video_clip_data = video_clip_data.permute(0, 3, 1, 2)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        cond_label = self.cond_labels[idx]
        return video_clip_data, label, cond_label


emotions_task = True
# Get Train-Validation Split
train_validation_split_file = "C:/Users/User/Downloads/Train_Validation_Split.csv"
train_validation_split = np.loadtxt(train_validation_split_file, delimiter=',', dtype=str)
train_validation_split = train_validation_split.tolist()
train_validation_vid_data_list, train_validation_label_list, train_validation_condition_list = get_split_data("Train-Validation", train_validation_split, emotions_task)
train_validation_dataset = VideoDataset(train_validation_vid_data_list, train_validation_label_list, train_validation_condition_list)

# Configure Model to 16 Frames
config = VivitConfig.from_pretrained("google/vivit-b-16x2-kinetics400")
config.num_frames = 16
model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400", config=config, ignore_mismatched_sizes=True)

# Freeze most of backbone
for param in model.parameters():
    param.requires_grad = False

criterion = nn.BCEWithLogitsLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if emotions_task is True:
    head = torch.nn.Linear(768, 7)
else:
    head = torch.nn.Linear(768, 3)
head = head.to(device)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(head.parameters()),
    lr=1e-4
)
if emotions_task is True: 
    num_epochs = 10
else:
    num_epochs = 20


for epoch in range(num_epochs):
    # Training
    model.train()
    print(f"Epoch: {epoch+1}")

    train_indices, valid_indices = return_train_and_valid_indices(train_validation_dataset, epoch)
    train_dataset = Subset(train_validation_dataset, train_indices)
    valid_dataset = Subset(train_validation_dataset, valid_indices)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=4)

    total_train_loss = 0.0
    train_pred_array = []
    train_label_array = []
    for video_data, label, cond_label in Bar(train_dataloader):
        video_data = video_data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(video_data)
        cls = output.last_hidden_state.mean(dim=1)
        pred = head(cls)
        loss = criterion(pred, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        prob = torch.sigmoid(pred)
        pred = (prob > 0.35).float()
        train_pred_array.append(pred.detach())
        train_label_array.append(label.detach())

        total_train_loss += loss.item()

    train_pred_temp = torch.cat(train_pred_array, dim=0)
    train_label_temp = torch.cat(train_label_array, dim=0)
    train_preds_array = train_pred_temp.view(-1).detach().cpu().numpy()
    train_labels_array = train_label_temp.view(-1).detach().cpu().numpy()

    train_accuracy = accuracy_score(train_labels_array, train_preds_array)
    train_loss = total_train_loss / len(train_dataloader)
    print(f"Video Epoch {epoch+1}: Train Accuracy: {train_accuracy:.5f}, Train Loss = {train_loss:.5f}")

    # Validation
    model.eval()
    total_val_loss = 0.0
    val_pred_array = []
    val_label_array = []
    with torch.no_grad():
        for video_data, label, cond_label in Bar(valid_dataloader):
            video_data = video_data.to(device)
            label = label.to(device)

            output = model(video_data)
            cls = output.last_hidden_state.mean(dim=1)
            pred = head(cls)
            loss = criterion(pred, label)

            prob = torch.sigmoid(pred)
            pred = (prob > 0.35).float()
            val_pred_array.append(pred.detach())
            val_label_array.append(label.detach())

            total_val_loss += loss.item()

        val_pred_temp = torch.cat(val_pred_array, dim=0)
        val_label_temp = torch.cat(val_label_array, dim=0)
        val_preds_array = val_pred_temp.view(-1).detach().cpu().numpy()
        val_labels_array = val_label_temp.view(-1).detach().cpu().numpy()

        val_accuracy = accuracy_score(val_labels_array, val_preds_array)
        val_loss = total_val_loss / len(valid_dataloader)
        print(f"Video Epoch {epoch+1}: Val Accuracy: {val_accuracy:.5f}, Validation Loss: {val_loss:.5f}")

    if epoch == (num_epochs - 1):
        torch.save({
            'model_state_dict': model.state_dict(),
            'head_state_dict': head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss
        }, f"video_weights_epoch{num_epochs}_emotions_mlc.pth")


# Load Model for Evaluation
if emotions_task is True:
    weights_file = f"video_weights_epoch{num_epochs}_emotions_mlc.pth"
else:
    weights_file = f"video_weights_epoch{num_epochs}_attributes_mlc.pth"
checkpoint = torch.load(weights_file)
model.load_state_dict(checkpoint['model_state_dict'])
head.load_state_dict(checkpoint['head_state_dict'])

# Get Test Split
test_split_file = "C:/Users/User/Downloads/Test_Split.csv"
test_split = np.loadtxt(test_split_file, delimiter=',', dtype=str)
test_split = test_split.tolist()
test_split_vid_data_list, test_split_label_list, test_split_condition_list = get_split_data("Test", test_split, emotions_task)

test_dataset = VideoDataset(test_split_vid_data_list, test_split_label_list, test_split_condition_list)
test_dataloader = DataLoader(test_dataset, batch_size=4)

# Testing
model.eval()
total_test_loss = 0.0
test_pred_array = []
test_label_array = []
with torch.no_grad():
    for video_data, label, cond_label in Bar(test_dataloader):
        video_data = video_data.to(device)
        label = label.to(device)

        output = model(video_data)
        cls = output.last_hidden_state.mean(dim=1)
        pred = head(cls)
        loss = criterion(pred, label)

        prob = torch.sigmoid(pred)
        pred = (prob > 0.35).float()
        test_pred_array.append(pred.detach())
        test_label_array.append(label.detach())

        for i in range(cond_label.shape[0]):
            if cond_label[i].item() == 1:
                neurodivergent_pred_array.append(pred[i].detach())
                neurodivergent_label_array.append(label[i].detach())
            elif cond_label[i].item() == 0:
                neurotypical_pred_array.append(pred[i].detach())
                neurotypical_label_array.append(label[i].detach())

        total_test_loss += loss.item()

test_pred_temp = torch.cat(test_pred_array, dim=0)
test_label_temp = torch.cat(test_label_array, dim=0)
test_preds_array = test_pred_temp.view(-1).detach().cpu().numpy()
test_labels_array = test_label_temp.view(-1).detach().cpu().numpy()

test_accuracy = accuracy_score(test_labels_array, test_preds_array)
test_f1_micro = f1_score(test_labels_array, test_preds_array, average='micro')
test_f1_macro = f1_score(test_labels_array, test_preds_array, average='macro')
test_f1_weighted = f1_score(test_labels_array, test_preds_array, average='weighted')
test_f1_per_label = f1_score(test_label_temp.squeeze(1).detach().cpu().numpy(), test_pred_temp.squeeze(1).detach().cpu().numpy(), average=None)
test_loss = total_test_loss / len(test_dataloader)

label_emotion_names = ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
label_attribute_names = ["Valence", "Arousal", "Dominance"]

print(f"Video Test Predictions: {pred}")
print(f"Video Test Labels: {label}")
print(f"Video Test Accuracy: {test_accuracy:.5f}")
print(f"Video Test Loss: {test_loss:.6f}")
print(f"Video Test F1_Micro: {test_f1_micro:.5f}")
print(f"Video Test F1_Macro: {test_f1_macro:.5f}")
print(f"Video Test F1_Weighted: {test_f1_weighted:.5f}")
print(f"Video Test F1 Per Emotion:")
for l, f in zip(label_emotion_names, test_f1_per_label):
    print(f"{l}: {f:.5f}")






