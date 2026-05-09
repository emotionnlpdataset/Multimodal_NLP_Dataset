import numpy as np
import pandas as pd
import os
import ast

labels_file = "C:/Users/User/PycharmProjects/Research Project/Labels_File.csv"
labels_data = np.genfromtxt(labels_file, delimiter=',')
new_labels_data = [arr[7:] for arr in labels_data]
new_labels_data = np.array(new_labels_data)

def revised_new_temp_label_fn(temp_label):
    new_temp_label = np.zeros(temp_label.shape)
    for i in range(temp_label.shape[0]):
        if temp_label[i] > 0:
            new_temp_label[i] = 1

        elif temp_label[i] <= 0:
            new_temp_label[i] = 0

    return new_temp_label

new_revised_temp_label_table = []
threshold = 0
for temp_label in new_labels_data:
    new_temp_label = revised_new_temp_label_fn(temp_label)
    new_revised_temp_label_table.append(new_temp_label)

export_file_csv = f"C:/Users/User/Downloads/Revised_New_Labels_By_Classification_Attributes.csv"
np.savetxt(export_file_csv, new_revised_temp_label_table, delimiter=',', fmt='%d')

export_file_npy = f"C:/Users/User/Downloads/Revised_New_Labels_By_Classification_Attributes.npy"
np.save(export_file_npy, new_revised_temp_label_table)





