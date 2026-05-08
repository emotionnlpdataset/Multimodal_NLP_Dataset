import numpy as np
import pandas as pd
import h5py
import os
from natsort import natsorted

# Labels File
labels_file = "C:/Users/User/Downloads/ResearchProjectLabels.csv"
labels_pd = pd.read_csv(labels_file)
result_array = labels_pd.iloc[:, 1:].to_numpy()

hdf5_folder = "C:/Users/User/OneDrive/Documents/ResearchProjectHDF5Files/"
i = 0
for file in natsorted(os.listdir(hdf5_folder)):
    temp_hdf5_file = os.path.join(hdf5_folder, file)
    print(file)

    # Video Clip Data
    with h5py.File(temp_hdf5_file, 'r') as f:
        data = f['video_data']
        temp_video_data_array = data[()]

    # Label
    temp_label = result_array[i]

    with h5py.File(temp_hdf5_file, 'w') as f:
        f.create_dataset("label", data=temp_label)
        f.create_dataset("video_data", data=temp_video_data_array, compression="gzip")
        f.attrs['fps'] = 2

    i += 1










