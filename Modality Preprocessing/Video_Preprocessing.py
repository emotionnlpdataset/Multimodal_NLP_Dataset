import numpy as np
import os
import math
from moviepy import VideoFileClip, ImageSequenceClip, CompositeVideoClip
import moviepy.video.fx as vfx
import pandas as pd
import h5py
import sys
from pathlib import Path


def frame_extraction_clips(video_clip, num_samples):
    # Generate 16 evenly spaced timestamps
    timestamps = np.linspace(0, video_clip.duration, num_samples)

    frames = []
    for t in timestamps:
        frame = video_clip.get_frame(t)
        frames.append(frame)

    # Create new video from sampled frames
    frame_extracted_clip = ImageSequenceClip(frames, fps=2)
    frame_extracted_clip = CompositeVideoClip([frame_extracted_clip])

    return frame_extracted_clip


def video_preprocess(video_file, output_folder):
    path1 = Path(video_file)
    video_name = path1.name
    video_base_name = video_name.split(".")[0]
    video_clip = VideoFileClip(video_file)
    width = video_clip.w; height = video_clip.h

    if width > height:
        diff = width - height
        left = 0; right = 0; top = int(round(diff/2)); bottom = int(round(diff/2)); color = (0, 0, 0)
    elif height > width:
        diff = height - width
        left = int(round(diff/2)); right = int(round(diff/2)); top = 0; bottom = 0; color = (0, 0, 0)

    video_clip = frame_extraction_clips(video_clip, num_samples=16)
    print("Frame Extraction Completed")

    padded_clip = video_clip.with_effects([
        vfx.Margin(top, bottom, left, right, color)
    ])
    print("Padded Clip Completed")
    
    resized_clip = padded_clip.resized(new_size=(224, 224))
    print("Resized Clip Completed")

    def video_to_hdf5(clip, hdf5_path):
        frames = []
        # Normalization per frame
        for frame in clip.iter_frames():
            normalized_frame = frame.astype(np.float32) / 255.0
            frames.append(normalized_frame)
        # Convert list to numpy array (frames, height, width, channels)
        video_data = np.stack(frames)
        # Save to HDF5
        with h5py.File(hdf5_path, 'w') as f:
            f.create_dataset('video_data', data=video_data, compression="gzip")
            f.attrs['fps'] = clip.fps
        return video_data

    output_file_name = "output_" + video_base_name + ".h5"
    hdf5_path = os.path.join(output_folder, output_file_name)
    hdf5_video_data = video_to_hdf5(resized_clip, hdf5_path)
    print(f"Output saved to {hdf5_path}")

    video_clip.close(); padded_clip.close(); resized_clip.close()


folder1 = "C:/Users/User/OneDrive/Documents/Research Project Data - Copy/"
output_folder = "C:/Users/User/OneDrive/Documents/Output_Preprocessed_Video_Clips/"
i = 0
for file in os.listdir(folder1):
    print(f"{i+1}) {file}")
    temp_video_file = os.path.join(folder1, file)
    video_preprocess(temp_video_file, output_folder)
    i += 1





