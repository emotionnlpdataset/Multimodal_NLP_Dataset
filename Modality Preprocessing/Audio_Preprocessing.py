import numpy as np
import pandas as pd
import os
import sys
from df.enhance import enhance, init_df, load_audio, save_audio
from pydub import AudioSegment


def clean_audio_deepfilternet(input_mp3, temp_wav, output_mp3, start_time, end_time):
    # Convert and resample the audio to 48 kHz WAV file (DeepFilterNet's required sample rate)
    audio = AudioSegment.from_mp3(input_mp3)
    start_time_ms = start_time*1000
    end_time_ms = end_time*1000
    audio = audio[start_time_ms:end_time_ms]
    audio = audio.set_frame_rate(48000).set_channels(1)
    audio.export(temp_wav, format="wav")

    # Use DeepFilterNet for Enhancement
    model, df_state, _ = init_df()
    audio_data, _ = load_audio(temp_wav, sr=df_state.sr())
    enhanced_audio = enhance(model, df_state, audio_data)
    save_audio("enhanced.wav", enhanced_audio, df_state.sr())

    # Convert back to MP3
    AudioSegment.from_wav("enhanced.wav").export(output_mp3, format="mp3")

    # Clean up the temporary file
    os.remove(temp_wav)
    os.remove("enhanced.wav")
    print(f"Enhanced audio saved to {output_mp3}")


folder1 = "C:/Users/User/OneDrive/Documents/Research Project Audio Files/"
video_folder1 = "C:/Users/User/OneDrive/Documents/Research Project Data - Copy/"
temp_wav_file = "C:/Users/User/Downloads/temp_file.wav"
output_folder1 = "C:/Users/User/OneDrive/Documents/Research Project Cleaned Audio Files/"
i = 0
for file in os.listdir(folder1):
    print(f"{i+1}) {file}")
    temp_audio_file = os.path.join(folder1, file)

    output_filename = "cleaned" + str(file.split(".")[0]) + ".mp3"
    temp_output_file = os.path.join(output_folder1, output_filename)
    clean_audio_deepfilternet(temp_audio_file, temp_wav_file, temp_output_file, start_time=None, end_time=None)
    
    i += 1







