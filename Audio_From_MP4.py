from moviepy.editor import VideoFileClip
import os

def extract_audio_from_mp4(video_path, output_audio_file):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_audio_file, codec="libmp3lame", bitrate="192k")
    audio_clip.close()
    video_clip.close()
    return audio_clip

input_folder = "C:/Users/User/OneDrive/Documents/Research Project Data - Copy"
output_folder = "C:/Users/User/OneDrive/Documents/Research Project Audio Files/"

for file in os.listdir(input_folder):
    temp_file = os.path.join(input_folder, file)
    file_name = file.split(".")[0] + ".mp3"
    print(file_name)
    output_file = os.path.join(output_folder, file_name)
    audio_file = extract_audio_from_mp4(temp_file, output_file)






