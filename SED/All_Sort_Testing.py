# import soundfile as sf

# # Load audio file
# file_path = 'C:/Users/issac/Documents/ML/Badminton_sound/audio_wav/1.wav'
# audio_data, sample_rate = sf.read(file_path)

# # Print audio data and sample rate
# print(audio_data)
# print(sample_rate)

# import torch
# # from my_model import MyModel # import your own model class

# # Step 1: Create an instance of the model
# # model = MyModel()

# # Step 2: Load the saved state dictionary
# path_to_saved_model = r"C:\Users\issac\Documents\ML\exps\new_exp_gpu=0_iter_0\best_student.pt"
# state_dict = torch.load(path_to_saved_model)

# print(state_dict)
# Step 3: Load the state dictionary into the model
# model.load_state_dict(state_dict)

# openfile = open('C:\\Users\\issac\\Documents\\ML\\exps\\200epc_new_exp_gpu=0_iter_0\\log.txt')
# print(openfile)

# import yt_dlp as youtube_dl
# url = "https://www.youtube.com/watch?v=uIj03RsGrJA"
# youtube_dl_options = {
#     "format": "mp4[height=1080]", # This will select the specific resolution typed here
#     "outtmpl": "%(title)s-%(id)s.%(ext)s",
#     "restrictfilenames": True,
#     "nooverwrites": True,
#     "writedescription": True,
#     "writeinfojson": True,
#     "writeannotations": True,
#     "writethumbnail": True,
#     "writesubtitles": True,
#     "writeautomaticsub": True
# }
# with youtube_dl.YoutubeDL(youtube_dl_options) as ydl:
#     ydl = youtube_dl.YoutubeDL()
#     r=ydl.extract_info(url,download=True)

# import wave

# with wave.open(r'C:\Users\issac\Documents\ML\Combine_test\separate\output_audio.aac', 'r') as wav_file:
#     num_channels = wav_file.getnchannels()
#     print("Number of audio channels:", num_channels)

# from moviepy.editor import *
# empty_audio_clip = AudioClip([], fps=30)
# empty_composite_audio_clip = CompositeAudioClip([])

# import wave

# def get_sampling_rate(audio_file):
#     with wave.open(audio_file, 'rb') as audio:
#         sampling_rate = audio.getframerate()
#     return sampling_rate

# # Example usage
# audio_file = r'C:\Users\issac\Documents\ML\Combine_test\silence.wav'
# sampling_rate = get_sampling_rate(audio_file)
# print(f"The audio file has a sampling rate of {sampling_rate} Hz.")

from moviepy.editor import VideoFileClip


video_clip = VideoFileClip(r'C:\Users\issac\Documents\ML\Combine_test\output\output1.mp4')
audio = video_clip.audio
audio_channels = audio.nchannels
video_clip.close()
print("Number of audio channels:", audio_channels)

import subprocess

def get_audio_channels(file_path):
    ffprobe_cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'stream=channels',
        '-select_streams', 'a:0',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        file_path
    ]

    result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
    output = result.stdout.strip()
    channels = int(output) if output.isdigit() else None
    return channels





# # Usage example
# audio_file = r'C:\Users\issac\Documents\ML\Combine_test\separate\output_audio1.aac'
# channels = get_audio_channels(audio_file)
# if channels is not None:
#     print(f"The audio file '{audio_file}' has {channels} channel(s).")
# else:
#     print(f"Failed to determine the number of channels for '{audio_file}'.")

# import subprocess

# video_file = r"C:\Users\issac\Documents\ML\Combine_test\Input_video\badminton1.mp4"
# output_file_left = r"C:\Users\issac\Documents\ML\Combine_test\testing\output_left.wav"
# output_file_right = r"C:\Users\issac\Documents\ML\Combine_test\testing\output_right.wav"


# output_video = r"C:\Users\issac\Documents\ML\Combine_test\testing\output_video.mp4"


# # Separate video from the input file
# video_command = [
#     "ffmpeg",
#     "-i", video_file,
#     "-c:v", "copy",
#     "-an",  # Disable audio
#     '-y',
#     output_video
# ]
# # Separate audio channels from the input file
# audio_command1 = [
#     "ffmpeg",
#     "-i", video_file,
#     "-map_channel", "0.0.0"  # Select audio channel 1
#     # "-c:a", "copy",
#     '-y',
#     output_file_left
# ]
# audio_command2 = [
#     "ffmpeg",
#     "-i", video_file,
#     "-map_channel", "0.0.1"  # Select audio channel 2
#     # "-c:a", "copy",
#     '-y',
#     output_file_right
# ]
# # Execute the video and audio separation commands
# subprocess.run(video_command)
# subprocess.run(audio_command1)
# subprocess.run(audio_command2)
# Split left channel

