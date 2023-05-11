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

# with wave.open(r'C:\Users\issac\Documents\ML\Combine_test\final_ouput.mp4', 'r') as wav_file:
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


video_clip = VideoFileClip(r'C:\Users\issac\Documents\ML\Combine_test\final_output.mp4')
audio = video_clip.audio
audio_channels = audio.nchannels
video_clip.close()


# Example usage

print("Number of audio channels:", audio_channels)
