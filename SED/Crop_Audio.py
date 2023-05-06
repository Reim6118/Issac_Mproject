from pydub import AudioSegment
import os

# set the length of each cropped audio file (in milliseconds)
chunk_length_ms = 10000

# load the audio file
audio = AudioSegment.from_file(r"C:\Users\issac\Documents\ML\Badminton_sound\1hour2\1.wav")

# get the total length of the audio in milliseconds
audio_length_ms = len(audio)

# create a directory to save the cropped audio files
os.makedirs("output_directory", exist_ok=True)

# iterate over the audio and crop every 10 seconds
for i in range(0, audio_length_ms, chunk_length_ms):
    # set the start and end time of the cropped audio
    start = i
    end = min(i + chunk_length_ms, audio_length_ms)
    ii = i/10000
    # crop the audio
    cropped_audio = audio[start:end]

    # set the filename of the cropped audio file
    filename = r"C:\Users\issac\Documents\ML\Badminton_sound\FDY-CRNN\1hour_chunk\1.{0}.wav".format(ii)

    # export the cropped audio file
    cropped_audio.export(filename, format="wav")
