import wave

with wave.open(r"C:\Users\issac\Documents\ML\Badminton_sound\FDY-CRNN\Train\Strong_sound\7.wav") as mywav:
    duration_seconds = mywav.getnframes() / mywav.getframerate()
    print(f"Length of the WAV file: {duration_seconds:.1f} s")