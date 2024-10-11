import librosa

audio, sr = librosa.load('./audio/proj_sound-of-norway_bugg_RPiID-10000000cc849698_conf_6f40914_2022-09-24T02_03_08.447Z.mp3', sr=None)

print(sr)