import soundfile as sf

data, samplerate = sf.read('lj_speech_22050/wavs/00001.wav')
print(f'Shape: {data.shape}')
print(f'Dtype: {data.dtype}')
print(f'Sample rate from header: {samplerate}')
