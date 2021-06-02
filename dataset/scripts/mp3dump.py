import aubio
import csv
import os
from madmom.audio.signal import SignalProcessor
from madmom.features.onsets import OnsetPeakPickingProcessor, RNNOnsetProcessor, CNNOnsetProcessor
from tqdm import tqdm

rnn = RNNOnsetProcessor()  # BLSTM, RNN
cnn = CNNOnsetProcessor()  # CNN

sig = {
    'blstm': SignalProcessor(norm=False, gain=0),
    'cnn': SignalProcessor(norm=False, gain=0),
    'll': SignalProcessor(gain=0),
}
nw = {
    'blstm': rnn,
    'cnn': cnn,
    'll': rnn
}
proc = {
    'blstm': OnsetPeakPickingProcessor(threshold=0.35, smooth=0.07),
    'cnn': OnsetPeakPickingProcessor(threshold=0.54, smooth=0.05),
    'll': OnsetPeakPickingProcessor(threshold=0.23),
}

for filename in tqdm(os.listdir('.')):
    if os.path.splitext(filename)[1] != '.mp3':
        continue
    
    artist, title = os.path.splitext(filename)[0].split('-')

    src = aubio.source(filename)
    
    o = {}
    out = {}
    
    o['energy'] = aubio.onset("energy", samplerate=src.samplerate)
    o['hfc'] = aubio.onset("hfc", samplerate=src.samplerate)
    o['complex'] = aubio.onset("complex", samplerate=src.samplerate)
    o['phase'] = aubio.onset("phase", samplerate=src.samplerate)
    o['specdiff'] = aubio.onset("specdiff", samplerate=src.samplerate)
    o['kl'] = aubio.onset("kl", samplerate=src.samplerate)
    o['mkl'] = aubio.onset("mkl", samplerate=src.samplerate)
    o['specflux'] = aubio.onset("specflux", samplerate=src.samplerate)
    
    for method in o:
        out[method] = []
    
    total_read = 0
    while True:
        samples, read = src()
        
        for method in o:
            if o[method](samples):
                out[method].append(o[method].get_last())
        
        total_read += read
        if read < src.hop_size:
            break
    
    for method in sig:
        out[method] = proc[method](nw[method](sig[method](filename))) * src.samplerate
    
    for method in out:
        with open(f"{title}-{method}.txt", 'w') as wf:
            wf.write(artist + "\n")
            wf.write(title + "\n")
            wf.write(str(src.samplerate) + "\n")
            wf.write(method + "\n")
            for t in out[method]:
                wf.write(str(int(round(t))) + '\n')