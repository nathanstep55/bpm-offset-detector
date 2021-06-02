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
        
    track_id = os.path.splitext(filename)[0]
    
    txt_fn = f"{track_id}.txt"
    with open(txt_fn) as infile:
        l = infile.readlines()
        artist = l[0].strip()
        title = l[1].strip()
        samplerate = int(l[2])
    
    
    out = {}
    
    for method in sig:
        out[method] = proc[method](nw[method](sig[method](filename))) * samplerate
    
    # for a in out:
    #     print(out[a])
    # break
    
    for method in out:
        with open(f"txt\\{track_id}-{method}.txt", 'w') as wf:
            wf.write(artist + "\n")
            wf.write(title + "\n")
            wf.write(str(samplerate) + "\n")
            wf.write(method + "\n")
            for t in out[method]:
                wf.write(str(int(round(t))) + '\n')