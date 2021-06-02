import os
from madmom.audio.signal import SignalProcessor
from madmom.features.beats import RNNBeatProcessor
from madmom.features.tempo import TempoEstimationProcessor
from tqdm import tqdm
import aubio
import csv
from numpy import median, diff

def get_bpm(s, title):
    win_s = 1024                # fft size
    hop_s = win_s // 2          # hop size

    samplerate = s.samplerate
    o = aubio.tempo("specdiff", win_s, hop_s, samplerate)
    # List of beats, in samples
    beats = []
    # Total number of frames read
    total_frames = 0

    while True:
        samples, read = s()
        is_beat = o(samples)
        if is_beat:
            this_beat = o.get_last_s()
            beats.append(this_beat)
            #if o.get_confidence() > .2 and len(beats) > 2.:
            #    break
        total_frames += read
        if read < hop_s:
            break

    def beats_to_bpm(beats, title):
        # if enough beats are found, convert to periods then to bpm
        if len(beats) > 1:
            if len(beats) < 4:
                print(f"few beats found in {title}")
            bpms = 60./diff(beats)
            return median(bpms)
        else:
            print(f"not enough beats found in {title}")
            return 0

    return beats_to_bpm(beats, title)

sig = {
    'madmom_comb': RNNBeatProcessor()
}
proc = {
    'madmom_comb': TempoEstimationProcessor(method='comb', min_bpm=40.,
                                            max_bpm=250., act_smooth=0.14,
                                            hist_smooth=9, hist_buffer=10.,
                                            alpha=0.79, fps=100),
}

with open('otherout.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    for filename in tqdm(os.listdir('.')):
        if os.path.splitext(filename)[1] != '.mp3':
            continue
            
        track_id = os.path.splitext(filename)[0]
        
        txt_fn = f"{track_id}.txt"
        with open(txt_fn) as infile:
            l = infile.readlines()
            artist = l[0].strip()
            title = l[1].strip()
        
        out = {}
        
        for method in sig:
            out[method] = proc[method](sig[method](filename))
        
        src = aubio.source(filename)
        
        out['aubio'] = get_bpm(src, title)
        
        # for a in out:
        #     print(out[a])
        # break
        
        for method in out:
            if hasattr(out[method], "__len__"):
                i = 1
                for tempo, strength in out[method]:
                    writer.writerow([artist, title, track_id, method, i, tempo, 0, strength])
                    i += 1
            else:
                writer.writerow([artist, title, track_id, method, 1, out[method], 0, 1])