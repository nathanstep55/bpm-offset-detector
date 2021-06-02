# offset and BPM calculator
# Implementation of the paper Non-causal Beat Tracking for Rhythm Games by Bram van de Wetering (Fietsemaker, who made ArrowVortex)

# direct from test implementation first, then i will modify it if necessary

import aubio
import weighted # for weighted median
import numpy as np
import copy
import math
import cProfile
#import matplotlib.pyplot
#from scipy import signal

enable_profiling = True
graph = False
subtract_polynomial = True

def ev(p, h):
    hamming_window = 64 # make this lower for faster performance but higher for better accuracy
    sumval = 0
    for a in range(1,hamming_window+1):
        sumval += (0.54-0.46*math.cos((2*math.pi*a)/(hamming_window-1)))*h[int(p-0.5*hamming_window+a)%len(h)]
    return sumval

def main():
    if enable_profiling:
        pr = cProfile.Profile()
        pr.enable()
    samplerate = 0  # use original source samplerate
    hop_size = 256 # number of frames to read in one block
    window_size = 1024
    #temporal_window = (-5, 2) # add one to end
    silence_threshold = -70 # in decibels
    #temporal_weight = 0.1
    bpm = (89, 205) # min and max (there isn't much of a reason to change because if it's close it will fix itself on the second run)
    intervalinterval = 20 # what interval to use when checking histogram values between the bpm interval (the "step")

    maxseconds = 120 # only analyze first x seconds if applicable

    desc = []
    tdesc = []
    allsamples_max = np.zeros(0,)
    downsample = 2  # to plot n samples / hop_s

    #filename = "pattyshort.wav" # 168 bpm
    filename = "jigokukakurenbo.wav"
    #filename = "lovesick.m4a"

    s = aubio.source(filename, samplerate, hop_size)
    maxframes = int(maxseconds*s.samplerate)
    total_frames = 0

    o = aubio.onset("specflux", window_size, hop_size, s.samplerate)
    #pv = aubio.pvoc(window_size, hop_size)
    o.set_silence(silence_threshold)
    frames = []
    onsets = []
    while True:
        samples, read = s()
        frames.append(copy.deepcopy(samples))
        #if o(samples):
        #    print("%f" % o.get_last_s())
        #    onsets.append(o.get_last())
        if graph:
            new_maxes = (abs(samples.reshape(hop_size//downsample, downsample))).max(axis=0)
            allsamples_max = np.hstack([allsamples_max, new_maxes])
        total_frames += read
        if read < hop_size or (total_frames > maxframes and maxframes >= 0): break
    fmt_string = "read {:d} frames at {:d}Hz from {:s}"
    print(fmt_string.format(total_frames, s.samplerate, filename))

#f = open("D:/stepbot/stepcharter/temp.txt", "w")
#f.write(str(frames))
#f.close()

#print("Samples: "+str(samples))
#print("Frames: "+str(frames[-1]))
#if samples.all() == frames[len(frames)-1].all():
#    print("nope")

    #all_onset = []

    for i in range(len(frames)):
        onset = o(frames[i])
        #all_onset.append(onset)
        if onset:
            #print("%f" % o.get_last_s())
            onsets.append(o.get_last())
        if graph:
            desc.append(o.get_descriptor())
            tdesc.append(o.get_thresholded_descriptor())

    #for i in range(len(all_onset)): # for everything in an array of onset vectors based by sample
    #    temp = [all_onset[i+j] for j in range(temporal_window[0], temporal_window[1]) if (i+j >= 0) and (i+j < len(frames))] # gets onset vector values between n-5 and n+1
    #    t = np.median(temp) + temporal_weight*np.mean(temp) # calculate threshold for frame using a temporal weight of 0.1
    #    if all_onset[i] < t: # if onset is below the threshold
    #        all_onset[i] = 0 # remove onset by setting its vector to 0
    #        print(i)
        
    print(onsets)

    if graph:
        # do plotting, taken from aubio python example
        import matplotlib.pyplot as plt
        allsamples_max = (allsamples_max > 0) * allsamples_max
        allsamples_max_times = [ float(t) * hop_size / downsample / s.samplerate for t in range(len(allsamples_max)) ]
        plt1 = plt.axes([0.1, 0.75, 0.8, 0.19])
        plt2 = plt.axes([0.1, 0.1, 0.8, 0.65], sharex = plt1)
        plt.rc('lines',linewidth='.8')
        plt1.plot(allsamples_max_times,  allsamples_max, '-b')
        plt1.plot(allsamples_max_times, -allsamples_max, '-b')
        for stamp in onsets:
            stamp /= float(s.samplerate)
            plt1.plot([stamp, stamp], [-1., 1.], '-r')
        plt1.axis(xmin = 0., xmax = max(allsamples_max_times) )
        plt1.xaxis.set_visible(False)
        plt1.yaxis.set_visible(False)
        desc_times = [ float(t) * hop_size / s.samplerate for t in range(len(desc)) ]
        desc_max = max(desc) if max(desc) != 0 else 1.
        desc_plot = [d / desc_max for d in desc]
        plt2.plot(desc_times, desc_plot, '-g')
        tdesc_plot = [d / desc_max for d in tdesc]
        for stamp in onsets:
            stamp /= float(s.samplerate)
            plt2.plot([stamp, stamp], [min(tdesc_plot), max(desc_plot)], '-r')
        plt2.plot(desc_times, tdesc_plot, '-y')
        plt2.axis(ymin = min(tdesc_plot), ymax = max(desc_plot))
        plt.xlabel('time (s)')
        plt.savefig('t.png', dpi=200)
        plt.show()

    #print(s.samplerate)
    interval = (int(s.samplerate*(60.0/bpm[1])), int(s.samplerate*(60.0/bpm[0])))
    print(interval)
    confidence = []
    percent = 0
    print(len(onsets))
    print(str(percent) + "%")
    for f in range(interval[0], interval[1], intervalinterval):
        #if f != 25817:
        #    continue
        if int((f-interval[0])/(interval[1]-interval[0])*100) > percent: # keeping track of percent finished for calculating initial confidence values for each interval
            percent = int((f-interval[0])/(interval[1]-interval[0])*100)
            print(str(percent) + "% initial confidence completed")
        #graph_interval_freqs = []
        interval_freqs = [0 for number in range(f)]
        for n in onsets:
            interval_freqs[n%f] += 1
            #graph_interval_freqs.append(n%f)
        #if max(interval_freqs) > 4:
        #    print(max(interval_freqs))
        #if f == interval[0]:
        #if f == 15747: # the most correct 60/BPM * 44100 for pattyshort.wav (168 bpm)
        #if f == 12907: # the most correct 60/BPM * 44100 for jigokukakurenbo.wav (205 bpm)
        #if f == 25817: # the most correct 60/BPM * 44100 for jigokukakurenbo.wav (102.5 bpm)
        #    for i in range(len(interval_freqs)):
        #        if interval_freqs[i]:
        #            print(str(i)+": "+str(interval_freqs[i]))
            #matplotlib.pyplot.hist(graph_interval_freqs, bins=200)
            #matplotlib.pyplot.ylabel("onset count")
            #matplotlib.pyplot.show()
        maxval = 0
        for n in range(len(onsets)):
            cval = ev(n, interval_freqs)+(ev(n+(0.5*f), interval_freqs)/2.0)
            if cval > maxval: maxval = cval
        confidence.append(maxval)

    #return

    # correct with polynomial function somehow
    print("Initial confidence values calculated")

    bestbpms = []
    percent = 0
    print(len(confidence))
    maxconf = max(confidence)
    print("Max initial confidence value: " + str(maxconf))
    confidencelen = len(confidence)
    onsetrange = range(len(onsets))
    for c in range(confidencelen):
        #print(c)
        if not c % 20: # hopefully slightly optimized by not constantly checking
            if int((c/confidencelen)*100) > percent: # keeping track of percent finished for calculating initial confidence values for each interval
                percent = int((c/confidencelen)*100)
                print(str(percent) + "% more refined confidence completed")
        if confidence[c] > 0.4*maxconf:
            print(str(c) + " " + str(interval[0]+c) + " " + str(confidence[c]))
            temp = []
            for f in range(interval[0]+c-9, interval[0]+c+10, 1):
                interval_freqs = [0 for number in range(f)]
                for n in onsetrange:
                    interval_freqs[onsets[n]%f] += 1
                #if max(interval_freqs) > 4:
                #    print(max(interval_freqs))
                maxval = 0
                for n in onsetrange:
                    cval = ev(n, interval_freqs)+(ev(n+0.5*f, interval_freqs))/2
                    if cval > maxval: maxval = cval
                temp.append(maxval)
            bestbpms.append([maxval, 60.0/(float(c+temp.index(maxval)-9+interval[0])/s.samplerate), c+temp.index(maxval)-9+interval[0]])

    if subtract_polynomial:
        print("Refined confidence values calculated, determining polynomial to subtract confidence values from...")
        polyfit = np.polyfit(range(len(bestbpms)), [a[0] for a in bestbpms], 3)

        print(polyfit)
        for b in range(len(bestbpms)):
            bestbpms[b][0] = bestbpms[b][0]-((polyfit[0]*(bestbpms[b][0]**3))+(polyfit[1]*(bestbpms[b][0]**2))+(polyfit[2]*(bestbpms[b][0]))+polyfit[3])
            #if abs(bestbpms[b][1]-int(bestbpms[b][1])) < 0.05:

    print(sorted(bestbpms))

    if enable_profiling:
        pr.disable()
        pr.dump_stats("bpm.profile")

if __name__ == "__main__":
    main()
