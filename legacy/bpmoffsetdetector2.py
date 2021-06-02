# Nathan Stephenson
# BPM and Offset Detector translated directly from ArrowVortex source code in C++ to Python
# (with some merged code from the original implementation attempt I made in the main() function)

import aubio, copy, math
import numpy as np
from multiprocessing import Process, Queue, Manager, cpu_count

MinimumBPM = 89.0
MaximumBPM = 205.0
IntervalDelta = 50 # try 100, 50, 25, and 10
IntervalDownsample = 3
try:
    MaxThreads = cpu_count() * 16
except:
    MaxThreads = 1

class TempoResult:
    def __init__(self, bpm=0, offset=0.0, fitness=0):
        self.bpm = bpm
        self.offset = offset
        self.fitness = fitness

    def __repr__(self):
        return self.getFull()

    def __str__(self):
        return self.getReadable()

    def getFull(self):
        return "(" + str(self.bpm) + ", " + str(self.offset) + ", " + str(self.fitness) + ")"

    def getReadable(self):
        return "(" + str(round(self.bpm, 1)) + ", " + str(round(self.offset, 2)) + ", " + str(round(self.fitness, 1)) + ")"

class TempoSort:
    def operator(self, a, b):
        return a.fitness - b.fitness

class Onset:
    def __init__(self, pos=0, strength=0):
        self.pos = pos
        self.strength = strength
    
    def __repr__(self):
        return "(" + str(self.pos) + ", " + str(self.strength) + ")" 

class IntervalRange:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

class SerializedTempo:
    samples = []
    samplerate = 0
    numFrames = 0
    numThreads = 0
    terminate = ''
    progress = 0
    result = [] # TempoResult vector

class GapData:
    onsets = None
    wrappedPos = None
    wrappedOnsets = None
    window = None
    bufferSize = 0
    numOnsets = 0
    windowSize = 0
    downsample = 0

    def __init__(self, numThreads, bufferSize, downsample, numOnsets, onsets):
        self.downsample = downsample
        self.numOnsets = numOnsets
        self.onsets = onsets
        self.bufferSize = int(bufferSize)
        self.windowSize = 2048 >> downsample
        if numThreads == 0:
            numThreads = 1
        self.wrappedPos = [0 for n in range(self.numOnsets * int(numThreads))]
        self.wrappedOnsets = [0 for n in range(self.bufferSize * int(numThreads))]
        self.window = CreateHammingWindow(self.windowSize)

# Returns the confidence value that indicates how many onsets are close to the given gap position.
def GapConfidence(gapdata, threadId, gapPos, interval):
    numOnsets = gapdata.numOnsets
    windowSize = gapdata.windowSize
    halfWindowSize = windowSize / 2
    window = gapdata.window
    wrappedOnsets = gapdata.wrappedOnsets #+ gapdata.bufferSize * threadId
    area = 0.0

    beginOnset = int(gapPos - halfWindowSize)
    endOnset = int(gapPos + halfWindowSize)

    if beginOnset < 0:
        wrappedBegin = int(beginOnset + interval)
        # area += sum([wrappedOnsets[i] * window[i - wrappedBegin] for i in range(wrappedBegin, interval)])
        for i in range(wrappedBegin, interval):
            windowIndex = i - wrappedBegin
            area += wrappedOnsets[i] * window[windowIndex]
        beginOnset = 0

    if endOnset > interval:
        wrappedEnd = int(endOnset - interval)
        indexOffset = windowSize - wrappedEnd
        # area += sum([wrappedOnsets[i] * window[i + indexOffset] for i in range(wrappedEnd)])
        for i in range(wrappedEnd):
            windowIndex = i + indexOffset
            area += wrappedOnsets[i] * window[windowIndex]
        endOnset = interval

    # area += sum([wrappedOnsets[i] * window[i - beginOnset] for i in range(int(beginOnset), int(endOnset))])
    for i in range(int(beginOnset), int(endOnset)):
        windowIndex = i - beginOnset
        area += wrappedOnsets[i] * window[windowIndex]

    return area

# Returns the confidence of the best gap value for the given interval.
def GetConfidenceForInterval(gapdata, threadId, interval):
    downsample = gapdata.downsample
    numOnsets = gapdata.numOnsets
    onsets = gapdata.onsets

    wrappedPos = gapdata.wrappedPos #+ gapdata.numOnsets * threadId
    wrappedOnsets = gapdata.wrappedOnsets #+ gapdata.bufferSize * threadId

    #for i in range(gapdata.numOnsets * threadId):
    #    wrappedPos.append(0)
    #for i in range(gapdata.bufferSize * threadId):
    #    wrappedOnsets.append(0)

    interval = int(interval) # idk if i have to but

    # Make a histogram of onset strengths for every position in the interval.
    reducedInterval = interval >> downsample
    for i in range(numOnsets):
        pos = (onsets[i].pos % interval) >> downsample
        wrappedPos[i] = pos
        wrappedOnsets[pos] += onsets[i].strength

    # Record the amount of support for each gap value.
    highestConfidence = 0.0
    for i in range(numOnsets):
        pos = wrappedPos[i]
        confidence = GapConfidence(gapdata, threadId, pos, reducedInterval)
        offbeatPos = (pos + reducedInterval / 2) % reducedInterval
        confidence += GapConfidence(gapdata, threadId, offbeatPos, reducedInterval) * 0.5

        if confidence > highestConfidence:
            highestConfidence = confidence
    
    #print("Highest confidence value: " + str(highestConfidence))
    return highestConfidence

# Returns the confidence of the best gap value for the given BPM value.
def GetConfidenceForBPM(gapdata, threadId, test, bpm):
    numOnsets = gapdata.numOnsets
    onsets = gapdata.onsets

    wrappedPos = gapdata.wrappedPos
    wrappedOnsets = gapdata.wrappedOnsets

    # Make a histogram of i strengths for every position in the interval.
    intervalf = test.samplerate * 60.0 / bpm
    interval = int(intervalf + 0.5)
    for i in range(numOnsets):
        pos = int(math.fmod(onsets[i].pos, intervalf)) # might be able to just modulo here
        wrappedPos[i] = pos
        wrappedOnsets[pos] += onsets[i].strength

    # Record the amount of support for each gap value.
    highestConfidence = 0.0
    for i in range(numOnsets):
        pos = wrappedPos[i]
        confidence = GapConfidence(gapdata, threadId, pos, interval)
        offbeatPos = (pos + interval / 2) % interval
        confidence += GapConfidence(gapdata, threadId, offbeatPos, interval) * 0.5

        if confidence > highestConfidence:
            highestConfidence = confidence

    highestConfidence = NormalizeFitness(highestConfidence, test.coefs, intervalf)

    return highestConfidence

class IntervalTester:
    minInterval = 0
    maxInterval = 0
    numIntervals = 0
    samplerate = 0
    gapWindowSize = 0
    numOnsets = 0
    onsets = None
    fitness = None
    coefs = None # this is probably where the third degree polynomial goes (coefs[4])

    def __init__(self, samplerate, numOnsets, onsets):
        self.samplerate = samplerate
        self.numOnsets = numOnsets
        self.onsets = onsets
        self.minInterval = int(samplerate * 60.0 / MaximumBPM + 0.5)
        self.maxInterval = int(samplerate * 60.0 / MinimumBPM + 0.5)
        self.numIntervals = int(self.maxInterval - self.minInterval)
        self.fitness = [0 for n in range(int(self.numIntervals))]

def IntervalToBPM(test, i):
    return (test.samplerate * 60.0) / (i + test.minInterval)

def FillCoarseIntervals(test, gapdata, numThreads):
    numCoarseIntervals = int((test.numIntervals + IntervalDelta - 1) / IntervalDelta)
    if numThreads > 1:
        with Manager() as manager:
            tempfitness = manager.list()
            for x in test.fitness:
                tempfitness.append(0)
            processes = []
            '''
            class IntervalThreads(Process):
                test = None
                gapdata = None

                def __init__(self, queue, idx, **kwargs):
                    super(Processor, self).__init__()
                    self.queue = queue
                    self.idx = idx
                    self.kwargs = kwargs

                def run(self, i, thread):
                    index = i * IntervalDelta
                    interval = self.test.minInterval + index
                    self.test.fitness[index] = max((0.001, GetConfidenceForInterval(gapdata, thread, interval))
            '''
            i = 0
            while i < numCoarseIntervals:
                if len(processes) < numThreads and i < numCoarseIntervals:
                    p = Process(target=ThreadInterval, args=(gapdata, tempfitness, i, test.minInterval, IntervalDelta))
                    p.start()
                    processes.append(p)
                    i += 1
                for j in range(len(processes)-1, 0, -1):
                    if not processes[j].is_alive():
                        del processes[j]
                    if j > len(processes)-1:
                        j = len(processes)-1
            for p in processes:
                p.join()
                
            test.fitness = [x for x in tempfitness]
            #threads = IntervalThreads()
            #threads.test = test
            #threads.gapdata = gapdata
            #threads.run(numCoarseIntervals, numThreads)
    else:
        for i in range(numCoarseIntervals):
            index = i * IntervalDelta
            interval = test.minInterval + index
            print(interval)
            test.fitness[index] = max((0.001, GetConfidenceForInterval(gapdata, 0, interval)))

def ThreadInterval(gapdata, fitness, i, minInterval, IntervalDelta):
    index = i * IntervalDelta
    interval = minInterval + index
    fitness[index] = max((0.001, GetConfidenceForInterval(gapdata, 0, interval)))
    return "success"

def FillIntervalRange(test, gapdata, begin, end, numThreads):
    begin = max((begin, 0))
    end = min((end, test.numIntervals))
    fit = test.fitness
    #for i in range(begin):
    #    fit.append(0)
    
    i = begin
    interval = test.minInterval + begin
    #percent = 0
    if numThreads > 1:
        with Manager() as manager:
            processes = []
            tempfitness = manager.list()
            for x in test.fitness:
                tempfitness.append(x)
            while i < end:
                if fit[i] == 0 and len(processes) < numThreads:
                    p = Process(target=ThreadRange, args=(gapdata, tempfitness, i, test.coefs, interval))
                    p.start()
                    processes.append(p)
                    i += 1
                    interval += 1
                elif len(processes) < numThreads:
                    i += 1
                    interval += 1
                for j in range(len(processes)-1, 0, -1):
                    if not processes[j].is_alive():
                        del processes[j]
                    if j > len(processes)-1:
                        j = len(processes)-1
            for p in processes:
                p.join()
            
            test.fitness = [x for x in tempfitness]
    else:
        while (i < end):
            if fit[i] == 0:
                fit[i] = GetConfidenceForInterval(gapdata, 0, interval)
                fit[i] = NormalizeFitness(fit[i], test.coefs, interval)
                fit[i] = max((fit[i], 0.1))
            i += 1
            interval += 1

    return IntervalRange(begin, end) # idk if this needs to be mutable or not

def ThreadRange(gapdata, fit, i, coefs, interval):
    fit[i] = GetConfidenceForInterval(gapdata, 0, interval)
    fit[i] = NormalizeFitness(fit[i], coefs, interval)
    fit[i] = max((fit[i], 0.1))

def FindBestInterval(fitness, begin, end):
    bestInterval = 0
    highestFitness = 0.0

    for i in range(begin, end):
        if fitness[i] > highestFitness:
            highestFitness = fitness[i]
            bestInterval = i

    return bestInterval

# Removes BPM values that are near-duplicates or multiples of a better BPM value.
def RemoveDuplicates(tempo):
    for i in range(len(tempo), 0, -1):
        if i > len(tempo)-1:
            i = len(tempo)-1
        bpm = tempo[i].bpm
        doubled = bpm * 2.0
        halved = bpm * 0.5
        for j in range(len(tempo)-1, i, -1):
            v = tempo[j].bpm
            if min((abs(v - bpm), abs(v - doubled), abs(v - halved))) < 0.1:
                del tempo[j]

# Rounds BPM values that are close to integer values.
def RoundBPMValues(test, gapdata, tempo):
    for t in tempo:
        roundBPM = round(t.bpm)
        diff = abs(t.bpm - roundBPM)
        if diff < 0.01:
            t.bpm = roundBPM
        elif diff < 0.05:
            old = GetConfidenceForBPM(gapdata, 0, test, t.bpm)
            cur = GetConfidenceForBPM(gapdata, 0, test, roundBPM)
            if cur > old * 0.99:
                t.bpm = roundBPM

# Finds likely BPM candidates based on the given note onset values.
def CalculateBPM(data, onsets, numOnsets):
    tempo = data.result

    # In order to determine the BPM, we need at least two onsets.
    if numOnsets < 2:
        tempo.append(TempoResult(100.0, 0.0, 1.0))
        return

    print(sProgressText[1])

    test = IntervalTester(data.samplerate, numOnsets, onsets)
    gapdata = GapData(data.numThreads, test.maxInterval, IntervalDownsample, numOnsets, onsets)

    # Loop through every 10th possible BPM, later we will fill in those that look interesting.
    #FillCoarseIntervals(test, gapdata, data.numThreads)
    FillCoarseIntervals(test, gapdata, MaxThreads)
    numCoarseIntervals = int((test.numIntervals + IntervalDelta - 1) / IntervalDelta)
    #MarkProgress(2, "Fill course intervals")
    

    #print(tempo)
    #print(test.fitness)

    # Determine the polynomial coefficients to approximate the fitness curve and normalize the current fitness values.
    test.coefs = np.polyfit(range(len(test.fitness)), test.fitness, 3) # unsure ab this, line 402
    maxFitness = 0.001
    for i in range(0, test.numIntervals, IntervalDelta):
        #if test.fitness[i] != 0:
        #    print("Old: " + str(test.fitness[i]))
        test.fitness[i] = NormalizeFitness(test.fitness[i], test.coefs, test.minInterval + i)
        #if test.fitness[i] != 0:
        #    print("New: " + str(test.fitness[i]))
        maxFitness = max((maxFitness, test.fitness[i]))

    #print("Fitness values normalized.")

    print(sProgressText[2])

    # Refine the intervals around the best intervals.
    fitnessThreshold = maxFitness * 0.4
    #percent = 0
    for i in range(0, test.numIntervals, IntervalDelta):
        #if int(i/test.numIntervals) > percent: # keeping track of percent finished for calculating initial confidence values for each interval
        #    percent = int(i/test.numIntervals)*100
        #    print(str(percent) + "% fitness refining completed")
        if test.fitness[i] > fitnessThreshold:
            interval_range = FillIntervalRange(test, gapdata, i - IntervalDelta, i + IntervalDelta, MaxThreads)
            best = FindBestInterval(test.fitness, interval_range.x, interval_range.y)
            result = TempoResult(IntervalToBPM(test, best), 0.0, test.fitness[best])
            #print(result)
            tempo.append(result)

    #MarkProgress(3, "Refine intervals")
    
    print(sProgressText[3])

    # At this point we stop the downsampling and upgrade to a more precise gap window.
    gapdata = GapData(data.numThreads, test.maxInterval, 0, numOnsets, onsets)

    # Round BPM values to integers when possible, and remove weaker duplicates.
    tempo.sort(key=lambda x: x.fitness) # key needs to be something
    RemoveDuplicates(tempo)
    RoundBPMValues(test, gapdata, tempo)

    # If the fitness of the first and second option is very close, we ask for a second opinion.
    if len(tempo) >= 2 and tempo[0].fitness / tempo[1].fitness < 1.05:
        for t in tempo:
            t.fitness = GetConfidenceForBPM(gapdata, 0, test, t.bpm)
        tempo.sort(key=lambda x: x.fitness) # idk

    # In all 300 test cases the correct BPM value was part of the top 3 choices,
	# so it seems reasonable to discard anything below the top 3 as irrelevant.
    if len(tempo) > 3:
        tempo = tempo[0:3]

    #data.result = tempo # just to make sure

def ComputeSlopes(samples, out, numFrames, samplerate):
    wh = samplerate // 20
    if numFrames < wh * 2: return

    # Initial sums of the left/right side of the window.
    sumL, sumR = 0, 0
    j = wh
    for i in range(wh):
        sumL += abs(samples[i])
        sumR += abs(samples[j])
        j += 1

    # Slide window over the samples.
    scalar = 1.0 / wh
    end = numFrames
    for i in range(wh, end):
        # Determine slope value.
        out[i] = max((0.0, ((sumR - sumL) * scalar)))

        # Move window.
        cur = abs(samples[i])
        sumL -= samples[abs(max((0, i - wh)))]
        sumL += cur
        sumR -= cur
        sumR += samples[abs(min((i + wh, len(samples)-1)))]

# Returns the most promising offset for the given BPM value.
def GetBaseOffsetValue(gapdata, samplerate, bpm):
    numOnsets = gapdata.numOnsets
    onsets = gapdata.onsets
    
    wrappedPos = gapdata.wrappedPos
    wrappedOnsets = gapdata.wrappedOnsets

    # Make a histogram of onset strengths for every position in the interval.
    intervalf = samplerate * 60.0 / bpm
    interval = int(intervalf + 0.5)
    for i in range(numOnsets):
        pos = int(math.fmod(onsets[i].pos, intervalf)) # might be able to just modulo here
        wrappedPos[i] = pos
        wrappedOnsets[pos] += 1.0

    # Record the amount of support for each gap value.
    highestConfidence = 0.0
    offsetPos = 0
    for i in range(numOnsets):
        pos = wrappedPos[i]
        confidence = GapConfidence(gapdata, 0, pos, interval)
        offbeatPos = (pos + interval / 2) % interval
        confidence += GapConfidence(gapdata, 0, offbeatPos, interval) * 0.5

        if confidence > highestConfidence:
            highestConfidence = confidence
            offsetPos = pos
    
    return offsetPos / samplerate

def AdjustForOffbeats(data, offset, bpm):
    samplerate = data.samplerate
    numFrames = data.numFrames

    # Create a slope representation of the waveform.
    slopes = [0 for x in range(numFrames)]
    ComputeSlopes(data.samples, slopes, numFrames, samplerate)

    # Determine the offbeat sample position.
    secondsPerBeat = 60.0 / bpm
    offbeat = offset + secondsPerBeat * 0.5
    if offbeat > secondsPerBeat:
        offbeat -= secondsPerBeat
    
    # Calculate the support for both sample positions.
    end = numFrames
    interval = secondsPerBeat * samplerate
    posA = offset * samplerate
    sumA = 0.0
    posB = offbeat * samplerate
    sumB = 0.0
    while posA < end and posB < end:
        sumA += slopes[int(posA)]
        sumB += slopes[int(posB)]
        posA += interval
        posB += interval
    
    # Return the offset with the highest support.
    return offset if sumA >= sumB else offbeat

def CalculateOffset(data, onsets, numOnsets, numThreads):
    tempo = data.result
    samplerate = data.samplerate

    # Create gapdata buffers for testing.
    maxInterval = max([samplerate * 60.0 / t.bpm for t in tempo])
    gapdata = GapData(1, int(maxInterval + 1.0), 1, numOnsets, onsets)

    if numThreads > 1:
        with Manager() as manager:
            processes = []
            temptempo = manager.list()
            for t in tempo:
                temptempo.append(t)
            for i in range(len(tempo)):
                if len(processes) < numThreads:
                    p = Process(target=ThreadOffset, args=(gapdata, samplerate, data, temptempo, i))
                    p.start()
                    processes.append(p)
                    i += 1
                for j in range(len(processes)-1, 0, -1):
                    if not processes[j].is_alive():
                        del processes[j]
                    if j > len(processes)-1:
                        j = len(processes)-1
            for p in processes:
                p.join()

            print(temptempo)

            processes = []
            for i in range(len(tempo)):
                if len(processes) < numThreads:
                    p = Process(target=ThreadOffsetAdjust, args=(gapdata, samplerate, data, temptempo, i))
                    p.start()
                    processes.append(p)
                    i += 1
                for j in range(len(processes)-1, 0, -1):
                    if not processes[j].is_alive():
                        del processes[j]
                    if j > len(processes)-1:
                        j = len(processes)-1
            for p in processes:
                p.join()

            print(temptempo)
            
            tempo = [x for x in temptempo]

            data.result = tempo # in case

    else:
        for t in tempo:
            t.offset = GetBaseOffsetValue(gapdata, samplerate, t.bpm)

        for t in tempo:
            t.offset = AdjustForOffbeats(data, t.offset, t.bpm)

def ThreadOffset(gapdata, samplerate, data, tempo, i):
    tempo[i].offset = GetBaseOffsetValue(gapdata, samplerate, tempo[i].bpm)

def ThreadOffsetAdjust(gapdata, samplerate, data, tempo, i):
    tempo[i].offset = AdjustForOffbeats(data, tempo[i].offset, tempo[i].bpm)

sProgressText = (
    "[1/6] Looking for onsets",
	"[2/6] Scanning intervals",
	"[3/6] Refining intervals",
	"[4/6] Selecting BPM values",
	"[5/6] Calculating offsets",
	"BPM detection results"
)

# Creates weights for a hamming window of length n.
def CreateHammingWindow(n):
    t = 6.2831853071795864 / (n - 1)
    return [(0.54 - 0.46 * math.cos(i*t)) for i in range(n)]

# Normalizes the given fitness value based the given 3rd order poly coefficients and interval.
def NormalizeFitness(fitness, coefs, interval):
    x = interval
    x2 = x*x
    x3 = x2*x
    return fitness - (coefs[0]*x3 + coefs[1]*x2 + coefs[2]*x + coefs[3])

def main():
    #if enable_profiling:
    #    pr = cProfile.Profile()
    #    pr.enable()
    samplerate = 0  # use original source samplerate
    hop_size = 256 # number of frames to read in one block
    window_size = 1024
    #temporal_window = (-5, 2) # add one to end
    silence_threshold = -70 # in decibels
    #temporal_weight = 0.1
    #bpm = (89, 205) # min and max (there isn't much of a reason to change because if it's close it will fix itself on the second run)
    #intervalinterval = 20 # what interval to use when checking histogram values between the bpm interval (the "step")

    maxseconds = 120 # only analyze first x seconds if applicable

    #desc = []
    #tdesc = []
    #allsamples_max = np.zeros(0,)
    #downsample = 2  # to plot n samples / hop_s

    #filename = "pattyshort.wav" # 168 bpm
    filename = "jigokukakurenbo.mp3"
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
        #if graph:
        #    new_maxes = (abs(samples.reshape(hop_size//downsample, downsample))).max(axis=0)
        #    allsamples_max = np.hstack([allsamples_max, new_maxes])
        total_frames += read
        if read < hop_size or (total_frames > maxframes and maxframes >= 0): break
    fmt_string = "read {:d} frames at {:d}Hz from {:s}"
    print(fmt_string.format(total_frames, s.samplerate, filename))

    data = SerializedTempo()
    data.progress = 0

    data.numThreads = 0 # for now (ParallelThreads::concurrency();)
    data.numFrames = total_frames
    data.samplerate = s.samplerate
    
    
    # Copy the input samples.
    data.samples = [sample for frame in frames for sample in frame]
    
    #for frame in frames:
    #    for sample in frame:
    #        data.samples.append(sample)
    #print(len(data.samples))

    print(sProgressText[0])

    for i in range(len(frames)):
        onset = o(frames[i])
        #all_onset.append(onset)
        if onset:
            #print("%f" % o.get_last_s())
            onsets.append(Onset(pos=o.get_last()))

    # MarkProgress(1, "Find onsets")
    
    # so in this implementation we need to make onsets their own class as well as TempoResult

    for i in range(min((len(onsets), 100))):
        a = max((0, onsets[i].pos - 100))
        b = min((data.numFrames, onsets[i].pos + 100))
        v = 0.0
        for j in range(a, b):
            v += abs(data.samples[j])
        v /= float(max((1, b - a)))
        onsets[i].strength = v

    #print(onsets)
    #print("Onset strength calculated.")

    # Find BPM values.
    CalculateBPM(data, onsets, len(onsets)) # there is onsets.begin() but that is for vectors, like a linked list sorta
    # MarkProgress(4, "Find BPM")

    #print(data.result)

    # Find offset values.
    print(sProgressText[4])
    CalculateOffset(data, onsets, len(onsets), MaxThreads)
    # MarkProgress(5, "Find offsets")

    print(sProgressText[5])

    print([str(x) for x in data.result])

if __name__ == "__main__":
    main()