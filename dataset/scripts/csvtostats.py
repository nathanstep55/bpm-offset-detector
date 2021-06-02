import csv
from collections import Counter

def mode(sample):
    c = Counter(sample)
    modes = [k for k, v in c.items() if v == c.most_common(1)[0][1]]
    return modes[0] if len(modes) == 1 else None

db = {}

with open('output1wspotifymadmomaubio.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile, fieldnames=['artist', 'title', 'id', 'method', 'ranking', 'bpm', 'offset', 'fitness'])
    for entry in reader:
        if int(entry['ranking']) != 1: continue
        if entry['title'] not in db:
            db[entry['title']] = {}
        db[entry['title']][entry['method']] = float(entry['bpm'])

methods_mode = {}
methods_mean = {}
methods_below_threshold = {}
methods_x2 = {}
for a in db:
    for method in db[a]:
        methods_mode[method] = []  # list of squared errors
        methods_mean[method] = []
        methods_below_threshold[method] = 0
        methods_x2[method] = 0
    break


for song in db:
    vals = db[song].values()
    m = mode(vals)
    if m is None:  # only look at less ambiguous cases
        continue
    mean = sum(vals)/len(vals)
    for method in db[song]:
        mode_diff = abs(db[song][method] - m)
        mode_diff_x2 = abs(db[song][method]*2 - m)
        if mode_diff_x2 < mode_diff:
            methods_x2[method] += 1
            mode_diff = mode_diff_x2
        if mode_diff < 0.05 and mode_diff != 0:
            methods_below_threshold[method] += 1
            mode_diff = 0
        mode_diff *= mode_diff
        methods_mode[method].append(mode_diff)
        
        mean_diff = abs(db[song][method] - mean)
        mean_diff_x2 = abs(db[song][method]*2 - mean)
        if mean_diff_x2 < mean_diff:
            mean_diff = mean_diff_x2
        mean_diff *= mean_diff
        methods_mean[method].append(mean_diff)

print('MSE using mode as "ground truth" + 0.05 threshold')
results_mode = sorted([(k, sum(v)/len(v)) for k, v in methods_mode.items()], key=lambda x: x[1])
for n, method in enumerate(results_mode):
    print(f"{n+1}. {method[0]}: {method[1]:.3f}")
    
print()

print('MSE using mean as "ground truth"')
results_mean = sorted([(k, sum(v)/len(v)) for k, v in methods_mean.items()], key=lambda x: x[1])
for n, method in enumerate(results_mean):
    print(f"{n+1}. {method[0]}: {method[1]:.3f}")
    
print()

print('error count')
results_error = sorted([(method, len([x for x in methods_mode[method] if x > 0])) for method in methods_mode], key=lambda x: x[1])
for n, method in enumerate(results_error):
    print(f"{n+1}. {method[0]}: {method[1]}/{len(methods_mode[method[0]])}")
    
print()

print('error below 0.05 threshold count')
for n, method in enumerate(sorted(methods_below_threshold.items(), key=lambda x: x[1])):
    print(f"{n+1}. {method[0]}: {method[1]}")
    
print()

print('count of half-BPM detections')
for n, method in enumerate(sorted(methods_x2.items(), key=lambda x: x[1])):
    print(f"{n+1}. {method[0]}: {method[1]}")