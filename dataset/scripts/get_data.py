import os
import os.path

for filename in os.listdir('.'):
    if os.path.splitext(filename)[1] != '.txt':
        continue
    fn = f"{os.path.splitext(filename)[0].split('-')[0]}.txt"
    if os.path.isfile(fn):
        continue
    with open(filename) as infile:
        with open(fn, 'w') as outfile:
            for line in infile.readlines()[:3]:
                outfile.write(line)