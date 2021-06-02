import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import csv

CLIENT_ID = ''
CLIENT_SECRET = ''

auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

ids = {}
for filename in os.listdir('.'):
    if os.path.splitext(filename)[1] != '.txt':
        continue
        
    track_id = os.path.splitext(filename)[0]
    
    ids[track_id] = {}
    txt_fn = f"{track_id}.txt"
    with open(txt_fn) as infile:
        l = infile.readlines()
        ids[track_id]['artist'] = l[0].strip()
        ids[track_id]['title'] = l[1].strip()

with open('spotifyout.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    id_list = list(ids.keys())
    for i in range(0, len(ids)//100 + 1):
        result = sp.audio_features(tracks=id_list[i*100:(i*100)+100])
        for d in result:
            if d is None: continue
            writer.writerow([ids[d['id']]['artist'], ids[d['id']]['title'], d['id'], 'spotify', 1, d['tempo'], 0, 1])