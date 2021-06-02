import spotipy, requests
from spotipy.oauth2 import SpotifyClientCredentials
import aubio
import csv
from pathvalidate import sanitize_filename
from os.path import isfile

artists = {
    # every artist in the Spotify Top 50
    'Lil Nas X',
    'Justin Bieber',
    'Doja Cat',
    'Masked Wolf',
    'The Weeknd',
    'Polo G',
    'Dua Lipa',
    'Bruno Mars',
    'Anderson .Paak',
    'Gera MX',
    'Christian Nodal',
    'Los Legendarios',
    'Wisin',
    'Jhay Cortez',
    'Olivia Rodrigo',
    'Giveon',
    'Kali Uchis',
    'Bad Bunny',
    'AURORA',
    'Tiësto',
    'Sebastian Yatra',
    'Myke Towers',
    'Nio Garcia',
    'Flow La Movie',
    'Cardi B',
    'The Kid LAROI',
    '24kGoldn',
    'Duncan Laurence',
    'Juhn',
    'Glass Animals',
    'ATB',
    'Topic',
    'A7S',
    'Harry Styles',
    'Sech',
    'Maroon 5',
    'Travis Scott',
    'HVME',
    'BTS',
    'Lil Tjay',
    '6LACK',
    'KAROL G',
    'Mariah Angeliq',
    'Lewis Capaldi',
    'ROSALÍA',
    'Joel Corry',
    'RAYE',
    'David Guetta',
    'Selena Gomez',
    'Saweetie',
    'SZA',
    'Ariana Grande',
    'MEDUZA',
    # top artists of the decade according to billboard and my guesses for top 5
    'Drake',
    'Mariah Carey',
    'Taylor Swift',
    'BTS',
    'Adele',
    'Ed Sheeran',
    'Justin Bieber',
    'Katy Perry',
    'Maroon 5',
    'Post Malone',
    'Lady Gaga',
    'Ariana Grande',
    'Imagine Dragons',
    'The Weeknd',
    'Nicki Minaj',
    'Eminem',
    'Luke Bryan',
    'P!nk',
    'One Direction',
    'Justin Timberlake',
    'Kendrick Lamar',
    'Lady A',
    'Beyonce',
    'Jason Aldean',
    'Sam Smith',
    'Kesha',
    'Florida Georgia Line',
    'twenty one pilots',
    'Lil Wayne',
    'Chris Brown',
    'Blake Shelton',
    'Travis Scott',
    'Khalid',
    'Shawn Mendes',
    'Cardi B',
    'Future',
    'Mumford & Sons',
    'Selena Gomez',
    'JAY-Z',
    'Meghan Trainor',
    'J. Cole',
    'Usher',
    'Coldplay',
    'The Black Eyed Peas',
    'Pitbull',
    'Flo Rida',
    'Michael Buble',
    'Zac Brown Band',
    'Jason Derulo',
    'The Chainsmokers',
    'Halsey',
    'Lorde',
    'Kanye West',
    'Kenny Chesney',
    'Miley Cyrus',
    'Carrie Underwood',
    'Wiz Khalifa',
    'Migos',
    'Kelly Clarkson',
    'OneRepublic',
    'Macklemore & Ryan Lewis',
    'XXXTENTACION',
    'Eric Church',
    'Juice WRLD',
    'fun.',
    'Billie Eilish',
    'LMFAO',
    'DJ Khaled',
    'Chris Stapleton',
    'Calvin Harris',
    'Britney Spears',
    'Fetty Wap',
    'Sia',
    'Pentatonix',
    'Kidz Bop Kids',
    'David Guetta',
    'U2',
    'Ellie Goulding',
    'The Lumineers',
    'Pharrell Williams',
    'The Rolling Stones',
    'Train',
    'Trey Songz',
    'Demi Lovato',
    'Sam Hunt',
    'Big Sean',
    'Camila Cabello',
    'Lil Uzi Vert',
    'Panic! At The Disco',
    'Miranda Lambert',
    'Bruce Springsteen',
    'John Legend',
    'B.o.B',
    '21 Savage',
    'Thomas Rhett',
    'Meek Mill',
    'Keith Urban',
    'Bon Jovi',
    '5 Seconds Of Summer',
    'Paul McCartney',
    'Megan Thee Stallion',
    # artists i want to test
    'Michael Jackson',
    'Jacob Collier',
    'Green Day',
    'tricot',
    'Sheena Ringo',
    'Kohmi Hirose',
    'ミドリ',
    'Kikuo',
    'Björk',
    'Death Grips',
    'Danny Brown',
    'Tyler, the Creator',
    'BROCKHAMPTON',
}

CLIENT_ID = ''
CLIENT_SECRET = ''

auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

for artist in artists:
    results = sp.search(q=f'artist:{artist}', type='artist')
    items = results['artists']['items']
    if len(items) == 0:
        print(f'Artist {artist} not found on Spotify, skipping...')
        continue
    uri = items[0]['uri']
        
    results = sp.artist_top_tracks(uri)

    for track in results['tracks'][:10]:
        temp_fn = f"{track['id']}.mp3"

        # s = "{}--{}--{}--{}--{}.txt"
        title = track['name']
        print(f'Loading track {title} by {artist}...')
        
        if isfile(temp_fn):
            print('File already downloaded, using local version...')
        else:
            if track['preview_url'] is None:
                print('No preview URL, skipping...')
                continue
            
            r = requests.get(track['preview_url'], allow_redirects=True)
            content_type = r.headers.get('content-type', None)
            
            if content_type is not None and 'text' not in content_type.lower() and 'html' not in content_type.lower():
                with open(temp_fn, 'wb') as wf:
                    wf.write(r.content)
                print('File downloaded.')
            else:
                print('Not audio, skipping...')
                continue
            
        src = aubio.source(temp_fn)
        
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
        
        for method in out:
            with open(f"{track['id']}-{method}.txt", 'w') as wf:
                wf.write(artist + "\n")
                wf.write(title + "\n")
                wf.write(str(src.samplerate) + "\n")
                wf.write(method + "\n")
                for t in out[method]:
                    wf.write(str(t) + '\n')