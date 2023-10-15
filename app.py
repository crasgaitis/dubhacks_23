from flask import Flask, render_template, session
from matplotlib import pyplot as plt
import random

app = Flask(__name__)
app.secret_key = 'cat_key'

from pylsl import StreamInlet, resolve_byprop
import pandas as pd
import numpy as np
from utils import *
import pickle
from itertools import chain
import json

BUFFER_LENGTH = 5
EPOCH_LENGTH = 1
OVERLAP_LENGTH = 0
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH
INDEX_CHANNEL = [0]

with open('model_building/eeg_to_mh_state.pkl', 'rb') as file:
    model = pickle.load(file)
    
with open('music/songs.json', 'r') as json_file:
    data = json.load(json_file)

list_of_song_lyrics = [song_record["lyrics"] for song_record in data]
list_of_song_lyrics = clean_all(list_of_song_lyrics)
list_of_words = clean_all(list(chain(*word_list.values())))
corpus = build_corpus(list_of_words, list_of_song_lyrics)
sentences = [word.split() for word in corpus]
modelw2v = word2vec.Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/select-genre-page.html')
def select_genre_page():
    return render_template('select-genre-page.html')

@app.route('/user-input-feelings-page.html')
def user_input_feelings_page():
    return render_template('user-input-feelings-page.html')

@app.route('/egg-page.html')
def egg_page():
    return render_template('egg-page.html')

@app.route('/loading-page.html')
def loading_page():
    return render_template('loading-page.html')

@app.route('/results.html')
def results_page():
    
    # Search for active LSL streams
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')
    else:
        print('Found it!')
        print(streams)
        
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)

    info = inlet.info()
    fs = int(info.nominal_srate())

    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state = None  # for use with the notch filter

    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                                SHIFT_LENGTH + 1))

    band_buffer = np.zeros((n_win_test, 4))

    eeg_data, timestamp = inlet.pull_chunk(
        timeout=1, max_samples=int(SHIFT_LENGTH * fs))

    ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]

    eeg_buffer, filter_state = update_buffer(
        eeg_buffer, ch_data, notch=True,
        filter_state=filter_state)

    data_epoch = get_last_data(eeg_buffer,
                                        EPOCH_LENGTH * fs)

    # compute band powers
    band_powers = compute_band_powers(data_epoch, fs)
    band_buffer, _ = update_buffer(band_buffer,
                                            np.asarray([band_powers]))
    bw_data = {
    'Delta': [band_powers[0]],
    'Theta': [band_powers[1]],
    'Alpha': [band_powers[2]],
    'Beta': [band_powers[3]]
    }

    df = pd.DataFrame(bw_data)
    
    # plot 
    x = [1, 2, 3, 4]
    plt.scatter(x, band_powers[0:5], c='blue', marker='o', label='Points')

    for i in range(len(x)):
        plt.plot([x[i], x[i]], [0, band_powers[0:5][i]], 'r--')
        
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
        
    custom_xticks = ['Alpha', 'Beta', 'Theta', 'Delta']
    ax.set_xticks(x)
    ax.set_xticklabels(custom_xticks)

    plt.xlabel('Brainwave Band')
    plt.ylabel('Avg Voltage')
    plt.ylim(0, 3)
    
    plt.savefig('static/plot_image.png', transparent=True)
    
    pred = model.predict(df)
    result = pred_to_mh(pred[0])
    response = response_list[result]
    icon = icon_list[result]
    
    list_of_words = word_list[result]
    # new_words = ["I am so sad and I want to cry"]
    
    # for word in new_words:
    #     list_of_words.append(word)
    
    filtered_data = filter_songs(str(result).lower(), data)
    list_of_song_lyrics = [song_record["lyrics"] for song_record in filtered_data]
    
    list_of_song_lyrics = clean_all(list_of_song_lyrics)
    
    list_of_words = clean_all(list_of_words)

    best_matching_song, spotifyLink, similarities = get_song(list_of_words, list_of_song_lyrics, filtered_data, modelw2v)
    print(similarities)
    similarities = len(similarities) - 1
    print(similarities)

    pattern = r"/track/(\w+)"
    match = re.search(pattern, str(spotifyLink))
    
    if match:
        track_id = match.group(1)
    else: 
        track_id = 'fail'
        
    spotifyLink = "https://open.spotify.com/embed/track/" + track_id + "?utm_source=generator"
    session['similarities'] = similarities
    session['icon'] = icon
    session['best_matching_song'] = best_matching_song
    session['spotifyLink'] = spotifyLink
    session['result'] = result
    session['response'] = response
    session['filtered_data'] = filtered_data
    
    return render_template('results.html', icon = icon, best_matching_song = best_matching_song, spotifyLink = spotifyLink, pred = result, response = response)

@app.route('/results-more.html')
def results_more_page():
    
    similarities = session.get('similarities')
    icon = session.get('icon') 
    result = session.get('result')
    response = session.get('response')
    filtered_data = session.get('filtered_data')
    
    random_index = random.randint(0, similarities)
    
    best_matching_song = filtered_data[random_index-1]['songName']
    spotifyLink = filtered_data[random_index-1]['spotifyLink']
    
    pattern = r"/track/(\w+)"
    match = re.search(pattern, str(spotifyLink))
    
    if match:
        track_id = match.group(1)
    else: 
        track_id = 'fail'
    
    spotifyLink = "https://open.spotify.com/embed/track/" + track_id + "?utm_source=generator"

    return render_template('results-more.html', icon = icon, best_matching_song = best_matching_song, spotifyLink = spotifyLink, pred = result, response = response)

if __name__ == '__main__':
    app.run(debug=True)