import numpy as np
import re
import gensim
from gensim.models import word2vec
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity


stop_words = set(stopwords.words("english"))
stop_words.update({'im', 'em', 'on', 'say', 'cant', 'bout', 'about', 
                   'that', 'thats', 'put', 'askin', 'weve', 'got', 'could'
                   'would', 'let'})

word_list = {
    "Stressed": 
        ["stress", "uneasy", "nervous", "discomfort", "inconvenient", "Anxious", "Exhausting", "Intense", 
         "Panic", "Unmanageable", "Breakdown"],
    
    "Depressed":
        ["Melancholic", "Somber", "Blue", "Gloomy", "depression", "rain", "Overwhelming", "Desperate", 
         "Suicidal", "Paralyzed", "Incapacitating", "Unbearable", "Devastated", "cry", "tears", "pain"],
    
    "Neutral":
        ["bored", "simple", "pleasant", "normal", "neutral"],
        
    "Happy":
        ["Stable", "Calm", "Satisfied", "Happy", "Content", "Manic", "Euphoric", "Ecstatic", "Party",
         "Dance", "excited"],
        
    "Studious":
        ["smart", "study", "brain", "school", "grades", "graduation", "homework", "scholar", "work",
         "writing", "teacher", "students", "reading", "intelligent", "education"]
}

def pred_to_mh(pred):
    label_mapping = {
        0: "Stressed",
        1: "Depressed",
        2: "Neutral",
        3: "Happy",
        4: "Studious"
    }

    return label_mapping[pred]

def remove_stopwords(text):
    text_without_newlines = text.replace("\n", " ")
    words = text_without_newlines.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)

def clean_all(list):    
    cleaned_lists = []
    for text in list:
        filtered_text_list = text.lower()
        filtered_text_list = re.sub(r'[^\w\s]', '', filtered_text_list)
        filtered_text_list = remove_stopwords(filtered_text_list)
        cleaned_lists.append(filtered_text_list)
    return cleaned_lists

def build_corpus(list_of_words, list_of_song_lyrics):
    corpus = []
    
    for word in list_of_words:
        corpus.append(word)
    for song_lyrics in list_of_song_lyrics:
        for word in song_lyrics.split():
            corpus.append(word)
            
    corpus = list(set(corpus))
    return corpus

def get_word2vec_model(corpus):
    sentences = [word.split() for word in corpus]

    model = word2vec.Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)

    model.save("word2vec.model")
    
    return model

def word_embed(list, model, song = True):
    word_embeddings = []
    song_embeddings = []
    sentences = [word.split() for word in list]
    
    for sentence in sentences:
        word_embeddings.append(model.wv[sentence])
        
    if song:
         for sentence_embeddings in word_embeddings:
            song_embedding = np.mean(sentence_embeddings, axis=0)     
            song_embeddings.append(song_embedding) 
         return song_embeddings  
    
    else:
        return word_embeddings
    
def get_song(list_of_words, list_of_song_lyrics, data):
    list_of_words_embeddings = word_embed(list_of_words)
    song_lyrics_embeddings = word_embed(list_of_song_lyrics)

    list_avg_embedding = np.mean(song_lyrics_embeddings, axis=0)

    similarities = [cosine_similarity([list_avg_embedding], [word_embedding])[0][0] for word_embedding in list_of_words_embeddings]

    best_match_index = np.argmax(similarities)

    best_matching_song = data[best_match_index]['songName']

    return best_matching_song, similarities

def filter_songs(song_type, data):
    filtered_data = [record for record in data if song_type in record.get('mood', '').lower()]
    return filtered_data
