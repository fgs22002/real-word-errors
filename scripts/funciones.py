"""
real word errors

helper-functions

@author Daniel Bravo daniel.bravo@um.es
@author Jesica López <jesica.lopez@um.es>
@author José Antonio García-Díaz joseantonio.garcia8@um.es
@author Fernando Molina-Molina <fernando.molina@vocali.net>
@author Francisco García Sánchez <frgarcia@um.es>
"""

import numpy as np
import math
import re
import string

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM, Dropout, SpatialDropout1D
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.metrics.distance import edit_distance



def load_doc (filename):
    """
    load text into memory
    
    @param filename String
    """
    fi = open(filename, 'r')
    text = fi.read()
    fi.close()
    return text
 

def clean_doc (doc):
    """
    clean text
    
    @param doc String
    """
    
    # replace '--' with a space ' '
    doc = doc.replace ('-', ' ')
    doc = doc.replace ('ENDOFARTICLE.', ' ')
    doc = doc.replace ( 'doc id', ' ')
    doc = doc.replace ( 'doc', ' ')
    doc = doc.replace ( 'title', ' ')
    doc = doc.replace ( 'nonfiltered', ' ')
    doc = doc.replace ( 'processed', ' ')
    doc = doc.replace ( 'dbindex', ' ')
    
    
    # split into tokens by white space
    tokens = doc.split()
    
    
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    
    
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    
    
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    
    
    # make lower case
    tokens = [word.lower() for word in tokens]
    
    return tokens


def save_doc (lines, filename):
    """ 
    save into new file per line
    
    @param lines List
    @param filename String
    """
    data = '\n'.join (lines)
    file = open (filename, 'w')
    file.write (data)
    file.close ()
 

def load_doc (filename):
    """
    LOAD SEQUENCES
    """
    file = open (filename, 'r')
    text = file.read()
    file.close()
    return text
 

def define_model (vocab_size, seq_length):
    """ 
    build model
    
    @param vocab_size int
    @param seq_length int
    """
    model = Sequential ()
    model.add (Embedding(vocab_size, 50, input_length=seq_length))
    model.add (LSTM(100, return_sequences=True))
    model.add (LSTM(100))
    model.add (Dense(100, activation='relu'))
    model.add (Dense(vocab_size, activation='softmax'))

    
    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    
    #summarize defined model
    model.summary()
    return model


def lev_dist(voc, next_word):
    """
    calculate levenshtein distance
    
    @param voc List
    @param next_word string
    
    """

    voc_len = len(voc)
    L_v = np.zeros(voc_len+1, dtype=np.int8)
    L_v[0] = 5
    c = 1
 
    for list_word in voc:
        dist = edit_distance(list_word, next_word, transpositions=True)
        L_v[c] = dist
        c += 1
  
    return L_v


def w_probabs (L_v, p_in, thres):
    """
    Function that computes final probabilities for detection
    
    @param L_v
    @param p_in
    @param thres float
    
    """

    L_shape = np.shape (L_v)
    p_shape = np.shape (p_in)
    greater_dim = max (p_shape)
    p_out = np.zeros (p_shape, dtype=np.float64)
 
    for i in range(greater_dim):
        if L_v[i,] <= thres:
            p_out[0,i] = p_in[0,i]/(1 + math.exp(L_v[i,]))
        else:
            p_out[0,i] = 0.0
 
    return p_out
 

def compute_LSTM_prob (model, tokenizer, seq_length, seed_text):
    """
    compute_LSTM_prob
    
    @param model
    @param tokenizer
    @param seq_lenth
    @param seed_text
    """

    # @var encoded
    encoded = tokenizer.texts_to_sequences ([seed_text])[0]
    encoded = pad_sequences ([encoded], maxlen=seq_length, truncating = 'pre')
    
    
    # @var y_prob
    y_prob = model.predict_proba(encoded, verbose=0)
    vocab_size = len(tokenizer.word_index) + 1
    return y_prob

def compute_LSTM_prob_embed (model, seq_length, seed_text, embedding):
    
    """
    compute_LSTM_prob_embed
    
    @param model
    @param seq_length
    @param seed_text String
    @param embedding String
    """
    
    input_array =  np.ndarray((1, seq_length, 300), dtype=float)
    seed_text = str(seed_text)
    words = seed_text.split()
    word_i = 0
    for term in words:
        if word_i==50:
            break
        
        emb_map = embedding[term]
        emb_array = from_map_to_array(emb_map, 300)
        input_array[0, word_i, :] = emb_array
        word_i = word_i + 1
 
    y_prob = model.predict_proba(input_array, verbose=0)
 
    return y_prob

def new_seed (seed_text, next_word):
    seed_text = str(seed_text)
    seed_text = seed_text.split()
    count = 0
    out = ''
 
    for word in seed_text:
        if count == 0:
            count = count + 1
        else:
            out = out + ' ' + word
            count = count + 1
 
    new_seed = out.split()
    new_seed = new_seed.append(next_word)
    return new_seed


def from_map_to_array (map_object, array_shape):
    """
    transforms a map object containing float numbers to a numpy array. 
    
    @param map_object: an integer.
    @param array_shape int
    """
    array = np.ndarray(array_shape, dtype=float)
    count = 0
    
    for element in map_object:
        array[count] = element
        count = count + 1
    return array
 


def get_word (p_out, tokenizer):
    """
    Returns the word with high proba
    
    @param p_out
    @param tokenizer
    
    """
    
    index= p_out.argmax()
    out_word = ''
 
    for word, i in tokenizer.items():
        if i == index:
            return out_word
            break
  
 