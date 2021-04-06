"""
real word errors

gen-test

@author Daniel Bravo daniel.bravo@um.es
@author Jesica López <jesica.lopez@um.es>
@author José Antonio García-Díaz joseantonio.garcia8@um.es
@author Fernando Molina-Molina <fernando.molina@vocali.net>
@author Francisco García Sánchez <frgarcia@um.es>
"""

from funciones import load_doc, clean_doc, compute_LSTM_prob, w_probabs, lev_dist, get_word, save_doc, new_seed, compute_LSTM_prob_embed
from pickle import load
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
import nltk
import io


def load_vectors (fname):
    """
    @param fname
    """
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
 
    return data


# @var doc Load raw and preprocessed text
doc = load_doc ('./../embeddings/test_wiki005_stopword_embedding.txt')


# @var doc_proc Load processed data
doc_proc = load_doc ('./../embeddings/sequences_test_wiki005_stopword_embedding.txt')
lines = doc_proc.split ('\n')
seq_length = len(lines[0].split()) - 1


# @var model Model
model = load_model ('./../models/model_Wiki_cc.es.300_lines05_stopwords.h5')


# @var tokenizer Tokenizer 
tokenizer = load (open ('./../tokenizers/tok_dict_Wiki_05_stopword.pkl', 'rb'))


# @var embedding Matrix
embedding = load_vectors ('./../embeddings/pretrained_model/cc.es.300.vec')


# @var lWords_in
lWords_in = doc.split ()


# Create a copy of the tokenizer
vocabulary = tokenizer


# @var lWords_out List
lWords_out = [i for i in lWords_in[0:seq_length]]


for i in range (seq_length, seq_length + 100): 
 
    # calculate lstm probabilities
    p_in = compute_LSTM_prob_embed(model, seq_length, seed_text, embedding)
 
 
    # calculate levenshtein distance
    dist = lev_dist(vocabulary, lWords_in[i])
    
    
    # calculate final probabilities
    thresh = 2
    p_out = w_probabs(dist, p_in, thresh)
 
 
    # get corrected word
    out = get_word(p_out, tokenizer)
 
 
    lWords_out.append(out)
    seed_text = new_seed(seed_text, lWords_in[i])


# Output
c = 0
for i in lWords_out:
    print(lWords_in[c], '/', lWords_out[c])
    c = c + 1

