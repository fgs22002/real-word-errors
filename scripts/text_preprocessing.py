"""
real word errors

text-preprocessing

@author Daniel Bravo daniel.bravo@um.es
@author Jesica López <jesica.lopez@um.es>
@author José Antonio García-Díaz joseantonio.garcia8@um.es
@author Fernando Molina-Molina <fernando.molina@vocali.net>
@author Francisco García Sánchez <frgarcia@um.es>
"""

from funciones import load_doc, clean_doc, save_doc
import io
from tensorflow.keras.utils import get_file
from nltk.corpus import stopwords
import nltk


# @var doc load document
doc = load_doc ('./../raw/stopwords.txt')
 
 
# @var tokens
tokens = clean_doc (doc)


# @var filt_tokens List
filt_tokens = [word for word in tokens if word not in set (stopwords.words ('spanish'))]


# SAVE CLEAN TEXT
# Organize into sequences of tokens
length = 50 + 1
sequences = list ()

for i in range(length, len(filt_tokens)):
    seq = filt_tokens[i-length:i]
    line = ' '.join(seq)
    sequences.append(line)

print('Total sequences: %d' % len(sequences))



## Save into new file per line
out_filename = './../out/sequences/test2_STOPWORDS_sequences.txt'
save_doc(sequences, out_filename)
