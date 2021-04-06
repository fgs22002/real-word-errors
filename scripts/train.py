"""
real word errors

training

@author Daniel Bravo daniel.bravo@um.es
@author Jesica López <jesica.lopez@um.es>
@author José Antonio García-Díaz joseantonio.garcia8@um.es
@author Fernando Molina-Molina <fernando.molina@vocali.net>
@author Francisco García Sánchez <frgarcia@um.es>
"""

from pickle import dump
from funciones import define_model, load_doc
from tensorflow.keras.models import Sequential
from data_generator import DataGenerator
from tensorflow.keras.preprocessing.text import Tokenizer
from numpy import array
from tensorflow.keras.utils import to_categorical
from pickle import load

import numpy as np
import os.path

 
# @var doc load sequences
doc = load_doc('./../input/spanishText_10000_15000_STOPWORDS.txt')
lines = doc.split('\n')

print (lines[:200])
lines = lines[0:round((len(lines))*0.01)]
print ('N lines: ')
print (len(lines))


# encode sequences: 
tokenizer = Tokenizer ()
tokenizer.fit_on_texts (lines)
sequences = tokenizer.texts_to_sequences (lines)


# vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print ('vocab_size:')
print (vocab_size)


# sequence input and labels: save .npy files
sequences = array (sequences)
X, y = sequences[:,:-1], sequences[:,-1]
seq_length = X.shape[1]


# Generate sequences
for x in range (X.shape[0]):
    ID = 'id-' + str(x+1)
    fi = './../npy_files/spanishText_10000_15000/' + ID + '.npy'
    
    if not os.path.exists (fi):
        np.save (fi, X[x,:])



# dictionaries
samp_ids = ['id-' + str(counter + 1) for counter, item in enumerate (lines)]


# @var train_ids Sample training
train_ids = samp_ids[0:round(len(samp_ids) * 0.8)]


# @var val_ids Sample validation
val_ids = samp_ids[round (len (samp_ids) * 0.8):len (samp_ids)]


# @var partition Dict
partition = {
    'train': train_ids, 
    'validation': val_ids
}


# @var labels Dict 
labels = {samp_ids[j]: y[j] for j in range(len(samp_ids))}


# Configure TRAINING parameters
# @var EPOCHS int
EPOCHS = 50


# @var BATCH_SIZE int
BATCH_SIZE = 32 


# @var dat_dim int
dat_dim = 50


# @var params Dict
params = {
    'dim': dat_dim,
    'batch_size': BATCH_SIZE,
    'n_classes': vocab_size,
    'shuffle': True
}
 

# @var training_generator DataGenerator
training_generator = DataGenerator (partition['train'], labels, **params)


# @var validation_generator DataGenerator
validation_generator = DataGenerator (partition['validation'], labels, **params)


# @var model 
model = define_model (vocab_size, seq_length)
 

# Fit model and validate
evaluation = model.fit_generator (generator=training_generator, epochs = EPOCHS, validation_data = validation_generator)
print(evaluation)


# Save model to file and save tokenizer
model.save ('./../models/model_test_Wiki_001.h5')
dump(tokenizer, open ('./../tokenizers/model_test_Wiki_001.pkl', 'wb'))