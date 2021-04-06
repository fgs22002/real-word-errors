"""
real word errors

gen-test

@author Daniel Bravo daniel.bravo@um.es
@author Jesica López <jesica.lopez@um.es>
@author José Antonio García-Díaz joseantonio.garcia8@um.es
@author Fernando Molina-Molina <fernando.molina@vocali.net>
@author Francisco García Sánchez <frgarcia@um.es>
"""

from funciones import load_doc, clean_doc, save_doc


# Load the document
in_filename = '../raw/texts.txt'
doc = load_doc (in_filename)


# CLEAN TEXT
tokens = clean_doc (doc)


# SAVE CLEAN TEXT
# Organize into sequences of tokens
length = 50 + 1
sequences = list ()
for i in range(length, len(tokens)):
    seq = tokens[i-length:i]
    line = ' '.join(seq)
    sequences.append(line)


print ('Total sequences: %d' % len(sequences))
print (sequences[0:200])

 
out_filename = '../output/texts.txt'
save_doc(sequences, out_filename)
