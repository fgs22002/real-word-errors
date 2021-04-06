# Automatic correction of real-word errors in Spanish clinical texts
Real-word errors are characterized by being actual terms in the dictionary. By providing context, real-word errors are detected. Traditional methods to detect and correct such errors are mostly based on counting the frequency of short word sequences in a corpus. Then, the probability of a word being a real-word error is computed. On the other hand, state-of-the-art approaches make use of Deep Learning models to learn context by extracting semantic features from text. In this work, a Deep Learning model were implemented for correcting real-word errors in clinical text. Specifi-cally, a Seq2seq Neural Machine Translation Model mapped erroneous sentences to correct them. For that, different types of errors were generated in correct sentences by using rules. Different Seq2seq models were trained and evaluated on two corpora: the Wikicorpus and a collection of three clinical datasets. The medicine corpus was much smaller than the Wikicorpus due to privacy issues when dealing with patient information. Moreover, Glove and Word2vec pretrained word embeddings were used to study their performance. Despite the medicine corpus was much smaller than the Wikicorpus, Seq2seq models trained on the medicine corpus performed better than those models trained on the Wikicorpus. Nevertheless, a larger amount of clinical text is required to improve the results.


## Folder structure
This folder contains the sentences used for training and testing the seq2seq model. Sentences are classified according to the corpus (Wikicorpus and medicine corpus) and the two dataset compilation strategies in the case of the training sentences. The same test sentences were used for each strategy. Finally, erroneous sentences (source) are provided in addition to correct sentences (target). 


## Install
Project was created using a virtual environment. The libraries within the environment are in ```requirements.txt```. To install, please create a new environment and install the dependencies

```
$ pip install -r requirements.txt
```

Create then the folders ```models```, ```tokenizers```, and ```raw```.

Due to size limitations, the datasets ae not stored here. You can download it from http://pln.inf.um.es/corpora/realworderrors/datasets.rar


## Note
Repository code is now refactoring because some of the urls and paths were absolute paths. Future installation instructions will be provided soon.