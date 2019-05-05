#!/usr/bin/env python3


from pathlib import Path
import numpy as np
import spacy
from sklearn.neighbors import KNeighborsClassifier

def spacy_vectorize(word):
    return [a for a in nlp("the")][0].vector


nlp = spacy.load("en")
data_file = Path.home() / "mldata" / "data.txt"
WORD_VECTOR_SIZE = 384


vocab = ["~~END~~"]

x = []
y = []
def get_training_data():
    with data_file.open() as f:
        for line in f:
            x_vals = [np.zeros(WORD_VECTOR_SIZE) for _ in range(10)]
            for sentence in nlp(line).sents:
                for token in sentence:
                    x_vals.pop(0) 
                    x_vals.append(token.vector)
                    x.append(np.append(np.array([]), x_vals))
    
                for token in list(sentence)[1:]:
                    # adding y data: it is using one hot encoding
                    if str(token) not in vocab:
                        vocab.append(str(token))
                    y.append(vocab.index(str(token)))
                y.append(0) # adding the "~~END~~" character

def chat(seed):
    training_count = int(len(x) * .8)
    
    neigh = KNeighborsClassifier()
    print("traininig kneighbors classifier")
    neigh.fit(x, y)
    start_word_vector = spacy_vectorize(seed)
    test_sentence = [np.zeros(WORD_VECTOR_SIZE) for _ in range(9)] + [start_word_vector]
    predicted = neigh.predict([np.append(np.array([]), test_sentence)])
    print("First prediction:", vocab[int(predicted)])

    count = 0
    while predicted != 0:
        print(vocab[predicted])
        count += 1
        if count > 50:
            break
        predicted = neight.predict( [np.append(np.array([]), test_sentence)] )
        test_sentence.pop(0)
        test_sentence.append(spacy_vectorize(vocab[predicted]))

get_training_data()
chat("The")
