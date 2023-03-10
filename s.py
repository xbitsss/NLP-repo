import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import numpy as np



###### Pre processing ########
def review_encoder(text):

    arr = [word_index[word] for  word in text]
    return arr

def encode_sentiments(sentiment):

    if sentiment == 'positive':
        return 1
    else:
        return 0

# Loading the model
imdb_reviews = pd.read_csv('imdb_reviews.csv')
test_reviews = pd.read_csv('test_reviews.csv')


# File after each word is converted to a distinct integer
word_index = pd.read_csv('word_indexes.csv')

# Dictionary mapping each word to its subsequent integer
word_index = dict(zip(word_index.Words, word_index.Indexes))


word_index["<PAD>"] = 0
word_index["<START"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3


train_data, train_labels = imdb_reviews['Reviews'], imdb_reviews['Sentiment']
test_data, test_labels = test_reviews['Reviews'], test_reviews['Sentiment']


# Tokenise each string
train_data = train_data.apply(lambda review: review.split())
test_data = test_data.apply(lambda review: review.split())

# Encode

train_data = train_data.apply(review_encoder)
test_data = test_data.apply(review_encoder)


# Encode sentiments

train_labels = train_labels.apply(encode_sentiments)
test_labels = test_labels.apply(encode_sentiments)

# Making reviews of equal length
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding='post',maxlen = 500)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding='post',maxlen = 500)


##### Word Embeddings #######

# Converts each word to a vector of length 16 + input length = conversion
model = keras.Sequential([keras.layers.Embedding(10000,16,input_length = 500), 
                         keras.layers.GlobalAveragePooling1D(), 
                         keras.layers.Dense(16,activation = 'relu'), 
                         keras.layers.Dense(1, activation='sigmoid')])

# Performance metrics =  Accuracy
# Loss function =  Binary crossentropy

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

history = model.fit(train_data,train_labels,epochs=30, batch_size=512, validation_data=(test_data,test_labels))

loss, accuracy = model.evaluate(test_data,test_labels)
