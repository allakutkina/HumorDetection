import numpy
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# fix random seed for reproducibility
numpy.random.seed(7)
df = pd.read_csv('haha_2021_train.csv')

# data inspection
print(df.head())
print(df.info())

# data preparation
# create the tokenizer
t = Tokenizer()
# fit the tokenizer on the documents
t.fit_on_texts(df['text'])
# summarize what was learned
print(t.word_counts)
print(t.document_count)
print(t.word_index)
print(t.word_docs)
# integer encode documents
encoded_docs = np.array(pad_sequences(t.texts_to_sequences(df['text'])))
max_len = max([len(text) for text in df['text']])
print(encoded_docs)

train_x = encoded_docs[:int(0.8 * df.shape[0])]
test_x = encoded_docs[int(0.8 * df.shape[0]):]
train_y = np.array(df[:int(0.8 * df.shape[0])]['is_humor'])
test_y = np.array(df[int(0.8 * df.shape[0]):]['is_humor'])

max_vocab = encoded_docs.max()
print(max_vocab)

# baseline model for task 1
# create the model
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(max_vocab + 1, embedding_vector_length, input_length=47))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
print(model.summary())
model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=30, batch_size=64)

# TODO: try to improve baseline model: hyperparameters, vector word embeddings, features
# TODO: the model overfits immediately
# TODO: Implement f1