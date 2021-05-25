
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K


with open('haha_2021_train.csv') as f:
    df = pd.read_csv(f)
print(df.head())
print(len(df))

# prepare data for different tasks
df2 = df.dropna(subset=['humor_rating'])

df3 = df.dropna(subset=['humor_mechanism'])

df4 = df.dropna(subset=['humor_target'])

# data preparation

# create the tokenizer
t = Tokenizer()
# fit the tokenizer on the documents
t.fit_on_texts(df['text'])

# integer encode documents
encoded_docs = np.array(pad_sequences(t.texts_to_sequences(df['text'])))
encoded_docs2 = np.array(pad_sequences(t.texts_to_sequences(df2['text'])))
encoded_docs3 = np.array(pad_sequences(t.texts_to_sequences(df3['text'])))
encoded_docs4 = np.array(pad_sequences(t.texts_to_sequences(df4['text'])))

max_len = max([len(text) for text in df['text']])

train_x = encoded_docs[:int(0.8 * df.shape[0])]
test_x = encoded_docs[int(0.8 * df.shape[0]):]
train_y = np.array(df[:int(0.8 * df.shape[0])]['is_humor'])
test_y = np.array(df[int(0.8 * df.shape[0]):]['is_humor'])

train_x2 = encoded_docs2[:int(0.8 * df2.shape[0])]
test_x2 = encoded_docs2[int(0.8 * df2.shape[0]):]
train_y2 = np.array(df2[:int(0.8 * df2.shape[0])]['humor_rating'])
test_y2 = np.array(df2[int(0.8 * df2.shape[0]):]['humor_rating'])








max_vocab = encoded_docs.max()


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# baseline model for task 1
# create the model
embedding_vector_length = 32
'''
model = Sequential()
model.add(Embedding(max_vocab + 1, embedding_vector_length, input_length=47))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])
print(model.summary())
model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=3, batch_size=32)
predictions_task1 = model.predict(encoded_docs)
'''
model2 = Sequential()
model2.add(Embedding(max_vocab + 1, embedding_vector_length, input_length=47))
model2.add(LSTM(100))
model2.add(Dropout(0.2))
model2.add(Dense(1))
model2.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.RootMeanSquaredError()])
print(model2.summary())
model2.fit(train_x2, train_y2, validation_data=(test_x2, test_y2), epochs=3, batch_size=32)
predictions_task2 = model2.predict(encoded_docs)
print(predictions_task2)

# TODO: try to improve baseline model: hyperparameters, vector word embeddings, features
# TODO: filter away NAN values; create separate datasets for all three tasks

