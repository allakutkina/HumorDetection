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
from tensorflow.keras.utils import to_categorical

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

# encode documents
# TODO: store text to sequence
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

# TODO: make a method
label3_set = set(df3['humor_mechanism'])
labels3 = {}
labels3_reverse = {}
i = 0
for label in label3_set:
    labels3[label] = i
    labels3_reverse[i] = label
    i += 1
print(labels3)

label4_set = set(df4['humor_target'])
labels4 = {}
labels4_reverse = {}
i = 0
for label in label4_set:
    labels4[label] = i
    labels4_reverse[i] = label
    i += 1
print(labels4)

train_x3 = encoded_docs3[:int(0.8 * df3.shape[0])]
test_x3 = encoded_docs3[int(0.8 * df3.shape[0]):]
train_y3_raw = np.array(df3[:int(0.8 * df3.shape[0])]['humor_mechanism'])
test_y3_raw = np.array(df3[int(0.8 * df3.shape[0]):]['humor_mechanism'])
train_y3 = to_categorical(np.array([labels3[x] for x in train_y3_raw]))
test_y3 = to_categorical(np.array([labels3[x] for x in test_y3_raw]))
print(train_y3)

train_x4 = encoded_docs4[:int(0.8 * df4.shape[0])]
test_x4 = encoded_docs4[int(0.8 * df4.shape[0]):]
y4_raw = np.array(df4['humor_target'])
y4 = to_categorical(np.array([labels4[x] for x in y4_raw]))
train_y4 = y4[:int(0.8 * df4.shape[0])]
test_y4 = y4[int(0.8 * df4.shape[0]):]
print(train_y4)


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

model3 = Sequential()
model3.add(Embedding(max_vocab + 1, embedding_vector_length, input_length=test_x3.shape[1]))
model3.add(LSTM(100))
model3.add(Dropout(0.2))
model3.add(Dense(12, activation='softmax'))
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model3.summary())
model3.fit(train_x3, train_y3, validation_data=(test_x3, test_y3), epochs=30, batch_size=32)
predictions_task3 = model3.predict(encoded_docs)


def cat_to_label(predictions, dict):
    predicted = []
    for label in predictions:
        predicted.append(dict[np.argmax(label)])
    return predicted

# TODO: put this in csv
print(cat_to_label(predictions_task3, labels3_reverse))
'''

# TODO: Model 3 SHAPE
# TODO: Model 3 organise right output (categorical prediction to labeld) done
# TODO: try to improve baseline model: hyperparameters, vector word embeddings, features
# TODO: filter away NAN values; create separate datasets for all three tasks
# TODO: implement Macro F1
# TODO: new labels

model4 = Sequential()
model4.add(Embedding(max_vocab + 1, embedding_vector_length, input_length=test_x4.shape[1]))
model4.add(LSTM(100))
model4.add(Dropout(0.2))
model4.add(Dense(53, activation='softmax'))
model4.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model4.summary())
model4.fit(train_x4, train_y4, validation_data=(test_x4, test_y4), epochs=30, batch_size=32)
predictions_task4 = model4.predict(encoded_docs)

# TODO: for model 4 fitting words can help (mentioned woman -> target etc).
# TODO: Model 4 overfits as well!!
