import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

# load train/test set
with open('haha_2021_train.csv') as f:
    df = pd.read_csv(f)

# load dev set
with open('haha_2021_test.csv') as f:
    dev = pd.read_csv(f)

print(df.columns)

# Formatting labels:
# is_humor already encoded: binary, 0,1
# humor_rating is encoded: set of integers or scalar

# creating a set of labels for humor mechanism (multiclass):
labels_hm = set(df['humor_mechanism'])

# creating a set of labels for humor target (multilabel):
labels_ht = []
for label in list(set(df['humor_target'])):
    labels_ht += (str(label).split('; '))
labels_ht = set(labels_ht)

# data preprocessing

# initialise the tokenizer
t = Tokenizer()
t.fit_on_texts(df['text'])
print(df.head())

# shuffling
df.sample(frac=1)

# excluding non-humorous tweets
df2 = df.dropna(subset=['humor_rating'])

# excluding tweets with nan-values for humor mechanism or humor target
df3 = df2.dropna(subset=['humor_mechanism'])
df4 = df2.dropna(subset=['humor_target'])

# Encoding tweets for Task 1
encoded1 = np.array(pad_sequences(t.texts_to_sequences(df['text']), maxlen=47))

train_x1 = encoded1[:int(0.8 * df.shape[0])]
test_x1 = encoded1[int(0.8 * df.shape[0]):]
train_y1 = np.array(df[:int(0.8 * df.shape[0])]['is_humor'])
test_y1 = np.array(df[int(0.8 * df.shape[0]):]['is_humor'])
max_vocab1 = encoded1.max()

# Encoding tweets  for Task 2
encoded2 = np.array(pad_sequences(t.texts_to_sequences(df2['text']), maxlen=47))
train_x2 = encoded2[:int(0.8 * df2.shape[0])]
test_x2 = encoded2[int(0.8 * df2.shape[0]):]
train_y2 = np.array(df2[:int(0.8 * df2.shape[0])]['humor_rating'])
test_y2 = np.array(df2[int(0.8 * df2.shape[0]):]['humor_rating'])
max_vocab2 = encoded2.max()

# Encoding tweets for Task 3
encoded3 = np.array(pad_sequences(t.texts_to_sequences(df3['text']), maxlen=47))

# Encoding tweets for Task 4
encoded4 = np.array(pad_sequences(t.texts_to_sequences(df4['text']), maxlen=47))
max_vocab4 = encoded4.max()

encoded_dev = np.array(pad_sequences(t.texts_to_sequences(dev['text']), maxlen=47))


# print(dev.head())


# method returning dictionaries for encoding and decoding labels (integer encoding)
# TODO: after label splitting label encoder doesn't work as expected

def label_encoder(label_set):
    labels = {}
    labels_reverse = {}
    i = 0
    for label in label_set:
        labels[label] = i
        labels_reverse[i] = label
        i += 1
    return labels, labels_reverse


labels3, labels3_reverse = label_encoder(labels_hm)
labels4, labels4_reverse = label_encoder(labels_ht)

print(labels3, labels4)

train_x3 = encoded3[:int(0.8 * df3.shape[0])]
test_x3 = encoded3[int(0.8 * df3.shape[0]):]
train_y3_raw = np.array(df3[:int(0.8 * df3.shape[0])]['humor_mechanism'])
test_y3_raw = np.array(df3[int(0.8 * df3.shape[0]):]['humor_mechanism'])
train_y3 = to_categorical(np.array([labels3[x] for x in train_y3_raw]))
test_y3 = to_categorical(np.array([labels3[x] for x in test_y3_raw]))

train_x4 = encoded4[:int(0.8 * df4.shape[0])]
test_x4 = encoded4[int(0.8 * df4.shape[0]):]

# encoding labels for Task 4. Here it gets a bit more tricky since it is multilabel task
# like one-hot encoding, but it is not one-hot if there are two or more labels.
# When predicting will need to think rather of prediction confidence threshold rather then argmax
y4 = np.zeros((df4.shape[0], len(labels_ht)))

for y, y_encoded in zip(df4['humor_target'], y4):
    y_list = y.split('; ')
    for l in y_list:
        y_encoded[labels4[l]] = 1

print(y4)

train_y4 = y4[:int(0.8 * df4.shape[0])]
test_y4 = y4[int(0.8 * df4.shape[0]):]


# evaluation methods

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
embedding_vector_length = 64
# TODO: early stopping best epoch?

model = Sequential()
model.add(Embedding(max_vocab1 + 1, embedding_vector_length, input_length=test_x1.shape[1]))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])
print(model.summary())
model.fit(train_x1, train_y1, validation_data=(test_x1, test_y1), epochs=3, batch_size=32)
predictions_task1 = model.predict(encoded_dev)
predictions_task1 = [int(pred > 0.5) for pred in predictions_task1]
print(predictions_task1)
dev['is_humor'] = predictions_task1

model2 = Sequential()
model2.add(Embedding(max_vocab2 + 1, embedding_vector_length, input_length=test_x2.shape[1]))
model2.add(LSTM(64))
model2.add(Dropout(0.2))
model2.add(Dense(1))
model2.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.RootMeanSquaredError()])
print(model2.summary())

model2.fit(train_x2, train_y2, validation_data=(test_x2, test_y2), epochs=3, batch_size=32)
predictions_task2 = model2.predict(encoded_dev)

dev['humor_rating'] = predictions_task2
print(dev.head())

model3 = Sequential()
model3.add(Embedding(max_vocab2 + 1, embedding_vector_length, input_length=test_x3.shape[1]))
model3.add(LSTM(64))
model3.add(Dropout(0.2))
model3.add(Dense(13, activation='softmax'))
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model3.summary())

model3.fit(train_x3, train_y3, validation_data=(test_x3, test_y3), epochs=3, batch_size=32)
predictions_task3 = model3.predict(encoded_dev)


def cat_to_label(predictions, dec: dict):
    predicted = []
    for label in predictions:
        predicted.append(dec[np.argmax(label)])
    return predicted


predictions_task3_decoded = cat_to_label(predictions_task3, labels3_reverse)

dev['humor_mechanism'] = predictions_task3_decoded
print(dev.head())

model4 = Sequential()
model4.add(Embedding(max_vocab4 + 1, embedding_vector_length, input_length=test_x4.shape[1]))
model4.add(LSTM(128))
model4.add(Dropout(0.2))
model4.add(Dense(len(labels_ht), activation='softmax'))
model4.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model4.summary())

model4.fit(train_x4, train_y4, validation_data=(test_x4, test_y4), epochs=6, batch_size=8)
predictions_task4 = model4.predict(encoded_dev)

threshold = 0.5

predictions_ind = [[i for i, v in enumerate(prediction) if v > threshold] for prediction in predictions_task4]
s = '; '
predictions_task4_decoded = [s.join([labels4_reverse[l] for l in pred_i]) for pred_i in predictions_ind]

dev['humor_target'] = predictions_task4_decoded

print(dev['humor_target'])

# TODO: Model 4 overfits to certain classes!
# TODO: Multilabeling is 0 in the output: model likes to predict single labels
dev = dev.drop('text', axis=1)

dev.to_csv('out.CSV', index=False)
