import os
import string
import emoji

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Concatenate, Dropout
from keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

# load train/test set
with open('haha_2021_train.csv') as f:
    df_train = pd.read_csv(f)
with open('haha_2021_dev_gold.csv') as f:
    df_test = pd.read_csv(f)

# appending the df_test to df_train so we could shuffle test and train instances later
df = df_train.append(df_test)

# load eval set
with open('haha_2021_test.csv') as f:
    eval_set = pd.read_csv(f)

print(df.columns)

# Formatting labels:
# is_humor already encoded: binary, 0,1
# humor_rating is encoded: set of integers or scalar

# creating a set of labels for humor mechanism (multiclass):
labels_hm = set(df['humor_mechanism'])

# creating a set of labels for humor target (multilabel):
labels_ht = set(df['humor_target'])


# helper functions

# some useful functions to eliminate the noise from texts and measure sequence length
def clean_text(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation))


def seq_len(text):
    return len(text.split(' '))


# label encoding

def label_encoder(label_set):
    labels = {}
    labels_reverse = {}
    j = 0
    for label in label_set:
        labels[label] = j
        labels_reverse[j] = label
        j += 1
    return labels, labels_reverse


# label decoding

def cat_to_label(predictions, dec: dict):
    predicted = []
    for label in predictions:
        predicted.append(dec[np.argmax(label)])
    return predicted


# multi label decoding

def cat_to_multi_label(predictions, dec: dict, threshold: int):
    predictions_ind = []
    for prediction in predictions:
        if np.max(prediction) > threshold:
            predictions_ind.append([i for i, v in enumerate(prediction) if v > threshold])
        else:
            predictions_ind.append([np.argmax(prediction)])
    s = ';'
    decoded = [s.join([dec[l] for l in pred_i]) for pred_i in predictions_ind]
    return decoded

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


# from kaggle https://www.kaggle.com/code/guglielmocamporese/macro-f1-score-keras/notebook

def macro_f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


# feature extraction method
def extract_features(texts):
    # 5 columns for 5 features. 0: length, 1: hashtags, 2: mentions, 3:emoji, 4: punctuation
    feature_matrix = np.zeros(shape=(len(texts), 8), dtype=int)
    for i, text in zip(range(len(texts)), texts):
        feature_matrix[i][0] = len(text)
        feature_matrix[i][1] = int('#' in text)
        feature_matrix[i][2] = int('@' in text)

        # FEATURE: Punctuation counter
        count = lambda l1, l2: sum([1 for x in l1 if x in l2])

        feature_matrix[i][3] = count(text, set(string.punctuation))
        # FEATURE: Emoji Count, decreases the results
        feature_matrix[i][4] = emoji.emoji_count(text)
        feature_matrix[i][5] = int('—' in text)
        feature_matrix[i][6] = int('-' in text)
        feature_matrix[i][7] = int('\n' in text)

    return feature_matrix


# df['text'] = df['text'].apply(lambda x: clean_text(x))
df['len'] = df['text'].apply(lambda x: seq_len(x))
df.head()

t = Tokenizer()
t.fit_on_texts(df['text'])
word_index = t.word_index
vocab_size = len(word_index) + 1

print(df.head())
# load pretrained vectors
embedding_index = {}

with open('SBW-vectors-300-min5.txt') as f:
    for line in f:
        word, coef = line.split(maxsplit=1)
        coef = np.fromstring(coef, 'f', sep=' ')
        embedding_index[word] = coef
    print('{} embeddings found'.format(len(embedding_index)))

embedding_dim = 300
hits = 0
misses = 0

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for w, i in word_index.items():
    embedding_vector = embedding_index.get(w)
    if embedding_vector is not None:
        hits += 1
        embedding_matrix[i] = embedding_vector
    else:
        misses += 1

print('{} words encoded with an embedding from SBW-vectors'.format(hits))
print('{} word embeddings not found'.format(misses))

# shuffling
df.sample(frac=1)
feats1 = extract_features(df['text'])
feats1_train, feats1_test = feats1[:int(0.8 * df.shape[0])], feats1[int(0.8 * df.shape[0]):]


# excluding tweets with nan-values for humor rating, mechanism and target
df2 = df.dropna(subset=['humor_rating'])
df3 = df.dropna(subset=['humor_mechanism'])
df4 = df.dropna(subset=['humor_target'])

# Encoding tweets for Task 1
encoded1 = np.array(pad_sequences(t.texts_to_sequences(df['text']), maxlen=47))

train_x1 = encoded1[:int(0.8 * df.shape[0])]
test_x1 = encoded1[int(0.8 * df.shape[0]):]
train_y1 = np.array(df[:int(0.8 * df.shape[0])]['is_humor'])
test_y1 = np.array(df[int(0.8 * df.shape[0]):]['is_humor'])

# Encoding tweets  for Task 2
encoded2 = np.array(pad_sequences(t.texts_to_sequences(df2['text']), maxlen=47))
train_x2 = encoded2[:int(0.8 * df2.shape[0])]
test_x2 = encoded2[int(0.8 * df2.shape[0]):]

train_y2 = np.array(df2[:int(0.8 * df2.shape[0])]['humor_rating'])
test_y2 = np.array(df2[int(0.8 * df2.shape[0]):]['humor_rating'])
feats2 = extract_features(df2['text'])
feats2_train, feats2_test = feats2[:int(0.8 * df2.shape[0])], feats2[int(0.8 * df2.shape[0]):]

# Encoding tweets for Task 3
encoded3 = np.array(pad_sequences(t.texts_to_sequences(df3['text']), maxlen=47))

# Encoding tweets for Task 4
encoded4 = np.array(pad_sequences(t.texts_to_sequences(df4['text']), maxlen=47))

encoded_eval = np.array(pad_sequences(t.texts_to_sequences(eval_set['text']), maxlen=47))
eval_feats = extract_features(eval_set['text'])

labels3, labels3_reverse = label_encoder(labels_hm)
labels4, labels4_reverse = label_encoder(labels_ht)

print(labels3, labels4)

train_x3 = encoded3[:int(0.8 * df3.shape[0])]
test_x3 = encoded3[int(0.8 * df3.shape[0]):]
train_y3_raw = np.array(df3[:int(0.8 * df3.shape[0])]['humor_mechanism'])
test_y3_raw = np.array(df3[int(0.8 * df3.shape[0]):]['humor_mechanism'])
train_y3 = to_categorical(np.array([labels3[x] for x in train_y3_raw]))
test_y3 = to_categorical(np.array([labels3[x] for x in test_y3_raw]))
feats3 = extract_features(df3['text'])
feats3_train, feats3_test = feats3[:int(0.8 * df3.shape[0])], feats3[int(0.8 * df3.shape[0]):]

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


train_y4 = y4[:int(0.8 * df4.shape[0])]
test_y4 = y4[int(0.8 * df4.shape[0]):]
feats4 = extract_features(df4['text'])
feats4_train, feats4_test = feats4[:int(0.8 * df4.shape[0])], feats4[int(0.8 * df4.shape[0]):]

# hyperparameters
train_embeds = True
btch = 16

# input model
aux_inp = Input(shape=(feats1.shape[1],))
aux_out = Dense(1)(aux_inp)
# binary classification model

inputs = Input(shape=(47,))
embed = Embedding(vocab_size, embedding_dim, embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                  input_length=47, trainable=train_embeds)(inputs)
lstm = LSTM(32, dropout=0.2)(embed)
lstm_out = Dense(5)(lstm)
concat = Concatenate()([lstm_out, aux_out])

# binary classification out layer
out = Dense(1, activation='sigmoid')(concat)
model = Model(inputs=[inputs, aux_inp], outputs=out)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])
print(model.summary())
model.fit([train_x1, feats1_train], train_y1, validation_data=([test_x1, feats1_test], test_y1), epochs=2,
          batch_size=btch)
predictions_task1 = model.predict([encoded_eval, eval_feats])
predictions_task1 = [int(pred > 0.5) for pred in predictions_task1]
eval_set['is_humor'] = predictions_task1

# regression model out layer
out2 = Dense(1, activation='relu')(concat)
model2 = Model(inputs=[inputs, aux_inp], outputs=out2)
model2.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.RootMeanSquaredError()])
print(model2.summary())

model2.fit([train_x2, feats2_train], train_y2, validation_data=([test_x2, feats2_test], test_y2), epochs=5,
           batch_size=32)
predictions_task2 = model2.predict([encoded_eval, eval_feats])
print(predictions_task2)

eval_set['humor_rating'] = predictions_task2

# multiclass model out layer
out3 = Dense(len(labels_hm), activation='softmax')(concat)
model3 = Model(inputs=[inputs, aux_inp], outputs=out3)

model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', macro_f1])
print(model3.summary())

model3.fit([train_x3, feats3_train], train_y3, validation_data=([test_x3, feats3_test], test_y3), epochs=10,
           batch_size=btch)
predictions_task3 = model3.predict([encoded_eval, eval_feats])
predictions_task3_decoded = cat_to_label(predictions_task3, labels3_reverse)
eval_set['humor_mechanism'] = predictions_task3_decoded
# multilabel model out layer

out4 = Dense(len(labels_ht), activation='softmax')(concat)
model4 = Model(inputs=[inputs, aux_inp], outputs=out4)

model4.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', macro_f1])
print(model4.summary())
model4.fit([train_x4, feats4_train], train_y4, validation_data=([test_x4, feats4_test], test_y4), epochs=10,
           batch_size=btch)
predictions_task4 = model4.predict([encoded_eval, eval_feats])

predictions_task4_decoded = cat_to_multi_label(predictions_task4, labels4_reverse, 0.5)
eval_set['humor_target'] = predictions_task4_decoded

print(eval_set.head())

eval_set = eval_set.drop('text', axis=1)

eval_set.to_csv('out.CSV', index=False)
