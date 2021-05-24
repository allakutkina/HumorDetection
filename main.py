# TODO Keras Tokenizer
# TODO put everything i will need for test set too in methods
# TODO rewrite LSTM model for SINGLE number output. Do I need LSTM here??? Which model???

import pandas as pd



# load data
df = pd.read_csv('haha_2021_train.csv')

# data inspection
print(df.head())
print(df.info())

# data split
train = df[:int(0.8 * df.shape[0])]
test = df[int(0.8 * df.shape[0]):]

print(len(train), len(test))

train_texts = test['text']

# tokenize data
token_data = [text.split() for text in train_texts]

print(len(token_data))

# encode data
# create a vocabulary
vocab = set()
for seq in token_data:
    for token in seq:
        vocab.add(token)

# 0 is reserved for out of vocabulary
word_to_ix = {word: i + 1 for i, word in enumerate(vocab)}


def simple_enc(seq):
    enc_sequence = []
    for tok in seq:
        if tok in word_to_ix:
            enc_token = word_to_ix[tok]
            enc_sequence.append(enc_token)
        else:
            enc_sequence.append(0)
    return enc_sequence


# word encoding
simple_encoding = [simple_enc(text) for text in token_data]

print(len(simple_encoding))

