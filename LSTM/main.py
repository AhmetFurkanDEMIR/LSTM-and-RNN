import os
import re
import string
import requests
import numpy as np
import collections
import random
import tensorflow as tf

import LSTM

min_word_freq = 5
rnn_size = 128
batch_size = 128
learning_rate = 0.001
training_seq_len = 50
embedding_size = rnn_size
prime_texts = ['ankara', 'kitap']
epochs = 10000

file = open("yeni.txt","w+")

data_dir = 'data'
data_file = 'eksidata.txt'

punctuation = string.punctuation
punctuation = ''.join([x for x in punctuation if x not in ['-', "'"]])

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

print('Data yÃ¼kleniyor.')
if not os.path.isfile(os.path.join(data_dir, data_file)):
    print('Data bulunamadÄ±. Ä°ndiriliyor...')
    data_url = 'https://raw.githubusercontent.com/hakanceb/eksi/master/eksidata.txt'
    response = requests.get(data_url)
    eksi_file = response.content
    s_text = eksi_file.decode('utf-8')
    s_text = s_text.replace('\r\n', '')
    s_text = s_text.replace('\n', '')

    with open(os.path.join(data_dir, data_file), 'w') as out_conn:
        out_conn.write(s_text)
else:
    with open(os.path.join(data_dir, data_file), 'r') as file_conn:
        s_text = file_conn.read().replace('\n', '')

print('Text temizleniyor.')
s_text = re.sub(r'[{}]'.format(punctuation), ' ', s_text)
s_text = re.sub('\s+', ' ', s_text).strip().lower()

def build_vocab(text, min_word_freq):
    word_counts = collections.Counter(text.split(' '))
    word_counts = {key: val for key, val in word_counts.items() if val > min_word_freq}
    words = word_counts.keys()
    vocab_to_ix_dict = {key: (ix + 1) for ix, key in enumerate(words)}
    vocab_to_ix_dict['unknown'] = 0
    ix_to_vocab_dict = {val: key for key, val in vocab_to_ix_dict.items()}

    return (ix_to_vocab_dict, vocab_to_ix_dict)

ix2vocab, vocab2ix = build_vocab(s_text, min_word_freq)
vocab_size = len(ix2vocab) + 1
print('Vocabulary Length = {}'.format(vocab_size))

assert (len(ix2vocab) == len(vocab2ix))

s_text_words = s_text.split(' ')
s_text_ix = []
for ix, x in enumerate(s_text_words):
    try:
        s_text_ix.append(vocab2ix[x])
    except:
        s_text_ix.append(0)
s_text_ix = np.array(s_text_ix)

lstm_model = LSTM.LSTM_Model(embedding_size, rnn_size, batch_size, learning_rate, training_seq_len, vocab_size)
with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    test_lstm_model = LSTM.LSTM_Model(embedding_size, rnn_size, batch_size, learning_rate, training_seq_len, vocab_size,
                                      infer_sample=True)

n_batches = int(len(s_text_ix)) / (batch_size * training_seq_len) + 1
batches = np.array_split(s_text_ix, n_batches)
batches = [np.resize(x, [batch_size, training_seq_len]) for x in batches]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

iterations = 1
for epoch in range (epochs):
    random.shuffle(batches)
    targets = [np.roll(x, -1, axis=1) for x in batches]
    print('Starting Epoch #{} of {}.'.format(epoch + 1, epochs))
    state = sess.run(lstm_model.initial_state)

    for ix, batch in enumerate(batches):
        feed_dict_train = {lstm_model.x_data: batch, lstm_model.y_output: targets[ix]}
        c, h = lstm_model.initial_state
        feed_dict_train[c] = state.c
        feed_dict_train[h] = state.h

        [train_loss, state, _] = sess.run([lstm_model.loss, lstm_model.final_state, lstm_model.train_op],
                                          feed_dict=feed_dict_train)

        if iterations % 10 == 0:
            summary_nums = (iterations, epoch + 1, ix + 1, n_batches + 1, train_loss)
            print('Iteration: {}, Epoch: {}, Batch: {} out of {}, Loss: {:.2f}'.format(*summary_nums))

        if iterations % 50 == 0:
            for sample in prime_texts:
                print(test_lstm_model.sample(sess, ix2vocab, vocab2ix, num=10, prime_text=sample))
                print(test_lstm_model.sample(sess, ix2vocab, vocab2ix, num=10, prime_text=sample),file=file,flush=True)

        iterations += 1
