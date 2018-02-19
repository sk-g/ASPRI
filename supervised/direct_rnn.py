import tensorflow as tf
from tensorflow.contrib import rnn
import os,re,time,sys
import numpy as np
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import backend as K
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
from sklearn.model_selection import train_test_split as split

#loading pre built data set with labels
paths = pd.read_csv('11012018.csv',sep='\t',low_memory = False,index_col = False)
del paths['Unnamed: 0']

#splitting into training and test set
train,test = split(paths,test_size = 0.2)

max_length = 30 # verified max ASes in a path for 11012018.txt <-- update this to detect max length
                # automatically. Use classes and make it easier to read
vocab_size = 24612 #unique tokens for this file <- same as above, make changes
encoded_train = [one_hot(d,vocab_size) for d in train['Paths']] # one hot vector representation for AS paths
                                                                # in training set (fake and real)
encoded_test = [one_hot(d,vocab_size) for d in test['Paths']]   # same ohv encoding for test set
train_lengths = [len(t) for t in encoded_train] #array of lengths so we can pad zeros later
test_lengths= [len(t) for t in encoded_test] #array of lengths for test set to be padded later
# two classes for data iteration and batch generation
# simple itereator and padded iterator

class SimpleDataIterator():
    def __init__(self, df):
        self.df = df
        self.size = len(self.df)
        self.epochs = 0
        self.shuffle()

    def shuffle(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.cursor = 0

    def next_batch(self, n):
        if self.cursor+n-1 > self.size:
            self.epochs += 1
            self.shuffle()
        res = self.df.ix[self.cursor:self.cursor+n-1]
        self.cursor += n
        return res['data'], res['labels'], res['length']

class PaddedDataIterator(SimpleDataIterator):
    def next_batch(self, n):
        if self.cursor+n > self.size:
            self.epochs += 1
            self.shuffle()
        res = self.df.ix[self.cursor:self.cursor+n-1]
        self.cursor += n

        # Pad sequences with 0s so they are all the same length
        maxlen = max(res['length'])
        x = np.zeros([n, maxlen], dtype=np.int32)
        for i, x_i in enumerate(x):
            x_i[:res['length'].values[i]] = res['data'].values[i]

        return x, res['labels'], res['length']
    
# showing what the returns from the above classes look like
labels_train = train['Fake']
train_dic={}
train_dic["data"] = encoded_train
train_dic["labels"] = labels_train#labels_train[0].ravel().tolist()
train_dic["length"] = train_lengths
train_len = len(train)
test_len = len(test)

train_ = pd.DataFrame.from_dict(data=train_dic, orient='columns', dtype=None)



test_dic={}
test_dic["data"] = encoded_test
test_dic["length"] = test_lengths
test_dic["labels"] = test['Fake']
test_ = pd.DataFrame.from_dict(data=test_dic, orient='columns', dtype=None)

test_input = test.values

#without padding
data = SimpleDataIterator(train_)
d = data.next_batch(3)
print('Input sequences\n', d[0], end='\n\n')
print('Target values\n', d[1], end='\n\n')
print('Sequence lengths\n', d[2])

#with padding
data = PaddedDataIterator(train_)
train_data = PaddedDataIterator(test_)
d = data.next_batch(3)
print('Input sequences\n', d[0], end='\n\n')

# func to reset tf graph
def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def build_graph(
    vocab_size = vocab_size,
    state_size = 30,
    batch_size = 189,
    num_classes = 2):

    reset_graph()

    # Placeholders
    x = tf.placeholder(tf.int32, [batch_size, None]) # [batch_size, num_steps]
    seqlen = tf.placeholder(tf.int32, [batch_size])
    y = tf.placeholder(tf.int32, [batch_size])
    keep_prob = tf.placeholder_with_default(1.0, [])

    # Embedding layer
    embeddings = tf.get_variable('embedding_matrix', [vocab_size, state_size])
    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

    # RNN
    cell = tf.contrib.rnn.GRUCell(state_size)
    #cell = tf.nn.rnn_cell.BasicLSTMCell(state_size,forget_bias = 1)
    #cell = tf.contrib.rnn.LSTMCell(state_size,forget_bias = 1)
    init_state = tf.get_variable('init_state', [1, state_size],
                                 initializer=tf.glorot_uniform_initializer(seed = 10, dtype = tf.float32))
    init_state = tf.tile(init_state, [batch_size, 1])
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs,dtype=tf.float32)#, sequence_length=seqlen,initial_state=init_state)

    # Add dropout, as the model otherwise quickly overfits
    rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)

    """
    Obtain the last relevant output. The best approach in the future will be to use:

        last_rnn_output = tf.gather_nd(rnn_outputs, tf.pack([tf.range(batch_size), seqlen-1], axis=1))

    which is the Tensorflow equivalent of numpy's rnn_outputs[range(30), seqlen-1, :], but the
    gradient for this op has not been implemented as of this writing.

    The below solution works, but throws a UserWarning re: the gradient.
    """
    idx = tf.range(batch_size)*tf.shape(rnn_outputs)[1] + (seqlen - 1)
    last_rnn_output = tf.gather(tf.reshape(rnn_outputs, [-1, state_size]), idx)

    # Softmax layer
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.glorot_uniform_initializer(seed = 10, dtype = tf.float32))
    logits = tf.matmul(last_rnn_output, W) + b
    preds = tf.nn.softmax(logits)
    correct = tf.equal(tf.cast(tf.argmax(preds,1),tf.int32), y)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    #decayed learning rate
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.1
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,5, 0.63, staircase = False)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = y))
    #train_step = (tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step = global_step))
    train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)
    lr_print = tf.Print(learning_rate,[learning_rate])
    return {
        'x': x,
        'seqlen': seqlen,
        'y': y,
        'dropout': keep_prob,
        'loss': loss,
        'ts': train_step,
        'preds': preds,
        'accuracy': accuracy,
        'lr_print': lr_print
    }
def train_graph(graph, sess, batch_size = 1024, num_epochs = 100, iterator = PaddedDataIterator):

    start = time.time()
    sess.run(tf.global_variables_initializer())
    tr = iterator(train_)
    te = iterator(test_)

    step, accuracy = 0, 0
    tr_losses, te_losses = [], []
    current_epoch = 0
    while current_epoch < num_epochs:
        step += 1
        batch = tr.next_batch(batch_size)
        feed = {g['x']: batch[0], g['y']: batch[1], g['seqlen']: batch[2], g['dropout']: 0.8}
        accuracy_, _,l_r = sess.run([g['accuracy'], g['ts'],g['lr_print']], feed_dict=feed)
        accuracy += accuracy_
        #if step >1 and accuracy/step >= 0.97:
        #    print("Accuracy after epoch", current_epoch, " - tr:", accuracy / step)
        #    break;

        if tr.epochs > current_epoch:
            current_epoch += 1
            tr_losses.append(accuracy / step)
            step, accuracy = 0, 0
            #eval test set
            
            te_epoch = te.epochs
            while te.epochs == te_epoch:
                step += 1
                batch = te.next_batch(batch_size)
                feed = {g['x']: batch[0], g['y']: batch[1], g['seqlen']: batch[2]}
                accuracy_ = sess.run([g['accuracy'],g['loss']], feed_dict=feed)[0]
                accuracy += accuracy_
            te_losses.append(accuracy / step)
            step, accuracy = 0,0
            
            if current_epoch%5 == 0:
                print(" - learning rate:", l_r," Accuracy after epoch", current_epoch, " - tr:", tr_losses[-1], "- te:", te_losses[-1])
    end = time.time()
    seconds = end - start
    minutes = seconds//60
    seconds = seconds % 60
    hours = 0
    if minutes > 60:
        hours = minutes//60
        minutes = minutes%60
    print("time taken for training: {0} hours, {1} minutes and {2} seconds".format(hours,minutes,seconds))
    return tr_losses, te_losses

# main run of the graph. Use tr,te = train_graph(g)
# if predicting on the test set
g = build_graph(batch_size = 1024)
sess = tf.Session()
tr_losses,te_losses = train_graph(g, sess)
plt.figure(figsize = (10,6))
plt.plot(tr_losses,label = 'training accuracy')
plt.plot(te_losses,label = 'testing accuracy')
plt.legend()