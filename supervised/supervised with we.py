
# coding: utf-8

# In[1]:


from __future__ import print_function
import os,re,time,sys,os,math,random,time,pickle,keras
#import pydot_ng as pydot
#import pydot
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pandas as pd
from keras import backend as K
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM,Flatten,Dense,Embedding,GRU,BatchNormalization
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
from sklearn.model_selection import train_test_split as split
start = time.time()

## loading the file with paths ##

if os.name != 'posix':
    f = open(r'M:\Course stuff\ASPRI\data\PCH\paths\11012018.txt','r')
else:
    f = open('../data/PCH/paths/11012018.txt','r')
lines = f.readlines()
lines = [i.strip() for i in lines]


# loading the word dictionaries, frequency counts
# and embedding vector form word2vec results


if os.name != 'posix':
    final_embeddings = pickle.load(open(r'M:\Course stuff\ASPRI\supervised\gcp_fe','rb'))
    dictionary = pickle.load(open(r'M:\Course stuff\ASPRI\supervised\gcp_dictionary','rb'))
    reverse_dictionary = pickle.load(open(r'M:\Course stuff\ASPRI\supervised\gcp_reverse_dictionary','rb'))
    count = pickle.load(open(r'M:\Course stuff\ASPRI\supervised\gcp_count','rb'))
else:
    final_embeddings = pickle.load(open('gcp_fe','rb'))
    dictionary = pickle.load(open('gcp_dictionary','rb'))
    reverse_dictionary = pickle.load(open('gcp_reverse_dictionary','rb'))
    count = pickle.load(open('gcp_count','rb'))
print(type(final_embeddings),type(dictionary),type(reverse_dictionary),type(count))


# saving the state of final embeddings


original_gcp_fe = final_embeddings.copy()


# creating a row of Zeroed vectors
# shape = dimensions x 1

nd_array = []
for i in range(30):
    nd_array.append(np.zeros(shape = (32,1)))

# creating a new array that contains above vectors
new_array = []
for i in range(len(lines)):
    new_array.append(nd_array)
new_array = np.asarray(new_array) # storing the list as ndarray


# modyifing the new_array to have word embeddings
# for every word in the sentence
# automatically padded to be of length 30
# because of 'new_array' construction from above

for i in range(len(lines)):
    splits = lines[i].split(' ')
    for j in range(len(splits)):
        new_array[i,j] = final_embeddings[dictionary[str(splits[j])]-1].reshape(32,1) # replace the zeroed
                                                                                      # vector with corresponding
                                                                                      # word vector

# sanity check: the shape should
# now be (32,1)
print("\nshape of word replaced with it's vector:{0}".format(new_array[0][0].shape))


# reading the labeled data from csv
if os.name != 'posix':
    paths = pd.read_csv(r'M:\Course stuff\ASPRI\supervised\11012018.csv',sep='\t',low_memory = False,index_col = False)
else:
    paths = pd.read_csv('11012018.csv',sep='\t',low_memory = False,index_col = False)
del paths['Unnamed: 0']
#paths.head()

train,test = split(paths,test_size = 0.3) # splitting into train,test


# uncomment to show plot of fake-ness
# sns.countplot(x = 'Fake',data = train)
# plt.show()
print("\nReal and fake in training set: {0}\nReal and fake in test set {1}".format(train.Fake.value_counts(),test.Fake.value_counts()))

# In[16]:


max_length = 30
vocab_size = 24612 #unique tokens for this file
#encoded_train = [one_hot(d,vocab_size) for d in train['Paths']]
#encoded_test = [one_hot(d,vocab_size) for d in test['Paths']]


# In[17]:


def encode_lines(arr):
    # function works on df
    # iterate over lines in df
    # iterate over splits of line
    # convert split word to embedding vector (32,1)
    # pad with (32,1) zeros
    
    
    new_array = []
    for i in range(len(arr)):
        new_array.append(nd_array)
    new_array = np.asarray(new_array)
    c = 0
    for i in arr['Paths']:
        splits = i.split(' ')
        for j in range(len(splits)):
            #print(new_array[i][j])
            new_array[c,j] = final_embeddings[dictionary[str(splits[j])]-1].reshape(32,1)
        c += 1
    assert(len(new_array) == len(arr))
    assert(len(new_array[0]) == 30)
    return new_array
encoded_train = encode_lines(train)


# In[18]:


len(encoded_train)


# In[19]:


encoded_test = encode_lines(test)


# In[20]:


train_lengths = [len(t) for t in encoded_train] #array of lengths so we can pad zeros later
test_lengths= [len(t) for t in encoded_test] #array of lengths for test set to be padded later


# In[21]:


y_test,y_train = test['Fake'],train['Fake']


# In[22]:


x_train = encode_lines(train)#['Paths'])
x_test = encode_lines(test)#['Paths'])


# In[23]:


max_length = 30
vocab_size = 24612 #unique tokens for this file
encoded_train = [one_hot(d,vocab_size) for d in train['Paths']]
encoded_test = [one_hot(d,vocab_size) for d in test['Paths']]
train_lengths = [len(t) for t in encoded_train] #array of lengths so we can pad zeros later
test_lengths= [len(t) for t in encoded_test] #array of lengths for test set to be padded later


# In[24]:


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


# In[25]:


x_train, x_test = train_["data"],test_["data"]
y_train,y_test = train['Fake'],test['Fake']


# In[26]:


max_features = vocab_size
maxlen = 30  # cut texts after this number of words (among top max_features most common words)
batch_size = 1024
epochs = 10 #training steps
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


# In[27]:


print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(30, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
print('Train...')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = 0,validation_data=(x_test, y_test))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score_lstm_sigmoid_FC)
print('Test accuracy:', acc_lstm_sigmoid_FC)


# In[28]:


print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(GRU(30, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose = 0,
          validation_data=(x_test, y_test))
score_gru_sigmoid_FC, acc_gru_sigmoid_FC = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score_gru_sigmoid_FC)
print('Test accuracy:', acc_gru_sigmoid_FC)


# In[29]:


print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(30, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='tanh'))
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose = 0,
          validation_data=(x_test, y_test))
score_lstm_tanh_FC, acc_lstm_tanh_FC = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score_lstm_tanh_FC)
print('Test accuracy:', acc_lstm_tanh_FC)


# In[30]:


print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(30, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='relu'))
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose = 0,
          validation_data=(x_test, y_test))
score_lstm_relu_FC, acc_lstm_relu_FC = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score_lstm_relu_FC)
print('Test accuracy:', acc_lstm_relu_FC)


# In[31]:


print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(30, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='elu'))
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose = 0,
          validation_data=(x_test, y_test))
score_lstm_elu_FC, acc_lstm_elu_FC = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score_lstm_elu_FC)
print('Test accuracy:', acc_lstm_elu_FC)


# In[32]:


print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(30, dropout=0.2, recurrent_dropout=0.2))
#model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='relu'))
model.add(Dense(1, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose = 0,
          validation_data=(x_test, y_test))
score_lstm_3FC_02dropout, acc_lstm_3FC_02dropout = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score_lstm_3FC_02dropout)
print('Test accuracy:', acc_lstm_3FC_02dropout)


# In[33]:


print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(30, dropout=0.8, recurrent_dropout=0.2))
model.add(Dense(1, activation='relu'))
model.add(Dense(1, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
print('Train...')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = 0,validation_data=(x_test, y_test))
score_lstm_3FC, acc_lstm_3FC = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score_lstm_3FC)
print('Test accuracy:', acc_lstm_3FC)


# In[34]:


print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(30, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
print('Train...')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = 0,validation_data=(x_test, y_test))
score_lstm_3FC_bn, acc_lstm_3FC_bn = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score_lstm_3FC_bn)
print('Test accuracy:', acc_lstm_3FC_bn)


# In[35]:


print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(GRU(30, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
print('Train...')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = 0,validation_data=(x_test, y_test))
score_gru_3FC_bn, acc_gru_3FC_bn = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score_gru_3FC_bn)
print('Test accuracy:', acc_gru_3FC_bn)


# In[36]:


#acc_gru_3FC_bn = 0.966
names = ['LSTM with sigmoid FC',
         'GRU with sigmoid FC',
         'LSTM with tanh FC',
         'LSTM with relu FC', 
         'LSTM with elu FC',
         'Denser LSTM with 0.2 dropout',
         'Denser LSTM with 0.8 dropout',
         'Denser LSTM with 0.2 dropout and batchnorm',
        'Denser GRU with 0.2 dropout and batchnorm']
accs = [acc_lstm_sigmoid_FC,
        acc_gru_sigmoid_FC,
        acc_lstm_tanh_FC,
        acc_lstm_relu_FC,
        acc_lstm_elu_FC,
        acc_lstm_3FC_02dropout,
        acc_lstm_3FC,
        acc_lstm_3FC_bn,
       acc_gru_3FC_bn]
for i in range(len(names)):
    print(names[i]," = ", accs[i],"\n" )
x = [i for i in range(len(names))]
plt.figure(figsize=(12,8))
pl.xticks(x, names)
pl.xticks(range(len(names)), names, rotation=60) #writes strings with 45 degree angle
pl.plot(x,accs,'*')
pl.show()
end = time.time()
seconds = end - start
minutes = seconds//60
seconds = seconds % 60
hours = 0
if minutes > 60:
    hours = minutes//60
    minutes = minutes%60
print("time taken for running the notebook:\n {0} hours, {1} minutes and {2} seconds".format(hours,minutes,seconds))


# In[38]:


### search for hyperparameters ###

### IN THIS BLOCK ###

## TO CHECK: RECURRENT DROPOUT RATE ##

# CHECK 0.2, 0.8 #

batch_size = 256

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(30, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score_lstm_sigmoid_FC)
print('Test accuracy:', acc_lstm_sigmoid_FC)
print('Build model...')

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(30, dropout=0.2, recurrent_dropout=0.8))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print('Train...')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = 0,validation_data=(x_test, y_test))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score_lstm_sigmoid_FC)
print('Test accuracy:', acc_lstm_sigmoid_FC)


# In[ ]:


### search for hyperparameters ###

### IN THIS BLOCK ###

## TO CHECK: number of LSTM units ##
# already have: 128 in github. 30 in this notebook #
# CHECK 64 #
from keras import backend as K

K.clear_session()
epochs = 5

batch_size = 32
print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
print('Train...')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = 1,validation_data=(x_test, y_test))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score_lstm_sigmoid_FC)
print('Test accuracy:', acc_lstm_sigmoid_FC)
print('Build model...')


# In[ ]:


batch_size = 32
K.clear_session()
print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(8, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
print('Train...')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = 0,validation_data=(x_test, y_test))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score_lstm_sigmoid_FC)
print('Test accuracy:', acc_lstm_sigmoid_FC)
print('Build model...')


# In[28]:


embedding_matrix = {}
for i in list(reverse_dictionary.keys()):
    embedding_matrix[i] = final_embeddings[i-1]


# In[29]:


final_embeddings.shape


# In[30]:


embedding_matrix = np.zeros((len(dictionary), 32))
for i in list(reverse_dictionary.keys()):
    embedding_vector = final_embeddings[i-1]#.reshape((32,1))
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i-1] = embedding_vector    


# In[31]:


len(embedding_matrix)


# In[33]:





# In[35]:


print('Building Trainable w2v embedding model...')
model = Sequential()
model.add(Embedding(len(embedding_matrix), 32,weights = [embedding_matrix],trainable = True))
model.add(LSTM(30, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
print('Train...')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = 0,validation_data=(x_test, y_test))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score_lstm_sigmoid_FC)
print('Test accuracy:', acc_lstm_sigmoid_FC)


# In[38]:


batch_size = 256
print('Building w2v embedding model with denser (16 units) FC layer ...')
model = Sequential()
model.add(Embedding(len(embedding_matrix), 32,weights = [embedding_matrix],trainable = False))
model.add(LSTM(30, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
print('Train...')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = 1,validation_data=(x_test, y_test))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score_lstm_sigmoid_FC)
print('Test accuracy:', acc_lstm_sigmoid_FC)
# increasing number of hidden units in dense layer helped
# ==>> extra layer with more hidden units

# try lstm : 128 -> Dense : 64 -> Dens: 1


# In[ ]:


print('Building w2v embedding model with denser (16 units) FC layer and 128 hidden units in LSTM ...')
model = Sequential()
model.add(Embedding(len(embedding_matrix), 32,weights = [embedding_matrix],trainable = False))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
print('Train...')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = 0,validation_data=(x_test, y_test))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score_lstm_sigmoid_FC)
print('Test accuracy:', acc_lstm_sigmoid_FC)


# In[ ]:


print('Building w2v embedding model with denser (32 units) FC layer and 128 hidden units in LSTM ...')
model = Sequential()
model.add(Embedding(len(embedding_matrix), 32,weights = [embedding_matrix],trainable = False))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
print('Train...')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = 0,validation_data=(x_test, y_test))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score_lstm_sigmoid_FC)
print('Test accuracy:', acc_lstm_sigmoid_FC)


# In[ ]:


print('Building trainable w2v embedding model with denser (32 units) FC layer and 128 hidden units in LSTM ...')
model = Sequential()
model.add(Embedding(len(embedding_matrix), 32,weights = [embedding_matrix],trainable = True))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
print('Train...')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = 0,validation_data=(x_test, y_test))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score_lstm_sigmoid_FC)
print('Test accuracy:', acc_lstm_sigmoid_FC)


# In[ ]:


print('Building trainable w2v embedding model with denser (64 units) FC layer and 128 hidden units in LSTM ...')
model = Sequential()
model.add(Embedding(len(embedding_matrix), 32,weights = [embedding_matrix],trainable = True))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
print('Train...')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = 0,validation_data=(x_test, y_test))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score_lstm_sigmoid_FC)
print('Test accuracy:', acc_lstm_sigmoid_FC)


# In[ ]:


print('Building trainable w2v embedding model with extra units for last model ...')
model = Sequential()
model.add(Embedding(len(embedding_matrix), 32,weights = [embedding_matrix],trainable = True))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
print('Train...')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = 0,validation_data=(x_test, y_test))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score_lstm_sigmoid_FC)
print('Test accuracy:', acc_lstm_sigmoid_FC)


# In[ ]:


print('Building trainable w2v embedding model with extra units for last model ...')
model = Sequential()
model.add(Embedding(len(embedding_matrix), 32,weights = [embedding_matrix],trainable = True))
model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(8, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

