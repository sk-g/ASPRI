from __future__ import print_function
import re,os,time,sys,pickle,nltk,statistics
import pandas as pd
import numpy as np
from collections import Counter
import os,re,time,sys,os,math,random,time,pickle,keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pandas as pd
from keras import backend as K
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM,Flatten,Dense,Embedding,GRU,BatchNormalization,TimeDistributed
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split as split
if os.name != 'posix':
	paths = pd.read_csv(r'M:\Course stuff\ASPRI\supervised\11012018.csv',sep='\t',low_memory = False,index_col = False)
else:
	paths = pd.read_csv('11012018.csv',sep='\t',low_memory = False,index_col = False)
del paths['Unnamed: 0']

fake_paths = [paths.loc[paths['Fake'] == 1]]
valid_paths = [paths.loc[paths['Fake'] == 0]]

print(len(fake_paths[0]))
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
#print(type(final_embeddings),type(dictionary),type(reverse_dictionary),type(count))

#train,test = split(paths,test_size = 0.3) # splitting into train,test


max_length = timesteps =  32
vocab_size = 24612 #unique tokens for this file


def encode_lines(arr):
	# function works on df
	# iterate over lines in df
	# iterate over splits of line
	# convert split word to embedding vector (32,1)
	# pad with (32,1) zeros
	
	nd_array = []
	for i in range(max_length):
		nd_array.append(np.zeros(shape = (32,1)))
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
	assert(len(new_array[0]) == max_length)
	return new_array
encoded_train = encode_lines(valid_paths[0])
encoded_train = encoded_train.reshape(encoded_train.shape[0],encoded_train.shape[1],encoded_train.shape[2])
#print(encoded_train.shape)
encoded_test = encode_lines(fake_paths[0])
encoded_test = encoded_test.reshape(encoded_test.shape[0],encoded_test.shape[1],encoded_test.shape[2])

vocab_size = 24612 #unique tokens for this file



x_train, x_test = encoded_train,encoded_test
y_train,y_test = valid_paths[0]['Fake'],fake_paths[0]['Fake']

max_features = vocab_size
batch_size = 1024
epochs = 10 #training steps



print("\n..... Extracting and building the word2vec results .....")
embedding_matrix = {}
for i in list(reverse_dictionary.keys()):
	embedding_matrix[i] = final_embeddings[i-1]


# In[29]:


print("Word2Vec embedding space has the shape: {0}\n".format(final_embeddings.shape))
print("\nTraining data shape:{} \nTesting data shape: {}".format(x_train[:-58].shape,x_test[:-33].shape))#encoded_train.shape,encoded_test.shape))
verbose = 1
batch_size = 128
epochs = 5

embedding_matrix = np.zeros((len(dictionary), 32))
for i in list(reverse_dictionary.keys()):
	embedding_vector = final_embeddings[i-1]#.reshape((32,1))
	if embedding_vector is not None:
		# words not found in embedding index will be all-zeros.
		embedding_matrix[i-1] = embedding_vector
timesteps = 30
data_dim = 30

model = Sequential()
#model.add(Embedding(len(embedding_matrix), 32,weights = [embedding_matrix],trainable = False))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2,batch_size = batch_size,\
	input_shape = (encoded_train.shape[1],encoded_train.shape[2])))
#model.add(TimeDistributed(Dense(1,activation = 'sigmoid')))
model.add(Dense(1, activation='sigmoid',batch_size = batch_size))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
print('\nTrain...\n')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = verbose,validation_data=(x_test[:-33], y_test[:-33]))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,batch_size=batch_size)
print('\nTest score:', score_lstm_sigmoid_FC)
print('\nTest accuracy:', acc_lstm_sigmoid_FC)