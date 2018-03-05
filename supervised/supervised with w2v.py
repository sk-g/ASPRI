from __future__ import print_function
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
from keras.layers import LSTM,Flatten,Dense,Embedding,GRU,BatchNormalization
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
#import matplotlib.pyplot as plt
#import seaborn as sns
#import pylab as pl
from sklearn.model_selection import train_test_split as split
start = time.time()
#sys.stdout = open('word2vec_print.txt','w')
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
	final_embeddings = pickle.load(open(r'M:\Course stuff\ASPRI\supervised\gcp_fe','rb'))#original 32 dims w2v
	#final_embeddings = pickle.load(open(r'M:\Course stuff\ASPRI\supervised\128dimsw2v','rb'))# 128 dims w2v
	dictionary = pickle.load(open(r'M:\Course stuff\ASPRI\supervised\gcp_dictionary','rb'))
	reverse_dictionary = pickle.load(open(r'M:\Course stuff\ASPRI\supervised\gcp_reverse_dictionary','rb'))
	count = pickle.load(open(r'M:\Course stuff\ASPRI\supervised\gcp_count','rb'))
else:
	final_embeddings = pickle.load(open('gcp_fe','rb'))
	dictionary = pickle.load(open('gcp_dictionary','rb'))
	reverse_dictionary = pickle.load(open('gcp_reverse_dictionary','rb'))
	count = pickle.load(open('gcp_count','rb'))
#print(type(final_embeddings),type(dictionary),type(reverse_dictionary),type(count))
max_length = 30
vocab_size = 24612 #unique tokens for this file
embedding_dim = final_embeddings.shape[1]
# saving the state of final embeddings


original_gcp_fe = final_embeddings.copy()




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
#print("\nReal and fake in training set: {0}\nReal and fake in test set {1}".format(train.Fake.value_counts(),test.Fake.value_counts()))

# In[16]:




def encode_lines(arr,embedding_dim = 32):
	# function works on df
	# iterate over lines in df
	# iterate over splits of line
	# convert split word to embedding vector (32,1)
	# pad with (32,1) zeros
	# defualt embedding dim = 32
	# can send in dim or can extract from the arr itself
	# creating a row of Zeroed vectors
	# shape = dimensions x 1

	nd_array = []
	for i in range(max_length):
		nd_array.append(np.zeros(shape = (embedding_dim,1)))

	# creating a new array that contains above vectors	
	_embedding_dim = embedding_dim
	print("encoding function, {} is the _embedding_dim".format(_embedding_dim))
	new_array = []
	for i in range(len(arr)):
		new_array.append(nd_array)
	new_array = np.asarray(new_array)
	c = 0
	for i in arr['Paths']:
		splits = i.split(' ')
		for j in range(len(splits)):
			#print(new_array[i][j])
			new_array[c,j] = final_embeddings[dictionary[str(splits[j])]-1].reshape(_embedding_dim,1)
		c += 1
	assert(len(new_array) == len(arr))
	assert(len(new_array[0]) == max_length)
	return new_array
encoded_train = encode_lines(train)
encoded_train = encoded_train.reshape(encoded_train.shape[0],encoded_train.shape[1],encoded_train.shape[2])
#print(encoded_train.shape)
encoded_test = encode_lines(test)
encoded_test = encoded_test.reshape(encoded_test.shape[0],encoded_test.shape[1],encoded_test.shape[2])

vocab_size = 24612 #unique tokens for this file

#test_input = test.values


x_train, x_test = encoded_train,encoded_test
y_train,y_test = train['Fake'],test['Fake']

max_features = vocab_size
batch_size = 1024
epochs = 10 #training steps



print("\n..... Extracting and building the word2vec results .....")
embedding_matrix = {}
for i in list(reverse_dictionary.keys()):
	embedding_matrix[i] = final_embeddings[i-1]


# In[29]:


print("Word2Vec embedding space has the shape: {0}\n".format(final_embeddings.shape))

verbose = 1
batch_size = 128
epochs = 5

embedding_matrix = np.zeros((len(dictionary), embedding_dim))
for i in list(reverse_dictionary.keys()):
	embedding_vector = final_embeddings[i-1]#.reshape((embedding_dim,1))
	if embedding_vector is not None:
		# words not found in embedding index will be all-zeros.
		embedding_matrix[i-1] = embedding_vector
timesteps = 30
data_dim = 30

"""
print('Build first model...')
model = Sequential()
#model.add(Embedding(len(embedding_matrix), embedding_dim,weights = [embedding_matrix],trainable = False))
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2,input_shape = (encoded_train.shape[1],encoded_train.shape[2])))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
print('\nTrain...\n')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = verbose,validation_data=(x_test, y_test))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,batch_size=batch_size)
print('\nTest score:', score_lstm_sigmoid_FC)
print('\nTest accuracy:', acc_lstm_sigmoid_FC)
# 84.84% training accuracy for 128 dims w2v
# testing accuracy 84.805%

print('\nBuilding above model but without dropout...\n')

model = Sequential()
model.add(LSTM(32, input_shape = (encoded_train.shape[1],encoded_train.shape[2])))
#model.add(Dense(16, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
print('\nTrain...\n')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = verbose,validation_data=(x_test, y_test))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,batch_size=batch_size)
print('\nTest score:', score_lstm_sigmoid_FC)
print('\nTest accuracy:', acc_lstm_sigmoid_FC)

print('\nBuilding above model with more hidden units in LSTM ...\n')
model = Sequential()
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape = (encoded_train.shape[1],encoded_train.shape[2])))
#model.add(Dense(16, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
print('\nTrain...\n')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = verbose,validation_data=(x_test, y_test))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,batch_size=batch_size)
print('\nTest score:', score_lstm_sigmoid_FC)
print('\nTest accuracy:', acc_lstm_sigmoid_FC)

print('\nBuilding above model with an additional dense layer ...\n')
model = Sequential()
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape = (encoded_train.shape[1],encoded_train.shape[2])))
model.add(Dense(16, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
print('\nTrain...\n')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = verbose,validation_data=(x_test, y_test))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,batch_size=batch_size)
print('\nTest score:', score_lstm_sigmoid_FC)
print('\nTest accuracy:', acc_lstm_sigmoid_FC)


print("Trying a stacked LSTM model ")

model = Sequential()
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape = (encoded_train.shape[1],encoded_train.shape[2]), return_sequences = True))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2,return_sequences = True))
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
model.add(BatchNormalization())
model.add(Dense(16, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
print('\nTrain...\n')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = verbose,validation_data=(x_test, y_test))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,batch_size=batch_size)
print('\nTest score:', score_lstm_sigmoid_FC)
print('\nTest accuracy:', acc_lstm_sigmoid_FC)

print("Trying a stacked GRU version of above model ")
model = Sequential()
model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2, input_shape = (encoded_train.shape[1],encoded_train.shape[2]), return_sequences = True))
model.add(GRU(64, dropout=0.2, recurrent_dropout=0.2,return_sequences = True))
model.add(GRU(32, dropout=0.2, recurrent_dropout=0.2))
model.add(BatchNormalization())
model.add(Dense(16, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
print('\nTrain...\n')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = verbose,validation_data=(x_test, y_test))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,batch_size=batch_size)
print('\nTest score:', score_lstm_sigmoid_FC)
print('\nTest accuracy:', acc_lstm_sigmoid_FC)
"""

print("\n\nRunning All the models but now with the 128 dimension word2vec ...\n\n")

if os.name != 'posix':
	#final_embeddings = pickle.load(open(r'M:\Course stuff\ASPRI\supervised\gcp_fe','rb'))#original 32 dims w2v
	final_embeddings = pickle.load(open(r'M:\Course stuff\ASPRI\supervised\128dimsw2v','rb'))# 128 dims w2v
embedding_dim = final_embeddings.shape[1]
encoded_train = encode_lines(train,embedding_dim = embedding_dim)
encoded_train = encoded_train.reshape(encoded_train.shape[0],encoded_train.shape[1],encoded_train.shape[2])
#print(encoded_train.shape)
encoded_test = encode_lines(test,embedding_dim = embedding_dim)
encoded_test = encoded_test.reshape(encoded_test.shape[0],encoded_test.shape[1],encoded_test.shape[2])
embedding_dim = final_embeddings.shape[1]
x_train, x_test = encoded_train,encoded_test
y_train,y_test = train['Fake'],test['Fake']

max_features = vocab_size
batch_size = 1024
epochs = 10 #training steps



print("\n..... Extracting and building the word2vec results .....")
embedding_matrix = {}
for i in list(reverse_dictionary.keys()):
	embedding_matrix[i] = final_embeddings[i-1]


# In[29]:


print("Word2Vec embedding space has the shape: {0}\n".format(final_embeddings.shape))

verbose = 1
batch_size = 128
epochs = 5

embedding_matrix = np.zeros((len(dictionary), embedding_dim))
for i in list(reverse_dictionary.keys()):
	embedding_vector = final_embeddings[i-1]#.reshape((embedding_dim,1))
	if embedding_vector is not None:
		# words not found in embedding index will be all-zeros.
		embedding_matrix[i-1] = embedding_vector
timesteps = 30
data_dim = 30
###################################################
print('Build first model...')
model = Sequential()
#model.add(Embedding(len(embedding_matrix), embedding_dim,weights = [embedding_matrix],trainable = False))
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2,input_shape = (encoded_train.shape[1],encoded_train.shape[2])))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
print('\nTrain...\n')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = verbose,validation_data=(x_test, y_test))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,batch_size=batch_size)
print('\nTest score:', score_lstm_sigmoid_FC)
print('\nTest accuracy:', acc_lstm_sigmoid_FC)
# 84.84% training accuracy for 128 dims w2v
# testing accuracy 84.805%

print('\nBuilding above model but without dropout...\n')

model = Sequential()
model.add(LSTM(32, input_shape = (encoded_train.shape[1],encoded_train.shape[2])))
#model.add(Dense(16, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
print('\nTrain...\n')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = verbose,validation_data=(x_test, y_test))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,batch_size=batch_size)
print('\nTest score:', score_lstm_sigmoid_FC)
print('\nTest accuracy:', acc_lstm_sigmoid_FC)

print('\nBuilding above model with more hidden units in LSTM ...\n')
model = Sequential()
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape = (encoded_train.shape[1],encoded_train.shape[2])))
#model.add(Dense(16, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
print('\nTrain...\n')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = verbose,validation_data=(x_test, y_test))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,batch_size=batch_size)
print('\nTest score:', score_lstm_sigmoid_FC)
print('\nTest accuracy:', acc_lstm_sigmoid_FC)

print('\nBuilding above model with an additional dense layer ...\n')
model = Sequential()
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape = (encoded_train.shape[1],encoded_train.shape[2])))
model.add(Dense(16, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
print('\nTrain...\n')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = verbose,validation_data=(x_test, y_test))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,batch_size=batch_size)
print('\nTest score:', score_lstm_sigmoid_FC)
print('\nTest accuracy:', acc_lstm_sigmoid_FC)


print("Trying a stacked LSTM model ")

model = Sequential()
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape = (encoded_train.shape[1],encoded_train.shape[2]), return_sequences = True))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2,return_sequences = True))
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
model.add(BatchNormalization())
model.add(Dense(16, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
print('\nTrain...\n')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = verbose,validation_data=(x_test, y_test))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,batch_size=batch_size)
print('\nTest score:', score_lstm_sigmoid_FC)
print('\nTest accuracy:', acc_lstm_sigmoid_FC)

print("Trying a stacked GRU version of above model ")
model = Sequential()
model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2, input_shape = (encoded_train.shape[1],encoded_train.shape[2]), return_sequences = True))
model.add(GRU(64, dropout=0.2, recurrent_dropout=0.2,return_sequences = True))
model.add(GRU(32, dropout=0.2, recurrent_dropout=0.2))
model.add(BatchNormalization())
model.add(Dense(16, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
print('\nTrain...\n')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = verbose,validation_data=(x_test, y_test))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,batch_size=batch_size)
print('\nTest score:', score_lstm_sigmoid_FC)
print('\nTest accuracy:', acc_lstm_sigmoid_FC)
"""

#from keras.layers import TimeDistributed
print('Building w2v embedding model with denser (16 units) FC layer ...')
model = Sequential()
model.add(Embedding(len(embedding_matrix), 32,weights = [embedding_matrix],trainable = False))
model.add(LSTM(30, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
print('\nTrain...\n')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = verbose,validation_data=(x_test, y_test))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,batch_size=batch_size)
print('\nTest score:', score_lstm_sigmoid_FC)
print('\nTest accuracy:', acc_lstm_sigmoid_FC)
## 0.924 , 0.232


print('Building w2v embedding model with denser LSTM-128, relu 32, sigmoid FC layer with BatchNormalization...')
model = Sequential()
model.add(Embedding(len(embedding_matrix), 32,weights = [embedding_matrix],trainable = False))
model.add(LSTM(128))#, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
print('\nTrain...\n')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = verbose,validation_data=(x_test, y_test))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,batch_size=batch_size)
print('\nTest score:', score_lstm_sigmoid_FC)
print('\nTest accuracy:', acc_lstm_sigmoid_FC)

# 0.952 , 0.174


## stacked LSTMs without stateful representation, without trainable embeddings
# note : in LSTM calls, default dropout is set 0.0!


model = Sequential()
model.add(Embedding(len(embedding_matrix), 32,weights = [embedding_matrix],trainable = False))
model.add(LSTM(64,return_sequences = True)) # if return sequences is set to False (default)
											# it will return a single a single vector of
											# dimension = # of units in LSTM
											# for stacked, we need a sequnce of vectors
											# so set return_sequences = True for 
											# stacked LSTM (RNN) layers
#model.add(LSTM(128,return_sequences = True, input_shape = (timesteps,data_dim)))
model.add(LSTM(64,return_sequences = True))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
print('\nTrain...\n')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = verbose,validation_data=(x_test, y_test))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,batch_size=batch_size)
print('\nTest score:', score_lstm_sigmoid_FC) # 0.1709
print('\nTest accuracy:', acc_lstm_sigmoid_FC) # 0.9511


## trainable
print("\n...Trying a trainable w2v space version of the previous model...\n")
model = Sequential()
model.add(Embedding(len(embedding_matrix), 32,weights = [embedding_matrix],trainable = True))
model.add(LSTM(64,batch_size = batch_size,return_sequences = True)) # if return sequences is set to False (default)
											# it will return a single a single vector of
											# dimension = # of units in LSTM
											# for stacked, we need a sequnce of vectors
											# so set return_sequences = True for 
											# stacked LSTM (RNN) layers
#model.add(LSTM(128,return_sequences = True, input_shape = (timesteps,data_dim)))
model.add(LSTM(64,return_sequences = True))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
print('\nTrain...\n')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = verbose,validation_data=(x_test, y_test))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,batch_size=batch_size)
print('\nTest score:', score_lstm_sigmoid_FC) # 0.145
print('\nTest accuracy:', acc_lstm_sigmoid_FC) # 0.962

print("\n...Trying a huge model with stacked LSTMs and trainable w2v embedding...\n")
model = Sequential()
model.add(Embedding(len(embedding_matrix), 32,weights = [embedding_matrix],trainable = True))
model.add(LSTM(128,batch_size = batch_size,return_sequences = True)) # if return sequences is set to False (default)
											# it will return a single a single vector of
											# dimension = # of units in LSTM
											# for stacked, we need a sequnce of vectors
											# so set return_sequences = True for 
											# stacked LSTM (RNN) layers
#model.add(LSTM(128,return_sequences = True, input_shape = (timesteps,data_dim)))
model.add(LSTM(128,return_sequences = True))
model.add(LSTM(128,return_sequences = True))
model.add(LSTM(128,return_sequences = True))
model.add(LSTM(128))
model.add(Dense(128, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
print('\nTrain...\n')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = verbose,validation_data=(x_test, y_test))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,batch_size=batch_size)
print('\nTest score:', score_lstm_sigmoid_FC) # 0.150
print('\nTest accuracy:', acc_lstm_sigmoid_FC) # 0.963

print("\n...Trying a huge model with stacked LSTMs and trainable w2v embedding...\n")
model = Sequential()
#model.add(Embedding(len(embedding_matrix), 32,weights = [embedding_matrix],trainable = True))
model.add(GRU(128,batch_size = batch_size,input_shape = (encoded_train.shape[1],encoded_train.shape[2]),return_sequences = True)) # if return sequences is set to False (default)
											# it will return a single a single vector of
											# dimension = # of units in LSTM
											# for stacked, we need a sequnce of vectors
											# so set return_sequences = True for 
											# stacked LSTM (RNN) layers
#model.add(LSTM(128,return_sequences = True, input_shape = (timesteps,data_dim)))
model.add(GRU(128,return_sequences = True))
model.add(GRU(128,return_sequences = True))
model.add(GRU(128,return_sequences = True))
model.add(GRU(128))
model.add(Dense(128, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
print('\nTrain...\n')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = verbose,validation_data=(x_test, y_test))
score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,batch_size=batch_size)
print('\nTest score:', score_lstm_sigmoid_FC) # 0.128
print('\nTest accuracy:', acc_lstm_sigmoid_FC) # 0.9633
"""

end = time.time()
seconds = end - start
minutes = seconds//60
seconds = seconds % 60
hours = 0
if minutes > 60:
	hours = minutes//60
	minutes = minutes%60
print("time taken for running the notebook:\n {0} hours, {1} minutes and {2} seconds".format(hours,minutes,seconds))
#sys.stdout = sys.__stdout__