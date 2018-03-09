'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k wordacters. ~1M is better.
'''

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import pandas as pd
import random,io,os,sys,itertools,collections,pickle,math,zipfile
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
import argparse

import os,re,time,sys,os,math,random,time,pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pandas as pd
from keras import backend as K
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM,Flatten,Dense,Embedding,GRU,BatchNormalization,RepeatVector
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import confusion_matrix

maxlen = 32
timesteps = 32 

def loader():
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
	max_length = 32
	vocab_size = 24612 #unique tokens for this file
	embedding_dim = final_embeddings.shape[1]
	# saving the state of final embeddings
	original_gcp_fe = final_embeddings.copy()

	# reading the labeled data from csv
	if os.name != 'posix':
		paths = pd.read_csv(r'M:\Course stuff\ASPRI\supervised\11012018.csv',sep='\t',low_memory = False,index_col = False)
	else:
		paths = pd.read_csv('../supervised/11012018.csv',sep='\t',low_memory = False,index_col = False)
	del paths['Unnamed: 0']
	#paths.head()

	train,test = split(paths,test_size = 0.3,shuffle = False) # splitting into train,test
	return train,test,final_embeddings,dictionary,reverse_dictionary

def drawProgressBar(percent, barLen = 50):
	sys.stdout.write("\r")
	progress = ""
	for i in range(barLen):
		if i<int(barLen * percent):
			progress += "="
		else:
			progress += " "
	sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
	sys.stdout.flush()

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
	embedding_dim = embedding_dim
	#print("embedding dimension shape in encode_lines = ", embedding_dim)
	nd_array = []
	for i in range(max_length):
		nd_array.append(np.zeros(shape = (embedding_dim,1)))

	# creating a new array that contains above vectors	
	_embedding_dim = embedding_dim
	#print("encoding function, {} is the _embedding_dim".format(_embedding_dim))
	new_array = []
	for i in range(len(arr)):
		new_array.append(nd_array)
	new_array = np.asarray(new_array)
	c = 0
	for i in arr['Paths']:
		splits = i.split(' ')
		for j in range(len(splits)):
			#print(new_array[i][j])
			if splits[j] in dictionary:
				new_array[c,j] = final_embeddings[dictionary[str(splits[j])]-1].reshape(_embedding_dim,1)
			else:
				new_array[c,j] = final_embeddings[dictionary['UNK']]#setting unkown word to UNK
		c += 1
	assert(len(new_array) == len(arr))
	assert(len(new_array[0]) == max_length)
	return new_array

train,test,final_embeddings,dictionary,reverse_dictionary = loader()
y_train,y_test = train['Fake'],test['Fake']
embedding_dim = final_embeddings.shape[1]
timesteps = max_length = data_dim = 32
vocab_size = len(dictionary) #unique tokens for this file
max_features = vocab_size
epochs = 5 #training steps
verbose = 1
batch_size = 128

def encode(train,test,embedding_dim = 32):
	#print("Embedding Dimension = ", embedding_dim)
	embedding_dim_ = embedding_dim
	encoded_train = encode_lines(train,embedding_dim = embedding_dim)
	encoded_train = encoded_train.reshape(encoded_train.shape[0],encoded_train.shape[1],encoded_train.shape[2])
	#print(encoded_train.shape)
	encoded_test = encode_lines(test,embedding_dim = embedding_dim)
	encoded_test = encoded_test.reshape(encoded_test.shape[0],encoded_test.shape[1],encoded_test.shape[2])
	return encoded_train,encoded_test

	#test_input = test.values

encoded_train,encoded_test = encode(train,test)

x_train, x_test = encoded_train,encoded_test

def load_embeddings(final_embeddings):

	print("\n..... Extracting and building the word2vec results .....")
	embedding_matrix = {}
	for i in list(reverse_dictionary.keys()):
		embedding_matrix[i] = final_embeddings[i-1]

	print("Word2Vec embedding space has the shape: {0}\n".format(final_embeddings.shape))

	embedding_matrix = np.zeros((len(dictionary), embedding_dim))
	for i in list(reverse_dictionary.keys()):
		embedding_vector = final_embeddings[i-1]#.reshape((embedding_dim,1))
		if embedding_vector is not None:
			# words not found in embedding index will be all-zeros.
			embedding_matrix[i-1] = embedding_vector
	return embedding_matrix
embedding_matrix = load_embeddings(final_embeddings)

def cnfmx(y_true,y_pred):

	####
	# 				True Conditions
	# 				valid invalid
	# predictions
	# valid 		tp 		fp
	# invalid 		fn 		tn
	####
	predictions = [i[0] for i in y_pred]
	p = [0]*len(predictions)
	print("\nGetting Predictions..\n")
	for i in range(len(predictions)):
		drawProgressBar(i/len(predictions))
		if predictions[i] >=0.5:
			p[i] = 1
	print("\n")
	tp,fp,fn,tn = 0,0,0,0
	i = 0
	for index,value in y_true.iteritems():
		if value == 0: # true condition valid
			if p[i] == 0:# predict valid
				tp += 1# tp
			else:# predict invalid
				fn += 1 # fn
		elif value == 1:# true condition invalid
			if p[i] == 1:# predict invalid
				tn += 1# true negative
			else:# predict valid
				fp += 1# false positive
		i += 1
	acc = (tp+tn)*100/(len(predictions))
	#tn,fp,fn,tp = confusion_matrix(y_test,p).ravel()

	#sys.stdout = open('confusion matrix.txt','a')
	sys.stdout = open('LSTM_hidden_units.txt','a')
	print("Confusion Matrix on test set:\n\
		true negatives = {}\n\
		false positives = {}\n\
		false negatives = {}\n\
		true positives = {}\n\
		accuracy = {}\n".format(tn,fp,fn,tp,acc))
	print("__________________________________")
	sys.stdout = sys.__stdout__
	return



def main(x_train,y_train,x_test,y_test):
		print(encoded_train.shape)
		model = Sequential()
		model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2,input_shape = (encoded_train.shape[1],encoded_train.shape[2])))#,return_sequences = True ))
		model.add(RepeatVector(encoded_train.shape[1]))
		model.add(LSTM(vocab_size, dropout=0.2, recurrent_dropout=0.2,return_sequences = True ))
		model.compile(loss='mean_squared_error',optimizer = 'adam')
		model.summary()
		model.fit(x_train,x_train, epochs = 1)
		preds = model.predict(x_test)
		print(preds.shape)
		#pickle.dump(preds,open('predictions','wb'))
if __name__ == '__main__':
	main(x_train, y_train,x_test,y_test)		