from __future__ import print_function
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
from keras.layers import LSTM,Flatten,Dense,Embedding,GRU,BatchNormalization
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import confusion_matrix

start = time.time()


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
timesteps = max_length = data_dim = 30
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
	sys.stdout = open('confusion matrix.txt','a')

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
	print('Build first model...')
	model = Sequential()
	model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2,input_shape = (encoded_train.shape[1],encoded_train.shape[2])))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	sys.stdout = open('confusion matrix.txt','a')
	model.summary()
	sys.stdout = sys.__stdout__
	print('\nTrain...\n')
	model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = verbose,validation_data=(x_test, y_test))
	score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,batch_size=batch_size)
	predictions = model.predict(x_train)
	cnfmx(y_train,predictions)
	print('\nTest score:', score_lstm_sigmoid_FC)
	print('\nTest accuracy:', acc_lstm_sigmoid_FC)


	print('\nBuilding above model but without dropout...\n')
	model = Sequential()
	model.add(LSTM(32, input_shape = (encoded_train.shape[1],encoded_train.shape[2])))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	sys.stdout = open('confusion matrix.txt','a')
	model.summary()
	sys.stdout = sys.__stdout__
	print('\nTrain...\n')
	model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = verbose,validation_data=(x_test, y_test))
	score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,batch_size=batch_size)
	predictions = model.predict(x_train)
	cnfmx(y_train,predictions)
	print('\nTest score:', score_lstm_sigmoid_FC)
	print('\nTest accuracy:', acc_lstm_sigmoid_FC)

	print('\nBuilding above model with more hidden units in LSTM ...\n')
	model = Sequential()
	model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape = (encoded_train.shape[1],encoded_train.shape[2])))
	#model.add(Dense(16, activation='sigmoid'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	sys.stdout = open('confusion matrix.txt','a')
	model.summary()
	sys.stdout = sys.__stdout__
	print('\nTrain...\n')
	model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = verbose,validation_data=(x_test, y_test))
	score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,batch_size=batch_size)
	predictions = model.predict(x_train)
	cnfmx(y_train,predictions)
	print('\nTest score:', score_lstm_sigmoid_FC)
	print('\nTest accuracy:', acc_lstm_sigmoid_FC)

	print('\nBuilding above model with an additional dense layer and BatchNormalization ...\n')
	model = Sequential()
	model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape = (encoded_train.shape[1],encoded_train.shape[2])))
	model.add(Dense(16, activation='sigmoid'))
	model.add(BatchNormalization())
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	sys.stdout = open('confusion matrix.txt','a')
	model.summary()
	sys.stdout = sys.__stdout__
	print('\nTrain...\n')
	model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = verbose,validation_data=(x_test, y_test))
	score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,batch_size=batch_size)
	predictions = model.predict(x_train)
	cnfmx(y_train,predictions)
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
	sys.stdout = open('confusion matrix.txt','a')
	model.summary()
	sys.stdout = sys.__stdout__
	print('\nTrain...\n')
	model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = verbose,validation_data=(x_test, y_test))
	score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,batch_size=batch_size)
	predictions = model.predict(x_train)
	cnfmx(y_train,predictions)
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
	sys.stdout = open('confusion matrix.txt','a')
	model.summary()
	sys.stdout = sys.__stdout__
	print('\nTrain...\n')
	model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = verbose,validation_data=(x_test, y_test))
	score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,batch_size=batch_size)
	predictions = model.predict(x_train)
	cnfmx(y_train,predictions)
	print('\nTest score:', score_lstm_sigmoid_FC)
	print('\nTest accuracy:', acc_lstm_sigmoid_FC)

	print("Trying a huge stacked LSTM model with many hidden units and FC Layers")
	model = Sequential()
	model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape = (encoded_train.shape[1],encoded_train.shape[2]), return_sequences = True))
	model.add(LSTM(128,return_sequences = True))
	model.add(LSTM(128,return_sequences = True))
	model.add(LSTM(128))
	model.add(Dense(128, activation = 'sigmoid'))
	model.add(BatchNormalization())
	model.add(Dense(128,activation = 'sigmoid'))
	model.add(BatchNormalization())
	model.add(Dense(64,activation = 'sigmoid'))
	model.add(BatchNormalization())
	model.add(Dense(32,activation = 'sigmoid'))
	model.add(BatchNormalization())
	model.add(Dense(1,activation = 'sigmoid'))
	model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics=['accuracy'])
	sys.stdout = open('confusion matrix.txt','a')
	model.summary()
	sys.stdout = sys.__stdout__
	print('\nTrain...\n')
	model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose = verbose,validation_data=(x_test, y_test))
	score_lstm_sigmoid_FC, acc_lstm_sigmoid_FC = model.evaluate(x_test, y_test,batch_size=batch_size)
	predictions = model.predict(x_train)
	cnfmx(y_train,predictions)
	print('\nTest score:', score_lstm_sigmoid_FC)
	print('\nTest accuracy:', acc_lstm_sigmoid_FC)

if __name__ == '__main__':
	main(x_train, y_train,x_test,y_test)
###################### above are models run on (26k,32) vector space ###############################



if os.name != 'posix':
	#final_embeddings = pickle.load(open(r'M:\Course stuff\ASPRI\supervised\gcp_fe','rb'))#original 32 dims w2v
	final_embeddings = pickle.load(open(r'M:\Course stuff\ASPRI\supervised\128dimsw2v','rb'))# 128 dims w2v
else:
	final_embeddings = pickle.load(open('128dimsw2v','wb'))

embedding_dim = final_embeddings.shape[1]
encoded_train,encoded_test = encode(train,test,embedding_dim)

x_train, x_test = encoded_train,encoded_test

y_train,y_test = train['Fake'],test['Fake']


embedding_matrix = load_embeddings(final_embeddings)

if __name__ == '__main__':
	print("\n\nRunning All the models but now with the 128 dimension word2vec ...\n\n")
	main(x_train, y_train,x_test,y_test)	


end = time.time()
seconds = end - start
minutes = seconds//60
seconds = seconds % 60
hours = 0
if minutes > 60:
	hours = minutes//60
	minutes = minutes%60
print("time taken for running the notebook:\n {0} hours, {1} minutes and {2} seconds".format(hours,minutes,seconds))
sys.stdout = open('confusion matrix.txt','a')
print("time taken for running the notebook:\n {0} hours, {1} minutes and {2} seconds".format(hours,minutes,seconds))
sys.stdout = sys.__stdout__