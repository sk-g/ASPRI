from __future__ import print_function
import os,re,time,sys,os,math,random,time,pickle,collections,keras
import pandas as pd
import numpy as np

import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.contrib import rnn
from keras import backend as K
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM,Flatten,Dense,Embedding,GRU,BatchNormalization
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split as split
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

#import matplotlib.pyplot as plt
#import seaborn as sns
#import pylab as pl

# we are working with labeled data
# so load csv files
# not raw text data
start = time.time()
class DataModeling():
	def __init__(self, path):
		self.path = str(path)
		self.lines = None
		self.maxlen = 32
		self.vocabSize = None
		self.EmbeddingDim = None
		self.dictionary = None
		self.reversed_dictionary = None
		self.timesteps = 32

	def load(self):
		self.paths = pd.read_csv(self.path,sep='\t',low_memory = False,index_col = False)
		del self.paths['Unnamed: 0']
		self.lines = self.paths['Paths']
		self.labels = self.paths['Fake']
		pass
	def buildData(self):
		# convert word sequences to
		# unique indices
		# add one extra index for unkown/0 pad
		print("\nMapping sequences to indices\n")
		if not self.lines:
			self.load()
		lines = self.lines
		self.n = len(lines)
		words = []
		for i in range(len(lines)):
			splits = lines[i].split(' ')
			for j in splits:
				words.append(j)
			self.drawProgressBar((i+1)/self.n)
		self.words = words
		self.uniqueWords = list(set(words))
		self.vocabSize = len(self.uniqueWords)
		print("\nVocabulary Size = {}".format(self.vocabSize))
		#print(self.uniqueWords)
		print("""\nProcess raw inputs into a dataset.""")
		count = [['UNK', -1]]
		#count = [[]]
		count.extend(collections.Counter(words).most_common(self.vocabSize+1))
		dictionary = dict()
		#print(count)
		for word, _ in count:
			dictionary[word] = len(dictionary)
		data = list()
		unk_count = 0
		for word in words:
			index = dictionary.get(word, 0)
			#print(word,index)
			if index == 0:  # dictionary['UNK']
				unk_count += 1
			data.append(index)
		count[0][1] = unk_count
		reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
		self.dictionary = dictionary
		self.reversed_dictionary = reversed_dictionary
		self.counts = count
		del dictionary,reversed_dictionary,count
		print("Size of dictionary = {}\nsize of vocabulary = {}".format(len(self.dictionary),self.vocabSize))

	def convert(self):
		# convert all sequences of words to 
		# sequences of indices
		print("\nMapping sequences\n")
		indexed_lines = []
		for i,v in self.lines.iteritems():
			temp = []
			splits = self.lines[i].split(' ')
			for j in splits:
				temp.append(self.dictionary[j])
			indexed_lines.append(temp)
			self.drawProgressBar(i/self.n)
		self.indexed_lines = indexed_lines
		del indexed_lines
		assert len(self.indexed_lines)==self.n,"Something Went wrong"
		#print("\n",indexed_lines[0]) < -- validated
		#print(self.reversed_dictionary[0]) 'UNK'
		pass
	def zeroPad(self):
		print("\nZero Padding the sequences\n")
		self.indexed_lines = pad_sequences(self.indexed_lines, maxlen = self.maxlen,padding = 'post')
		print("\nshape of indexed_lines = {}".format(self.indexed_lines.shape))

	def getTrainTestData(self):
		# split data into training and test set
		self.x_train,self.x_test,self.y_train,self.y_test = split(self.indexed_lines,self.labels,random_state = 694)
		print(type(self.x_train))
		#print(self.x_train[0],self.y_train[0])
		pass
	def drawProgressBar(self,percent, barLen = 50):
		sys.stdout.write("\r")
		progress = ""
		for i in range(barLen):
			if i<int(barLen * percent):
				progress += "="
			else:
				progress += " "
		sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
		sys.stdout.flush()


	def cnfmx(self,y_true,y_pred,strx):

		what = str(strx)
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
			self.drawProgressBar(i/len(predictions))
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

		sys.stdout = open('we_results.txt','a')
		print("Confusion Matrix on " + str(what)+":\n\
			true negatives = {}\n\
			false positives = {}\n\
			false negatives = {}\n\
			true positives = {}\n\
			accuracy = {}\n".format(tn,fp,fn,tp,acc))
		print("__________________________________")
		sys.stdout = sys.__stdout__
		return


	def train(self):
		epochs = 5
		verbose = 1
		batch_size = 1024
		callbacks = [keras.callbacks.TensorBoard(log_dir = r'.\logs')]

		model = Sequential()
		model.add(Embedding(self.vocabSize+1, 128))
		model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
		model.add(MaxPooling1D(pool_size=2))
		model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2,return_sequences = True))
		model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2,return_sequences = True))
		model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
		model.add(Dense(16, activation='sigmoid'))
		model.add(BatchNormalization())
		model.add(Dense(1, activation='sigmoid'))
		model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
		model.fit(self.x_train,self.y_train,batch_size=batch_size,epochs=epochs,verbose = verbose,callbacks = callbacks)
		predictions = model.predict(self.x_train)
		self.cnfmx(self.y_train,predictions,"train set,  128 ED + CNN + 3xLSTM(12) + Dense 16 + BN + Dense")
		predictions = model.predict(self.x_test)
		self.cnfmx(self.y_test,predictions,"train set, 128 ED + CNN + 3xLSTM(12) + Dense 16 + BN + Dense")


def main():
	start = time.time()
	dm = DataModeling(r'M:\Course stuff\ASPRI\supervised\11012018.csv')
	dm.buildData()
	dm.convert()
	dm.zeroPad()
	dm.getTrainTestData()
	dm.train()
	end = time.time()
	seconds = end - start
	minutes = seconds//60
	seconds = seconds % 60
	hours = 0
	if minutes > 60:
		hours = minutes//60
		minutes = minutes%60
	print("time taken:\n{0} hours, {1} minutes and {2} seconds".format(hours,minutes,seconds))
	return None

if __name__ == '__main__':
	main()
