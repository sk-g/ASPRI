from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random,io,os,sys,itertools,collections,pickle

class preprocess(object):
	def __init__(self,path):
		self.path = str(path)
		self.empty = None
		## constansts go here
		self.timesteps = self.maxlen = 32
			
	def load_text(self):
		# call this function with absoulute
		# path to text file
		# return lines as array

		return ([i.strip() for i in open(self.path,'r').readlines()])

	def unique(self,arr):
		# function to return the following:
		# unique lines
		# print total number of words and
		# numbder of uniqe words
		# set of unique words aka unique tokens
		self.lines = arr
		def wordify(l):
			return ([word for line in l for word in line.split(' ')])
		unique_lines = list(set(arr))
		unique_tokens = wordify(unique_lines)
		tokens = wordify(arr)
		self.U_TOKENS = len(unique_lines)
		self.U_LINES = len(unique_lines)
		self.LINES = len(arr)
		self.TOKENS = len(tokens)
		
		print("\nNumber of unique ASes = {}\nNumber of unique paths = {}\nNumber of total paths = {}\nNumber of total ASes = {}\n\
			".format(self.U_TOKENS,self.U_LINES,self.LINES,self.TOKENS))
		return unique_lines,unique_tokens
	def preprocess(self):
		lines = self.lines
		for i in range(len(lines)):
			temp = lines[i].split(' ')
			temp.insert(0,'START')
			temp.append('END')
			temp = ' '.join([str(i) for i in temp])
			lines[i] = temp
		return lines

	def onehotcoding(self,arr):
		# return one hot 
		return arr

	def loadEmbedding(self,embedding_dim = 32):
		self.embedding_dim = embedding_dim
		if self.embedding_dim == 32:
			self.final_embeddings = pickle.load(open(r'..\supervised\gcp_fe','rb'))
		elif self.embedding_dim == 128:
			self.final_embeddings = pickle.load(open(r'..\supervised\128dimsw2v','rb'))
		self.dictionary = pickle.load(open(r'..\supervised\gcp_dictionary','rb'))
		self.reverse_dictionary = pickle.load(open(r'..\supervised\gcp_reverse_dictionary','rb'))
		
		print("Word2Vec embedding space has the shape: {0}\n".format(self.final_embeddings.shape))

		embedding_matrix = np.zeros((len(self.dictionary), embedding_dim))
		for i in list(self.reverse_dictionary.keys()):
			embedding_vector = self.final_embeddings[i-1]#.reshape((embedding_dim,1))
			if embedding_vector is not None:
				# words not found in embedding index will be all-zeros.
				embedding_matrix[i-1] = embedding_vector
		self.embedding_matrix = embedding_matrix	
	def encode(self,arr):
		nd_array = []
		for i in range(self.maxlen):
			nd_array.append(np.zeros(shape=(self.embedding_dim,1)))
		new_array = []
		for i in range(len(arr)):
			new_array.append(nd_array)
		new_array = np.asarray(new_array)
		c = 0
		for i in arr:
			splits = i.split(' ')
			for j in range(len(splits)):
				if splits[j] in self.dictionary:
					new_array[c,j] = self.final_embeddings[self.dictionary[str(splits[j])]-1].reshape(self.embedding_dim,1)
			else:
				new_array[c,j] = self.final_embeddings[self.dictionary['UNK']].reshape(self.embedding_dim,1)#setting unkown word to UNK
			c += 1
		self.encoded = new_array.reshape(new_array.shape)
	def build_model(self):
		model = Sequential()
		model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2,input_shape = (self.encoded.shape[1],self.encoded.shape[2]),return_sequences = True ))
		model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
		model.compile(loss='categorical_crossentropy',optimizer = 'adam')
		model.summary()
		model.fit(self.encoded,self.encoded, nb_epoch = 10)

if __name__ == '__main__':
		f1 = preprocess(r'M:\Course stuff\ASPRI\data\PCH\paths\11012018.txt')
		f2 = preprocess(r'M:\Course stuff\ASPRI\data\PCH\paths\24012018.txt')
		paths = f1.load_text()
		unique_paths, unique_as = f1.unique(paths)
		f1.loadEmbedding()
		f1.encode(paths)
		f1.build_model()