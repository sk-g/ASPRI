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
	def createTrainingData(self):
		X = np.zeros(len(self.lines), self.timesteps , self.maxlen) 
		y = np.zeros(len(self.lines), self.timesteps , self.maxlen) 
	def onehotcoding(self,arr):
		# return one hot 
		return arr

	def loadEmbedding(self,embedding_dim = 32):
		self.embedding_dim = embedding_dim
		if self.embedding_dim == 32:
			self.final_embeddings = pickle.load(open(r'..\supervised\gcp_fe','rb'))
		elif self.embedding_dim == 32:
			self.final_embeddings = pickle.load(open(r'..\supervised\128dimsw2v','rb'))
		return self.final_embeddings
	
	def encode(arr):
		pass



if __name__ == '__main__':
		f1 = preprocess(r'M:\Course stuff\ASPRI\data\PCH\paths\11012018.txt')
		f2 = preprocess(r'M:\Course stuff\ASPRI\data\PCH\paths\24012018.txt')
		paths = f1.load_text()
		unique_paths, unique_as = f1.unique(paths)
		"""								
		paths = load_text(r'M:\Course stuff\ASPRI\data\PCH\paths\11012018.txt')
		unique_paths, unique_as = unique(paths)
		print(max([len(i.split(' ')) for i in unique_paths]))
		paths = load_text(r'M:\Course stuff\ASPRI\data\PCH\paths\24012018.txt')
		unique_paths, unique_as = unique(paths)
		print(max([len(i.split(' ')) for i in unique_paths]))

		"""