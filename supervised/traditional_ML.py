# let's try traditional ML on encoded data
from __future__ import print_function
import os,re,time,sys,os,math,random,time,pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
from sklearn.model_selection import train_test_split as split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import log_loss,confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report
start = time.time()



class Experiment:
	def __init__(self,fname = '11012018.txt',embedding_dim = 32):
		if os.name != 'posix':
			self.fpath = str(r'..\data\PCH\paths\\' )
		else:
			self.fpath = '../data/PCH/paths/'

		self.fname = str(fname)
		self.embedding_dim = embedding_dim
		self.max_length = 32
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

	def loader(self):

		self.lines = [i.strip() for i in open(os.path.join(self.fpath,self.fname),'r').readlines()]

		# loading labeled data
		if os.name != 'posix':
			#os.chdir(r'M:\Course stuff\ASPRI\supervised\ ')
			self.paths = pd.read_csv('11012018.csv',sep='\t',low_memory = False,index_col = False)
		else:
			self.paths = pd.read_csv('11012018.csv',sep='\t',low_memory = False,index_col = False)
		del self.paths['Unnamed: 0']

		return (self.lines,self.paths)
	def pickle_load(self):
		if self.embedding_dim == 32:
			self.final_embeddings = pickle.load(open('gcp_fe','rb'))#original 32 dims w2v
		elif self.embedding_dim == 128:
			self.final_embeddings = pickle.load(open('128dimsw2v','rb'))
		self.dictionary = pickle.load(open('gcp_dictionary','rb'))
		self.reverse_dictionary = pickle.load(open('gcp_reverse_dictionary','rb'))
		self.count = pickle.load(open('gcp_count','rb'))

	def encode_lines(self,arr,embedding_dim = 32):
		# function works on df
		# iterate over lines in df
		# iterate over splits of line
		# convert split word to embedding vector (32,1)
		# pad with (32,1) zeros
		# defualt embedding dim = 32
		# can send in dim or can extract from the arr itself
		# creating a row of Zeroed vectors
		# shape = dimensions x 1
		_,_ = self.loader()
		nd_array = []
		for i in range(self.max_length):
			nd_array.append(np.zeros(shape = (self.embedding_dim,1)))

		# creating a new array that contains above vectors	
		_embedding_dim = self.embedding_dim
		#print("encoding function, {} is the _embedding_dim".format(_embedding_dim))
		new_array = []
		for i in range(len(arr)):
			new_array.append(nd_array)
		new_array = np.asarray(new_array)
		c = 0
		t = len(arr)
		print("\nEncoding..\n")
		for i in arr:
			self.drawProgressBar(c/t)
			splits = i.split(' ')
			for j in range(len(splits)):
				
				new_array[c,j] = self.final_embeddings[self.dictionary[str(splits[j])]-1].reshape(_embedding_dim,1)
			c += 1
		assert(len(new_array) == len(arr))
		assert(len(new_array[0]) == self.max_length)
		return new_array

	def train_test_split(self):
		return(split(self.paths,test_size = 0.3))

	def data(self):
		train,test = self.train_test_split()
		x_train = self.encode_lines(train['Paths'],self.embedding_dim)
		x_test = self.encode_lines(test['Paths'],self.embedding_dim)
		self.x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
		self.x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])
		self.y_train = train['Paths']
		self.y_test = test['Fake']

	def LogReg(self):
		self.data()
		print("\nRunning LogisticRegression ...\n")
		clf = LogisticRegression(penalty = 'l2',n_jobs = -1)
		clf.fit(self.x_train,self.y_train)
		self.y_pred = clf.predict(self.x_train)
		acc_train = metrics.accuracy(self.y_pred,self.y_train)*100

		y_pred_test = clf.predict(self.x_test)
		acc_test = metrics.accuracy(self.y_pred_test,self.y_test)*100

		print("\nLogistic Regression results:\nTraining Accuracy = {}\nTesting Accuracy = {}".format(acc_train,acc_test))

LR = Experiment()
text, csv = LR.loader()
LR.pickle_load()
#LR.data()
LR.LogReg()
