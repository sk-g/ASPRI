import re,os,time,sys,pickle,nltk,statistics
import pandas as pd
import numpy as np
from collections import Counter

class LanguageModel:
	""" A simple model to measure 'unusualness' of sentences. 
	delta is a smoothing parameter. 
	The larger delta is, the higher is the penalty for unseen words.
	"""
	def __init__(self, delta=0.01):
		self.delta = delta
	def preprocess(self, sentence):
		# returns words in each sentence
		words = []
		splits = sentence.split(' ')
		[words.append(i) for i in splits]
		return words
	def fit(self, corpus):
		# corpus = paths array
		""" Estimate counts from an array of texts """
		self.counter_ = Counter(word 
								 for sentence in corpus 
								 for word in self.preprocess(sentence))
		self.lines = [line.split(' ') for line in corpus]
		#self.counter_ = Counter(self.lines)
		self.total_count_ = sum(self.counter_.values())
		self.vocabulary_size_ = len(self.counter_.values())
	def perplexity(self, sentence):
		""" Calculate negative mean log probability of a word in a sentence 
		The higher this number, the more unusual the sentence is.
		"""
		words = self.preprocess(sentence)
		mean_log_proba = 0.0
		for word in words:
			# use a smoothed version of "probability" to work with unseen words
			word_count = self.counter_.get(word, 0) + self.delta
			total_count = self.total_count_ + self.vocabulary_size_ * self.delta
			word_probability = word_count / total_count
			mean_log_proba += np.log(word_probability) / len(words)
		return -mean_log_proba

	def relative_perplexity(self, sentence):
		""" Perplexity, normalized between 0 (the most usual sentence) and 1 (the most unusual)"""
		return (self.perplexity(sentence) - self.min_perplexity) / (self.max_perplexity - self.min_perplexity)

	@property
	def max_perplexity(self):
		""" Perplexity of an unseen word """
		return -np.log(self.delta / (self.total_count_ + self.vocabulary_size_ * self.delta))

	@property
	def min_perplexity(self):
		""" Perplexity of the most likely word """
		return self.perplexity(self.counter_.most_common(1)[0][0])
if os.name != 'posix':
	f = open(r'..\data\PCH\paths\11012018.txt')
	lines = f.readlines()
	lines = [i.strip() for i in lines]
else:
	f = open('../data/PCH/paths/11012018.txt')
	lines = f.readlines()
	lines = [i.strip() for i in lines]
	
lm = LanguageModel()

lm.fit(lines)

"""
test = lines[-10:]

for sent in test:
	print(lm.perplexity(sent).round(3),sent)

test = ['219 4826 63199 4809 1021','0000 10101 4826 2836 63199 4809 4809 4809 4809 4809 094']

for sent in test:
	print(lm.perplexity(sent).round(3),sent)
"""

arr = []

for sent in lines:
	arr.append(lm.perplexity(sent).round(3))
print("Maximum perplexity:{}\nMinimum Perplexity:{}\nAverage Perplexity:{}\n\
	Perplexity var:{}\nPerplexity std dev:{}".format(max(arr),\
		min(arr),statistics.mean(arr),statistics.variance(arr),\
		statistics.stdev(arr)))

if os.name != 'posix':
    paths = pd.read_csv(r'M:\Course stuff\ASPRI\supervised\11012018.csv',sep='\t',low_memory = False,index_col = False)
else:
    paths = pd.read_csv('11012018.csv',sep='\t',low_memory = False,index_col = False)
del paths['Unnamed: 0']

test_data = paths.loc[paths['Fake'] == 0]
arr_ = []

for sent in test_data:
	arr_.append(lm.perplexity(sent).round(3))

preds = [i>min(arr) for i in arr_]

print("Accuracy with test on true data = ",preds.count(1)/len(preds))
test_data = paths.loc[paths['Fake'] == 1]
arr_ = []

for sent in test_data:
	arr_.append(lm.perplexity(sent).round(3))

preds = [i>min(arr) for i in arr_]

print("Accuracy with test on fake data = ",preds.count(1)/len(preds))