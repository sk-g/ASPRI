import re,os,time,sys,pickle,nltk,statistics,random
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


if os.name != 'posix':
	f = open(r'..\data\PCH\paths\11012018.txt')
	lines = f.readlines()
	lines = [i.strip() for i in lines]
else:
	f = open('../data/PCH/paths/11012018.txt')
	lines = f.readlines()
	lines = [i.strip() for i in lines]
	
lm = LanguageModel()
print("Fitting the language model:\n")
lm.fit(lines)
print("Done!\n")

#test = lines[-10:]

#for sent in test:
#	print(lm.perplexity(sent).round(3),sent)

test = ['219 4826 63199 4809 1021','0000 10101 4826 2836 63199 4809 4809 4809 4809 4809 094']

#for sent in test:
#	print(lm.perplexity(sent).round(3),sent)


arr = []

for sent in lines:
	arr.append(lm.perplexity(sent).round(3))
"""
print("Maximum perplexity:{}\nMinimum Perplexity:{}\nAverage Perplexity:{}\n\
	Perplexity var:{}\nPerplexity std dev:{}".format(max(arr),\
		min(arr),statistics.mean(arr),statistics.variance(arr),\
		statistics.stdev(arr)))
"""
thresh = max(arr)
print("Perplexity Threshold = {}\n".format(thresh))

# loading labeled data
if os.name != 'posix':
    paths = pd.read_csv(r'M:\Course stuff\ASPRI\supervised\11012018.csv',sep='\t',low_memory = False,index_col = False)
else:
    paths = pd.read_csv('11012018.csv',sep='\t',low_memory = False,index_col = False)
del paths['Unnamed: 0']

print("Testing on a subset of training data. Accuracy should be 100!\n")
test_data = lines[-101:]
arr_ = []

for sent in test_data:
	arr_.append(lm.perplexity(sent).round(3))

preds = [i<thresh for i in arr_] # < because we are checking if they are likely

print("Accuracy with test on validation data = ",preds.count(1)/len(preds))
# extracting fake paths
fake_paths = [paths.loc[paths['Fake'] == 1]][0]['Paths'].tolist()
test_data = fake_paths
ldata = len(test_data)
arr_ = []
print("\nTesting on fake paths:\n")
for i in range(len(test_data)):
	drawProgressBar(i/ldata)
	arr_.append(lm.perplexity(test_data[i]).round(3))
print("\nTested on {} paths with a Perplexity thresh = {}\n".format(len(arr_),thresh))
preds = [i>thresh for i in arr_]# > because we want to predict these as fake
print("Accuracy with test on fake data = ",preds.count(1)/len(preds))
print(arr_[-5:])
#print("\nAnd the predictions are:{}".format(arr_))
random_path = '094 I am an invalid route 4826 4809'
print("Let's see Perplexity of random path \'{}\' with invalid numbers:\n".format(random_path))
print(lm.perplexity(random_path).round(3))
print(lm.relative_perplexity(random_path).round(3))
print("perplexity of a random path from training set:",lm.perplexity(lines[random.randint(10000,15000)]).round(3))
print(lm.min_perplexity,lm.max_perplexity)