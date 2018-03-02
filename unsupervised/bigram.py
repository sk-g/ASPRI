import os,time,sys,pickle,nltk,statistics
import pandas as pd
import numpy as np

if os.name != 'posix':
	f = open(r'..\data\PCH\paths\11012018.txt')
	lines = f.readlines()
	lines = [i.strip() for i in lines]
else:
	f = open('../data/PCH/paths/11012018.txt')
	lines = f.readlines()
	lines = [i.strip() for i in lines]

def unique_lines(arr):
	return (list(set(arr)))

def build_table(arr):
	words = []
	for i in range(len(arr)):
		splits = arr[i].split(' ')
		[words.append(k) for k in splits]
	return((words),(list(set(words))))
words, unique_words = build_table(lines)

bigrams = list(nltk.bigrams(words))

print(len(lines))
#print(len(words),len(bigrams))

from collections import Counter

bigram_lookup = Counter(bigrams)

bigram_keys,bigram_values = list(bigram_lookup.keys()),list(bigram_lookup.values())


bigram_lines = [list(nltk.bigrams(line.split(' '))) for line in lines]
"""print(len(bigram_lines))
test_path = bigram_lines[0]
test_path_counts = [bigram_lookup[i] for i in test_path]
print("bigram line:{}\ncorresponding counts:{}".format(test_path,test_path_counts))
#[print(bigram_lookup[i]) for i in test_path]

print("sum of bigram counts:{0} and total bigrams:{1}".format(sum(bigram_values),len(bigram_values)))"""

# bigram_values -> bigram frequencies:
# bigram_values[i]/=sum(bigram_values)
sum_of_bigrams = sum(bigram_values)
bigram_values = [i/sum_of_bigrams for i in bigram_values] # TF
print("After normalizing coutns with respect to sum of counts - \nmax count = {}\nmin count = {}".format(max(bigram_values)*1000,min(bigram_values)*1000))
"""print("After normalizing\n \
	Mean = {}\n \
	Mode = {}\n \
	Population std = {}\n \
	population variance = {}\n \
	stdev = {}\n \
	variance = {}\n".format(statistics.mean(bigram_values),statistics.mode(bigram_values),statistics.pstdev(bigram_values)\
		,statistics.pvariance(bigram_values),statistics.stdev(bigram_values),statistics.variance(bigram_values)))"""
var = statistics.variance(bigram_values)
bigram_values=[i/var for i in bigram_values]
"""print("After re normalizing\n \
	Mean = {}\n \
	Mode = {}\n \
	Population std = {}\n \
	population variance = {}\n \
	stdev = {}\n \
	variance = {}\n".format(statistics.mean(bigram_values),statistics.mode(bigram_values),statistics.pstdev(bigram_values)\
		,statistics.pvariance(bigram_values),statistics.stdev(bigram_values),statistics.variance(bigram_values)))"""
print("After re normalizing coutns with respect to variance - \nmax count = {}\nmin count = {}".format(max(bigram_values)*1000,min(bigram_values)*1000))

# min = approx 1 micro
# max = 0.011



def probabilities(arr): #send in bigram representations
	# number of bigrams = total_bigrams 
	total_bigrams = len(bigram_lookup)
	for i in range(len(arr)):
		# for each sentence
		if len(arr[i]) == 1:
			arr[i] = bigram_lookup[arr[i][0]]/total_bigrams

		p1 = bigram_lookup[arr[i][0]]/total_bigrams
		arr[i][0] = p1
		for j in range(1,len(arr[i])):
			# for bigrams in ith sentence:
			#print(arr[i][0])
			prev_prob = arr[i][j-1]
			current_prb = bigram_lookup[j]/total_bigrams
			#print(type(prev_prob),type(current_prb))
			arr[i][j] = prev_prob*current_prb
	return arr

prob_rep = probabilities(bigram_lines[:10])
print([i for i in prob_rep])

