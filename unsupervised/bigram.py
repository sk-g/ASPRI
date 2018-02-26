import os,time,sys,pickle,nltk
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
print(len(bigram_lines))
test_path = bigram_lines[0]
[print(bigram_lookup[i]) for i in test_path]

print("sum of bigram counts:{0}\t{1}".format(sum(bigram_values),len(bigram_values)))

#path = 