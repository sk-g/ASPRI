import os,time,sys,pickle,nltk,math
import pandas as pd
import numpy as np
from collections import Counter

if os.name != 'posix':
	import matplotlib.pyplot
	import seaborn as sns
	import pylab as pl
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

words, unique_words = build_table(lines)
count_dict = Counter(words) # dictionary of counts
values,counts = list(count_dict.keys()),list(count_dict.values())
sum_counts = sum(counts)
tfs = [i/sum_counts for i in counts] # term frequencies

#print(count_dict)
print(np.mean(tfs),np.std(tfs),max(tfs),min(tfs))

# every line, sum of frequencies of all words
print(len(tfs),len(count_dict))
sum_of_frequencies = []

# building a list of sum of frequencies of AS numbers in each path
print("\nbuilding a list of sum of frequencies of AS numbers in each path\n")
for i in range(len(lines)):
	c = 0
	_ = len(lines)
	drawProgressBar((i+1)/_)
	splits = lines[i].split(' ')
	for j in splits:
		c += tfs[values.index(j)]
	sum_of_frequencies.append(c)
print("\n")
print(sum_of_frequencies[:10])
print("Average frequency: {0},\nleast frequency: {1},\nmaximum frequency: {2}".format(np.mean(sum_of_frequencies),\
	min(sum_of_frequencies),max(sum_of_frequencies)))
