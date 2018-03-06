'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k wordacters. ~1M is better.
'''

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import pandas as pd
import random,io,os,sys,itertools,collections,pickle,math,zipfile
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
import argparse
maxlen = 32
timesteps = 32 
lines = [i.strip() for i in open(r'M:\Course stuff\ASPRI\data\PCH\paths\11012018.txt','r').readlines()]
print('total lines:', len(lines))
lines_ = list(set(lines)) # building dictionaries from unique lines is faster
#word_indices = dict((c, i) for i, c in enumerate(words))
#indices_word = dict((i, c) for i, c in enumerate(words)) #ix_to_char
start_token = 'START'
end_token = 'END'
unkown_token = 'UNK'
for i in range(len(lines)):
	temp = lines[i].split(' ')
	temp.insert(0,'START')
	temp.append('END')
	temp = ' '.join([str(i) for i in temp])
	lines[i] = temp 
words = [word for line in lines_ for word in line.split(' ')]
vocab_size = len(list(set(words)))  # another constant
count = [['UNK',-1]]#,['START',-2],['END',-3]]
count.extend(collections.Counter(words))
dictionary = dict()
#print(count)
for word,_ in count:
	dictionary[word] = len(dictionary)
data = list()
unk_count = 0
start_count, end_count = 0,0
for word in words:
	index = dictionary.get(word, 0)
	#print(word,index)
	if index == 0:  # dictionary['UNK']
		unk_count += 1
	if index == 1:
		start_count += 1
		end_count += 1
	data.append(index)
count[0][1] = unk_count
reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
print(len(dictionary))