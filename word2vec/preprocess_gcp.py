from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

import time,sys,os,itertools,cv2,keras,math,random,zipfile,collections
from tempfile import gettempdir
from collections import *
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%matplotlib inline
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import sys, time, os, itertools
from collections import *
import numpy as np
#if os.name != 'posix':
#    os.chdir(r'M:\Course stuff\ASPRI\Practice')
#os.chdir("M:\Course stuff\ASPRI\Practice")
df = pd.DataFrame(columns = ['Valid','Prefix'])
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
f = open("initial_data")
lines = f.readlines()
lines = lines[1:-2]
lines = [i.strip() for i in lines]

def classify(strx):
    #call this on each line
    next_hop = 0
    prefix = ""
    #path = ""
    path = []
    for i in strx.split(' '):
        slash_counter,dot_counter = i.count('/'),i.count('.')
        #print(i.count('.'),i.count('/'))
        if dot_counter == 3 and slash_counter == 1:
            prefix = i
        elif dot_counter == 3 and slash_counter == 0:
            next_hop = i
        elif len(i) > 1 and dot_counter == 0 and slash_counter == 0:
            
            #path = path.join(i)
            path.append(i)
            #print("Sanity check on paths",path,"\t",i)
            #path = path.join(' ')
    path_ = path
    path = " ".join([str(i) for i in path_])
    protocol = strx[-1]
    #print(protocol)
    #print("prefix = {0}, next hop = {1}, path = {2}".format(prefix,next_hop,path))
    return(prefix,next_hop,path,protocol)
n_f_h = open("preprocessed_data.txt",'w')
n_f_h_ = open("preprocessed_data_without_protocol.txt",'w')
counter = 0
total = len(lines)
for line in lines:
    prefix,next_hop,path,protocol = classify(line)
    if not prefix:
        prefix = current_prefix
    else:
        current_prefix = prefix
    n_f_h.write(str(current_prefix)+"\t"+str(next_hop)+"\t"+str(path)+"\t"+str(protocol)+"\n")
    n_f_h_.write(str(current_prefix)+"\t"+str(next_hop)+"\t"+str(path)+"\n")
    counter += 1
    drawProgressBar(counter/total)
n_f_h.close()
n_f_h_.close()