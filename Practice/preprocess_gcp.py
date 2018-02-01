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

%matplotlib inline
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
f = open('initial_data')
current_prefix = None
for index,line in enumerate(f):
    drawProgressBar(int(index)/1058274)
    if index != 0 and index != 1058274:
        if (line.split(' ')[2].startswith(str(1)) or 
        line.split(' ')[2].startswith(str(2)) or
        line.split(' ')[2].startswith(str(3)) or
        line.split(' ')[2].startswith(str(4)) or 
        line.split(' ')[2].startswith(str(5)) or
        line.split(' ')[2].startswith(str(6)) or
        line.split(' ')[2].startswith(str(7)) or
        line.split(' ')[2].startswith(str(8)) or
        line.split(' ')[2].startswith(str(9))):                                 
            current_prefix = line.split(' ')[2]
            df.loc[index,'Valid']= line.split(' ')[0]
            df.loc[index,'Prefix'] = current_prefix
            #print(index,line)
        else:
            df.loc[index,'Valid'],df.loc[index,'Prefix'] = line.split(' ')[0],current_prefix

print(df)