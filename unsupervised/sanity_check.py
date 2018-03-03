import re,os,time,sys,pickle,nltk,statistics
import pandas as pd
import numpy as np
from collections import Counter
if os.name != 'posix':
    paths = pd.read_csv(r'M:\Course stuff\ASPRI\supervised\11012018.csv',sep='\t',low_memory = False,index_col = False)
else:
    paths = pd.read_csv('11012018.csv',sep='\t',low_memory = False,index_col = False)
del paths['Unnamed: 0']

fake_paths = [paths.loc[paths['Fake'] == 1]]

print(type(fake_paths))
#print(fake_paths)
print(type(fake_paths[0]['Paths'].tolist()))
fake_paths[0]['Paths'].tolist()