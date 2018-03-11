import tensorflow as tf
import pandas as pd
import numpy as np
import random,math,sys,os,itertools
print(os.listdir())
f = open('preprocessed_data.txt')
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(f).toarray()
print(x_train.shape)
