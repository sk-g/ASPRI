import os,time,sys,math,random
import numpy as np
class pathExtractor(object):
	def __init__(self,strx):
		self.fname = str(strx)
		self._lines = []
		self._misconfiguredPaths = []
		self._hijackedPaths = []
		self._variedLengths = []
		self._originalPaths = []
		self.__combined__ = []
		self.__IntPaths = []
	def getPaths(self):
		#strx is the absolute path of file
		fname = self.fname
		f = open(fname)
		self._lines = f.readlines()
		for i in range(len(self._lines)):
			self._originalPaths.append(str(''.join([str(j) for j in self._lines[i].strip().split('\t')[-1]])))
		return self._originalPaths
	def getInts(self):
		
		#if len(self._orignalPaths) == 0 :
		if not self._originalPaths:
			print("run the getPaths method first")
			return
		arr = self._originalPaths
		for i in range(len(arr)):
			#[int(i) for i in obj._originalPaths[100].split(' ')]
			self.__IntPaths.append([int(j) for j in arr[i].split('\t')])
		return self.__IntPaths
	def misconfiguredPaths(self,strx):
		if len(self._originalPaths) == 0:
			print("run the getPaths method first")
		pass
		#should take entire file or line by line?
	def hijackedPaths(self):
		if len(self._originalPaths) == 0:
			print("run the getPaths method first")
		random_paths = np.random.choice(self._originalPaths,3,replace = False)
		for i in random_paths:
			print(i,len(i.split(' ')))
		#print(random_selection)
		pass
	def variedLengths(self,strx):
		if len(self._originalPaths) == 0:
			print("run the getPaths method first")
		pass
	def showAllPaths(self,strx):
		#call this after finishing up above three functions
		#otherwise,legnth overflow
		# for i in range
		# print originalpaths[i],etc[i]
		if len(self._originalPaths) == 0:
			print("run the getPaths method first")
		pass
	def writePaths(self):
		#write with same filename
		#all arrays
		if len(self._originalPaths) == 0:
			print("run the getPaths method first")
		pass

#obj = pathExtractor('24012018_.txt')
#obj.getInts() < - correctly fetches error asking to call getPaths first
#obj.getPaths()
#tw = open('24012018.txt','w')
#for i in range(len(obj._originalPaths)):
#	tw.write(obj._originalPaths[i]+"\n")