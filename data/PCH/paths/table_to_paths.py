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
		if not self._originalPaths:
			print("run the getPaths method first")
			return
		arr = self._originalPaths
		for i in range(len(arr)):
			#[int(i) for i in obj._originalPaths[100].split(' ')]
			self.__IntPaths.append([int(j) for j in arr[i].split('\t')])
		return self.__IntPaths

	def misconfiguredPaths(self):
		if len(self._originalPaths) == 0:
			print("run the getPaths method first")
		how_many = int(0.1*len(self._originalPaths))
		random_paths = np.random.choice(self._originalPaths,how_many,replace = False)
		for i in range(len(random_paths)):
			splits = random_paths[i].split(' ')
			if len(splits) >=2:
				self._misconfiguredPaths.append(' '.join(random.sample(splits,len(splits))))
		print(self._misconfiguredPaths)

	def hijackedPaths(self):
		if len(self._originalPaths) == 0:
			print("run the getPaths method first")
		if len(self._originalPaths) == 0:
			print("run the getPaths method first")
		how_many = int(0.1*len(self._originalPaths))
		random_paths = np.random.choice(self._originalPaths,how_many,replace = False)
		#print(how_many)
		for i in range(len(random_paths)):
			if len(random_paths[i].split(' '))> 3:
				self._hijackedPaths.append(random_paths[i]+" "+self._originalPaths[100].split(' ')[1])
	def variedLengths(self):
		if len(self._originalPaths) == 0:
			print("run the getPaths method first")
		pass
	def showAllPaths(self):
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
		#write with same filename + fake
		#all arrays
		if len(self._hijackedPaths) == 0:
			self.hijackedPaths()
		if len(self._misconfiguredPaths) == 0:
			self.misconfiguredPaths()
		self.__combined__ = self._hijackedPaths + self._misconfiguredPaths
		f = open(self.fname+"_f.txt",'a')
		for i in self.__combined__:
			f.write(i+"\n")
		f.close()
		return

#obj = pathExtractor('24012018_.txt')
#obj.getInts() < - correctly fetches error asking to call getPaths first
#obj.getPaths()
#tw = open('24012018.txt','w')
#for i in range(len(obj._originalPaths)):
#	tw.write(obj._originalPaths[i]+"\n")
obj = pathExtractor('11012018.txt')
retPaths = obj.getPaths()
hjp = obj.hijackedPaths()
obj.writePaths()