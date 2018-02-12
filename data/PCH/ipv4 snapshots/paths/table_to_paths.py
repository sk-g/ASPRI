import os,time,sys
files = os.listdir()
class pathExtractor(object):
	def __init__(self,strx):
		self.fname = str(strx)
	
		self._misconfiguredPaths = []
		self._hijackedPaths = []
		self._variedLengths = []
		self._originalPaths = []
		self.__combined__ = []
	def misconfiguredPaths(self,strx):
		pass
		#should take entire file or line by line?
	def hijackedPaths(self,strx):
		pass
	def variedLengths(self,strx):
		pass
	def showAllPaths(self,strx):
		#call this after finishing up above three functions
		#otherwise,legnth overflow
		# for i in range
		# print originalpaths[i],etc[i]
		pass
	def writePaths(self):
		#write with same filename
		#all arrays
		pass
	def getPaths(self):
		#strx is the absolute path of file
		fname = self.fname
		f = open(fname)
		self._lines = f.readlines()
		for i in range(len(self._lines)):
			self._originalPaths.append(str(' '.join([int(j) for j in self._lines[i].strip().split(' ')[1:]])))
		return self._originalPaths