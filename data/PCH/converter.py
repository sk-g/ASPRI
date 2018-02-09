import pandas as pd
import numpy as np
#import tensorflow as tf
#import keras
import sys,os,time,random,math,itertools
from collections import *
from sklearn.metrics import *

class converter(object):
	def __init__(self,path):
		self.__path__ = path
		self.__fname__ = str(path)+".txt"
		self.__filehandle = open(str(path))
		self._lines = self.__filehandle.readlines()
		self.__rawlines = [i.strip() for i in self._lines ]
		self._content = self.__rawlines[6:-2]
		self.next_hops = []
		self.prefixes = []
		self.paths = []
		self.protocols = []		
		

	def drawProgressBar(self,percent, barLen = 50):
		sys.stdout.write("\r")
		progress = ""
		for i in range(barLen):
			if i<int(barLen * percent):
				progress += "="
			else:
				progress += " "
		sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
		sys.stdout.flush()
	def counters(self):
		prefix_counter = 0
		print('\npreprocessing')
		itercounter = 0
		current_prefix = ""
		for line in self._content:
			#print(line)
			splits = line.split(' ')
			#print(line,splits)
			dot_counter = 0
			self.next_hop,c = 0,0
			self.prefix = ""
			self.path = ""
			c = 0
			path_ = [] # for every line there is a path which could be multiple
			for split in splits:
				if split.count('.') == 3:
					dot_counter += 1
			#print(line,dot_counter)

			if dot_counter == 2:
				#prefix_counter += 1
				for split in splits:
					if len(split) != 0:
						c += 1
						if c ==2:

							#print(line+"this is prefix:\t",split)
							
							#if self.prefixes[-1] != split:
							self.prefix,current_prefix = split,split
							#else:
						if split.count('.') != 0 and c == 3:
							self.next_hop = split
							#print("this is the next_hop:\t",split)
						if c != 1 and split.count('.') == 0 and len(split)>1:
							path_.append(split)
				self.path = " ".join([i for i in path_])
				#print('{0}: \n\t the prefix is: {1} \n\t the next hop is: {2}\n\tpath is {3}'.format(line,self.prefix,self.next_hop,self.path))
				self.next_hops.append(self.next_hop)
				self.prefixes.append(self.prefix)
				self.paths.append(self.path)
				self.protocols.append(splits[-1])
				itercounter += 1
				#print("in dot counter 2, number of prefixes =",prefix_counter)
				self.drawProgressBar(itercounter/len(self._content))
			if dot_counter == 1:
				#prefix_counter += 1
				self.prefixes.append(current_prefix)
				for split in splits:
					if len(split) != 0:
						c+=1
					if split.count('.') != 0:
						self.next_hop = split
						#print(split)
						#print("this is the next_hop:\t",split)
					if c != 1 and split.count('.') == 0 and len(split)>1:
						path_.append(split)
				self.path = " ".join([i for i in path_])
				#print('{0}: \n\t the prefix is: {1} \n\t the next hop is: {2}\n\tpath is {3}'.format(line,self.prefix,self.next_hop,self.path))
				self.next_hops.append(self.next_hop)
				self.paths.append(self.path)
				self.protocols.append(splits[-1])
				itercounter += 1
				self.drawProgressBar(itercounter/len(self._content))
		#print("in dot counter = 1, number of prefixes =",prefix_counter)				
		print(len(self.prefixes),len(self.next_hops),len(self.paths),len(self.protocols))
		return (self.prefixes,self.next_hops,self.paths,self.protocols)
	def writer(self):
		if len(self.prefixes) != 0:
			#print("\n",len(self.prefixes))
			writer = open(self.__fname__,"w")
			for i in range(len(self.prefixes)):
				self.drawProgressBar(i/len(self.protocols))
				writer.write(self.prefixes[i]+'\t'+self.next_hops[i]+'\t'+self.paths[i]+'\n')
			writer.close()
		if len(self.prefixes)==0:
			a,b,c,d = self.counters()#self.__path__)
			print("\nwriting a new file: ",self.__fname__)
			#print(len(a),len(b))
			self.writer()
def main(strx):
	path = str(strx)
	if __name__ == '__main__':
		#obj = converter(r'M:\Course stuff\ASPRI\Route Collector\route-collector.akl.pch.net-ipv4_bgp_routes.2017.12.28')
		print(path.split(os.sep)[-1])
		obj = converter(path)
		#prefix, next_hop,path,protocol = obj.counters() #fixed to call within writer block
		obj.writer()
		#print("\n",len(prefix),len(next_hop),len(path),len(protocol))
os.chdir(r"M:\Course stuff\ASPRI\data\PCH\ipv4 snapshots")
files = os.listdir()
for file in files:
	if os.path.splitext(file)[1].strip() != str('.py') and os.path.splitext(file)[1].strip() != '.txt':
		#print(os.path.splitext(file)[1])
		main(file)

#temp testing
#main(r"M:\Course stuff\ASPRI\data\PCH\ipv4 snapshots\route-collector.abj.pch.net-ipv4_bgp_routes.2017.11.30")