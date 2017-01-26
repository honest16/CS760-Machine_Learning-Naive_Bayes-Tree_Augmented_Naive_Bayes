# Importing libraries
import sys
import math
import random
import collections
from collections import deque
import scipy
from scipy.io import arff
import numpy as np

class dataContents(object):
	def __init__(self):
		self.attributes = None
		self.instances = None

def getContents(filename):	
	f = open(str(filename),'r')
	data, meta = arff.loadarff(f)
	f.close()
	contents = dataContents()
	contents.instances = data
	contents.attributes = []
	
	
	# Names of the attributes
	for i in range(len(data.dtype.names)):
		contents.attributes.append([data.dtype.names[i]])
		
	
	metaSpl = str(meta).split('\n')
	attrInd = 0
	for i in range(len(metaSpl)):
		if 'type' in metaSpl[i] and not 'range' in metaSpl[i]:
			for j in range(len(metaSpl[i])):
				k = len(metaSpl[i])-1-j
	
				if metaSpl[i][k] == ' ':
					 contents.attributes[attrInd].append(metaSpl[i][k+1:])
					 contents.attributes[attrInd].append(None)
					 attrInd += 1
					 break
		elif  'type' in metaSpl[i] and 'range' in metaSpl[i]:
			furthSpl = metaSpl[i].split(',')
			
			for j in range(len(furthSpl[0])):
				k = len(furthSpl[0])-1-j
				if furthSpl[0][k] == ' ':
					 contents.attributes[attrInd].append(furthSpl[0][k+1:])
					 break
					 
			rangeVals = []
	
			rangeStIn = metaSpl[i].index('(')
			rangeEnIn = metaSpl[i].index(')')
			vals = metaSpl[i][rangeStIn+1:rangeEnIn].split(',')
			rangeVals = [vals[i].strip()[1:-1] for i in range(len(vals))]
			contents.attributes[attrInd].append(rangeVals)
			attrInd += 1
	
	
	return contents
	
	
	
	
class bayes(object):
	def __init__(self):
		
		self.train_set = None
		self.test_set = None
		self.netw = None
		self.probs = None
		
		self.mutualInf = None
		
		self.retVal = []
		self.Vnew = None
		self.Enew = None
		self.dirGr = None
		self.parentGr = None
		
		self.cpts = None
		
		self.pseudoCnt = 1.0
		self.obtainCLIArgs()
		
		if self.netw == 'n':
			self.NaiveBayes()
		if self.netw == 't':
			self.TAN()
		
	# Method to obtain data from command line arguments
	def obtainCLIArgs(self):
		
		trainFileName = sys.argv[1]
		testFileName = sys.argv[2]
		self.netw = sys.argv[3]
		self.train_set = getContents(trainFileName)  
		self.test_set = getContents(testFileName)
		
		
	# Method to implement Naive Bayes
	def NaiveBayes(self):
		
		self.estimateProbs()
		self.classifyTestIns()
		
	
	
	
	# Method to implement TAN
	def TAN(self):
		
		self.MutualInfo()
		self.MaxSpanningTree()
		self.AddY()
		self.estimateProbs()
		self.classifyTestIns()
		
		

	def AddY(self):
		self.Vnew.add(len(self.train_set.attributes)-1)
		self.dirGr[len(self.train_set.attributes)-1] = []
		for fi in range(len(self.train_set.attributes)-1):
			self.dirGr[len(self.train_set.attributes)-1].append(fi)
			edg = (fi,len(self.train_set.attributes)-1)
			self.Enew.add(edg)
			
			
		
	
	# Maximum spanning tree
	def MaxSpanningTree(self):
		
		Vnew = set([])
		Vnew.add(0)
		Enew = set([])
		dirGr = {}
		parentGr = {}
		
		while len(Vnew) != len(self.train_set.attributes)-1:
			
			maxDist = -10000
			dists = {}
			
			for u in Vnew:
				
				for v in range(len(self.train_set.attributes)-1):
					if v == u or v in Vnew:
						continue
					else:
						
						if u < v:
							tup = (u,v)
						else:
							tup = (v,u)
						
						ind = self.IJlocMap[tup]
						d = self.mutualInf[ind]
						if d > maxDist:
						
							if maxDist in dists:
								del dists[maxDist]
								
							e1 = set([self.ReverseLocIJMap[ind]])
							try:
								dists[d] = dists[d].union(e1)
							except KeyError:
								dists[d] = e1
							
							maxDist = d
						
								
						elif d == maxDist:
						
							e1 = set([self.ReverseLocIJMap[ind]])
							try:
								dists[maxDist] = dists[maxDist].union(e1)
							except KeyError:
								dists[maxDist] = e1
								
						
						elif d < maxDist:
						
							continue
			
		
			if len(dists[maxDist]) == 1:
				
				for item in dists[maxDist]:
					newEdg = item
				
				if newEdg[0] in Vnew:
					Vnew.add(newEdg[1])
					try: 
						dirGr[newEdg[0]].append(newEdg[1])
					except KeyError:
						dirGr[newEdg[0]] = [newEdg[1]]
					parentGr[newEdg[1]] = newEdg[0]	
					
				elif newEdg[1] in Vnew:
					Vnew.add(newEdg[0])
					try:
						dirGr[newEdg[1]].append(newEdg[0])
					except KeyError:
						dirGr[newEdg[1]] = [newEdg[0]]
					parentGr[newEdg[0]] = newEdg[1]	
						
				Enew.add(newEdg)
					
					
			elif len(dists[maxDist]) > 1:
				
				relevantEdges = dists[maxDist]
				lstdFir = sys.maxint
				
				for item in relevantEdges:
					if self.IJlocMap[item] < lstdFir:
						lstdFir = self.IJlocMap[item]
					
				newEdg = self.ReverseLocIJMap[lstdFir]
				
				if newEdg[0] in Vnew:
					Vnew.add(newEdg[1])
					try:
						dirGr[newEdg[0]].append(newEdg[1])
					except KeyError:
						dirGr[newEdg[0]] = [newEdg[1]]
					parentGr[newEdg[1]] = newEdg[0]
					
				elif newEdg[1] in Vnew:
					Vnew.add(newEdg[0])	
					try:
						dirGr[newEdg[1]].append(newEdg[0])
					except KeyError:
						dirGr[newEdg[1]] = [newEdg[0]]
					parentGr[newEdg[1]] = newEdg[0]	
				Enew.add(newEdg)
				
			
		
		self.Vnew = Vnew
		self.Enew = Enew
		self.dirGr = dirGr
		self.parentGr = parentGr
		
	
	# Method to estimate class probability	
	def estimateProbs(self):
		if self.netw == 'n':
			pYey = []
			pXexGYey = []
	
			for i in range(len(self.train_set.attributes[-1][-1])):
				pYey.append(0)
		
		
			for att in range(len(self.train_set.attributes)-1):
				pX = []			
				for atVal in range(len(self.train_set.attributes[att][-1])): 
					a = pYey[:]
					pX.append(a)				
				pXexGYey.append(pX)

		
			# Loop over training instances
			for trIn in range(len(self.train_set.instances)):
				yInd = self.train_set.attributes[-1][-1].index(self.train_set.instances[trIn][-1])
				pYey[yInd] += 1
				for X in range(len(self.train_set.instances[trIn])-1):
					x = self.train_set.instances[trIn][X]
					xind = self.train_set.attributes[X][-1].index(x)
					pXexGYey[X][xind][yInd] += 1

		
			for j in range(len(pYey)):
				for X in range(len(self.train_set.instances[trIn])-1):
					for xind in range(len(pXexGYey[X])):
						pXexGYey[X][xind][j] = ((pXexGYey[X][xind][j] + self.pseudoCnt)*1.0)/(pYey[j] + (self.pseudoCnt*len(pXexGYey[X])))
			
			for i in range(len(pYey)):
				pYey[i] = ((pYey[i] + self.pseudoCnt)*1.0)/(len(self.train_set.instances)+ (self.pseudoCnt*len(pYey)))
			
			
			self.probs = [pYey, pXexGYey]
			
		elif self.netw == 't':
			
			self.cpts = []
			for i in range(len(self.train_set.attributes)):
				self.cpts.append([])
			
			
			
			
			for j in range(1,len(self.train_set.attributes)-1):
				parentId = self.parentGr[j]
			
				for parValId in range(len(self.train_set.attributes[parentId][2])):
					parValList = []
					for yInd in range(len(self.train_set.attributes[-1][2])):
						attrValList = []
						for attrInd in range(len(self.train_set.attributes[j][2])):
							attrValList.append(0)
						parValList.append(attrValList)
					self.cpts[j].append(parValList)
			
			
			for j in range(len(self.train_set.attributes[-1][2])):
				self.cpts[-1].append(0)
				attrValList = []
				for attrInd in range(len(self.train_set.attributes[0][2])):
					attrValList.append(0)
				self.cpts[0].append(attrValList)
			
			
			for trInd in range(len(self.train_set.instances)):
				ins = self.train_set.instances[trInd]
				ylabel = ins[-1]
				ylabelInd = self.train_set.attributes[-1][2].index(ylabel)
				
				self.cpts[-1][self.train_set.attributes[-1][2].index(ins[-1])] += 1
				
			
				self.cpts[0][ylabelInd][self.train_set.attributes[0][2].index(ins[0])]	+= 1
				
			
				
				for fIn in range(1,len(self.train_set.attributes)-1):
					fVal = ins[fIn]
					fValIn = self.train_set.attributes[fIn][2].index(fVal)
				
					pFInd = self.parentGr[fIn]
					pFVal = ins[pFInd]
					pFValInd = self.train_set.attributes[pFInd][2].index(pFVal)
				
					self.cpts[fIn][pFValInd][ylabelInd][fValIn] += 1
				
			for k in range(len(self.cpts[-1])):
				self.cpts[-1][k] += self.pseudoCnt
				self.cpts[-1][k] /= (len(self.train_set.instances) + self.pseudoCnt*len(self.train_set.attributes[-1][2]))
				
				sumgY = sum(self.cpts[0][k])
				for fvalIn in range(len(self.train_set.attributes[0][2])):
					self.cpts[0][k][fvalIn] += self.pseudoCnt
					self.cpts[0][k][fvalIn] /= (sumgY + self.pseudoCnt*len(self.train_set.attributes[0][2]))


			for fId in range(1,len(self.train_set.attributes)-1):
				for pfId in range(len(self.cpts[fId])):
					for yId in range(len(self.train_set.attributes[-1][2])):
						sumgPY = sum(self.cpts[fId][pfId][yId])
						
						for fvalIn in range(len(self.cpts[fId][pfId][yId])):
							self.cpts[fId][pfId][yId][fvalIn] += self.pseudoCnt
							self.cpts[fId][pfId][yId][fvalIn] /= ( sumgPY + self.pseudoCnt*len(self.train_set.attributes[fId][2]) )
							
							
			
			
		
	def MutualInfo(self):
		if self.netw == 't':
			
			IndCount = []
			
			for i in range(len(self.train_set.attributes)-1):
				attr = []
				for j in range(len(self.train_set.attributes[i][-1])):
					attrVal = []
					for k in range(len(self.train_set.attributes[-1][-1])):
						attrVal.append(0)	
					attr.append(attrVal)	
				IndCount.append(attr)
				
			
			
			# Pair wise counts
			PairCount = []
			IJlocMap = {}
			ReverseLocIJMap = {}
			index = 0
			
			
			# Make (i,j) pairs
			for i in range(len(self.train_set.attributes)-2):
				for j in range(i+1,len(self.train_set.attributes)-1):
					IJlocMap[(i,j)] =  index
					ReverseLocIJMap[index] = (i,j)
					
					thisPair = []
					
					for ivalInd in range(len(self.train_set.attributes[i][-1])):
						ival = []
			
						
						for jvalInd in range(len(self.train_set.attributes[j][-1])):
							jval = []
							for yvalInd in range(len(self.train_set.attributes[-1][-1])):
								jval.append(0)
							ival.append(jval)
							
						thisPair.append(ival)
					
					PairCount.append(thisPair)
					index += 1
						
			
			
			for trInInd in range(len(self.train_set.instances)):
				ins = self.train_set.instances[trInInd]
				yval = ins[-1]
				yvalInd = self.train_set.attributes[-1][-1].index(yval)
				for fi in range(len(ins)-1):
					fval = ins[fi]
					fvalInd = self.train_set.attributes[fi][-1].index(fval)
					IndCount[fi][fvalInd][yvalInd] += 1
				
				
			
				for fi in range(len(ins)-2):
					fiVal = ins[fi]	
					fiValInd = 	self.train_set.attributes[fi][-1].index(fiVal)
					for fj in range(fi+1,len(ins)-1):
						fjVal = ins[fj]
						fjValInd = self.train_set.attributes[fj][-1].index(fjVal)
						 
						PairCount[IJlocMap[(fi,fj)]][fiValInd][fjValInd][yvalInd] += 1 
						
			numInsYp = 0
			numInsYn = 0
			for p in range(len(IndCount[-1])):
				numInsYp += IndCount[-1][p][0]
				numInsYn += IndCount[-1][p][1]
			
			
			self.mutualInf = []
			for fi in range(len(ins)-2):
			
				for fj in range(fi+1,len(ins)-1): 
					IXiXjgY = [0]
					self.mutualInf.append(IXiXjgY)
			
			for pairInd in range(len(PairCount)):
				mij = 0
				xi = ReverseLocIJMap[pairInd][0]
				xj = ReverseLocIJMap[pairInd][1]
				for xiInd in range(len(self.train_set.attributes[xi][-1])):
					for xjInd in range(len(self.train_set.attributes[xj][-1])):
						for yInd in range(len(self.train_set.attributes[-1][-1])):
							Pxixjy =  (PairCount[pairInd][xiInd][xjInd][yInd] + self.pseudoCnt)/(len(self.train_set.instances) + (self.pseudoCnt*len(self.train_set.attributes[xi][-1])*len(self.train_set.attributes[xj][-1])*len(self.train_set.attributes[-1][-1])))
							NumInsy = 0
							for p in range(len(IndCount[-1])):
								NumInsy += IndCount[-1][p][yInd]
								
							Pxixjgy = (PairCount[pairInd][xiInd][xjInd][yInd] + self.pseudoCnt)/(NumInsy + (self.pseudoCnt*len(self.train_set.attributes[xi][-1])*len(self.train_set.attributes[xj][-1])))
							
			
							Numxigy = IndCount[xi][xiInd][yInd]
								
							Pxigy = (Numxigy + self.pseudoCnt)/(NumInsy + (self.pseudoCnt*len(self.train_set.attributes[xi][-1])))
							
			
							Numxjgy = IndCount[xj][xjInd][yInd]
							
							Pxjgy = (Numxjgy + self.pseudoCnt)/(NumInsy + (self.pseudoCnt*len(self.train_set.attributes[xj][-1])))
							
							mij += Pxixjy*(math.log((Pxixjgy/(Pxigy*Pxjgy)),2))	
							
				self.mutualInf[pairInd] = mij 
			
			
			
		self.IJlocMap = IJlocMap.copy()	
		self.ReverseLocIJMap = ReverseLocIJMap.copy()
		
		
	# Method to clasify test instances	
	def classifyTestIns(self):
		if self.netw == 'n':
			for ai in range(len(self.train_set.attributes)-1):
				print self.train_set.attributes[ai][0] + ' ' + 'class'
			print ''
			
			numCorrClass = 0
			pYey = self.probs[0]
			pXexGYey = self.probs[1]
			for tstInsInd in range(len(self.test_set.instances)):
		
				tstIns = self.test_set.instances[tstInsInd]
				actLab = self.test_set.instances[tstInsInd][-1]
			
				labPrCFeat = []
				PXgY =[]
				PyTimesPXgY = []
				for i in range(len(pYey)):
					labPrCFeat.append(0)
					PXgY.append(0)
					PyTimesPXgY.append(0)
			
				for i in range(len(PXgY)):
					prod = 1
					for j in range(len(pXexGYey)):
						prod *= pXexGYey[j][self.train_set.attributes[j][-1].index(tstIns[j])][i]
					PXgY[i] = prod
			
				for i in range(len(PyTimesPXgY)):
					PyTimesPXgY[i] = pYey[i]*PXgY[i]
			
				for i in range(len(labPrCFeat)):
					labPrCFeat[i] = (PyTimesPXgY[i]*1.0)/sum(PyTimesPXgY)
				
				
				predLab = self.test_set.attributes[-1][-1][labPrCFeat.index(max(labPrCFeat))]
				print predLab.replace("'", "")+ ' '+ actLab.replace("'", "") + ' '+ 	'%.12f'%max(labPrCFeat)#, labPrCFeat
				if predLab == actLab:
					numCorrClass += 1
			print ''
			print numCorrClass
			
		elif self.netw == 't':
			print self.train_set.attributes[0][0] + ' ' + 'class'
			
			for ai in range(1,len(self.train_set.attributes)-1):
				print self.train_set.attributes[ai][0] + ' ' + self.train_set.attributes[self.parentGr[ai]][0] + 'class'
			print ''
			
			numCorrClass = 0
		
			for tstInd in range(len(self.test_set.instances)):
				ins = self.test_set.instances[tstInd]
				PygxList = []
				for yind in range(len(self.train_set.attributes[-1][2])):
					Pxgy = 1.0*self.cpts[0][yind][self.train_set.attributes[0][2].index(ins[0])]
					
					q = deque([])
				
					q.append(self.dirGr[0][0])
					while q:
						feat = q.popleft()
				
						if feat in self.dirGr:
							for child in self.dirGr[feat]:
								q.append(child)
						
						featVal = ins[feat]
						featValInd = self.train_set.attributes[feat][2].index(featVal)
						
						par = self.parentGr[feat]
						parVal = ins[par]
						parValInd = self.train_set.attributes[par][2].index(parVal)
						
						Pxgy *= self.cpts[feat][parValInd][yind][featValInd]
						
						
					PyTPxgy = self.cpts[-1][yind]*Pxgy
					PygxList.append(PyTPxgy)
				
				Px = sum(PygxList)
				
				for i in range(len(PygxList)):
					PygxList[i] /= (1.0*Px) 
					
				predLab = self.test_set.attributes[-1][-1][PygxList.index(max(PygxList))]
				
				actLab = ins[-1]
				print predLab.replace("'", "")+ ' '+ actLab.replace("'", "") + ' '+ 	'%.12f'%max(PygxList)
				if predLab == actLab:
					numCorrClass += 1
			print ''
			print numCorrClass
				
				
				
				
bayes()		
	
	