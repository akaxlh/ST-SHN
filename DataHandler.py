import pickle
import numpy as np
from scipy.sparse import csr_matrix
from Params import args
import scipy.sparse as sp
from Utils.TimeLogger import log

class DataHandler:
	def __init__(self):
		if args.data == 'NYC':
			predir = 'Datasets/NYC_crime/'
		elif args.data == 'CHI':
			predir = 'Datasets/CHI_crime/'
		elif args.data == 'NYC_1kilo':
			predir = 'Datasets/NYC_crime/1kilo/'
		self.predir = predir
		with open(predir + 'trn.pkl', 'rb') as fs:
			trnT = pickle.load(fs)
		with open(predir + 'val.pkl', 'rb') as fs:
			valT = pickle.load(fs)
		with open(predir + 'tst.pkl', 'rb') as fs:
			tstT = pickle.load(fs)
		args.row, args.col, _, args.offNum = trnT.shape
		args.areaNum = args.row * args.col
		args.trnDays = trnT.shape[2]
		args.valDays = valT.shape[2]
		args.tstDays = tstT.shape[2]
		args.decay_step = args.trnDays//args.batch
		self.mean = np.mean(trnT)
		self.std = np.std(trnT)
		rspFunc = (lambda tensor: np.reshape(tensor, [args.areaNum, -1, args.offNum]))
		self.trnT = rspFunc(trnT)# row*col, days, offNum
		self.valT = rspFunc(valT)
		self.tstT = rspFunc(tstT)

		self.constructGraph()
		self.getTestAreas()
		print('Row:', args.row, ', Col:', args.col)
		print('Sparsity:', np.sum(trnT!=0) / np.reshape(trnT, [-1]).shape[0])

	@classmethod
	def idEncode(cls, x, y):
		return x * args.col + y

	@classmethod
	def idDecode(cls, node):
		return node // args.col, node % args.col

	def zScore(self, data):
		# return np.log2(data + 1)
		return (data - self.mean) / self.std

	def zInverse(self, data):
		return data * self.std + self.mean

	def constructGraph(self):
		mx = [-1, 0, 1, 0, -1, -1, 1, 1, 0]
		my = [0, -1, 0, 1, -1, 1, -1, 1, 0]
		def illegal(x, y):
			return x < 0 or y < 0 or x >= args.row or y >= args.col
		edges = list()
		for i in range(args.row):
			for j in range(args.col):
				n1 = self.idEncode(i, j)
				for k in range(len(mx)):
					temx = i + mx[k]
					temy = j + my[k]
					if illegal(temx, temy):
						continue
					n2 = self.idEncode(temx, temy)
					edges.append([n1, n2])
		edges.sort(key=lambda x: x[0]*1e5+x[1]) # 1e5 should be bigger than the number of areas
		rowTot, colTot = [[0] * args.areaNum for i in range(2)]
		for e in range(len(edges)):
			rowTot[edges[e][0]] += 1
			colTot[edges[e][1]] += 1
		vals = np.ones(len(edges))
		for e in range(len(vals)):
			vals[e] /= np.sqrt(rowTot[edges[e][0]] * colTot[edges[e][1]])
		edges = np.array(edges)
		self.rows = edges[:, 0]
		self.cols = edges[:, 1]
		self.vals = vals

	def getTestAreas(self):
		posTimes = np.sum(1 * (self.trnT!=0), axis=1)
		percent = posTimes / args.trnDays
		self.tstLocs = (percent > 0.2) * (percent < 0.8) * 1
		print('Negative/Positive Rate', args.negRate)
		print('Number of locations to test', np.sum(self.tstLocs), 'out or', self.trnT.shape[0])
		valRes = np.sum(np.sum(self.valT==0, axis=1) * self.tstLocs) / (np.sum(self.tstLocs) * args.valDays)
		tstRes = np.sum(np.sum(self.tstT==0, axis=1) * self.tstLocs) / (np.sum(self.tstLocs) * args.tstDays)
		print('Val Trivial Acc', valRes)
		print('Tst Trivial Acc', tstRes)
