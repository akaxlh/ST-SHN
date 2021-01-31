import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from Params import args
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
import Utils.NNLayers as NNs
from Utils.NNLayers import FC, Regularize, Activate, Bias, defineParam, defineRandomNameParam
from DataHandler import DataHandler
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle

class Model:
	def __init__(self, sess, handler):
		self.sess = sess
		self.handler = handler

		self.metrics = dict()
		mets = ['preLoss', 'microF1', 'macroF1']
		for i in range(args.offNum):
			mets.append('F1_%d' % i)
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')
		if args.load_model != None:
			self.loadModel()
			stloc = len(self.metrics['TrainpreLoss']) * args.tstEpoch
		else:
			stloc = 0
			init = tf.global_variables_initializer()
			self.sess.run(init)
			log('Variables Inited')
		bestRes = None
		for ep in range(stloc, args.epoch):
			test = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, test))
			if test:
				reses = self.testEpoch(self.handler.tstT, np.concatenate([self.handler.trnT, self.handler.valT], axis=1))
				if bestRes is None or args.task == 'r' and bestRes['MAPE'] > reses['MAPE'] or args.task == 'c' and bestRes['macroF1'] > reses['macroF1']:
					bestRes = reses
			if ep % args.tstEpoch == 0:
				self.saveHistory()
			print()
		reses = self.testEpoch(self.handler.tstT, np.concatenate([self.handler.trnT, self.handler.valT], axis=1))
		log(self.makePrint('Test', args.epoch, reses, True))
		if bestRes is None or args.task == 'r' and bestRes['MAPE'] > reses['MAPE'] or args.task == 'c' and bestRes['macroF1'] > reses['macroF1']:
			bestRes = reses
		log(self.makePrint('Best', args.epoch, bestRes, True))
		self.saveHistory()

	def spacialModeling(self, rows, cols, vals, embeds):
		# edge, time, offense, latdim
		rowEmbeds = tf.nn.embedding_lookup(embeds, rows)
		colEmbeds = tf.nn.embedding_lookup(embeds, cols)
		Q = defineRandomNameParam([args.latdim, args.latdim], reg=False)
		K = defineRandomNameParam([args.latdim, args.latdim], reg=False)
		V = defineRandomNameParam([args.latdim, args.latdim], reg=False)
		q = tf.reshape(tf.einsum('etod,dl->etol', rowEmbeds, Q), [-1, args.temporalRange, args.offNum, 1, args.head, args.latdim//args.head])
		k = tf.reshape(tf.einsum('etod,dl->etol', colEmbeds, K), [-1, args.temporalRange, 1, args.offNum, args.head, args.latdim//args.head])
		v = tf.reshape(tf.einsum('etod,dl->etol', colEmbeds, V), [-1, args.temporalRange, 1, args.offNum, args.head, args.latdim//args.head])
		att = tf.nn.softmax(tf.reduce_sum(q * k, axis=-1, keep_dims=True) / tf.sqrt(float(args.latdim//args.head)), axis=3)
		attV = tf.reshape(tf.reduce_sum(att * v, axis=3), [-1, args.temporalRange, args.offNum, args.latdim])
		ret = tf.math.segment_sum(attV * tf.nn.dropout(vals, rate=self.dropRate), rows)
		return Activate(ret, 'leakyRelu') # area, time, offense, latdim

	def temporalModeling(self, rows, cols, vals, embeds):
		prevTEmbeds = tf.slice(embeds, [0, 0, 0, 0], [-1, args.temporalRange-1, -1, -1])
		nextTEmbeds = tf.slice(embeds, [0, 1, 0, 0], [-1, args.temporalRange-1, -1, -1])
		rowEmbeds = tf.nn.embedding_lookup(nextTEmbeds, rows)
		colEmbeds = tf.nn.embedding_lookup(prevTEmbeds, cols)
		Q = defineRandomNameParam([args.latdim, args.latdim], reg=False)
		K = defineRandomNameParam([args.latdim, args.latdim], reg=False)
		V = defineRandomNameParam([args.latdim, args.latdim], reg=False)
		q = tf.reshape(tf.einsum('etod,dl->etol', rowEmbeds, Q), [-1, args.temporalRange-1, args.offNum, 1, args.head, args.latdim//args.head])
		k = tf.reshape(tf.einsum('etod,dl->etol', colEmbeds, K), [-1, args.temporalRange-1, 1, args.offNum, args.head, args.latdim//args.head])
		v = tf.reshape(tf.einsum('etod,dl->etol', colEmbeds, V), [-1, args.temporalRange-1, 1, args.offNum, args.head, args.latdim//args.head])
		att = tf.nn.softmax(tf.reduce_sum(q * k, axis=-1, keep_dims=True) / tf.sqrt(float(args.latdim//args.head)), axis=3)
		attV = tf.reshape(tf.reduce_sum(att * v, axis=3), [-1, args.temporalRange-1, args.offNum, args.latdim])
		ret = tf.math.segment_sum(attV * tf.nn.dropout(vals, rate=self.dropRate), rows)
		ret = tf.concat([tf.slice(embeds, [0, 0, 0, 0], [-1, 1, -1, -1]), ret], axis=1)
		return Activate(ret, 'leakyRelu') # area, time, offense, latdim

	def hyperGNN(self, adj, embeds):
		tpadj = tf.transpose(adj)
		hyperEmbeds = Activate(tf.einsum('hn,ntod->htod', tf.nn.dropout(adj, rate=self.dropRate), embeds), 'leakyRelu')
		retEmbeds = Activate(tf.einsum('nh,htod->ntod', tf.nn.dropout(tpadj, rate=self.dropRate), hyperEmbeds), 'leakyRelu')
		return retEmbeds

	def ours(self):
		offenseEmbeds = defineParam('offenseEmbeds', [1, 1, args.offNum, args.latdim], reg=False)
		initialEmbeds = offenseEmbeds * tf.expand_dims(self.feats, axis=-1) # area, time, offense, latdim
		areaEmbeds = defineParam('areaEmbeds', [args.areaNum, 1, 1, args.latdim], reg=False)
		embeds = [initialEmbeds]# + areaEmbeds]
		for l in range(args.spacialRange):
			embed = embeds[-1]
			embed = self.spacialModeling(self.rows, self.cols, self.vals, embed)
			embed = self.hyperGNN(self.hyperAdj, embed) + embed
			embeds.append(embed)
		embed = tf.add_n(embeds) / args.spacialRange
		embeds = [embed]
		for l in range(args.temporalGnnRange):
			embeds.append(self.temporalModeling(self.rows, self.cols, self.vals, embeds[-1]))
		embed = tf.add_n(embeds) / args.temporalGnnRange
		embed = tf.reduce_mean(embed, axis=1) # area, offense, latdim
		W = defineParam('predEmbeds', [1, args.offNum, args.latdim], reg=False)
		if args.task == 'c':
			allPreds = Activate(tf.reduce_sum(embed * W, axis=-1), 'sigmoid') # area, offense
		elif args.task == 'r':
			allPreds = tf.reduce_sum(embed * W, axis=-1)
		return allPreds, embed

	def prepareModel(self):
		self.rows = tf.constant(self.handler.rows)
		self.cols = tf.constant(self.handler.cols)
		self.vals = tf.reshape(tf.constant(self.handler.vals, dtype=tf.float32), [-1, 1, 1, 1])
		self.hyperAdj = defineParam('hyperAdj', [args.hyperNum, args.areaNum], reg=True)
		self.feats = tf.placeholder(name='feats', dtype=tf.float32, shape=[args.areaNum, args.temporalRange, args.offNum])
		self.dropRate = tf.placeholder(name='dropRate', dtype=tf.float32, shape=[])

		self.labels = tf.placeholder(name='labels', dtype=tf.float32, shape=[args.areaNum, args.offNum])
		self.preds, embed = self.ours()

		if args.task == 'c':
			posInd = tf.cast(tf.greater(self.labels, 0), tf.float32)
			negInd = tf.cast(tf.less(self.labels, 0), tf.float32)
			posPred = tf.cast(tf.greater_equal(self.preds, args.border), tf.float32)
			negPred = tf.cast(tf.less(self.preds, args.border), tf.float32)
			NNs.addReg('embed', embed * tf.expand_dims(posInd + negInd, axis=-1))
			self.preLoss = tf.reduce_sum(-(posInd * tf.log(self.preds + 1e-8) + negInd * tf.log(1 - self.preds + 1e-8))) / (tf.reduce_sum(posInd) + tf.reduce_sum(negInd))
			self.truePos = tf.reduce_sum(posPred * posInd, axis=0)
			self.falseNeg = tf.reduce_sum(negPred * posInd, axis=0)
			self.trueNeg = tf.reduce_sum(negPred * negInd, axis=0)
			self.falsePos = tf.reduce_sum(posPred * negInd, axis=0)
		elif args.task == 'r':
			self.mask = tf.placeholder(name='mask', dtype=tf.float32, shape=[args.areaNum, args.offNum])
			self.preLoss = tf.reduce_sum(tf.square(self.preds - self.labels) * self.mask) / tf.reduce_sum(self.mask)
			self.sqLoss = tf.reduce_sum(tf.square(self.preds - self.labels) * self.mask, axis=0)
			self.absLoss = tf.reduce_sum(tf.abs(self.preds - self.labels) * self.mask, axis=0)
			self.tstNums = tf.reduce_sum(self.mask, axis=0)
			posMask = self.mask * tf.cast(tf.greater(self.labels, 0.5), tf.float32)
			self.apeLoss = tf.reduce_sum(tf.abs(self.preds - self.labels) / (self.labels + 1e-8) * posMask, axis=0)
			self.posNums = tf.reduce_sum(posMask, axis=0)
			NNs.addReg('embed', embed * tf.expand_dims(self.mask, axis=-1))

		self.regLoss = args.reg * Regularize() + args.spreg * tf.reduce_sum(tf.abs(self.hyperAdj))
		self.loss = self.preLoss + self.regLoss

		globalStep = tf.Variable(0, trainable=False)
		learningRate = tf.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
		self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

	def sampleTrainBatch(self, batIds):
		idx = batIds[0]
		label = self.handler.trnT[:, idx, :]# area, offNum
		if args.task == 'c':
			negRate = args.negRate#np.random.randint(1, args.negRate*2)
		elif args.task == 'r':
			negRate = 0
		posNums = np.sum(label != 0, axis=0) * negRate
		retLabels = (label != 0) * 1
		if args.task == 'r':
			mask = retLabels
			retLabels = label
		for i in range(args.offNum):
			temMap = label[:, i]
			negPos = np.reshape(np.argwhere(temMap==0), [-1])
			sampedNegPos = np.random.permutation(negPos)[:posNums[i]]
			# sampedNegPos = negPos
			if args.task == 'c':
				retLabels[sampedNegPos, i] = -1
			elif args.task == 'r':
				mask[sampedNegPos, i] = 1
		feat = self.handler.trnT[:, idx-args.temporalRange: idx, :]
		if args.task == 'c':
			return self.handler.zScore(feat), retLabels
		elif args.task == 'r':
			return self.handler.zScore(feat), retLabels, mask

	def trainEpoch(self):
		ids = np.random.permutation(list(range(args.temporalRange, args.trnDays)))
		epochLoss, epochPreLoss, epochAcc = [0] * 3
		num = len(ids)

		steps = int(np.ceil(num / args.batch))
		for i in range(steps):
			st = i * args.batch
			ed = min((i+1) * args.batch, num)
			batIds = ids[st: ed]

			tem = self.sampleTrainBatch(batIds)
			if args.task == 'c':
				feats, labels = tem
			elif args.task == 'r':
				feats, labels, mask = tem

			targets = [self.optimizer, self.preLoss, self.loss]
			feeddict = {self.feats: feats, self.labels: labels, self.dropRate: args.dropRate}
			if args.task == 'r':
				feeddict[self.mask] = mask
			res = self.sess.run(targets, feed_dict=feeddict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

			preLoss, loss = res[1:]

			epochLoss += loss
			epochPreLoss += preLoss
			log('Step %d/%d: preLoss = %.4f         ' % (i, steps, preLoss), save=False, oneline=True)
		ret = dict()
		ret['Loss'] = epochLoss / steps
		ret['preLoss'] = epochPreLoss / steps
		return ret

	def sampTestBatch(self, batIds, tstTensor, inpTensor):
		idx = batIds[0]
		label = tstTensor[:, idx, :]# area, offNum
		if args.task == 'c':
			retLabels = ((label > 0) * 1 + (label == 0) * (-1)) * self.handler.tstLocs
		elif args.task == 'r':
			retLabels = label
			mask = self.handler.tstLocs * (label > 0)
		if idx - args.temporalRange < 0:
			temT = inpTensor[:, idx-args.temporalRange:, :]
			temT2 = tstTensor[:, :idx, :]
			feats = np.concatenate([temT, temT2], axis=1)
		else:
			feats = tstTensor[:, idx-args.temporalRange: idx, :]
		if args.task == 'c':
			return self.handler.zScore(feats), retLabels
		elif args.task == 'r':
			return self.handler.zScore(feats), retLabels, mask

	def testEpoch(self, tstTensor, inpTensor):
		ids = np.random.permutation(list(range(tstTensor.shape[1])))
		epochLoss, epochPreLoss,  = [0] * 2
		if args.task == 'c':
			epochTp, epochFp, epochTn, epochFn = [np.zeros(4) for i in range(4)]
		elif args.task == 'r':
			epochSqLoss, epochAbsLoss, epochTstNum, epochApeLoss, epochPosNums = [np.zeros(4) for i in range(5)]
		num = len(ids)

		steps = int(np.ceil(num / args.batch))
		for i in range(steps):
			st = i * args.batch
			ed = min((i+1) * args.batch, num)
			batIds = ids[st: ed]

			tem = self.sampTestBatch(batIds, tstTensor, inpTensor)
			if args.task == 'c':
				feats, labels = tem
			elif args.task == 'r':
				feats, labels, mask = tem

			if args.task == 'c':
				targets = [self.preLoss, self.regLoss, self.loss, self.truePos, self.falsePos, self.trueNeg, self.falseNeg]
				feeddict = {self.feats: feats, self.labels: labels, self.dropRate: 0.0}
			elif args.task == 'r':
				targets = [self.preds, self.preLoss, self.regLoss, self.loss, self.sqLoss, self.absLoss, self.tstNums, self.apeLoss, self.posNums]
				feeddict = {self.feats: feats, self.labels: labels, self.dropRate: 0.0, self.mask: mask}
			res = self.sess.run(targets, feed_dict=feeddict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
			if args.task == 'c':
				preLoss, regLoss, loss, tp, fp, tn, fn = res
				epochTp += tp
				epochFp += fp
				epochTn += tn
				epochFn += fn
			elif args.task == 'r':
				preds, preLoss, regLoss, loss, sqLoss, absLoss, tstNums, apeLoss, posNums = res
				epochSqLoss += sqLoss
				epochAbsLoss += absLoss
				epochTstNum += tstNums
				epochApeLoss += apeLoss
				epochPosNums += posNums
			epochLoss += loss
			epochPreLoss += preLoss
			log('Step %d/%d: loss = %.2f, regLoss = %.2f         ' % (i, steps, loss, regLoss), save=False, oneline=True)
		ret = dict()
		ret['preLoss'] = epochPreLoss / steps
		if args.task == 'c':
			temSum = 0
			for i in range(args.offNum):
				ret['F1_%d' % i] = epochTp[i] * 2 / (epochTp[i] * 2 + epochFp[i] + epochFn[i])
				temSum += ret['F1_%d' % i]
			ret['microF1'] = temSum / args.offNum
			ret['macroF1'] = np.sum(epochTp) * 2 / (np.sum(epochTp) * 2 + np.sum(epochFp) + np.sum(epochFn))
		elif args.task == 'r':
			for i in range(args.offNum):
				ret['RMSE_%d' % i] = np.sqrt(epochSqLoss[i] / epochTstNum[i])
				ret['MAE_%d' % i] = epochAbsLoss[i] / epochTstNum[i]
				ret['MAPE_%d' % i] = epochApeLoss[i] / epochPosNums[i]
			ret['RMSE'] = np.sqrt(np.sum(epochSqLoss) / np.sum(epochTstNum))
			ret['MAE'] = np.sum(epochAbsLoss) / np.sum(epochTstNum)
			ret['MAPE'] = np.sum(epochApeLoss) / np.sum(epochPosNums)
		return ret

	def calcRes(self, preds, temTst, tstLocs):
		hit = 0
		ndcg = 0
		for j in range(preds.shape[0]):
			predvals = list(zip(preds[j], tstLocs[j]))
			predvals.sort(key=lambda x: x[0], reverse=True)
			shoot = list(map(lambda x: x[1], predvals[:args.shoot]))
			if temTst[j] in shoot:
				hit += 1
				ndcg += np.reciprocal(np.log2(shoot.index(temTst[j])+2))
		return hit, ndcg
	
	def saveHistory(self):
		if args.epoch == 0:
			return
		with open('History/' + args.save_path + '.his', 'wb') as fs:
			pickle.dump(self.metrics, fs)

		saver = tf.train.Saver()
		saver.save(self.sess, 'Models/' + args.save_path)
		log('Model Saved: %s' % args.save_path)

	def loadModel(self):
		saver = tf.train.Saver()
		saver.restore(sess, 'Models/' + args.load_model)
		with open('History/' + args.load_model + '.his', 'rb') as fs:
			self.metrics = pickle.load(fs)
		log('Model Loaded')

if __name__ == '__main__':
	logger.saveDefault = True
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	log('Start')
	handler = DataHandler()
	log('Load Data')

	with tf.Session(config=config) as sess:
		model = Model(sess, handler)
		model.run()