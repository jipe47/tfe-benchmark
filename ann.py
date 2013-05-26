#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pylab as pl
import MySQLdb as mdb
import sys
from sla import DerivateRegressor
import math
from functions import *
from preprocessing import *

from pybrain.datasets import *
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import *
from pybrain.tools.shortcuts import buildNetwork

def makeAndTrainANN(ls_x, ls_y, window_size, hiddenSize=5, epochs=1):
	
	# Build Pybrain dataset
	ds = SupervisedDataSet(window_size, 1)

	for i, sample in enumerate(ls_x):
		ds.addSample(sample, ls_y[i])

	inLayer = LinearLayer(window_size)
	hiddenLayer = SigmoidLayer(hiddenSize)
	outLayer = LinearLayer(1)
	constant = BiasUnit()

	n = FeedForwardNetwork()

	n.addInputModule(inLayer)
	n.addModule(hiddenLayer)
	n.addOutputModule(outLayer)
	n.addModule(constant)
	
	n.addConnection(FullConnection(inLayer, hiddenLayer))
	n.addConnection(FullConnection(hiddenLayer, outLayer))
	n.addConnection(FullConnection(constant, inLayer))
	n.addConnection(FullConnection(constant, hiddenLayer))
	n.addConnection(FullConnection(constant, outLayer))
	
	n.sortModules()
	n.reset()
	#trainer = BackpropTrainer(n, dataset=ds)
	trainer = BackpropTrainer(n, dataset=ds, momentum=0.2)
	#trainer.trainEpochs(epochs)
	trainer.trainUntilConvergence(maxEpochs=epochs)
	
	return n

