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

window_size = 6
x = np.arange(0, 15, .1)
xt = x[130+window_size:]
y = [math.cos(i) for i in x]
y_original = y
#y = [math.cos(i) for i in x]

# Log transformation
points_min, y = preprocessing_logtransformation(y)

# Scale the function to [-1, 1]
points_mn, points_mx, y = preprocessing_scale(y)

overlapping_x, overlapping_y = timeserieMakeOverlappingSet(y, window_size)

n_sample = len(overlapping_x)
size_ls = int(math.floor(0.7*n_sample))

ls_x = overlapping_x[:size_ls]
ls_y = overlapping_y[:size_ls]

test_x = overlapping_x[size_ls:]
test_y = overlapping_y[size_ls:]

# Build Pybrain dataset
ds = SupervisedDataSet(window_size, 1)

for i, sample in enumerate(ls_x):
	ds.addSample(sample, ls_y[i])	
	
net = FeedForwardNetwork() 
inp = LinearLayer(ds.indim) 
h1 = SigmoidLayer(1) 
outp = LinearLayer(ds.outdim)
bias = BiasUnit()

# add modules 
net.addOutputModule(outp) 
net.addInputModule(inp) 
net.addModule(h1)
net.addModule(bias)

# create connections 
net.addConnection(IdentityConnection(inp, h1)) 
net.addConnection(FullConnection(h1, outp))
#net.addConnection(FullConnection(bias, outp))
#net.addConnection(FullConnection(bias, h1))

# finish up 
net.sortModules()

# initialize the backprop trainer and train 
trainer = BackpropTrainer(net, ds, momentum=.99, learningrate=0.01)
trainer.trainOnDataset(ds,10)
trainer.testOnData(verbose=True)

print 'Final weights:',net.params
		
print net

test_x = test_x[0]
preds_y = []
index_inpoints = size_ls

for i in range(len(xt)):
	pred_y = net.activate(test_x)
	
	test_x = test_x[1:]
	test_x.append(pred_y[0])
	
	pred_y = postprocessing_scale(pred_y, points_mn, points_mx)[0]
	pred_y = postprocessing_logtransformation([pred_y], points_min)[0]
	preds_y.append(pred_y)
	
	index_inpoints = index_inpoints + 1

#display_connections(n)

for mod in net.modules:
	print "Module:", mod.name
	if mod.paramdim > 0:
		print "--parameters:", mod.params
	for conn in net.connections[mod]:
		print "-connection to", conn.outmod.name
		if conn.paramdim > 0:
			print "- parameters", conn.params
	if hasattr(net, "recurrentConns"):
		print "Recurrent connections"
		for conn in net.recurrentConns:             
			print "-", conn.inmod.name, " to", conn.outmod.name
			if conn.paramdim > 0:
				print "- parameters", conn.params

print "preds = " , preds_y
pl.plot(x, y_original)
pl.plot(xt, preds_y)
pl.show()


