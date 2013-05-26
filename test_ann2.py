#!/usr/bin/python
# -*- coding: utf-8 -*-
import pylab as pl
import numpy as np
from ann import *
import neurolab as nl
from preprocessing import *

window_size = 5
X = np.arange(0, 6, 0.01)
y = [math.cos(x)+np.random.random()/50 for x in X]

pl.plot(y)

ls_x, ls_y = timeserieMakeOverlappingSet(y, window_size)

for h in [10]:
	net = makeAndTrainANN(ls_x, ls_y, window_size, hiddenSize=h, epochs=10)

	preds_y = []
	preds_y2 = []
	
	test_size = int(len(X) * 0.1);
	test_x = y[len(y) - test_size - window_size:len(y) - test_size]
	
	for w in range(window_size):
		preds_y.append(None)
		
	for i in range(len(X) - test_size):
		preds_y2.append(None)
		
	for x in ls_x:
		pred_y = net.activate(x)
		preds_y.append(pred_y[0])
	print "test_size = " , test_size
	print "init test_x = " , test_x
	print "Y = " , y
	for x in range(test_size):
		pred_y2 = net.activate(test_x)
		preds_y2.append(pred_y2[0])
		test_x = test_x[1:]
		test_x.append(pred_y2[0])

	pl.plot(preds_y)
	pl.plot(preds_y2)


pl.show()