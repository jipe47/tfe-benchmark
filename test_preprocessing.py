#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pylab as pl
import MySQLdb as mdb
import random
import sys
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Perceptron
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sla import DerivateRegressor
import math
from functions import *
from preprocessing import *


x = np.arange(0, 12, .1)
y = [math.cos(i) for i in x]

y_gp = preprocessing_gp(y)

y2 = postprocessing_gp(y_gp)

pl.plot(x, y, 'b')
pl.plot(x, y_gp, 'r')
pl.plot(x, y2, 'g')

#seas = findSeasonality(y)
#print "seas = " , seas

#y_m = movingAverage(y, seas)
#y_d = [y[i] - y_m[i] for i in range(len(x))]
#y = [random.random() for i in x]
#mn, mx, y_preprocessed = preprocessing_scale(y)
#y_postprocessed = postprocessing_scale(y_preprocessed, mn, mx)
#y_min, y_preprocessed = preprocessing_logtransformation(y)
"""
lags = []
autocor = []
for i in range(1, len(x)):
    lags.append(i)
    autocor.append(autocorrelation(y, i))
pl.plot(lags, autocor)
pl.show()
"""
#y_postprocessed = postprocessing_logtransformation(y_preprocessed, y_min)
"""
pl.plot(x, y, 'b')
pl.plot(x, y_m, 'r')
pl.plot(x, y_d, 'g')
"""
pl.show()
"""

pl.plot(x, y_preprocessed)
pl.plot(x, y_postprocessed)
pl.show()


"""
