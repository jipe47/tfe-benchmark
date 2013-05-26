#! /usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import sys

def display_connections(n):
	for mod in n.modules:
		for conn in n.connections[mod]:
			print conn
			for cc in range(len(conn.params)):
				print conn.whichBuffers(cc), conn.params[cc]

def findBestParameters(errors, parameters):

	# Old code used to handle TS and VS and retreive the parameters with close errors on those sets
	
	"""
	i = 0
	nbr_param = len(parameters)

	while i < nbr_param:
	
		min_index = errors_te.index(min(errors_te))

		error_te = errors_te[min_index]
		error_vs = errors_vs[min_index]

		#print "error_te = " , error_te , ", error_vs = " , error_vs

		if error_vs <= 1.1*error_te or i == nbr_param - 1:
			best_params = parameters[min_index]
			break
		else:
			errors_te.remove(error_te)
			errors_vs.remove(error_vs)
			parameters.remove(parameters[min_index])
		
		i = i + 1
	"""
	
	return parameters[errors.index(min(errors))]

def findSeasonality(y, threshold=0.8, getAutoCorrelation=False):
	lags = []
	autocor = []
	max_index = -1
	max_ac = -1
	for i in range(2, len(y)):
		lags.append(i)
		ac = autocorrelation(y, i)
		autocor.append(ac)
		
		if ac > max_ac:
			max_ac = ac
			max_index = i
	
	if max_ac > threshold:
		if getAutoCorrelation:
			return max_index, autocor
		else:
			return max_index
	if getAutoCorrelation:	
		return -1, autocor
	else:
		return -1

# From http://argandgahandapandpa.wordpress.com/2011/02/24/python-numpy-moving-average-for-data/
def movingAverage(data, WINDOW):
	#weightings = np.repeat(1.0, WINDOW) / WINDOW
	#return np.convolve(data, weightings)[WINDOW-1:-(WINDOW-1)]
	extended_data = np.hstack([[data[0]] * (WINDOW- 1), data])
	weightings = np.repeat(1.0, WINDOW) / WINDOW
	return np.convolve(extended_data, weightings)[WINDOW-1:-(WINDOW-1)]

def timeserieMakeOverlappingSet(Ys, window_size):
	ls_x = []
	ls_y = []
	
	i = window_size

	while i < len(Ys):
		ls_x.append(Ys[i-window_size : i])
		ls_y.append(Ys[i])
		i = i + 1
		
	return ls_x, ls_y

def computeError(xs, ys):
	return quadError(xs, ys)
	
def quadError(xs, ys):
	sum = 0
	for i, x in enumerate(xs):
		sum = sum + ((x - ys[i])*(x - ys[i]))
	return sum / len(xs)
	
def SMAPE(ys_true, ys_pred):   
	num = 0
	den = 0

	for i, y_true in enumerate(ys_true):
		y_pred = ys_pred[i]
		num = num + math.fabs(y_true - y_pred)
		den = den + (math.fabs(y_true) + math.fabs(y_pred))/2
	return num / den
	
def autocorrelation(ys, k):
	"""
    n = len(ys)
    sigma2 = np.var(ys)
    nu = np.mean(ys)

    s = 0
    for t in range(0, n - k-1):
        s = s + (ys[t] - nu)*(ys[t + k] - nu)
    return s / ((n - k) * sigma2)
	"""
	
	# Formula from http://shadow.eas.gatech.edu/~jean/paleo/Meko_Autocorrelation.pdf
	sumt = 0
	sumt2 = 0
	mean1 = np.mean(ys[:k])
	mean2 = np.mean(ys[len(ys)-k])
	
	for t in range(0, len(ys) - k):
		sumt += (ys[t] - mean1)*(ys[t + k] - mean2)
		sumt2 += math.sqrt((ys[t] - mean1)*(ys[t] - mean1))*math.sqrt((ys[t+k] - mean2)*(ys[t+k] - mean2))

	return (sumt+0j) / (sumt2+0j)