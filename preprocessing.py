#! /usr/bin/env python
# -*- coding: utf-8 -*-

import math
from functions import *
import sys

def preprocessing_gp(ys):
	out = []
	
	for i, y in enumerate(ys):
		out.append(i + y)

	return out
	
def postprocessing_gp(ys, start = 0):
	out = []
	for i, y in enumerate(ys):
		out.append(y-start)
		start = start + 1
	return out

def preprocessing_logtransformation(ys):
    y_min = min(ys)
    y_min = -1 if y_min > 0 else y_min
    #y_min = y_min + 1
    out = []
    #print "y_min = " , y_min
    for i,y in enumerate(ys):
        y = (y - y_min)+1
        out.append(math.log(y))
        #print "processing " , y , " -> " , ys[i]
    return y_min, out


def postprocessing_logtransformation(ys, y_min):
	out = []
	for i,y in enumerate(ys):
		#print "processing " , y
		try:
			val = math.exp(y) + (y_min - 1)
		except:
			val = sys.float_info.max
		out.append(val)
	return out

def preprocessing_scale(ys):
    mx = max(ys)
    mn = min(ys)
    out = []
    for y in ys:
        out.append((2 * (y - mn) / (mx - mn)) - 1)
    return mn, mx, out


def postprocessing_scale(ys, mn, mx):
    out = []
    for y in ys:
        out.append((mx-mn)*(y + 1)/2 + mn)
    return out


#http://docs.oracle.com/cd/E12825_01/epm.111/cb_statistical/frameset.htm?ch05s04s05.html
# http://docs.oracle.com/cd/E12825_01/epm.111/cb_statistical/frameset.htm?ch05s04s04s01.html
def isSeasonal(ys, k):
    # Calculate the standard error of autocorrelation:
    n = len(ys)

    se = 0
    for i in range(1, k):
        se = se + math.pow(autocorrelation(ys, i), 2)
    se = (1 + 2 * se) / n
    t = autocorrelation(ys, k) / se
    
    autocor = autocorrelation(ys, k)
    print "Autocor = " , autocor
    return True
