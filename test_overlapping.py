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

window_size = 2

y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
overlapping_x, overlapping_y = timeserieMakeOverlappingSet(y, window_size)

print overlapping_x
print overlapping_y
