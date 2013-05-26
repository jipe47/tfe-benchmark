#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class ConstantRegressor(BaseEstimator, ClassifierMixin):
	def __init__(self):
		self.lastY = 42

	def fit(self, X, y):
		if len(y) >= 2:
			self.lastY = values = y[-1:][0]
		return self

	def fit_transform(self, X, y):
		return X.toarray()

	def predict(self, X):
		return [self.lastY]*len(X)
		
	def transform(self, X):
		return X.toarray()

