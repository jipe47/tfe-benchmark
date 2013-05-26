#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class DerivateRegressor(BaseEstimator, ClassifierMixin):
	def __init__(self):
		self.delta = 42

	def fit(self, X, y):
		if len(y) > 2:
			self.delta = y[-1] - y[-2]
		return self

	def fit_transform(self, X, y):
		return X.toarray()

	def predict(self, X):
		y = [(x[-1:][0] + self.delta) for x in X]
		return y

	def transform(self, X):
		return X.toarray()