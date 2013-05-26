#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pylab as pl
import MySQLdb as mdb
import sys
import random
from sklearn.gaussian_process import GaussianProcess
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Perceptron
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sla import *
import math
from ann import *
from functions import *
from preprocessing import *
import json

###############################
#          CONSTANTS          #
###############################

# SQL constants
TABLE_ZONE			= "curve_zone"
TABLE_CURVE			= "curve"
TABLE_PREDICTION	= "prediction_algorithm"
TABLE_ALGO			= "algorithm"
TABLE_CATEGORY		= "category"
TABLE_GROUP			= "curve_group"

################################
#		  PARAMETERS		  #
################################

M3C_only = False
range_window_size = range(1, 20)
#range_window_size = range(5, 10)

display_step		= 10
run_dry				= False
testset_set_size 	= 70 # Percentage of available data for LS in the test set method

testset_use_ownprediction = False # True to use algorithm's predictions in the validation step, false otherwise

process_svr 		= False
process_svr_poly 	= False
process_svr_rbf 	= True
process_svr_linear 	= False

process_dtr 		= False
process_knn 		= False
process_gp	 		= False
process_derivation	= False
process_constant	= False
process_extratree	= True

preprocessing_enable 					= True
preprocessing_enable_seasonality 		= True
preprocessing_enable_logtransformation 	= True
preprocessing_enable_scale				= True

correct_seasonality = True# Set to True to use the seasonality reported by the detection function, False otherwise
if not correct_seasonality:
	ids_algo = {"SVR-rbf" : 1, "SVR-linear" : 8, "DTR" : 2, "ANN" : 3, "KNN" : 4, "GP" : 5, "CONSTANT" : 6, "DERIVATION" : 7, "ET" : 9} # 27
else: # Experimental IDs
	ids_algo = {"SVR-rbf" : 17, "SVR-linear" : 18, "DTR" : 11, "KNN" : 15, "GP" : 14, "CONSTANT" : 13, "DERIVATION" : 12, "ET" : 16} # 28


################################
####    MYSQL CONNECTION    ####
################################

db = None

try:
	db = mdb.connect('localhost', 'root', '', 'jipe_tfe')
except _mysql.Error, e:
	print "Error %d: %s" % (e.args[0], e.args[1])
	sys.exit(1)

cur = db.cursor(mdb.cursors.DictCursor)
cur.execute("SELECT z.*, c.points, c.tag, c.id as id_curve, parent_cat.name as period, g.id as id_group \
			  FROM "+ TABLE_ZONE +" z \
			  LEFT JOIN " + TABLE_CURVE + " c ON c.id = z.id_curve \
			  LEFT JOIN " + TABLE_CATEGORY + " cat ON cat.id = c.id_category \
			  LEFT JOIN " + TABLE_CATEGORY + " parent_cat ON cat.id_parent = parent_cat.id \
			  LEFT JOIN " + TABLE_GROUP + " g ON g.id_category = cat.id \
			  WHERE c.position <= g.nbr_curve \
			  ORDER BY g.position ASC, c.position ASC \
			  ")

rows = cur.fetchall()
maxCurve = -1
startCurve = 0
currentCurve = 0
for row in rows:
	if startCurve != 0:
		startCurve = startCurve - 1
		continue
		
	if maxCurve == 0:
		break
		
	#print "\n--------\n"
	maxCurve		= maxCurve - 1
	currentCurve	= currentCurve + 1
	
	# Treat data from database
	points = row["points"].split(";")
	
	for k, p in enumerate(points):
		points[k] = float(p) + random.random()/1000000; # Avoid errors in gaussian processes
	
	points_original = list(points)
	
	if preprocessing_enable:
	
		#######################
		#### PREPROCESSING ####
		#######################
		
		# Seasonality removal
		if preprocessing_enable_seasonality:
			seas = findSeasonality(points)
			period = len(points)
			#pl.plot(points_original, 'b')
			
			if seas == -1 or row["period"] == "Year":
				points_diff = [0 for i in range(len(points))]
				month_means = np.zeros(period)
			else:
				if not correct_seasonality:
					period = 4 if row["period"] == "Quart" else 12
				else:
					period = seas
				points_ma = movingAverage(points, period).tolist()
				points_diff = [points[i] - points_ma[i] for i in range(len(points_ma))]
				
				month_means = np.zeros(period)
				#print "months_means = " , month_means
				counts = np.zeros(period)
				index = 0

				for p in points_diff:
					counts[index] = counts[index] + 1
					month_means[index] = month_means[index] + p
					index = (index + 1)%period
				
				for i, m in enumerate(month_means):
					month_means[i] = m / float(counts[i])
				
				index = 0
				points_deseas = []
				for i, p in enumerate(points):
					points[i] = p - month_means[index]
					index = (index + 1) % period
				
				
		#points = preprocessing_gp(points)
		
		# Log transformation
		if preprocessing_enable_logtransformation:
			points_min, points = preprocessing_logtransformation(points)
		
		# Scale the function to [-1, 1]
		if preprocessing_enable_scale:
			points_mn, points_mx, points = preprocessing_scale(points)
	
	######################################
	####		ZONE SEPARATION	   ####
	######################################
		
	zone_start  = int(row["start"])
	zone_end	= int(row["end"])
		
	zone_length = zone_end - zone_start
	
	test_xs	 = range(zone_start, zone_end)
	ls_points   = points[:zone_start]
	ls_xs	   = range(0, zone_start)
	
	######################################
	####   BEST PARAMETER DISCOVERY   ####
	######################################
	
	##########################
	### GAUSSIAN PROCESSES ###
	##########################
	
	if process_gp:
		#cur.execute("DELETE FROM " + TABLE_PREDICTION + " WHERE id_algorithm='"+str(ids_algo["GP"])+"' \
		#			  ")
		#db.commit()	
	
		if not run_dry:
			# Check if a prediction exists. If yes, skip the time serie, otherwise reserve the entry
			cur.execute("SELECT COUNT(*) as cnt \
						  FROM "+ TABLE_PREDICTION +" p \
							WHERE id_algorithm='"+str(ids_algo["GP"])+"' AND id_zone='"+str(row["id"])+"'\
						  ")
			
			response = cur.fetchone()
			if response["cnt"] >= 1:
				print "Skipped curve " , currentCurve , " id = " , row["id_curve"]
				continue
			#else:
				#cur.execute("INSERT INTO "+ TABLE_PREDICTION +" (id_algorithm, id_zone, parameters, prediction) \
				#		 VALUES('"+str(ids_algo["GP"])+"', '"+str(row["id"])+"', '', '')\
				#	  ")
				#db.commit()	
				
		# Parameters ranges
		range_regr = ['constant', 'linear']
		#range_corr = ['absolute_exponential', 'squared_exponential', 'generalized_exponential', 'cubic', 'linear']
		range_corr = ['absolute_exponential','squared_exponential','cubic','linear']
		
		current_params = {}
		
		parameters = []
		errors = []

		total_test = len(range_regr) * len(range_corr) * len(range_window_size)

		it = 0
		for window_size in range_window_size:
			
			if window_size > zone_start:
				continue

			ls_x, ls_y = timeserieMakeOverlappingSet(ls_points, window_size)
			
			nbr_sample = len(ls_x)
			nbr_sample_half = int(math.floor(nbr_sample/2))
			half_size = int(math.floor(nbr_sample*(float(testset_set_size)/200)))

			# Separation of LS and TS
					
			if window_size > 2 * half_size:
				continue
			
			size_ls = int(math.floor(nbr_sample * testset_set_size/100))
			
			x_ls	= ls_x[:size_ls]
			y_ls	= ls_y[:size_ls]
													  
			x_ts	= ls_x[size_ls:]
			y_ts	= ls_y[size_ls:]
			
			if len(x_ls) <= window_size:
				continue
			
			# Test all combinations of parameters
			for regr in range_regr:
				for corr in range_corr:
					
					current_params["regr"] = regr
					current_params["corr"] = corr
					current_params["window_size"] = window_size
					
					
					if it % display_step == 0:
						print "GP - curve " , currentCurve , " (id_curve = " , row["id_curve"] , ", id_zone = " , row["id"] , ") - " , it , " / " , total_test

					clf = GaussianProcess(regr=regr, corr=corr)
					
					clf.fit(x_ls, y_ls)
													   
					test_x = x_ts[0]
					
					preds_y = []
					preds_y_original = []
					
					y_ts_original = []
					for i, y in enumerate(y_ts):
						if testset_use_ownprediction:
							pred_y = clf.predict(np.array(test_x))[0]
							test_x = test_x[1:]
							test_x.append(pred_y)
						else:
							pred_y = clf.predict(np.array(x_ts[i]))[0]
						preds_y.append(pred_y)

					parameters.append(dict(current_params))
					errors.append(computeError(y_ts, preds_y))

					it = it + 1

		# Find parameters than minimise the error on TS and VS
		
		if len(errors) == 0:
			continue
		else:
			best_params = findBestParameters(errors, parameters)
		
		######################################
		####	  COMPUTE PREDICTION	  ####
		######################################
		
		clf = GaussianProcess(regr=best_params["regr"], corr=best_params["corr"])
		ls_x, ls_y = timeserieMakeOverlappingSet(ls_points, best_params["window_size"])
		clf = clf.fit(ls_x, ls_y)
		
		
		test_x = points[zone_start-best_params["window_size"]+1:zone_start+1]
		test_y = points[zone_start+2:]
		
		preds_y = []
		index_inpoints = zone_start+1
		for y in range(zone_length):
			pred_y = clf.predict(np.array(test_x))
			pred_y = pred_y[0]
			test_x = test_x[1:]
			test_x.append(pred_y)
			
			# Reverse the preprocessing
			if preprocessing_enable_scale:
				pred_y = postprocessing_scale([pred_y], points_mn, points_mx)[0]
			
			if preprocessing_enable_logtransformation:
				pred_y = postprocessing_logtransformation([pred_y], points_min)[0]
			
			
			if preprocessing_enable_seasonality:
				pred_y = pred_y + points_diff[index_inpoints]
			preds_y.append(str(pred_y))
			
			index_inpoints = (index_inpoints + 1) % period

		#Serialize prediction and best parameters
		preds_string = ';'.join(preds_y)
		parameters_string = json.dumps(best_params)
		
		if not run_dry:
			#print "Inserting " , preds_string
			cur.execute("INSERT INTO " + TABLE_PREDICTION +  " VALUES('"+ str(row["id"]) +"', '"+str(ids_algo["GP"])+"', '"+preds_string+"', '"+parameters_string+"')")
			#cur.execute("UPDATE " + TABLE_PREDICTION +  " SET prediction='"+preds_string+"', parameters='"+parameters_string+"' \
			#	WHERE id_algorithm='"+str(ids_algo["GP"])+"' AND id_zone='"+str(row["id"])+"'")
			db.commit()
		"""
		# Creation of copies to avoid potential value rewriting
		known_points_y = points_original[:zone_start]
		known_points_x = range(len(known_points_y))
		
		unknown_points_y = points_original[zone_start-1:]
		unknown_points_x = range(zone_start-1, zone_start -1 + len(unknown_points_y))
				
		test_xs_cp = [zone_start-1]
		test_xs_cp.extend(test_xs)
		
		test_y_cp = [points_original[zone_start-1]]
		test_y_cp.extend(test_y)
		
		preds_y_cp = [points_original[zone_start-1]]
		preds_y_cp.extend(preds_y)
		
		print "\n"
		print "test_x_cp = " , test_xs_cp
		print "test_y_cp = " , test_y_cp
		print "preds_y_cp = " , preds_y_cp
		
		print np.array(unknown_points_x).shape
		print np.array(unknown_points_y).shape
				
		# Plot results
		
		pl.plot(known_points_x, known_points_y, 'g')
		pl.plot(unknown_points_x, unknown_points_y, 'b')
		pl.plot(test_xs_cp, preds_y_cp, 'r')
		pl.show()
		"""
	################
	### CONSTANT ###
	################
	
	if process_constant:
	
		if not run_dry:
			# Check if a prediction exists. If yes, skip the time serie, otherwise reserve the entry
			cur.execute("SELECT COUNT(*) as cnt \
						  FROM "+ TABLE_PREDICTION +" p \
							WHERE id_algorithm='"+str(ids_algo["CONSTANT"])+"' AND id_zone='"+str(row["id"])+"'\
						  ")
			
			response = cur.fetchone()
			if response["cnt"] >= 1:
				print "Skipped curve " , row["id_curve"]
				continue
		
		# Parameters ranges
		current_params = {}
		
		parameters = []
		errors = []

		total_test = 1

		it = 0
				
		######################################
		####	  COMPUTE PREDICTION	  ####
		######################################
		
		# Window size set to 3, but that does not matter regarding the learning algorithm
		clf = ConstantRegressor()
		ls_x, ls_y = timeserieMakeOverlappingSet(ls_points, 3)
		clf = clf.fit(ls_x, ls_y)
		
		test_x = points[zone_start-3+1:zone_start+1]
		test_y = points[zone_start+1:]
		index_inmonths = (zone_start+1)%period
		preds_y = []
		
		raw_y_pred = []
		for y in range(zone_length):
			#print "test_x = " , test_x
			pred_y = clf.predict(np.array(test_x).reshape(1, 3))
			pred_y = pred_y[0]
			test_x = test_x[1:]
			test_x.append(pred_y)
			
			raw_y_pred.append(pred_y)
			
			# Reverse the preprocessing
			if preprocessing_enable_scale:
				pred_y = postprocessing_scale([pred_y], points_mn, points_mx)[0]
			
			if preprocessing_enable_logtransformation:
				pred_y = postprocessing_logtransformation([pred_y], points_min)[0]
				
			if preprocessing_enable_seasonality:
				#pred_y = pred_y + points_diff[index_inpoints]
				pred_y = pred_y + month_means[index_inmonths]
			
			preds_y.append(str(pred_y))
			
			index_inmonths = (index_inmonths + 1)%period

		#Serialize prediction and best parameters
		preds_string = ';'.join(preds_y)
		parameters_string = ""
		
		if not run_dry:
			cur.execute("INSERT INTO " + TABLE_PREDICTION +  " VALUES('"+ str(row["id"]) +"', '"+str(ids_algo["CONSTANT"])+"', '"+preds_string+"', '"+parameters_string+"')")
			db.commit()
			
		# Creation of copies to avoid potential value rewriting
		
		"""
		known_points_y = points_original[:zone_start]
		known_points_x = range(len(known_points_y))
		
		unknown_points_y = points_original[zone_start-1:]
		unknown_points_x = range(zone_start-1, zone_start -1 + len(unknown_points_y))
		
		
		test_xs_cp = [zone_start-1]
		test_xs_cp.extend(test_xs)
		
		test_y_cp = [points_original[zone_start-1]]
		test_y_cp.extend(test_y)
		
		preds_y_cp = [points_original[zone_start-1]]
		preds_y_cp.extend(preds_y)
		
		print "\n"
		print "test_x_cp = " , test_xs_cp
		print "test_y_cp = " , test_y_cp
		print "preds_y_cp = " , preds_y_cp
		
		print np.array(unknown_points_x).shape
		print np.array(unknown_points_y).shape
		
		# Plot results
		
		pl.plot(known_points_x, known_points_y, 'g')
		pl.plot(unknown_points_x, unknown_points_y, 'b')
		pl.plot(test_xs_cp, preds_y_cp, 'r')
		pl.show()

		break
		"""
	
	##################
	### DERIVATION ###
	##################
	
	if process_derivation:
	
		if not run_dry:
			# Check if a prediction exists. If yes, skip the time serie, otherwise reserve the entry
			cur.execute("SELECT COUNT(*) as cnt \
						  FROM "+ TABLE_PREDICTION +" p \
							WHERE id_algorithm='"+str(ids_algo["DERIVATION"])+"' AND id_zone='"+str(row["id"])+"'\
						  ")
			
			response = cur.fetchone()
			if response["cnt"] >= 1:
				print "Skipped curve " , row["id_curve"]
				continue
		
		# Parameters ranges
		current_params = {}
		
		parameters = []
		errors = []

		total_test = 1

		it = 0
				
		######################################
		####	  COMPUTE PREDICTION	  ####
		######################################
					
		clf = DerivateRegressor()
		ls_x, ls_y = timeserieMakeOverlappingSet(ls_points, 3)
		clf = clf.fit(ls_x, ls_y)
		"""
		print "ls_x = " , np.array(ls_x).shape
		print "ls_y = " , np.array(ls_y).shape
		"""
		test_x = points[zone_start-3+1:zone_start+1]
		test_y = points[zone_start+1:]
		index_inmonths = (zone_start+1)%period
		preds_y = []
		
		raw_y_pred = []
		
		for y in range(zone_length):
			#print "test_x = " , test_x
			pred_y = clf.predict(np.array(test_x).reshape(1, 3))
			pred_y = pred_y[0]
			test_x = test_x[1:]
			test_x.append(pred_y)
			
			raw_y_pred.append(pred_y)
			
			# Reverse the preprocessing
			if preprocessing_enable_scale:
				pred_y = postprocessing_scale([pred_y], points_mn, points_mx)[0]
			
			if preprocessing_enable_logtransformation:
				pred_y = postprocessing_logtransformation([pred_y], points_min)[0]
				
			if preprocessing_enable_seasonality:
				#pred_y = pred_y + points_diff[index_inpoints]
				pred_y = pred_y + month_means[index_inmonths]
			
			preds_y.append(str(pred_y))
			
			index_inmonths = (index_inmonths + 1)%period

		#Serialize prediction and best parameters
		preds_string = ';'.join(preds_y)
		parameters_string = ""
		
		if not run_dry:
			cur.execute("INSERT INTO " + TABLE_PREDICTION +  " VALUES('"+ str(row["id"]) +"', '"+str(ids_algo["DERIVATION"])+"', '"+preds_string+"', '"+parameters_string+"')")
			db.commit()
		
		"""		
		# Creation of copies to avoid potential value rewriting
		
		known_points_y = points_original[:zone_start]
		known_points_x = range(len(known_points_y))
		
		known_points_y_pr = points[:zone_start]
		
		unknown_points_y = points_original[zone_start-1:]
		unknown_points_x = range(zone_start-1, zone_start -1 + len(unknown_points_y))
		
		
		test_xs_cp = [zone_start-1]
		test_xs_cp.extend(test_xs)
		
		test_y_cp = [points_original[zone_start-1]]
		test_y_cp.extend(test_y)
		
		preds_y_cp = [points_original[zone_start-1]]
		preds_y_cp.extend(preds_y)
		
		print "\n"
		print "test_x_cp = " , test_xs_cp
		print "test_y_cp = " , test_y_cp
		print "preds_y_cp = " , preds_y_cp
		
		print np.array(unknown_points_x).shape
		print np.array(unknown_points_y).shape
		
		# Plot results
		
		raw_y_pred2 = known_points_y_pr
		raw_y_pred2.extend(raw_y_pred)
		raw_y_pred = raw_y_pred2
		
		
		raw_y_true = points
		
		pl.plot(raw_y_pred)
		pl.plot(raw_y_true)
		#pl.plot(known_points_x, known_points_y, 'g')
		#pl.plot(unknown_points_x, unknown_points_y, 'b')
		#pl.plot(test_xs_cp, preds_y_cp, 'r')
		pl.show()
		"""
	
	###################
	### EXTRA TREES ###
	###################
	
	if process_extratree:
	
		if not run_dry:
			# Check if a prediction exists. If yes, skip the time serie, otherwise reserve the entry
			cur.execute("SELECT COUNT(*) as cnt \
						  FROM "+ TABLE_PREDICTION +" p \
							WHERE id_algorithm='"+str(ids_algo["ET"])+"' AND id_zone='"+str(row["id"])+"'\
						  ")
			
			response = cur.fetchone()
			if response["cnt"] >= 1:
				print "Skipped curve " , row["id_curve"]
				continue
			else:
				cur.execute("INSERT INTO "+ TABLE_PREDICTION +" (id_algorithm, id_zone, parameters, prediction) \
						 VALUES('"+str(ids_algo["ET"])+"', '"+str(row["id"])+"', '', '')\
					  ")
				db.commit()
		
		# Parameters ranges
		range_n_estimators = range(1, 30)
		#range_min_samples_split = range(1, 100, 10)
		range_min_samples_split = [2]
		
		current_params = {}
		
		parameters = []
		errors = []

		total_test = len(range_window_size)* len(range_n_estimators)* len(range_min_samples_split)

		it = 0
		for window_size in range_window_size:
			if window_size > zone_start:
				continue

			ls_x, ls_y = timeserieMakeOverlappingSet(ls_points, window_size)

			# Separation of LS, TS and VS
			nbr_sample = len(ls_x)
			nbr_sample_half = int(math.floor(nbr_sample/2))
			half_size = int(math.floor(nbr_sample*(float(testset_set_size)/200)))
			
			if window_size > 2 * half_size:
				continue
			
			size_ls = int(math.floor(nbr_sample * testset_set_size/100))
			
			x_ls	= ls_x[:size_ls]
			y_ls	= ls_y[:size_ls]
													  
			x_ts	= ls_x[size_ls:]
			y_ts	= ls_y[size_ls:]

			# Test all combinations of parameters
			for n_estimators in range_n_estimators:
				for min_samples_split in range_min_samples_split:
				
					current_params["min_samples_split"] = min_samples_split
					current_params["n_estimators"] = n_estimators
					current_params["window_size"] = window_size
						
					if it % display_step == 0:
						print "ET - curve " , currentCurve , " (id_curve = " , row["id_curve"] , ", id_zone = " , row["id"] , ") - " , it , " / " , total_test

					clf = ExtraTreesRegressor(n_estimators=n_estimators, min_samples_split=min_samples_split, n_jobs=1)
					
					clf.fit(x_ls, y_ls)
					
					test_x = x_ts[0]
					
					preds_y = []
					preds_y_original = []
					
					y_ts_original = []
					for i, y in enumerate(y_ts):
						if testset_use_ownprediction:
							pred_y = clf.predict(np.array(test_x))[0]
							test_x = test_x[1:]
							test_x.append(pred_y)
						else:
							pred_y = clf.predict(np.array(x_ts[i]))[0]
						preds_y.append(pred_y)
				
					parameters.append(dict(current_params))
					errors.append(computeError(y_ts, preds_y))
					
					it = it + 1


		# Find parameters than minimise the error on TS
		best_params = findBestParameters(errors, parameters)
		
		######################################
		####	  COMPUTE PREDICTION	  ####
		######################################
					
		clf = ExtraTreesRegressor(n_estimators=best_params["n_estimators"], min_samples_split=best_params["min_samples_split"], n_jobs=1)
		ls_x, ls_y = timeserieMakeOverlappingSet(ls_points, best_params["window_size"])
		clf = clf.fit(ls_x, ls_y)
		
		test_x = points[zone_start-best_params["window_size"]+1:zone_start+1]
		test_y = points[zone_start+1:]
		index_inpoints = zone_start+1
		preds_y = []
		for y in range(zone_length):
			pred_y = clf.predict(np.array(test_x))
			pred_y = pred_y[0]
			test_x = test_x[1:]
			test_x.append(pred_y)
			
			# Reverse the preprocessing
			if preprocessing_enable_scale:
				pred_y = postprocessing_scale([pred_y], points_mn, points_mx)[0]
			
			if preprocessing_enable_logtransformation:
				pred_y = postprocessing_logtransformation([pred_y], points_min)[0]
				
			if preprocessing_enable_seasonality:
				pred_y = pred_y + points_diff[index_inpoints]

			preds_y.append(str(pred_y))
			
			index_inpoints = (index_inpoints + 1)%period

		#Serialize prediction and best parameters
		preds_string = ';'.join(preds_y)
		parameters_string = json.dumps(best_params)
		
		if not run_dry:
			#cur.execute("INSERT INTO " + TABLE_PREDICTION +  " VALUES('"+ str(row["id"]) +"', '"+str(ids_algo["ET"])+"', '"+preds_string+"', '"+parameters_string+"')")
			cur.execute("UPDATE " + TABLE_PREDICTION +  " SET prediction='"+preds_string+"', parameters='"+parameters_string+"' WHERE id_algorithm='"+str(ids_algo["ET"])+"' AND id_zone='"+str(row["id"])+"'")
			db.commit()
			
		"""
		# Creation of copies to avoid potential value rewriting
		known_points_y = points_original[:zone_start]
		known_points_x = range(len(known_points_y))
		
		unknown_points_y = points_original[zone_start-1:]
		unknown_points_x = range(zone_start-1, zone_start -1 + len(unknown_points_y))
		
		
		test_xs_cp = [zone_start-1]
		test_xs_cp.extend(test_xs)
		
		test_y_cp = [points_original[zone_start-1]]
		test_y_cp.extend(test_y)
		
		preds_y_cp = [points_original[zone_start-1]]
		preds_y_cp.extend(preds_y)
		
		print "\n"
		print "test_x_cp = " , test_xs_cp
		print "test_y_cp = " , test_y_cp
		print "preds_y_cp = " , preds_y_cp
		
		print np.array(unknown_points_x).shape
		print np.array(unknown_points_y).shape
		
		
		# Plot results
		
		pl.plot(known_points_x, known_points_y, 'g')
		pl.plot(unknown_points_x, unknown_points_y, 'b')
		pl.plot(test_xs_cp, preds_y_cp, 'r')
		pl.show()

		continue
		"""

	###########
	### KNN ###
	###########
	
	if process_knn:
		
		# Parameters ranges
		range_k = range(1, 10)
		
		current_params = {}
		
		parameters = []
		errors = []

		total_test = len(range_window_size)* len(range_k)

		it = 0
		for window_size in range_window_size:
			if window_size > zone_start:
				continue

			ls_x, ls_y = timeserieMakeOverlappingSet(ls_points, window_size)

			# Separation of LS, TS and VS
			nbr_sample = len(ls_x)
			nbr_sample_half = int(math.floor(nbr_sample/2))
			half_size = int(math.floor(nbr_sample*(float(testset_set_size)/200)))
			
			if window_size > 2 * half_size:
				continue
			
			size_ls = int(math.floor(nbr_sample * testset_set_size/100))
			
			x_ls	= ls_x[:size_ls]
			y_ls	= ls_y[:size_ls]
													  
			x_ts	= ls_x[size_ls:]
			y_ts	= ls_y[size_ls:]

			# Test all combinations of parameters
			for k in range_k:
				
				current_params["k"] = k
				current_params["window_size"] = window_size
					
				if it % display_step == 0:
					print "KNN - curve " , currentCurve , " (id_curve = " , row["id_curve"] , ", id_zone = " , row["id"] , ") - " , it , " / " , total_test

				clf = KNeighborsRegressor(n_neighbors=k, warn_on_equidistant=False)
				
				clf.fit(x_ls, y_ls)
				
				test_x = x_ts[0]
				
				preds_y = []
				preds_y_original = []
				
				y_ts_original = []
				for i, y in enumerate(y_ts):
					if testset_use_ownprediction:
						pred_y = clf.predict(np.array(test_x))[0]
						test_x = test_x[1:]
						test_x.append(pred_y)
					else:
						pred_y = clf.predict(np.array(x_ts[i]))[0]
						
					preds_y.append(pred_y)
			
				parameters.append(dict(current_params))
				errors.append(computeError(y_ts, preds_y))
				
				it = it + 1

		# Find parameters than minimise the error on TS
		best_params = findBestParameters(errors, parameters)
		
		######################################
		####	  COMPUTE PREDICTION	  ####
		######################################
					
		clf = KNeighborsRegressor(n_neighbors=best_params["k"], warn_on_equidistant=False)
		ls_x, ls_y = timeserieMakeOverlappingSet(ls_points, best_params["window_size"])
		clf = clf.fit(ls_x, ls_y)
		
		test_x = points[zone_start-best_params["window_size"]+1:zone_start+1]
		test_y = points[zone_start+1:]
		index_inpoints = zone_start+1
		preds_y = []
		for y in range(zone_length):
			pred_y = clf.predict(np.array(test_x))
			pred_y = pred_y[0]
			test_x = test_x[1:]
			test_x.append(pred_y)
			
			# Reverse the preprocessing
			if preprocessing_enable_scale:
				pred_y = postprocessing_scale([pred_y], points_mn, points_mx)[0]
			
			if preprocessing_enable_logtransformation:				
					pred_y = postprocessing_logtransformation([pred_y], points_min)[0]
			
			if preprocessing_enable_seasonality:
				pred_y = pred_y + points_diff[index_inpoints]

			preds_y.append(str(pred_y))
			
			index_inpoints = index_inpoints + 1

		#Serialize prediction and best parameters
		preds_string = ';'.join(preds_y)
		parameters_string = json.dumps(best_params)
		
		if not run_dry:
			cur.execute("INSERT INTO " + TABLE_PREDICTION +  " VALUES('"+ str(row["id"]) +"', '"+str(ids_algo["KNN"])+"', '"+preds_string+"', '"+parameters_string+"')")
			db.commit()
			
		"""
		# Creation of copies to avoid potential value rewriting
		known_points_y = points_original[:zone_start]
		known_points_x = range(len(known_points_y))
		
		unknown_points_y = points_original[zone_start-1:]
		unknown_points_x = range(zone_start-1, zone_start -1 + len(unknown_points_y))
		
		
		test_xs_cp = [zone_start-1]
		test_xs_cp.extend(test_xs)
		
		test_y_cp = [points_original[zone_start-1]]
		test_y_cp.extend(test_y)
		
		preds_y_cp = [points_original[zone_start-1]]
		preds_y_cp.extend(preds_y)
		
		print "\n"
		print "test_x_cp = " , test_xs_cp
		print "test_y_cp = " , test_y_cp
		print "preds_y_cp = " , preds_y_cp
		
		print np.array(unknown_points_x).shape
		print np.array(unknown_points_y).shape
		
		
		# Plot results
		
		pl.plot(known_points_x, known_points_y, 'g')
		pl.plot(unknown_points_x, unknown_points_y, 'b')
		pl.plot(test_xs_cp, preds_y_cp, 'r')
		pl.show()

		continue
		"""

	###########
	### DTR ###
	###########
	
	if process_dtr:
		
		# Parameters ranges
		range_max_features = [None]
		range_max_depth = [None]
		range_min_samples_split = range(1, 100, 1)
		range_min_samples_leaf = [1]
		#range_min_samples_leaf = range(1, 100, 10)
		
		current_params = {}
		best_params = {}
		min_error = -1
		
		parameters = []
		errors = []

		total_test = len(range_window_size)* len(range_max_features)* len(range_max_depth) * len(range_min_samples_split) * len(range_min_samples_leaf)

		it = 0
		print "period = " , period
		print "len of months : " , len(month_means)
		for window_size in range_window_size:
			if window_size > zone_start:
				continue

			ls_x, ls_y = timeserieMakeOverlappingSet(ls_points, window_size)

			# Separation of LS, TS and VS
			nbr_sample = len(ls_x)
			nbr_sample_half = int(math.floor(nbr_sample/2))
			half_size = int(math.floor(nbr_sample*(float(testset_set_size)/200)))
			
			if window_size > 2 * half_size:
				continue
			
			size_ls = int(math.floor(nbr_sample * testset_set_size/100))
			
			x_ls	= ls_x[:size_ls]
			y_ls	= ls_y[:size_ls]
													  
			x_ts	= ls_x[size_ls:]
			y_ts	= ls_y[size_ls:]
			
			#y_ts_original = points_original[size_ls:zone_start]

			# Test all combinations of parameters
			for max_features in range_max_features:
				for max_depth in range_max_depth:
					for min_samples_split in range_min_samples_split:
						for min_samples_leaf in range_min_samples_leaf:
							 
							current_params["max_features"] = max_features
							current_params["max_depth"] = max_depth
							current_params["min_samples_split"] = min_samples_split
							current_params["min_samples_leaf"] = min_samples_leaf
							current_params["window_size"] = window_size
								
							if it % display_step == 0:
								print "DTR - curve " , currentCurve , " (id_curve = " , row["id_curve"] , ", id_zone = " , row["id"] , ") - " , it , " / " , total_test

							clf = DecisionTreeRegressor(max_features=max_features, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
							
							clf.fit(x_ls, y_ls)
													   
							test_x = x_ts[0]
							
							preds_y = []
							preds_y_original = []
							
							y_ts_original = []
							for i, y in enumerate(y_ts):
								if testset_use_ownprediction:
									pred_y = clf.predict(np.array(test_x))[0]
									test_x = test_x[1:]
									test_x.append(pred_y)
								else:
									pred_y = clf.predict(np.array(x_ts[i]))[0]
								preds_y.append(pred_y)
								
							current_error = computeError(y_ts, preds_y)
							
							if min_error == -1 or min_error > current_error:
								min_error = current_error
								best_params["max_features"] = current_params["max_features"]
								best_params["max_depth"] = current_params["max_depth"]
								best_params["min_samples_split"] = current_params["min_samples_split"]
								best_params["min_samples_leaf"] = current_params["min_samples_leaf"]
								best_params["window_size"] = current_params["window_size"]
		
							it = it + 1
		
		# Find parameters than minimise the error on TS
		#best_params = findBestParameters(errors, parameters)
				
		######################################
		####	  COMPUTE PREDICTION	  ####
		######################################
					
		clf = DecisionTreeRegressor(max_features=best_params["max_features"], max_depth=best_params["max_depth"], min_samples_split=best_params["min_samples_split"], min_samples_leaf=best_params["min_samples_leaf"])
		ls_x, ls_y = timeserieMakeOverlappingSet(ls_points, best_params["window_size"])
		clf = clf.fit(ls_x, ls_y)
		
		test_x = points[zone_start-best_params["window_size"]+1:zone_start+1]
		test_y = points[zone_start+1:]
		index_inmonths = (zone_start+1)%period
		preds_y = []
		for y in range(zone_length):
			pred_y = clf.predict(np.array(test_x))
			pred_y = pred_y[0]
			test_x = test_x[1:]
			test_x.append(pred_y)
			
			# Reverse the preprocessing
			if preprocessing_enable_scale:
				pred_y = postprocessing_scale([pred_y], points_mn, points_mx)[0]
			
			if preprocessing_enable_logtransformation:
				pred_y = postprocessing_logtransformation([pred_y], points_min)[0]
				
			if preprocessing_enable_seasonality:
				#pred_y = pred_y + points_diff[index_inpoints]
				pred_y = pred_y + month_means[index_inmonths]

			preds_y.append(str(pred_y))
			
			index_inmonths = (index_inmonths + 1)%period

		#Serialize prediction and best parameters
		preds_string = ';'.join(preds_y)
		parameters_string = json.dumps(best_params)
		
		if not run_dry:
			cur.execute("INSERT INTO " + TABLE_PREDICTION +  " VALUES('"+ str(row["id"]) +"', '"+str(ids_algo["DTR"])+"', '"+preds_string+"', '"+parameters_string+"')")
			db.commit()
			
		"""
		# Creation of copies to avoid potential value rewriting
		
		known_points_y = points_original[:zone_start]
		known_points_x = range(len(known_points_y))
		
		unknown_points_y = points_original[zone_start-1:]
		unknown_points_x = range(zone_start-1, zone_start -1 + len(unknown_points_y))
		
		
		test_xs_cp = [zone_start-1]
		test_xs_cp.extend(test_xs)
		
		test_y_cp = [points_original[zone_start-1]]
		test_y_cp.extend(test_y)
		
		preds_y_cp = [points_original[zone_start-1]]
		preds_y_cp.extend(preds_y)
		
		print "\n"
		print "test_x_cp = " , test_xs_cp
		print "test_y_cp = " , test_y_cp
		print "preds_y_cp = " , preds_y_cp
		
		print np.array(unknown_points_x).shape
		print np.array(unknown_points_y).shape
		
		
		# Plot results
		
		pl.plot(known_points_x, known_points_y, 'g')
		pl.plot(unknown_points_x, unknown_points_y, 'b')
		pl.plot(test_xs_cp, preds_y_cp, 'r')
		pl.show()

		continue
		"""
		
	###########
	### SVR ###
	###########
	
	if process_svr:
		
		range_C = [1.0, 0.1, 0.01, 0.001, 0.00001]
		range_epsilon = np.arange(0.1, 1, 0.1)
		range_shrinking = [True]
		range_tol = [0.001, 0.0001, 0.00001, 0.000001]

		range_degree = [2, 3, 4, 5]		  # rbf,  poly, sigmoid
		range_gamma = [0.0, 0.5, 1.0]   # rbf,  poly
		range_coef0 = [0.0]  #	   poly, sigmoid

		current_params = {}
		
		parameters = []
		errors = []

		# First kernel: rbf
		if process_svr_rbf:
		
			if not run_dry:
				# Check if a prediction exists. If yes, skip the time serie, otherwise reserve the entry
				cur.execute("SELECT COUNT(*) as cnt \
							  FROM "+ TABLE_PREDICTION +" p \
								WHERE id_algorithm='"+str(ids_algo["SVR-rbf"])+"' AND id_zone='"+str(row["id"])+"'\
							  ")
				
				response = cur.fetchone()
				
				if response["cnt"] == 1:
					print "Skipped curve " , row["id_curve"]
					continue
				#else:
				#	cur.execute("INSERT INTO "+ TABLE_PREDICTION +" (id_algorithm, id_zone, parameters, prediction) \
				#			 VALUES('"+str(ids_algo["SVR-rbf"])+"', '"+str(row["id"])+"', '', '')\
				#		  ")
				#	db.commit()
					
			total_test = len(range_C)* len(range_epsilon) * len(range_shrinking) * len(range_tol) * len(range_gamma) *len(range_degree)*len(range_window_size)
			it = 0
			for window_size in range_window_size:
				if window_size > zone_start:
					continue

				ls_x, ls_y = timeserieMakeOverlappingSet(ls_points, window_size)

				# Separation of LS, TS and VS
				nbr_sample = len(ls_x)
				nbr_sample_half = int(math.floor(nbr_sample/2))
				half_size = int(math.floor(nbr_sample*(float(testset_set_size)/200)))
				
				if window_size > 2 * half_size:
					continue
				
				size_ls = int(math.floor(nbr_sample * testset_set_size/100))
		
				x_ls	= ls_x[:size_ls]
				y_ls	= ls_y[:size_ls]
														  
				x_ts	= ls_x[size_ls:]
				y_ts	= ls_y[size_ls:]

				# Test all combinations of parameters
				for C in range_C:
					for epsilon in range_epsilon:
						for shrinking in range_shrinking:
							for tol in range_tol:
								for gamma in range_gamma:
									for degree in range_degree:

										current_params["C"] = C
										current_params["epsilon"] = epsilon
										current_params["kernel"] = 'rbf'
										current_params["shrinking"] = shrinking
										current_params["tol"] = tol
										current_params["degree"] = degree
										current_params["gamma"] = gamma
										current_params["window_size"] = window_size
										current_params["kernel"] = 'rbf'
										
										if it % display_step == 0:
											print "SVR (rbf) - curve " , currentCurve , " (id_curve = " , row["id_curve"] , ", id_zone = " , row["id"] , " , id_group = " , row["id_group"] , " ) - " , it , " / " , total_test

										clf = SVR(C=C, epsilon=epsilon, shrinking=shrinking, tol=tol, degree=degree, gamma=gamma, kernel='rbf')
										
										clf.fit(x_ls, y_ls)
												   
										test_x = x_ts[0]
										
										preds_y = []
										preds_y_original = []
										
										y_ts_original = []
										for i, y in enumerate(y_ts):
											if testset_use_ownprediction:
												pred_y = clf.predict(np.array(test_x))[0]
												test_x = test_x[1:]
												test_x.append(pred_y)
											else:
												pred_y = clf.predict(np.array(x_ts[i]))[0]
											preds_y.append(pred_y)
									
										parameters.append(dict(current_params))
										errors.append(computeError(y_ts, preds_y))
					
										it = it + 1
							
		# Second kernel: poly
		if process_svr_poly:
			total_test = len(range_C)* len(range_epsilon) * len(range_shrinking) * len(range_tol) * len(range_gamma) *len(range_degree)*len(range_coef0)*len(range_window_size)
			it = 0
			for window_size in range_window_size:
				if window_size > zone_start:
					continue

				ls_x, ls_y = timeserieMakeOverlappingSet(ls_points, window_size)

				# Separation of LS, TS and VS
				nbr_sample = len(ls_x)
				nbr_sample_half = int(math.floor(nbr_sample/2))
				half_size = int(math.floor(nbr_sample*(float(testset_set_size)/200)))
				
				if window_size > 2 * half_size:
					continue
				
				size_ls = int(math.floor(nbr_sample * testset_set_size/100))
		
				x_ls	= ls_x[:size_ls]
				y_ls	= ls_y[:size_ls]
														  
				x_ts	= ls_x[size_ls:]
				y_ts	= ls_y[size_ls:]

				# Test all combinations of parameters
				for C in range_C:
					for epsilon in range_epsilon:
						for shrinking in range_shrinking:
							for tol in range_tol:
								for gamma in range_gamma:
									for degree in range_degree:
										for coef0 in range_coef0:

											current_params["C"] = C
											current_params["epsilon"] = epsilon
											current_params["shrinking"] = shrinking
											current_params["tol"] = tol
											current_params["degree"] = degree
											current_params["gamma"] = gamma
											current_params["coef0"] = coef0
											current_params["kernel"] = 'poly'
											
											if it % display_step == 0:
												print "SVR (poly) - curve " , currentCurve , " (id_curve = " , row["id_curve"] , ", id_zone = " , row["id"] , ") - " , it , " / " , total_test

											clf = SVR(C=C, epsilon=epsilon, shrinking=shrinking, tol=tol, degree=degree, gamma=gamma, coef0=coef0, kernel='poly')
											
											clf.fit(x_ls, y_ls)
												   
											test_x = x_ts[0]
											
											preds_y = []
											preds_y_original = []
											
											y_ts_original = []
											for i, y in enumerate(y_ts):
												if testset_use_ownprediction:
													pred_y = clf.predict(np.array(test_x))[0]
													test_x = test_x[1:]
													test_x.append(pred_y)
												else:
													pred_y = clf.predict(np.array(x_ts[i]))[0]
												preds_y.append(pred_y)
										
											parameters.append(dict(current_params))
											errors.append(computeError(y_ts, preds_y))
						
											it = it + 1
											
		# Third kernel: linear
		if process_svr_linear:
			total_test = len(range_C)* len(range_epsilon) * len(range_shrinking) * len(range_tol)*len(range_window_size)
			it = 0
			for window_size in range_window_size:
				if window_size > zone_start:
					continue

				ls_x, ls_y = timeserieMakeOverlappingSet(ls_points, window_size)

				# Separation of LS, TS and VS
				nbr_sample = len(ls_x)
				nbr_sample_half = int(math.floor(nbr_sample/2))
				half_size = int(math.floor(nbr_sample*(float(testset_set_size)/200)))
				
				if window_size > 2 * half_size:
					continue
				
				size_ls = int(math.floor(nbr_sample * testset_set_size/100))
		
				x_ls	= ls_x[:size_ls]
				y_ls	= ls_y[:size_ls]
														  
				x_ts	= ls_x[size_ls:]
				y_ts	= ls_y[size_ls:]

				# Test all combinations of parameters
				for C in range_C:
					for epsilon in range_epsilon:
						for shrinking in range_shrinking:
							for tol in range_tol:
								
								current_params["C"] = C
								current_params["epsilon"] = epsilon
								current_params["shrinking"] = shrinking
								current_params["tol"] = tol
								current_params["kernel"] = 'linear'
								current_params["window_size"] = window_size;
								
								if it % display_step == 0:
									print "SVR (linear) - curve " , currentCurve , " (id_curve = " , row["id_curve"] , ", id_zone = " , row["id"] , ") - " , it , " / " , total_test

								clf = SVR(C=C, epsilon=epsilon, shrinking=shrinking, tol=tol, kernel='linear')
								
								clf.fit(x_ls, y_ls)
												   
								test_x = x_ts[0]
								
								preds_y = []
								preds_y_original = []
								
								y_ts_original = []
								for i, y in enumerate(y_ts):
									#print i , " - test_x1 = " , test_x1
									pred_y = clf.predict(np.array(test_x))[0]
									#print "pred_y1 = " , pred_y1
									test_x = test_x[1:]
									test_x.append(pred_y)
									preds_y.append(pred_y)
							
								parameters.append(dict(current_params))
								errors.append(computeError(y_ts, preds_y))
			
								it = it + 1

		
		best_params = findBestParameters(errors, parameters)
		
		print "best_params = " , best_params
		
		######################################
		####	  COMPUTE PREDICTION	  ####
		######################################
		
		if best_params["kernel"] == 'rbf':
			#clf = DecisionTreeRegressor(max_features=best_params["max_features"], max_depth=best_params["max_depth"], min_samples_split=best_params["min_samples_split"], min_samples_leaf=best_params["min_samples_leaf"])
			clf = SVR(C=best_params["C"], epsilon=best_params["epsilon"], shrinking=best_params["shrinking"], tol=best_params["tol"], degree=best_params["degree"], gamma=best_params["gamma"], kernel='rbf')
		elif best_params["kernel"] == 'linear':
			clf = SVR(C=best_params["C"], epsilon=best_params["epsilon"], shrinking=best_params["shrinking"], tol=best_params["tol"], kernel='linear')
		else: # poly
			clf = SVR(C=best_params["C"], epsilon=best_params["epsilon"], shrinking=best_params["shrinking"], tol=best_params["tol"], degree=best_params["degree"], gamma=best_params["gamma"], coef0=best_params["coef0"], kernel='poly')
			
		ls_x, ls_y = timeserieMakeOverlappingSet(ls_points, best_params["window_size"])
		clf = clf.fit(ls_x, ls_y)
		
		test_x = points[zone_start-best_params["window_size"]+1:zone_start+1]
		test_y = points[zone_start+1:]
		index_inpoints = zone_start+1
		preds_y = []
		for y in range(zone_length):
			pred_y = clf.predict(np.array(test_x))
			pred_y = pred_y[0]
			test_x = test_x[1:]
			test_x.append(pred_y)
			
			# Reverse the preprocessing
			if preprocessing_enable_scale:
				pred_y = postprocessing_scale([pred_y], points_mn, points_mx)[0]
			
			if preprocessing_enable_logtransformation:
				pred_y = postprocessing_logtransformation([pred_y], points_min)[0]
				
			if preprocessing_enable_seasonality:
				pred_y = pred_y + points_diff[index_inpoints]

			preds_y.append(str(pred_y))
			
			index_inpoints = (index_inpoints + 1) % period

		#Serialize prediction and best parameters
		preds_string = ';'.join(preds_y)
		parameters_string = json.dumps(best_params)
		
		if not run_dry:
			id_algo = ids_algo["SVR-linear"] if best_params["kernel"] == "linear" else ids_algo["SVR-rbf"]
			#print "id_algo = " , id_algo
			cur.execute("INSERT INTO " + TABLE_PREDICTION +  " VALUES('"+ str(row["id"]) +"', '"+str(id_algo)+"', '"+preds_string+"', '"+parameters_string+"')")
			
			#cur.execute("UPDATE " + TABLE_PREDICTION +  " SET prediction='"+preds_string+"', parameters='"+parameters_string+"' WHERE id_algorithm='"+str(id_algo)+"' AND id_zone='"+str(row["id"])+"'")

			db.commit()

		
		# Creation of copies to avoid potential value rewriting
		"""
		known_points_y = points_original[:zone_start]
		known_points_x = range(len(known_points_y))
		
		unknown_points_y = points_original[zone_start-1:]
		unknown_points_x = range(zone_start-1, zone_start -1 + len(unknown_points_y))
		
		
		test_xs_cp = [zone_start-1]
		test_xs_cp.extend(test_xs)
		
		test_y_cp = [points_original[zone_start-1]]
		test_y_cp.extend(test_y)
		
		preds_y_cp = [points_original[zone_start-1]]
		preds_y_cp.extend(preds_y)
		
		print "\n"
		print "test_x_cp = " , test_xs_cp
		print "test_y_cp = " , test_y_cp
		print "preds_y_cp = " , preds_y_cp
		
		print np.array(unknown_points_x).shape
		print np.array(unknown_points_y).shape
		
		
		# Plot results
		
		pl.plot(known_points_x, known_points_y, 'g')
		pl.plot(unknown_points_x, unknown_points_y, 'b')
		pl.plot(test_xs_cp, preds_y_cp, 'r')
		pl.show()

		continue
		"""
		

if db:
	cur.close()
	db.commit()
	db.close()
