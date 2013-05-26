#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
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
TABLE_PREDICTION_USER= "prediction"
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

display_step		= 100
run_dry				= True
testset_set_size 	= 70 # Percentage of available data for LS in the test set method

preprocessing_enable 					= True
preprocessing_enable_seasonality 		= True
preprocessing_enable_logtransformation 	= True
preprocessing_enable_scale				= True

correct_seasonality = False # Set to True to use the seasonality reported by the detection function, False otherwise
if not correct_seasonality:
	id_algo = 19
else: # Experimental IDs
	id_algo = 20


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
	#print "original points = " , points
	for k, p in enumerate(points):
		points[k] = float(p)# + random.random()/1000000;
	#print "noised points = " , points
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
	
	if not run_dry:
		# Check if a prediction exists. If yes, skip the time serie, otherwise reserve the entry
		cur.execute("SELECT COUNT(*) as cnt \
					  FROM "+ TABLE_PREDICTION +" p \
						WHERE id_algorithm='"+str(id_algo)+"' AND id_zone='"+str(row["id"])+"'\
					  ")
		
		response = cur.fetchone()
		if response["cnt"] >= 1:
			print "Skipped curve " , currentCurve , " id = " , row["id_curve"]
			continue
		
	
	######################################
	####	  COMPUTE PREDICTION	  ####
	######################################
	
	window_size = 5
	
	# Get user's predictions
	
	cur.execute("\
			\
			SELECT p.* \
			FROM " + TABLE_PREDICTION_USER + " p \
			LEFT JOIN " + TABLE_ZONE + " z ON p.id_zone = z.id \
			LEFT JOIN " + TABLE_CURVE + " c ON c.id = z.id_curve \
			LEFT JOIN " + TABLE_CATEGORY + " cat ON cat.id = c.id_category \
			LEFT JOIN " + TABLE_CATEGORY + " cat_par ON cat.id_parent = cat_par.id \
			LEFT JOIN " + TABLE_GROUP + " g ON g.id_category = cat.id \
			WHERE z.id='"+str(row["id"])+"' AND p.input='TREND' AND cat_par.name != 'Year'\
			  ")

	rows_user = cur.fetchall()
	if len(rows_user) == 0:
		continue
	user_x = []
	user_y = []
	
	for r in rows_user:
		#print "data = " , r["data"]
		points_user = r["data"].split(";")
		for k, p in enumerate(points_user):
			points_user[k] = float(p)
		
		# Preprocessing
		print "avant preprocess = " , points_user
		if preprocessing_enable:
			index = (zone_start + 1)%period
			for i, p in enumerate(points_user):
				if preprocessing_enable_seasonality:
					p = p - month_means[index]
					
				if preprocessing_enable_logtransformation:
					if p < points_min:
						p = points_min
					p = math.log((p - points_min)+1)
					
				if preprocessing_enable_scale:
					p = (2 * (p - points_mn) / (points_mx - points_mn)) - 1
					
				points_user[i] = p
				
				index = (index + 1) % period
		
		print "après preprocess = " , points_user
		ls_user = points[zone_start-window_size:zone_start];
		ls_user.extend(points_user);
		#print "ls_user = " , ls_user
		ls_x_user, ls_y_user = timeserieMakeOverlappingSet(ls_user, window_size)
		
		user_x.extend(ls_x_user);
		user_y.extend(ls_y_user);
		
	clf = KNeighborsRegressor(n_neighbors=10)
	#ls_x, ls_y = timeserieMakeOverlappingSet(ls_points, window_size)
	#clf = clf.fit(ls_x, ls_y)
	clf.fit(user_x, user_y)
	
	test_x = points[zone_start-window_size:zone_start]
	test_y = points[zone_start+1:]
	preds_y = []
	index_inpoints = zone_start+1
	for y in range(zone_length):
		pred_y = clf.predict(np.array(test_x))
		pred_y = pred_y[0]
		test_x = test_x[1:]
		test_x.append(pred_y)
		#print "pred = " , pred_y
		
		# Reverse the preprocessing
		if preprocessing_enable_scale:
			#print "ok"
			pred_y = postprocessing_scale([pred_y], points_mn, points_mx)[0]
		
		if preprocessing_enable_logtransformation:
			#print "ok"
			pred_y = postprocessing_logtransformation([pred_y], points_min)[0]
		
		#pred_y = postprocessing_gp([pred_y], zone_start + len(preds_y))[0]
		
		if preprocessing_enable_seasonality:
			#print "ok"
			pred_y = pred_y + points_diff[index_inpoints]
		#print "pred reversed = " , pred_y
		preds_y.append(str(pred_y))
		
		index_inpoints = (index_inpoints + 1) % period

	#Serialize prediction and best parameters
	preds_string = ';'.join(preds_y)
	parameters_string = json.dumps({"window_size": window_size})
	print "\n\n\nPredictions = " , preds_y
	if not run_dry:
		#print "Inserting " , preds_string
		cur.execute("INSERT INTO " + TABLE_PREDICTION +  " VALUES('"+ str(row["id"]) +"', '"+str(id_algo)+"', '"+preds_string+"', '"+parameters_string+"')")
		#cur.execute("UPDATE " + TABLE_PREDICTION +  " SET prediction='"+preds_string+"', parameters='"+parameters_string+"' \
		#	WHERE id_algorithm='"+str(ids_algo["GP"])+"' AND id_zone='"+str(row["id"])+"'")
		db.commit()
	
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
	"""
	print "\n"
	print "test_x_cp = " , test_xs_cp
	print "test_y_cp = " , test_y_cp
	print "preds_y_cp = " , preds_y_cp
	
	print np.array(unknown_points_x).shape
	print np.array(unknown_points_y).shape
	"""
	# Plot results
	
	pl.plot(known_points_x, known_points_y, 'g')
	pl.plot(unknown_points_x, unknown_points_y, 'b')
	pl.plot(test_xs_cp, preds_y_cp, 'r')
	pl.show()
	
	
		

if db:
	cur.close()
	db.commit()
	db.close()
