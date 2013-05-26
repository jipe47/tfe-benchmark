#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pylab as pl
import MySQLdb as mdb
import sys
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


correct_seasonality = True

if not correct_seasonality:
	#ids_algo = {"ANN" : 23}
	ids_algo = {"ANN" : 3}
else:
	#ids_algo = {"ANN" : 24}
	ids_algo = {"ANN" : 10}

################################
#		  PARAMETERS		  #
################################

range_window_size = range(1, 20)

display_step		= 10
run_dry				= False
testset_set_size 	= 70 # Percentage of available data for LS in the test set method
testset_use_ownprediction = False # True to use algorithm's predictions in the validation step, false otherwise

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
currentCurve = 0

epochs = 15
range_hiddenSize = range(1, 25)
nbr_iter = 10

for row in rows:
	currentCurve = currentCurve + 1
	
	if currentCurve < 2:
		continue
		
	if not run_dry:
		# Check if a prediction exists. If yes, skip the time serie, otherwise reserve the entry
		cur.execute("SELECT COUNT(*) as cnt \
					  FROM "+ TABLE_PREDICTION +" p \
						WHERE id_algorithm='"+str(ids_algo["ANN"])+"' AND id_zone='"+str(row["id"])+"'\
					  ")
		
		response = cur.fetchone()
		
		if response["cnt"] == 1:
			print "Skipped curve " , row["id_curve"]
			continue
		else:
			cur.execute("INSERT INTO "+ TABLE_PREDICTION +" (id_algorithm, id_zone, parameters, prediction) \
					 VALUES('"+str(ids_algo["ANN"])+"', '"+str(row["id"])+"', '', '')\
				  ")
			db.commit()

	# Treat data from database
	points = row["points"].split(";")
	for k, p in enumerate(points):
		points[k] = float(p)
		
	
	points_original = list(points)
	
	# Seasonality removal
	seas = findSeasonality(points)
	period = len(points)
	#pl.plot(points_original, 'b')
	
	if seas == -1 or row["period"] == "Year":
		points_diff = [0 for i in range(len(points))]
		month_means = np.zeros(period)
	else:
		if not correct_seasonality:
			period = 4 if row["period"] == "Quart" else 12
		#period = seas
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
	
	# Log transformation
	points_min, points = preprocessing_logtransformation(points)
		
	# Scale the function to [-1, 1]
	points_mn, points_mx, points = preprocessing_scale(points)
	
	######################################
	####        ZONE SEPARATION       ####
	######################################

	zone_start = int(row["start"])
	zone_end = int(row["end"])
	
	zone_length = zone_end - zone_start
	
	test_xs	 = range(zone_start, zone_end)
	ls_points   = points[:zone_start]
	ls_xs	   = range(0, zone_start)
	
	######################################
	####   BEST PARAMETER DISCOVERY   ####
	######################################
	
	current_params = {}
	
	parameters = []
	errors = []

	total_test = len(range_window_size) * len(range_hiddenSize)*nbr_iter

	it = 0
	for window_size in range_window_size:
		if window_size > zone_start:
			continue

		ls_x, ls_y = timeserieMakeOverlappingSet(ls_points, window_size)

		# Separation of LS and TS
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
		
		for hiddenSize in range_hiddenSize:
		
			current_params["window_size"] = window_size
			current_params["hiddenSize"] = hiddenSize
			
			preds_y = np.zeros(len(y_ts))
			
			for j in range(nbr_iter):
				if it % display_step == 0:
					print "ANN - curve " , currentCurve , " (id_curve = " , row["id_curve"] , ", id_zone = " , row["id"] , ") - " , it , " / " , total_test

				clf = makeAndTrainANN(x_ls, y_ls, window_size, hiddenSize=hiddenSize, epochs=epochs)
				test_x = x_ts[0]
					
				y_ts_original = []

				for i, y in enumerate(y_ts):
					if testset_use_ownprediction:
						pred_y = clf.activate(np.array(test_x))[0]
						test_x = test_x[1:]
						test_x.append(pred_y)
					else:
						pred_y = clf.activate(np.array(x_ts[i]))[0]
					preds_y[i] = preds_y[i] + pred_y
				
				it = it + 1
					
			for i, y in enumerate(y_ts):
				preds_y[i] = preds_y[i] / float(nbr_iter)
		
			parameters.append(dict(current_params))
			errors.append(computeError(y_ts, preds_y))

	# Find parameters than minimise the error on TS
	best_params = findBestParameters(errors, parameters)
	
	######################################
	####	  COMPUTE PREDICTION	  ####
	######################################
						
	ls_x, ls_y = timeserieMakeOverlappingSet(ls_points, best_params["window_size"])
					
	preds_y = np.zeros(zone_length)
	for j in range(nbr_iter):
	
		index_inpoints = zone_start % period
		clf = makeAndTrainANN(ls_x, ls_y, best_params["window_size"], hiddenSize=best_params["hiddenSize"], epochs=epochs)
		
		test_x = points[zone_start-best_params["window_size"]+1:zone_start+1]
		
		print "len test_x = " , len(test_x)
		print "len points = " , len(points)
		print "zone_start = ",  zone_start
		print "zone_end = " , zone_end
		print "test_x = " , test_x
		print "points = " ,  points
		
		
		for i, y in enumerate(range(zone_length)):
			pred_y = clf.activate(np.array(test_x))
			pred_y = pred_y[0]
			test_x = test_x[1:]
			test_x.append(pred_y)
			
			# Reverse the preprocessing
			pred_y = postprocessing_scale([pred_y], points_mn, points_mx)[0]
			pred_y = postprocessing_logtransformation([pred_y], points_min)[0]
			pred_y = pred_y + points_diff[index_inpoints]
			index_inpoints = (index_inpoints + 1) % period

			# Store the prediction
			preds_y[i] = preds_y[i] + pred_y
			
	# Compute prediction means
	preds_y_str = []
	for j, pred_y in enumerate(preds_y):
		preds_y[j] = pred_y / float(nbr_iter)
		preds_y_str.append(str(preds_y[j]))
	print "len predictions = " , len(preds_y_str)
	# Serialize prediction and best parameters
	preds_string = ';'.join(preds_y_str)
	parameters_string = json.dumps(best_params)
	
	if not run_dry:
		#cur.execute("INSERT INTO " + TABLE_PREDICTION +  " VALUES('"+ str(row["id"]) +"', '"+str(ids_algo["ANN"])+"', '"+preds_string+"', '"+parameters_string+"')")
		cur.execute("UPDATE " + TABLE_PREDICTION +  " SET prediction='"+preds_string+"', parameters='"+parameters_string+"' \
				WHERE id_algorithm='"+str(ids_algo["ANN"])+"' AND id_zone='"+str(row["id"])+"'")
		db.commit()
	"""
	pl.plot(points_original)
	
	#preds_y = ([None]*zone_start).extend(preds_y)
	ys = [None]*(zone_start + 1)
	ys.extend(preds_y)
	pl.plot(ys)
	pl.show()
	break;
	"""

if db:
	cur.close()
	db.commit()
	db.close()
