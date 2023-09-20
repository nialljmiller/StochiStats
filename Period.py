# -*##- coding: utf-8 -*-
#OS and General
import sys
import os
from os import listdir
import os.path
from os.path import isfile, join
import pathlib
from time import sleep
import traceback

#computing
from multiprocessing import Pool, Process, Queue
import multiprocessing
import gc;gc.enable()
import csv
#from streamlit import caching

#Maths stuff
#import today
from datetime import date
import time
from time import time as timer
from random import random
from random import randrange
import random
from random import seed;seed(69420)#
import math as maths			#because im not a troglodyte
import numpy as np

#scipy
from scipy import stats
from scipy import signal as sg
from scipy.optimize import curve_fit

#matplotlib
import matplotlib;matplotlib.rcParams['agg.path.chunksize'] = 10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
import matplotlib.pyplot as plt
import matplotlib.colors as mplcol
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from matplotlib import gridspec



try:

#	import CE as CE
	import cupy as xp
	run_ce = 1
except:
	print('NO GPU!')
	run_ce = 0
	import numpy as xp



#import my stuff
#import PDM as PDM
#import GP as GP
#import LS as LS


# == == == == == = #
#Global Vars#
# == == == == == = #

image_dpi = 300

dpi = 666  # 200-300 as per guidelines
maxpix = 3000  # max pixels of plot
width = maxpix / dpi  # max allowed with
matplotlib.rcParams.update({'axes.labelsize': 'small', 'axes.titlesize': 'small',  # the size of labels and title
                 'xtick.labelsize': 'small', 'ytick.labelsize': 'small',  # the size of the axes ticks
                 'legend.fontsize': 'x-small', 'legend.frameon': False,  # legend font size, no frame
                 'legend.facecolor': 'none', 'legend.handletextpad': 0.25,
                 # legend no background colour, separation from label to point
                 'font.serif': ['Computer Modern', 'Helvetica', 'Arial',  # default fonts to try and use
                                'Tahoma', 'Lucida Grande', 'DejaVu Sans'],
                 'font.family': 'serif',  # use serif fonts
                 'mathtext.fontset': 'cm', 'mathtext.default': 'regular',  # if in math mode, use these
                 'figure.figsize': [width, 0.7 * width], 'figure.dpi': dpi,
                 # the figure size in inches and dots per inch
                 'lines.linewidth': .75,  # width of plotted lines
                 'xtick.top': True, 'ytick.right': True,  # ticks on right and top of plot
                 'xtick.minor.visible': True, 'ytick.minor.visible': True,  # show minor ticks
                 'text.usetex': True, 'xtick.labelsize':'medium',
                 'ytick.labelsize':'medium'})  # process text with LaTeX instead of matplotlib math mode


def lightcurve_clip(mag = None, magerr = None, time = None,error_clip = False, s_fit = False, time_res = 0):
	if error_clip:
		mag, magerr, time = self.error_clip_xy(mag, magerr, time, ast_res_chi, chi, ambi_match, sigma = 4, err_max = 0.5)

	if hasattr(mag, '__len__') and len(mag) > 10:		

		if s_fit:
			mag, magerr, time = self.spline_fit(mag, magerr, time)
		
		if time_res != 0:
			mag, magerr, time = self.time_res_reduce(mag, magerr, time, resolution = self.time_res)

	return mag, magerr, time





def peak_analysis(x, y, peak_min = 0):


	def del_list(l, id_to_del):
		arr = np.array(l)
		return list(np.delete(arr, id_to_del))



	def weird_check(x, y):
		#find and remove all shite in it
		where_are_NaNs = np.isnan(y)
		where_are_infs = np.isinf(y)
		where_are_high = np.where(np.logical_or(y>10000, y<-10000))[0]

		y[where_are_high] = np.median(y)
		y[where_are_NaNs] = np.median(y)
		y[where_are_infs] = np.median(y)
		return x, y

	def remove_observing_sampling(x, y):

		exclusion_periods = [[0.0,0.0001],[0.99,1.01],[0.664,0.666],[1.994,1.995],
				[0.498,0.54],[27,34],[363,368],[0.14246,0.1428],[0.16620,0.16622],[0.19944,0.19946],
				[0.12464,0.12466],[0.11080,0.11082],[0.24930,0.24935],[0.33241,0.33243],[0.66483,0.68486],
				[2.990,3.0015],[426,426.5],[496.5,497],[506.7,507],[331.5,332],[298.5,299]]
					
		observing_del_range = []
		for eperiods in exclusion_periods:
			observing_del_range.extend(list(np.where(np.logical_and(1/x>=eperiods[0], 1/x<=eperiods[1]))[0]))

		y_new = del_list(y, observing_del_range)
		x_new = del_list(x, observing_del_range)
		return x_new[4:], y_new[4:]

	def identify_peak(x, y, peak_min = 0):

		if peak_min == 1:
			#PROPERLY IDENTIFY PEAK
			peak_id = int(np.argmin(y))	#get min id
			#print(peak_id)
			#find adding edge
			yip = y[peak_id]
			add_edge = peak_id
			add_edge_list = list(range(peak_id, len(y)-1))
			add_edge_list.sort()
			y_add = [y[i] for i in add_edge_list]
			#print(y_add)
			for i, yi in enumerate(y_add):
				if yi > yip:
					yip = yi
					add_edge = add_edge_list[i]
				else:
					add_edge = add_edge_list[i]
					break

			#find subtracting edge
			yip = y[peak_id]
			sub_edge = peak_id
			sub_edge_list = list(range(0, peak_id))
			sub_edge_list.sort(reverse=True)
			y_sub = [y[i] for i in sub_edge_list]
			for i, yi in enumerate(y_sub):
				if yi > yip:
					yip = yi
					sub_edge = sub_edge_list[i]
				else:
					sub_edge = sub_edge_list[i]
					break

		else:	
			#PROPERLY IDENTIFY PEAK
			peak_id = int(np.argmax(y))	#get min id
			#print(peak_id)
			#find adding edge
			yip = y[peak_id]
			add_edge = peak_id
			add_edge_list = list(range(peak_id, len(y)-1))
			add_edge_list.sort()
			y_add = [y[i] for i in add_edge_list]
			#print(y_add)
			for i, yi in enumerate(y_add):
				if yi < yip:
					yip = yi
					add_edge = add_edge_list[i]

				else:
					add_edge = add_edge_list[i]
					break

			#find subtracting edge
			yip = y[peak_id]
			sub_edge = peak_id
			sub_edge_list = list(range(0, peak_id))
			sub_edge_list.sort(reverse=True)
			y_sub = [y[i] for i in sub_edge_list]
			for i, yi in enumerate(y_sub):
				if yi < yip:
					yip = yi
					sub_edge = sub_edge_list[i]

				else:
					sub_edge = sub_edge_list[i]
					break

		#https://arxiv.org/pdf/1512.01611.pdf
		# We assume the half of the full width is the period uncertainty.
		
		peak_width = abs(x[sub_edge] - x[add_edge]) 
		return peak_id, peak_width



	def delete_peak(x, y, peak_id):

		ids = np.where(np.logical_and(x>=x[peak_id]*0.9, x<=x[peak_id]*1.1))[0]

		if len(ids) < 2:
			ids = [peak_id-2,peak_id-1,peak_id,peak_id+1,peak_id+2]

		if len(ids) > 0.4*len(x):
			ids = np.where(np.logical_and(x>=x[peak_id]*0.99, x<=x[peak_id]*1.11))[0]
			if len(ids) > 0.45*len(x):
				ids = [peak_id]
		y_new = []
		x_new = []

		del_range = range(min(ids)-5, max(ids)+6, 1)
		del_range = [item for item in del_range if item >= 0 and item < len(y)]
		y_new = del_list(y, del_range)
		x_new = del_list(x, del_range)

		return x_new, y_new


	#ensure sorted for easy peak stuff
	sort = np.argsort(x)
	x = np.array(x)[sort]	
	y = np.array(y)[sort]

	#print(x, y)
	x, y = weird_check(x, y)
	#print('NOT REMOVING WEIRD')
	x, y = remove_observing_sampling(x, y)
	#print(x, y)

	q001, q01, q1, q25, q50, q75, q99, q999, q9999 = np.percentile(y, [0.01, 0.1, 1, 25, 50, 75, 99, 99.9, 99.99])
	ymean = np.mean(y)
	ystd = np.std(y)


	peak_id_0, peak_width_0 = identify_peak(x, y, peak_min = peak_min)
	y_y_0 = y[peak_id_0]
	x_y_0 = x[peak_id_0]
	x,y = delete_peak(x, y, peak_id_0)

	peak_id_1, peak_width_1 = identify_peak(x, y, peak_min = peak_min)
	y_y_1 = y[peak_id_1]
	x_y_1 = x[peak_id_1]
	x,y = delete_peak(x, y, peak_id_1)

	peak_id_2, peak_width_2 = identify_peak(x, y, peak_min = peak_min)
	y_y_2 = y[peak_id_2]
	x_y_2 = x[peak_id_2]
	x,y = delete_peak(x, y, peak_id_2)

	# Create dictionaries with variable names as keys
	peak0 = {'Peak Height': y_y_0, 'Period': 1/x_y_0, 'Peak Width': peak_width_0}
	peak1 = {'Peak Height': y_y_1, 'Period': 1/x_y_1, 'peak Width': peak_width_1}
	peak2 = {'Peak Height': y_y_2, 'Period': 1/x_y_2, 'peak Width': peak_width_2}

	xystats = [ymean, ystd, q001, q01, q1, q25, q50, q75, q99, q999, q9999]
	return peak0, peak1, peak2, xystats









def time_res_reduce(self, mag = None, magerr = None, time = None, resolution = 0.5):
		
	if mag is None:
		mag = self.mag
	if magerr is None:
		magerr = self.magerr
	if time is None:
		time = self.time
		
	mag_out = []; magerr_out = []; time_out = []; ignore = []
	removed = 0
	group_count = 0
	merge_count= 0 
	for i in range(len(time)):
		if i not in ignore:
			#find datapoints withing rannge
			group_id = np.where(np.logical_and(time<=time[i]+resolution, time>=time[i]-resolution))[0]

			if len(group_id) > 1:
				merge_count = merge_count + 1
				group_count = group_count + len(group_id)

				#append to ignore list
				group_mag=[]; group_magerr=[]; group_time=[]
				for ig in group_id:
					ignore.append(ig)
					group_mag.append(mag[ig])
					group_magerr.append(magerr[ig])
					group_time.append(time[ig])

				#if theres more than 1, append group to new lists
				#if theyre not within eachothers error bars, delete them, else, append to output lists
				if group_mag[0] - group_magerr[0] < group_mag[-1] + group_magerr[-1]*2 or group_mag[0] + group_magerr[0] > group_mag[-1] - group_magerr[-1]*2:
					mag_out.append(float(np.median(group_mag)))
					magerr_out.append(float(np.median(group_magerr)/np.sqrt(len(group_mag))))
					time_out.append(float(np.median(group_time)))
				else:
					removed = removed + 1
			else:	
				ignore.append(i)
				mag_out.append(float(mag[i]))
				magerr_out.append(float(magerr[i]))
				time_out.append(float(time[i]))
	if self.IO > 2:
		print('\t~~~~~TIME RES REDUCE~~~~~~')
		print("\tMerged :", group_count)
		print("\tDeleted :", removed)
		print("\tNew list:", len(mag_out))
		print("\tOld list:", len(mag))
		print('\t~~~~~~~~~~~~~~~~~~~~~~~~~~')
		
	return np.squeeze(mag_out), np.squeeze(magerr_out), np.squeeze(time_out)


