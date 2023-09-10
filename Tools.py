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

	import CE as CE
	import cupy as xp
	run_ce = 1
except:
	print('NO GPU!')
	run_ce = 0
	import numpy as xp



#import my stuff
import PDM as PDM
import GP as GP
import LS as LS

import Synth_LC as Synth_LC
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






#os.environ["OMP_NUM_THREADS"] = "1" #turn of the auto parellization
#THIS WAS TURNED OFF FOR NO REASON IF CODE DOESNT WORK TRY THIS!!!!!!!!!!!!!!!

############################################
#ŠŠŠŠŠŠŠŠ  ŠŠŠŠŠŠ      ŠŠŠŠŠŠ   ŠŠŠŠŠŠ     #
#   ŠŠ    ŠŠ    ŠŠ     ŠŠ   ŠŠ ŠŠ    ŠŠ ŠŠ #
#   ŠŠ    ŠŠ    ŠŠ     ŠŠ   ŠŠ ŠŠ    ŠŠ    #
#   ŠŠ    ŠŠ    ŠŠ     ŠŠ   ŠŠ ŠŠ    ŠŠ ŠŠ #
#   ŠŠ     ŠŠŠŠŠŠ      ŠŠŠŠŠŠ   ŠŠŠŠŠŠ     #
#					   #
#ŠŠŠŠŠ ŠŠŠŠŠ ŠŠŠŠŠ ŠŠŠŠŠ ŠŠŠŠŠ ŠŠŠŠŠ       #
############################################
                                
###TODO		change lists to numpy ONLY when performing list calculations. NOT when appending or anyhting like that

###TODO		change all numpy to cupy where possible. NOTE how either one of the other are imported as 'xp' this allows it to switch back when gpu not available

###TODO		make it so the whole thing works WITHOUT GPU. CE does run on cpu i think...

###TODO 	/dt 

###TODO		svm to do the mira YSO, scikit learn

###TODO 	if tool io > 0, print final stuff

###TODO		Normalise phase folded  diagrams for LSTM and others, start a minima always 

###TODO		FAP as binned running scatter of phase fold

###TODO		FAP as LC vs same LC with synthetic signal for randomized and not randomized


class Tools:

######################################################
#  ______ _____ _      ______   _____ ____  
# |  ____|_   _| |    |  ____| |_   _/ __ \ 
# | |__    | | | |    | |__      | || |  | |
# |  __|   | | | |    |  __|     | || |  | |
# | |     _| |_| |____| |____   _| || |__| |
# |_|    |_____|______|______| |_____\____/ 
######################################################                                           
                                           
	####=================####
	####TOOLS SELF ASSIGN####
	####=================####

	def __init__(self):
		pass


	def lightcurve(self, mag = None, magerr = None, time = None,error_clip = False, s_fit = False, time_res = 0):

		if mag.all() == None and magerr.all() == None and time.all() == None:
			mag, magerr, time = self.synthesize()

		#===============#
		#ERROR_CLIP	#
		#===============#
		if error_clip:
			mag, magerr, time = self.error_clip_xy(mag, magerr, time, ast_res_chi, chi, ambi_match, sigma = 4, err_max = 0.5)

		if hasattr(mag, '__len__') and len(mag) > 10:		
			#===============#
			#SPLINE FIT	#
			#===============#
			if s_fit:
				mag, magerr, time = self.spline_fit(mag, magerr, time)
			
			#=========#
			#time_res #
			#=========#
			if time_res != 0:
				mag, magerr, time = self.time_res_reduce(mag, magerr, time, resolution = self.time_res)

			self.mag = np.squeeze(mag)
			self.magerr = np.squeeze(magerr)
			self.time = np.squeeze(time)
			

	def test_frequencies(self, F_start, F_stop, F_N, log=True):
		if log:
			return np.logspace(np.log10(F_start), np.log10(F_stop), F_N)
		else:
			return 1/np.linspace(1/F_start, 1/F_stop, F_N)




	def periodogram_save(self,x,y):
		OUTPUT = [x,y]
		if os.path.exists(self.output_dir+method) == False:
			os.mkdir(self.output_dir+method)
		f = open(self.output_dir+method+'/'+self.name+'_'+str(class_mod)+'.csv', "w+")
		for i in range(len(x)):
			f.write(str(x[i])+' '+str(y[i])+'\n')
		f.close()



			
	def make_dat(self, mag = None, magerr = None, time = None):

		if mag is None:
			mag = self.mag
		if magerr is None:
			magerr = self.magerr
		if time is None:
			time = self.time

		lc_fp = os.path.abspath(os.path.dirname(sys.argv[0])) + "/LC_0.dat"
		f = open(lc_fp, "w+")
		f.write('time mag magerr\n')
		f.close()
		f = open(lc_fp, "a")
		for i in range(len(mag)):
			f.write(str(time[i])+' '+str(mag[i])+' '+str(np.abs(magerr[i]))+'\n')
		f.close()
###############################################################
#           _   _          _  __     _______ _____  _____ 
#     /\   | \ | |   /\   | | \ \   / / ____|_   _|/ ____|
#    /  \  |  \| |  /  \  | |  \ \_/ / (___   | | | (___  
#   / /\ \ | . ` | / /\ \ | |   \   / \___ \  | |  \___ \ 
#  / ____ \| |\  |/ ____ \| |____| |  ____) |_| |_ ____) |
# /_/    \_\_| \_/_/    \_\______|_| |_____/|_____|_____/ 
################################################################                                                         
                                                         


	def pdm(self, mag = None, magerr = None, time = None):
		if self.do_pdm == 1:
			try:
				return PDM.PDM(self, mag, magerr, time)
			except Exception as e:
				print(repr(e))
				print("-----------------------")
				traceback.print_exc()
				print("!!!!!!!!!!!!!!!!!!!!!!!\n\n")
		else:
			return [0,0]


	def ls(self, mag = None, magerr = None, time = None):
		if self.do_ls == 1:
			try:
				return LS.LS(self, mag, magerr, time)
			except Exception as e:
				print(repr(e))
				print("-----------------------")
				traceback.print_exc()
				print("!!!!!!!!!!!!!!!!!!!!!!!\n\n")
				return [0,0]
		else: 
			return [0,0]




	def ce(self, mag = None, magerr = None, time = None):
		if self.do_ce == 1:
			ce_array = []
			#start CE, as cuda does NOT like running with anything else. cant see a fix for this
			#add infinite try and except to mitigate GPU ram issue 
			try_num = 0
			wait_time = 2
			try_flag = 0
			while try_flag == 0:
				try:
					ce_array = CE.CE(self, mag, magerr, time)
					try_flag = 1
				except Exception as e:
					print("\n\n!!!!!!!!CE ERROR!!!!!!")
					print("\n\n!!!!!!!!!!!!!!!!!!!!!!")
					print(repr(e))
					print("----------------------")
					#traceback.print_exc()
					print("!!!!!!!!!!!!!!!!!!!!!!\n\n")
					try_num += 1
					print('Failed on attept:', try_num, '\t waiting', wait_time, 'seconds...')
					sleep(wait_time)
					if try_num == 10:
						print("!!!!!!!!!!!!!!!!!!!!")
						ce_array = 0
						try_flag = 1
		else:
			ce_array = [0,0]
		return ce_array



	def run_all(self, mag, magerr, time):


		#t00 = timer()


		def pdm(self, PDM_array, mag = None, magerr = None, time = None):
			PDM_array[0] = self.pdm(mag, magerr, time)
			return PDM_array

		def ls(self, LS_array, mag = None, magerr = None, time = None):
			LS_array[0] = self.ls(mag, magerr, time)
			return LS_array

		def gp(self, GP_array, mag = None, magerr = None, time = None):
			GP_array[0] = self.gp(mag, magerr, time)
			return GP_array

		#list best for returning data when using parralell stuff
		manager = multiprocessing.Manager()
		PDM_array = manager.list(range(1))	#this is very stupid but it wont work otherwise
		LS_array = manager.list(range(1))	#this whole paralell stuff is wank but needed
		GP_array = manager.list(range(1))
		#print('\t\tC00\tTime', round(timer()-t00, 4))
		#start all the none cuda processes
		p_pdm = Process(target = pdm, args = (self, PDM_array, mag, magerr, time))
		p_ls = Process(target = ls, args = (self, LS_array, mag, magerr, time))
		p_gp = Process(target = gp, args = (self, GP_array, mag, magerr, time))
		#print('\t\tC01\tTime', round(timer()-t00, 4))
		p_pdm.start()
		p_ls.start()
		p_gp.start()
		#print('\t\tC02\tTime', round(timer()-t00, 4))
		#start them
		p_pdm.join()
		p_ls.join()
		p_gp.join()

		#print('\t\tC03\tTime', round(timer()-t00, 4))

		LS_array = LS_array[0]
		PDM_array = PDM_array[0]
		GP_array = GP_array[0]
		#t00 = timer()
		#print('\t\tC04\tTime', round(timer()-t00, 4))
		gc.collect()
		return LS_array, PDM_array, GP_array



	def run_all_nogp(self, mag, magerr, time):


		#t00 = timer()


		def pdm(self, PDM_array, mag = None, magerr = None, time = None):
			PDM_array[0] = self.pdm(mag, magerr, time)
			return PDM_array

		def ls(self, LS_array, mag = None, magerr = None, time = None):
			LS_array[0] = self.ls(mag, magerr, time)
			return LS_array

		#list best for returning data when using parralell stuff
		manager = multiprocessing.Manager()
		PDM_array = manager.list(range(1))	#this is very stupid but it wont work otherwise
		LS_array = manager.list(range(1))	#this whole paralell stuff is wank but needed
		#print('\t\tC00\tTime', round(timer()-t00, 4))
		#start all the none cuda processes
		p_pdm = Process(target = pdm, args = (self, PDM_array, mag, magerr, time))
		p_ls = Process(target = ls, args = (self, LS_array, mag, magerr, time))
		#print('\t\tC01\tTime', round(timer()-t00, 4))
		p_pdm.start()
		p_ls.start()
		#print('\t\tC02\tTime', round(timer()-t00, 4))
		#start them
		p_pdm.join()
		p_ls.join()

		#print('\t\tC03\tTime', round(timer()-t00, 4))

		LS_array = LS_array[0]
		PDM_array = PDM_array[0]
		#t00 = timer()
		#print('\t\tC04\tTime', round(timer()-t00, 4))
		gc.collect()
		return LS_array, PDM_array



	def print_short(self, name, period, peak):
		if self.IO > 2:
			print('\t#################\n\t', name)
			print('\tPeak:\t', peak)
			print('\tPeriod:\t', period,'[day]')
			if period < 1:
				print('\tPeriod:\t', period*24,'[hr]')
				print('\tPeriod:\t', period*24*60,'[min]')

	def gp_assign(self, GP_OUTPUT):
		if self.do_gp == 1:
			self.gp_lnlike = GP_OUTPUT[0]
			self.gp_b = GP_OUTPUT[1]
			self.gp_c = GP_OUTPUT[2]
			self.gp_p = abs(GP_OUTPUT[3])
			self.print_short('GP', self.gp_p, self.gp_lnlike)


	def ce_assign(self, CE_OUTPUT):
		if self.do_ce == 1:
			self.ce_y_y_0, self.ce_x_y_0, self.ce_peak_width_0, self.ce_y_y_1, self.ce_x_y_1, self.ce_peak_width_1, self.ce_y_y_2, self.ce_x_y_2, self.ce_peak_width_2, self.ce_q001, self.ce_q01, self.ce_q1, self.ce_q25, self.ce_q50, self.ce_q75, self.ce_q99, self.ce_q999, self.ce_q9999 = self.peak_analysis(CE_OUTPUT[0], CE_OUTPUT[1], method='CE')
			self.ce_p = 1.0/self.ce_x_y_0
			self.ce_period1 = 1.0/self.ce_x_y_1
			self.ce_period2 = 1.0/self.ce_x_y_2
			self.print_short('CE', self.ce_p, self.ce_y_y_0)


	def ls_assign(self, LS_OUTPUT):
		if self.do_ls == 1:
			try:
				self.ls_y_y_0, self.ls_x_y_0, self.ls_peak_width_0, self.ls_y_y_1, self.ls_x_y_1, self.ls_peak_width_1, self.ls_y_y_2, self.ls_x_y_2, self.ls_peak_width_2, self.ls_q001, self.ls_q01, self.ls_q1, self.ls_q25, self.ls_q50, self.ls_q75, self.ls_q99, self.ls_q999, self.ls_q9999 = self.peak_analysis(LS_OUTPUT[0], LS_OUTPUT[1], method='LS')
				self.ls_p = 1.0/self.ls_x_y_0
				self.ls_period1 = 1.0/self.ls_x_y_1
				self.ls_period2 = 1.0/self.ls_x_y_2

				self.ls_bal_fap = LS_OUTPUT[2]
				self.print_short('LS', self.ls_p, self.ls_y_y_0)
			except:
				pass

	def pdm_assign(self, PDM_OUTPUT):
		if self.do_pdm == 1:
			self.pdm_y_y_0, self.pdm_x_y_0, self.pdm_peak_width_0, self.pdm_y_y_1, self.pdm_x_y_1, self.pdm_peak_width_1, self.pdm_y_y_2, self.pdm_x_y_2, self.pdm_peak_width_2, self.pdm_q001, self.pdm_q01, self.pdm_q1, self.pdm_q25, self.pdm_q50, self.pdm_q75, self.pdm_q99, self.pdm_q999, self.pdm_q9999 = self.peak_analysis(PDM_OUTPUT[0], PDM_OUTPUT[1], method='PDM')
			self.pdm_p = 1.0/self.pdm_x_y_0
			self.pdm_period1 = 1.0/self.pdm_x_y_1
			self.pdm_period2 = 1.0/self.pdm_x_y_2
			self.print_short('PDM', self.pdm_p, self.pdm_y_y_0)


		
######################################################		
#  _      _____   _______ ____   ____  _       _____ #
# | |    / ____| |__   __/ __ \ / __ \| |     / ____|#
# | |   | |         | | | |  | | |  | | |    | (___  #
# | |   | |         | | | |  | | |  | | |     \___ \ #
# | |___| |____     | | | |__| | |__| | |____ ____) |#
# |______\_____|    |_|  \____/ \____/|______|_____/ #
######################################################


	def HB_period_unvertainty():
		#https://arxiv.org/pdf/1608.00650.pdf page 6
		pass


	def get_period_uncertainty(self, fx, fy, jmax, fx_width=100):

		#https://arxiv.org/pdf/1512.01611.pdf

		"""
		Get uncertainty of a period.
		The uncertainty is defined as the half width of the frequencies
		around the peak, that becomes lower than average + standard deviation
		of the power spectrum.
		Since we may not have fine resolution around the peak,
		we do not assume it is gaussian. So, no scaling factor of
		2.355 (= 2 * sqrt(2 * ln2)) is applied.
		Parameters
		----------
		fx : array_like
		    An array of frequencies.
		fy : array_like
		    An array of amplitudes.
		jmax : int
		    An index at the peak frequency.
		fx_width : int, optional
		    Width of power spectrum to calculate uncertainty.
		Returns
		-------
		p_uncertain : float
		    Period uncertainty.
		"""

		# Get subset
		start_index = jmax - fx_width
		end_index = jmax + fx_width
		if start_index < 0:
			start_index = 0
		if end_index > len(fx) - 1:
			end_index = len(fx) - 1

		fx_subset = fx[start_index:end_index]
		fy_subset = fy[start_index:end_index]
		fy_mean = np.median(fy_subset)
		fy_std = np.std(fy_subset)

		# Find peak
		max_index = np.argmax(fy_subset)

		# Find list whose powers become lower than average + std.
		index = np.where(fy_subset <= fy_mean + fy_std)[0]

		# Find the edge at left and right. This is the full width.
		left_index = index[(index < max_index)]
		if len(left_index) == 0:
			left_index = 0
		else:
			left_index = left_index[-1]
		right_index = index[(index > max_index)]
		if len(right_index) == 0:
			right_index = len(fy_subset) - 1
		else:
			right_index = right_index[0]

		# We assume the half of the full width is the period uncertainty.
		half_width = (1. / fx_subset[left_index]
			      - 1. / fx_subset[right_index]) / 2.
		period_uncertainty = half_width

		return period_uncertainty






	def NP_check(self, mag = None, magerr = None, time = None):

		def period_dupli_check(p_pdm, p_ls, p_ce):
			if p_pdm == p_ls:
				if p_ls == p_ce:
					periods = [(self.pdm_p+self.ce_p+self.ls_p)/3.]	
				else:
					periods = [(self.pdm_p+self.ls_p+self.ls_p)/2., self.ce_p]	
			elif p_ls == p_ce:
				periods = [(self.ce_p+self.ls_p)/2., self.pdm_p]	
			elif p_pdm == p_ce:
				periods = [(self.pdm_p+self.ce_p)/2., self.ls_p]	
			else:				
				periods = [self.pdm_p, self.ls_p, self.ce_p]
			return periods
		
		PDM_Array_New = []
		LS_Array_New = []
		CE_Array_New = []
		GP_Array_New = []
		original_ps = 1./self.test_freqs

		if mag is None:
			mag = self.mag
		if magerr is None:
			magerr = self.mag
		if time is None:
			time = self.time
			
		N = 10000		
		
		#if you round to 2 sig fig and then set the f grids to +/- 10% youll never miss anything with this method... i dont think...				
		round_to_n = lambda x, n: x if x == 0 else round(x, -int(maths.floor(maths.log10(np.abs(x)))) + (n - 1))

		p_pdm = round_to_n(self.pdm_p, 2)	
		p_ls = round_to_n(self.ls_p, 2)
		p_ce = round_to_n(self.ce_p, 2)			

		p1_ce = round_to_n(self.ce_period1, 2)
		p1_ls = round_to_n(self.ls_period1, 2)
		p1_pdm = round_to_n(self.pdm_period1, 2)


		p2_ce = round_to_n(self.ce_period2, 2)
		p2_ls = round_to_n(self.ls_period2, 2)
		p2_pdm = round_to_n(self.pdm_period2, 2)

		#decode what np check was wanted, this could be better...
		if self.np_check == 1:
			test_ranges = [[0.9, 1.1]]
		elif self.np_check == 2:
			test_ranges = [[0.4, 0.6], [0.9, 1.1]]
		elif self.np_check == 3:
			test_ranges = [[0.9, 1.1], [1.9, 2.1]]
		elif self.np_check == 4:
			test_ranges = [[0.4, 0.6], [1.9, 2.1]]
		elif self.np_check == 5:
			test_ranges = [[0.4, 0.6], [0.9, 1.1], [1.9, 2.1]]
		else:
			test_ranges = [[0.9, 1.1]]

		periods = period_dupli_check(p_pdm, p_ls, p_ce)

		#if NP CHECK > 5 then check minor periods too
		if self.np_check == 6:
			periods.extend(period_dupli_check(p1_pdm, p1_ls, p1_ce))
		if self.np_check > 6:
			periods.extend(period_dupli_check(p2_pdm, p2_ls, p2_ce))
		else:
			test_ranges = [[0.9, 1.1]]

		if self.tempio > 2:
			print('\n\t#################')
			print('\tSTARTING NP Check!\n\tN Periods', len(periods), '\n\tTests', len(periods)*len(test_ranges))
			
		pdm_np_x = self.pdm_period_x; pdm_np_y = self.pdm_period_y; ls_np_x = self.ls_period_x; ls_np_y = self.ls_period_y; ce_np_x = self.ce_period_x; ce_np_y = self.ce_period_y
		
		#dont run GP for this, tests showed its pointless
		temp_gp = 0
		if self.do_gp == 1:
			temp_gp = 1
			self.do_gp = 0

		for fs in test_ranges:
			for period in periods:
				if period == 0:
					pass
				else:
					try:
						period_start = max(original_freqs[np.where(original_ps <  period*fs[0])[0]])
					except:
						period_start = period*fs[0]
					try:
						period_stop = min(original_freqs[np.where(original_ps > period*fs[1])[0]]) 
					except:
						period_stop = period*fs[1]
					
					self.time_range = period_stop - period_start
					self.F_start = 1./(period_stop)
					self.F_stop = 1./(period_start)
					self.F_N = 30000
					self.test_freqs = self.test_frequencies(self.F_start, self.F_stop, self.F_N)
						
					LS_array, PDM_array, GP_array = self.run_all(mag, magerr, time)


					if self.do_pdm == 1:
						pdm_np_x, pdm_np_y = self.xy_merge(pdm_np_x, pdm_np_y, PDM_array[0], PDM_array[1])
					if self.do_ce == 1:
						ce_np_x, ce_np_y = self.xy_merge(ce_np_x, ce_np_y, CE_array[0], CE_array[1])
					if self.do_ls == 1:
						ls_np_x, ls_np_y = self.xy_merge(ls_np_x, ls_np_y, LS_array[0], LS_array[1])

		self.pdm_period_x = pdm_np_x; self.pdm_period_y = pdm_np_y; self.ls_period_x = ls_np_x; self.ls_period_y = ls_np_y; self.ce_period_x = ce_np_x; self.ce_period_y = ce_np_y

		self.ls_assign([self.ls_period_x, self.ls_period_y])
		self.ce_assign([self.ce_period_x, self.ce_period_y])
		self.pdm_assign([self.pdm_period_x, self.pdm_period_y])
		self.do_gp = temp_gp

	def xy_merge(self, x_old, y_old, x_new, y_new):

		x_comb = list(x_old) + list(x_new)
		y_comb = list(y_old) + list(y_new)

		sort = np.argsort(x_comb)
		x_out = np.array(x_comb)[sort]
		y_out = np.array(y_comb)[sort]		

		return x_out, y_out

	def del_list(self, l, id_to_del):
		arr = np.array(l)
		return list(np.delete(arr, id_to_del))


	def smooth(self,y, box_pts):
		box = np.ones(box_pts)/box_pts
		y_smooth = np.convolve(y, box, mode='same')
		return y_smooth


	def peak_analysis(self, x, y, method = None, peak_min = 0):

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
						
			observing_del_range = []
			for eperiods in self.exclusion_periods:
				observing_del_range.extend(list(np.where(np.logical_and(1/x>=eperiods[0], 1/x<=eperiods[1]))[0]))

			y_new = self.del_list(y, observing_del_range)
			x_new = self.del_list(x, observing_del_range)
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
			y_new = self.del_list(y, del_range)
			x_new = self.del_list(x, del_range)


			if self.IO > 2:
				print('\t~~~~~DELETE PEAK~~~~~~')
				print("\tDeleted :", len(del_range))
				print("\tNew list:", len(x_new))
				print("\tOld list:", len(x))
				print('\t~~~~~~~~~~~~~~~~~~~~~~~~~~')
				
			return x_new, y_new

		if method != None:

			if method == 'LS':
				peak_min = 0
				self.ls_period_x = x
				self.ls_period_y = y

			elif method == 'CE':
				peak_min = 1
				self.ce_period_x = x
				self.ce_period_y = y

			elif method == 'PDM':
				peak_min = 1
				self.pdm_period_x = x
				self.pdm_period_y = y

		try:
			#ensure sorted for easy peak stuff
			sort = np.argsort(x)
			x = np.array(x)[sort]	
			y = np.array(y)[sort]
		except:
			print(y)
			print('THIS HAS CRASHED')
			exit()

		#print(x, y)
		x, y = weird_check(x, y)
		#print('NOT REMOVING WEIRD')
		x, y = remove_observing_sampling(x, y)
		#print(x, y)

		q50 = np.median(y)

		q001, q01, q1, q25, q75, q99, q999, q9999 = self.IQR(y)

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

		#return Y peak value, X peak value (IE, extracted period), Y peak width for first 3 peaks
		#also return IQR stats
		return y_y_0, x_y_0, peak_width_0, y_y_1, x_y_1, peak_width_1, y_y_2, x_y_2, peak_width_2, q001, q01, q1, q25, q50, q75, q99, q999, q9999






	def norm_data(self,data):
		return (data - np.min(data)) / (np.max(data) - np.min(data))






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



	def dummy_LC_gen(self, mag, magerr):
		#gaussian of 
		mag = np.random.normal(loc=mag, scale=magerr)
		dummy_lc = Synth_LC.LCSim()
		# estimate observation quality
		_pc = dummy_lc.get_pc(mag, magerr)
		# get appropriate mag errors
		dummy_magerr = dummy_lc.get_emag(mag, _pc)
		# scatter model mags
		dummy_mag = np.random.normal(loc=mag, scale=dummy_magerr)
		return dummy_mag, dummy_magerr



	def spline_fit(self, mag, magerr, time, do_it_anyway = 0):
		def sl(x, A, B): # this is your 'straight line' y=f(x)
			return A*x + B
				
		y = np.array(mag)		# to fix order when called, (better to always to mag, time) CONSISTENCY!!!!!!!!!)
		yerr = np.array(magerr)
		x = np.array(time)

		res = 10
		rq50 = np.empty(res)
		rq25 = np.empty(res)
		rq75 = np.empty(res)
		Q75, Q25 = np.percentile(y, [75, 25])
		rx = np.linspace(min(x), max(x), res)
		rdelta = (max(x) - min(x))/(2*res)

		##bin need to have X points
		for i in range(res):
			check = []
			rdelta_temp = rdelta						
			while len(check) < 1:
				check = np.where((x < rx[i]+rdelta_temp) & (x > rx[i]-rdelta_temp))[0]
				rdelta_temp = rdelta_temp + 0.2*rdelta
			rq50[i] = np.median(y[check])
			try:
				rq75[i], rq25[i] = np.percentile(y[check], [75, 25])
			except:
				rq75[i], rq25[i] = rq75[i-1], rq25[i-1]


		RQ75, RQ25 = np.percentile(rq50, [75, 25])
		RIQR = abs(RQ75 - RQ25)

		
		#if the range of IQR of binned data changes alot when a single bin is removed, its probably transient
		IQRs = []
		for i in range(1,res):
			tq75, tq25 = np.percentile(np.delete(rq50,i), [75, 25])
			IQRs.append(abs(tq75-tq25))
		
		if abs(max(IQRs)-min(IQRs)) > 0.1 * RIQR:
			self.trans_flag = 1 

		try:
			popt, pcov = curve_fit(sl, rx, rq50) # your data x, y to fit
			self.grad = popt[0]
			self.intercept = popt[1]
			#generate fit

			y_fit = sl(x, popt[0], popt[1])
			#compare
			y_diff = y - y_fit
			#residual sum of squares
			ss_res = np.sum((y_diff) ** 2)
			#total sum of squares
			ss_tot = np.sum((y - np.mean(y)) ** 2)
			# r-squared
			r2 = 1 - (ss_res / ss_tot)
		
			if self.IO > 20:

				print('\t~~~~~~~SPLINE FIT~~~~~~~~~')
				print("\tCoefficient of determination = ", r2)
				print('\t~~~~~~~~~~~~~~~~~~~~~~~~~~')

				plt.clf()
				x_line = np.arange(min(x), max(x))
				plt.plot(x_line, sl(x_line, self.grad, self.intercept), 'k', label="$R^2\, =\, $"+str(r2))
				plt.plot(x, y, 'rx', label = 'Before')
				plt.errorbar(rx, rq50, yerr = (rq75-rq50, rq50-rq25), xerr = rdelta, ecolor = 'g', label = 'Running Median')			
				plt.plot(x, y_diff+np.median(y), 'b+', label = 'After')
				plt.xlabel('Time [JD]')
				plt.ylabel('Magnitude [mag]')
				plt.grid()
				plt.legend()
				plt.gca().invert_yaxis()
				plt.savefig(self.info_figure+self.name+'_spline.png', format = 'png', dpi = image_dpi) 
				plt.clf()
		except:
			r2 = -999
		if r2 > 1 or do_it_anyway == 1:
			return y_diff+np.median(y), yerr, x
		else:
			return y, yerr, x


	def sine_fit(self, mag, time, period):
		try:
			def sinus(x, A, B, C): # this is your 'straight line' y=f(x)
			    return (A * np.sin((2.*np.pi*x)+C))+B

			y = np.array(mag)		# to fix order when called, (better to always to mag, time) CONSISTENCY!!!!!!!!!)
			x = self.phaser(time, period)
			popt, pcov = curve_fit(sinus, x, y, bounds=((self.true_amplitude*0.3, self.mag_avg*0.3, -2), (self.true_amplitude*3.0, self.mag_avg*3, 2)))#, method = 'lm') # your data x, y to fit
			
			y_fit = sinus(x, popt[0], popt[1], popt[2])
			#compare
			y_diff = y - y_fit
			#residual sum of squares
			ss_res = np.sum((y_diff) ** 2)
			#total sum of squares
			ss_tot = np.sum((y - np.mean(y)) ** 2)
			# r-squared
			r2 = 1 - (ss_res / ss_tot)		#coefficient of determination
			
			
			if make_plot:

				plt.clf()
		
				text_font = {'family': 'serif', 'color':  'darkred', 'weight': 'normal', 'size': 8, }
				text_pos = 0.001
				sort = np.argsort(x)
				m = np.array(y)[sort]			
				
				plt.title('Amplitude = '+str(round(popt[0], 3))+'  Offset = '+str(round(popt[1], 3))+'  Phase Shift = '+str(round(popt[2], 3)))
								
				x_line = np.arange(0, 1.01, 0.01)
				plt.plot(x, y, 'rx', markersize=4, label = 'Before')
				plt.plot(x+1., y, 'rx', markersize=4)

				plt.plot(x_line, sinus(x_line, popt[0], popt[1], popt[2]), 'k', label="$R^2\, =\, $"+str(round(r2,3)))
				plt.plot(x_line+1., sinus(x_line, popt[0], popt[1], popt[2]), 'k')

				plt.plot(x, y_diff+np.median(y), 'b+', label = 'After')
				plt.plot(x+1., y_diff+np.median(y), 'b+')
				
				plt.xlabel('Phase')
				plt.ylabel('Magnitude [mag]')
				plt.grid()
				plt.legend()
				plt.gca().invert_yaxis()
				name = str('{:0>8d}'.format(int(period * 100000)))
				plt.savefig(str(name)+'_sine.png', format = 'png', dpi = image_dpi)

			return y_diff+np.median(y), x
		except Exception as e:
			print("\n\n!!!!!!!!!!!!?????????????????!!!!!!!!!!")
			print(repr(e))
			traceback.print_exc()
			print("\n\n!!!!!!!!!!!!?????????????????!!!!!!!!!!")
			return y, x



	def error_clip_xy(self, values_y, values_y_err, values_x, sigma = 3, exit_cond = 5, err_max = 1):

		def idx_clip(y, yerr, x, idxs):
			bad_y = xp.squeeze(y[idxs])
			bad_y_err = xp.squeeze(yerr[idxs])
			bad_x = xp.squeeze(x[idxs])
			gud_y = xp.delete(y,idxs)
			gud_y_err = xp.delete(yerr,idxs)
			gud_x = xp.delete(x,idxs)
			return gud_y, gud_y_err, gud_x, bad_y, bad_y_err, bad_x
			
		mag_nans = np.argwhere(~np.isnan(np.array(values_y)))
		values_y = values_y[mag_nans]
		values_y_err = values_y_err[mag_nans]
		values_x = values_x[mag_nans]

		med = xp.median(values_y)
		std = xp.std(values_y)
		mederr = xp.median(values_y_err)
		stderr = xp.std(values_y_err)

		sigma_clip_ind_0 = xp.where((values_y > med + sigma*std))[0]
		sigma_clip_ind_1 = xp.where((values_y < med - sigma*std))[0]
		sigma_clip_ind_2 = xp.where((values_y_err > err_max))[0]
		sigma_clip_ind_3 = xp.where((values_y_err > mederr + sigma*stderr))[0]
		normal_clip_ind_0 = xp.where((values_y < 5))[0]
		normal_clip_ind_1 = xp.where((values_y > 100))[0]
		err_clip = xp.where((values_y_err > mederr*3))[0]



		deletes = np.unique(list(sigma_clip_ind_0) + list(sigma_clip_ind_1) + list(sigma_clip_ind_2) + list(sigma_clip_ind_3) + list(normal_clip_ind_0) + list(normal_clip_ind_1) + list(err_clip))      

		try:
			gud_y, gud_y_err, gud_x, bad_y, bad_y_err, bad_x = idx_clip(values_y, values_y_err, values_x, deletes)
		except:
			if len(deletes) < 1:
				return values_y, values_y_err, values_x

		return np.squeeze(gud_y), np.squeeze(gud_y_err), np.squeeze(gud_x)

