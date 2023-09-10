from astropy.timeseries import LombScargle

import numpy as np
from time import time as timer
try:
	import cupy as xp
except:
	import numpy as xp
from joblib import Parallel, delayed
#from .mle import design_matrix



def LS(TOOL,mag,magerr,time):

	#print("Starting LS...")

	"""
	Parameters
	----------
	mag : list
		list of magnitudes for given star
	magerr : list
		list of magnitude errors for given star
	time : list
		list of magnitudes for given star
	LS_array : numpy array
		Array where [0] is time taken to run and [1] is period
	TOOL : class
		Class from Tools.py, holds lc meta data 
	Returns
	-------
	numpy array with second element as LS period
	"""

	ls = LombScargle(time, mag)
	ls_spectrum = ls.power(TOOL.test_freqs)

	BAL_LS_FAP = ls.false_alarm_probability(ls_spectrum.max(), method = 'baluev')  
	LS_OUT = [0,0,0]
	LS_OUT[0]=TOOL.test_freqs
	LS_OUT[1]=ls_spectrum
	LS_OUT[2]=BAL_LS_FAP

	return LS_OUT



