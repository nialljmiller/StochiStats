# -*- coding: utf-8 -*-
#OS and General
from os import listdir
from os.path import isfile, join
import os
import os.path
from multiprocessing import Pool
import multiprocessing
import sys
import gc;gc.disable()

import argparse

#Maths stuff
import time as Timetime
from random import seed;seed(69420)
from random import random
import random
import math
import numpy as np
#import subprocess
from time import time as timer


try:
	import cupy as xp
except:
	import numpy as xp


import matplotlib;matplotlib.rcParams['agg.path.chunksize'] = 10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


from multiprocessing import Process
from multiprocessing import Array
import subprocess



# == == == == == == == == == == == == == == = #
#Phase Dispersion Minimisation#
# == == == == == == == == == == == == == == = #
from PyAstronomy.pyTiming import pyPDM
#import Tools as T
import ctypes
# == == == == == = #
#Global Vars#
# == == == == == = #

image_dpi = 200

lc_meta_dir = "/beegfs/car/lsmith/virac_v2/data/output/agg_tables/"
lc_file_dir = "/beegfs/car/lsmith/virac_v2/data/output/ts_tables/"
lc_FITS_dir = "/home/njm/FITS/"
lc_graph_dir = "/home/njm/L_S/"
gp_post_dir = "/home/njm/GP_POST/"
os.environ["OMP_NUM_THREADS"] = "1" #turn of the auto parellization


# == == == == == == = 
#PDM##########::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# == == == == == == = 

def PDM(TOOL,mag,magerr,time, par_version = 0):

	current_file_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
	#bashCommand = current_file_dir+"/pdm2 "+str(TOOL.F_start)+' '+str(TOOL.F_stop)+' '+str(TOOL.df)+''
	if TOOL.pdm_p > 50:
		F_start = 2.0 / TOOL.pdm_p

	bashCommand = [current_file_dir+"/pdm2_"+str(par_version),str(TOOL.F_start),str(TOOL.F_stop),str(TOOL.df)]
	tpdm = timer()
	subprocess.call(bashCommand)
	attempt = 0
	cont = True
	while cont:
		try:
			freqs=[]
			pdm_spectrum=[]
			#os.system(bashCommand)
			#open the output from the C script 
			pdm_output_file = current_file_dir+"/pdmplot_"+str(par_version)+".csv"
			freqs,pdm_spectrum = np.genfromtxt(pdm_output_file, dtype = 'float', converters = None, unpack = True, comments = '#', delimiter = ',', skip_header = 1, skip_footer = 1)
			cont = False
		except Exception as e:
			print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
			print(repr(e))
			print(bashCommand)
			print(pdm_output_file)
			print('\t==========================')
			print('\tName (id):',TOOL.name)
			print('\tN:',len(mag))
			print('\tMag:',np.median(mag))
			print('\tMag error:',np.median(magerr))
			print('\t==========================\n')
			cont = True
			attempt = attempt +1
			if attempt > 5:
				cont = False

	PDM_OUT = [0,0]
	PDM_OUT[0]=freqs
	PDM_OUT[1]=pdm_spectrum

	os.remove(pdm_output_file)	
	return PDM_OUT

