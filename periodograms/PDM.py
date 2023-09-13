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

import numpy as np
import subprocess



#os.environ["OMP_NUM_THREADS"] = "1" #turn of the auto parellization


def make_dat(time, mag, magerr, par_version):
        lc_fp = os.path.abspath(os.path.dirname(sys.argv[0])) + "/LC_"+str(par_version)+".dat"
        f = open(lc_fp, "w+")
        f.write('time mag magerr\n')
        f.close()
        f = open(lc_fp, "a")
        for i in range(len(mag)):
            f.write(str(time[i])+' '+str(mag[i])+' '+str(np.abs(magerr[i]))+'\n')
        f.close()

def PDM(mag,magerr,time, par_version = 0, F_start = None, F_stop = 10, df = 0.0005, makedat = True):

    if F_start == None:
        F_start = 2/(max(time) - min(time))

    if makedat:
        make_dat(time, mag, magerr, par_version)


    current_file_dir = os.path.abspath(os.path.dirname(sys.argv[0]))

    bashCommand = [current_file_dir+"/pdm2_"+str(par_version),str(F_start),str(F_stop),str(df)]
    subprocess.call(bashCommand)
    attempt = 0
    cont = True
    while cont:
        try:
            freqs=[]
            pdm_spectrum=[]
            pdm_output_file = current_file_dir+"/pdmplot_"+str(par_version)+".csv"
            freqs,pdm_spectrum = np.genfromtxt(pdm_output_file, dtype = 'float', converters = None, unpack = True, comments = '#', delimiter = ',', skip_header = 1, skip_footer = 1)
            cont = False
        except Exception as e:
            print(repr(e))
            print(bashCommand)
            print(pdm_output_file)
            cont = True
            attempt = attempt +1
            if attempt > 5:
                cont = False

    PDM_OUT = [0,0]
    PDM_OUT[0]=freqs
    PDM_OUT[1]=pdm_spectrum

    os.remove(pdm_output_file)    
    return PDM_OUT

