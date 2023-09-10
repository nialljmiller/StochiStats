 # -*- coding: utf-8 -*-
         #OS and General
from os import listdir
from os.path import isfile, join
import os
import os.path
from multiprocessing import Pool
import multiprocessing
import sys
import gc;gc.enable()
import argparse

#Maths stuff
import time
from random import seed;seed(69420)
from random import random
import random
import math
#import numpy as np
import autograd.numpy as np


try:
	import cupy as xp
except:
	import numpy as xp


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl;mpl.rcParams['agg.path.chunksize'] = 1000000
import pylab as pl
# == == == == == == == == #
#Gaussian Process#
# == == == == == == == == #
import corner
#import george
#from george.kernels import ExpSquaredKernel, ExpSine2Kernel#, WhiteKernel
import celerite
from celerite import terms
import emcee
from scipy import stats
from scipy import optimize #Leastsq Levenberg-Marquadt Algorithm
from scipy.optimize import minimize
from scipy.optimize import least_squares
from time import time as timer
#import Tools as T

# == == == == == = #
#Global Vars#
# == == == == == = #

image_dpi = 500
os.environ["OMP_NUM_THREADS"] = "1" #turn of the auto parellization

def GP(TOOL,mag,magerr,time):#,CE_period):		<dont know if this will break it, !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	class CustomTerm(terms.Term):
		parameter_names = ("log_b", "log_c", "log_xx", "log_P")

		def get_real_coefficients(self, params):
			log_b, log_c, log_xx, log_P = params
			c = np.exp(log_c)
			return (
				np.exp(log_c) * (1.0 + c) / (2.0 + c), np.exp(log_xx),
			)

		def get_complex_coefficients(self, params):
			log_b, log_c, log_xx, log_P = params
			c = np.exp(log_c)
			return (
				np.exp(log_b) / (2.0 + c), 0.0,	np.exp(log_xx), 2*np.pi*np.exp(-log_P),
			)


	def nll(p, y, gp):
		
		# Update the kernel parameters:
		gp.set_parameter_vector(p)
		
		#  Compute the loglikelihood:
		ll = gp.log_likelihood(y)
		
		# The scipy optimizer doesn’t play well with infinities:
		return -ll if np.isfinite(ll) else 1e25

	def grad_nll(p, y, gp):
		
		# Update the kernel parameters:
		gp.set_parameter_vector(p)
		
		#  Compute the gradient of the loglikelihood:
		gll = gp.grad_log_likelihood(y)[1]
		
		return -gll

	# set the logprior
	def lnprior(p):
	 
		# These ranges are adapted from Table 3
		# of https://arxiv.org/pdf/1703.09710.pdf

		lnB = p[0]
		lnC = p[1]
		lnL = p[2]
		P = p[3]

		return gp.log_prior()


	def lnprob(p, x, y):
		#print(p,'aaaaaaaaaa')
		lp = lnprior(p)
		#print(lp + lnlike(p, x, y))
		return lp + lnlike(p, x, y) if np.isfinite(lp) else -np.inf

	# set the loglikelihood:

	def lnlike(p, x, y):
	 
		ln_a = p[0]
		ln_b = p[1]
		ln_c = p[2]      # we pass ln(c) to the CustomKernel, ln(c) = -ln(L)
		ln_p = np.log(p[3])  # we're sampling linearly from P so we need to log it

		p0 = np.array([ln_a,ln_b,ln_c,ln_p])

		# update kernel parameters:
		gp.set_parameter_vector(p0)

		# calculate the likelihood:

		try:
			ll = gp.log_likelihood(y)
		except:
			ll = 1e25
			print('#########')
		return ll if np.isfinite(ll) else 1e25



	def plot_psd(gp):

		plt.loglog(GP_periods, gp.kernel.get_psd(GP_omega), ":k", label = "model")
		plt.xlim(GP_periods.min(), GP_periods.max())
		plt.legend()
		plt.xlabel("Period [day]")
		plt.ylabel("Power [day ppt$^2$]")

	"""
	Parameters
	----------
	mag : list
		list of magnitudes for given star
	magerr : list
		list of magnitude errors for given star
	time : list
		list of magnitudes for given star
	GP_array : numpy array
		Array where [0] is time taken to run and [1] is period
	TOOL : class
		Class from Tools.py, holds lc meta data 
	Returns
	-------
	numpy array with second element as GP period
	"""
	#gpt0 = timer()


	#setup the data
	sort = xp.argsort(time)

	y = (np.array(mag)-min(mag))/(max(mag)-min(mag))
	yerr = np.array(magerr)[sort]
	t = np.array(time)[sort]
	t_full = np.arange(xp.min(time),xp.max(time),1) 	#<<for plotting

	#define priors
	log_b = 0.0						#A
	log_c = 0.0						#B
	log_xx = 0.0						#ln(xx) = -ln(L)
	log_P = np.log(10)					#Period

	#define bounds

	period_sigma = 2.0					#<<This is needed  #
	freqs = TOOL.test_freqs					#<<For plotting    #
	GP_periods = 1/freqs
	GP_omega = (2 * np.pi)/GP_periods			####################
	b_start = 0.01
	b_stop = 2.0
	c_start = 0.1
	c_stop = 100.0
	xx_start = 0.0001
	xx_stop = 10.0
	bnds = ((np.log(b_start),np.log(b_stop)),
		(np.log(c_start),np.log(c_stop)),
		(np.log(xx_start),np.log(xx_stop)),
		(np.log(1.0/TOOL.F_stop),np.log(1.0/TOOL.F_start)))

	# Setup the GP class
	kernel = CustomTerm(log_b, log_c, log_xx, log_P)
	#global gp
	gp = celerite.GP(kernel, mean=0.0)

	gp.compute(t,yerr)

	#define initial params that are given to minimize
	p0 = gp.get_parameter_vector()

	# run optimization:
	results = minimize(nll, p0, method='L-BFGS-B', jac=grad_nll, args=(y, gp), bounds = bnds)
	
	gp.set_parameter_vector(results.x)

	found_b_post = np.median(np.exp(results.x[0]))			#B
	found_c_post = np.median(np.exp(results.x[1]))			#C
	found_l_post = -1.0 * np.median(np.exp(results.x[2]))		#L
	found_period_post = np.median(np.exp(results.x[3]))		#P


	if TOOL.IO > 2:
		print('\n\t#################\n','\tGP SciPy Results\n\tb:\t',found_b_post,'\n\tc:\t',found_c_post,'\n\tl:\t',found_l_post,'\n\tp:\t',found_period_post,'\n\tLog-likelihood:\t',float(gp.log_likelihood(y)),'\n\t#################\n')
	if TOOL.IO > 3:

		plt.clf()
		plt.title("scipy psd")
		plot_psd(gp)
		plt.savefig(TOOL.info_figure+TOOL.name+'_scipy_psd.png', format = 'png',dpi = image_dpi)###################################### == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == 
		plt.clf()


		pl.clf()
		mu, cov = gp.predict(y, t_full)
		std = np.sqrt(np.diag(cov))
		ax = pl.subplot(111)
		pl.plot(t_full,mu)
		ax.fill_between(t_full,mu-std,mu+std,facecolor='lightblue', lw=0, interpolate=True)
		pl.scatter(t,y,s=2)
		
		text_pos = 0.
		if xp.argsort(time)[0] < 0.5:
			text_pos = 0.6

		pl.text(min(t_full) + 10, text_pos, 'b:'+str(found_b_post)+'\nc:'+str(found_c_post)+'\nl:'+str(found_l_post)+'\np:'+str(found_period_post)+'\nLog-likelihood:'+str(float(gp.log_likelihood(y))), family="serif")
		#pl.axis([0.,60.,-1.,1.])
		pl.ylabel("Relative flux [ppt]")
		pl.xlabel("Time [days]")
		pl.savefig(TOOL.info_figure+TOOL.name+'_scipy_pred.png', format = 'png',dpi = image_dpi)###################################### == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == 
		plt.clf()
		pl.clf()
                                                                                                        
         
	#if not doing posterior inference, return scipy period
	post_emcee = 0
	if post_emcee == 0:
		#return found_period_post
		#print('\t\t\tGP Time', round(timer()-gpt0, 4))
		return(gp.log_likelihood(y),found_b_post,found_c_post,found_period_post)
	else:

										                                                                
										                                                                
#					MMMMMMMM               MMMMMMMM        CCCCCCCCCCCCCMMMMMMMM               MMMMMMMM        CCCCCCCCCCCCC
#					M:::::::M             M:::::::M     CCC::::::::::::CM:::::::M             M:::::::M     CCC::::::::::::C
#					M::::::::M           M::::::::M   CC:::::::::::::::CM::::::::M           M::::::::M   CC:::::::::::::::C
#					M:::::::::M         M:::::::::M  C:::::CCCCCCCC::::CM:::::::::M         M:::::::::M  C:::::CCCCCCCC::::C
#					M::::::::::M       M::::::::::M C:::::C       CCCCCCM::::::::::M       M::::::::::M C:::::C       CCCCCC
#					M:::::::::::M     M:::::::::::MC:::::C              M:::::::::::M     M:::::::::::MC:::::C              
#					M:::::::M::::M   M::::M:::::::MC:::::C              M:::::::M::::M   M::::M:::::::MC:::::C              
#					M::::::M M::::M M::::M M::::::MC:::::C              M::::::M M::::M M::::M M::::::MC:::::C              
#					M::::::M  M::::M::::M  M::::::MC:::::C              M::::::M  M::::M::::M  M::::::MC:::::C              
#					M::::::M   M:::::::M   M::::::MC:::::C              M::::::M   M:::::::M   M::::::MC:::::C              
#					M::::::M    M:::::M    M::::::MC:::::C              M::::::M    M:::::M    M::::::MC:::::C              
#					M::::::M     MMMMM     M::::::M C:::::C       CCCCCCM::::::M     MMMMM     M::::::M C:::::C       CCCCCC
#					M::::::M               M::::::M  C:::::CCCCCCCC::::CM::::::M               M::::::M  C:::::CCCCCCCC::::C
#					M::::::M               M::::::M   CC:::::::::::::::CM::::::M               M::::::M   CC:::::::::::::::C
#					M::::::M               M::::::M     CCC::::::::::::CM::::::M               M::::::M     CCC::::::::::::C
#					MMMMMMMM               MMMMMMMM        CCCCCCCCCCCCCMMMMMMMM               MMMMMMMM        CCCCCCCCCCCCC
								            


                                                    
		scipy_period = found_period_post
		prior_sigma = 2.0
		p = gp.get_parameter_vector()
		initial = np.array([p[0],p[1],-1.*p[2],np.exp(p[3])])
		###################################
		#POSTERIOR INTERERENCE USING EMCEE#
		###################################

		#define MCMC parameters
		walkers = 32#coords.shape[0]			#key parameter, worth investigating
		dimensions = len(initial)			#dont touch
		burn_in = 200					#100 burn in is usefull but not neccesary 
		run_in = 5000					#graphs suggest 5000 is probably overkill
		discard_in = 0#burn_in				#how many of the  datapoints need to be removed, not really used right now
		corner_samples = 500				#for corner plot, also used in period extraction


		data = (t,y)
		p0 = [np.array(initial) + 1e-5 * np.random.randn(dimensions)
		      for i in range(walkers)]

		if TOOL.IO == 1:
			print('\tWalkers:\t',walkers)
			print('\tDimensions:\t',dimensions)
			print('\tBurn in:\t',burn_in)
			print('\tRun iterations:\t',run_in)
			print('\tDiscard:\t',discard_in)
			print('\tLog-likelihood:\t',float(gp.log_likelihood(y)))
		np.random.seed(6942069)

		sampler = emcee.EnsembleSampler(walkers, dimensions, lnprob, args=data)
		p0, lnp, _ = sampler.run_mcmc(p0, burn_in)#, progress = True)
		#sampler.reset()

		#p = p0[np.argmax(lnp)]
		#p0 = [p + 1e-5 * np.random.randn(dimensions) for i in range(walkers)]
		p0, lnp, _ = sampler.run_mcmc(p0, run_in)#, progress = True)

		gc.collect()

		######################################################################################################
		#Below lines extract period from MCMC, current method isnt that sophisticated and should be revisited#
		######################################################################################################

		# Find the maximum likelihood ys:
		ml = p0[np.argmax(lnp)]

		MLlnB = ml[0]
		MLlnC = ml[1]
		MLlnL = ml[2]
		MLlnP = np.log(ml[3])

		found_b_post = np.exp(ml[0])
		found_c_post = np.exp(ml[1])
		found_l_post = np.exp(ml[2])
		found_period_post = ml[3]

		p = np.array([MLlnB,MLlnC,MLlnL,MLlnP])
		gp.set_parameter_vector(p)




		print('\n-------------------------------------\n','\tGP MCMC Results\n\tb:\t',found_b_post,'\n\tc:\t',found_c_post,'\n\tl:\t',found_l_post,'\n\tp:\t',found_period_post,'\n\tLog-likelihood:\t',float(gp.log_likelihood(y)),'\n-------------------------------------\n')
		if TOOL.IO == 1:

			#Period Walker Chain
			plt.clf()
			plt.plot(np.exp(sampler.chain[:,:,2].transpose()))
			plt.ylabel('Period')
			plt.xlabel('Steps')
			plt.savefig(TOOL.info_figure+TOOL.name+'_GP_PCHAIN.png', format = 'png',dpi = image_dpi)###################################### == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == 
			#plt.show()
			plt.clf()

			#post prediction
			gc.collect()
			pl.clf()
			mu, cov = gp.predict(y, t_full)
			std = np.sqrt(np.diag(cov))
			ax = pl.subplot(111)
			pl.plot(t_full,mu)
			ax.fill_between(t_full,mu-std,mu+std,facecolor='lightblue', lw=0, interpolate=True)
			pl.scatter(t,y,s=2)
			
			text_pos = 0.
			if xp.argsort(time)[0] < 0.5:
				text_pos = 0.6

			pl.text(min(t_full) + 10, text_pos, 'b:'+str(found_b_post)+'\nc:'+str(found_c_post)+'\nl:'+str(found_l_post)+'\np:'+str(found_period_post)+'\nLog-likelihood:'+str(float(gp.log_likelihood(y))), family="serif")
			pl.ylabel("Relative flux [ppt]")
			pl.xlabel("Time [days]")
			pl.savefig(TOOL.info_figure+TOOL.name+'_mcmc_pred.png', format = 'png',dpi = image_dpi)###################################### == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == 
			plt.clf()
			pl.clf()



			#posterior PSD
			plt.clf()
			psds = sampler.get_blobs(discard = 100, flat = True)
			q = np.percentile(psds, [16, 50, 84], axis = 0)
			plt.loglog(np.log(GP_periods), q[1], color = "C0")
			plt.fill_between(np.log(GP_periods), q[0], q[2], color = "C0", alpha = 0.1)
			#plt.xlim(freq.min(), freq.max())
			plt.xlabel("Period [day]")
			plt.ylabel("Power [day ppt$^2$]")
			plt.title("posterior psd using emcee")
			plt.savefig(TOOL.info_figure+TOOL.name+'_mcmc_psd.png', format = 'png',dpi = image_dpi)###################################### == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == 
			plt.clf()
			

			# Corner
			samples = sampler.chain[:, :, :].reshape((-1, dimensions))#.transpose()
			# Find the maximum likelihood ys:
			ml = p0[np.argmax(lnp)]

			# Plot it.
			figure = corner.corner(samples, labels=[r"$lnB$", r"$lnC$", r"$lnL$", r"$P$"],
											truths=ml,
											quantiles=[0.16,0.5,0.84],
											levels=[0.39,0.86,0.99],
											#levels=[0.68,0.95,0.99],
											title="KIC 1430163",
											show_titles=True, title_args={"fontsize": 12})

			plt.savefig(TOOL.info_figure+TOOL.name+'_GP_CORNER.png', format = 'png',dpi = image_dpi)###################################### == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == 
			#plt.show()
			plt.clf()

			#corner messes with matplotlib
			font = {'size'   : 10}
			plt.rc('font', **font)




		return(gp.log_likelihood(y),found_b_post,found_c_post,found_period_post)
