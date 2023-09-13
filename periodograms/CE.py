# -*##- coding: utf-8 -*-
from __future__ import print_function, division
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import copy
from os import listdir
from os.path import isfile, join
import os
os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda/bin/"

import os.path
from multiprocessing import Process
from multiprocessing import Array

import sys
import gc;gc.enable()
import argparse

#Maths stuff
import time
from random import seed;seed(69420)
from random import random
import random
import math
import numpy as np
from time import time as timer
try:
	import cupy as xp
except:
	import numpy as xp

import matplotlib;matplotlib.rcParams['agg.path.chunksize'] = 10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# == == == == == == == == == = #
#Conditional Entropy#
# == == == == == == == == == = #

#import cuvarbase.ce as ce
#import Tools as T

from builtins import range
from builtins import object
import numpy as np
#from .utils import gaussian_window, tophat_window, get_autofreqs

try:
	import pycuda.driver as cuda
	from pycuda.compiler import SourceModule
	import pycuda.driver as cuda
	import pycuda.gpuarray as gpuarray
	import pycuda.autoinit
	from pycuda.compiler import SourceModule
except:
	print("!!!pycuda not found. Can't use Conditional Entropy!!!")




from builtins import zip
from builtins import range
from builtins import object

import numpy as np
import pkg_resources

#from .core import GPUAsyncProcess
#from .utils import _module_reader, find_kernel


import resource
import warnings

# == == == == == = #
#Global Vars#
# == == == == == = #

image_dpi = 250

os.environ["OMP_NUM_THREADS"] = "10" #turn of the auto parellization

def phase(t, freq, phi0 = 0.):
	phi = (t * freq - phi0)
	phi -= np.floor(phi)

	return phi

def CE(TOOL,mag,magerr,time):
	#print("Starting CE...")
	"""
	Parameters
	----------
	mag : list
		list of magnitudes for given star
	magerr : list
		list of magnitude errors for given star
	time : list
		list of magnitudes for given star
	CE_array : numpy array
		Array where [0] is time taken to run and [1] is period
	TOOL : class
		Class from Tools.py, holds lc meta data 
	Returns
	-------
	numpy array with second element as CE period
	"""
	#tce = timer()

	#try freedman-driaconis rule but just average for phased bins as on the fly calculation would be slow
	test_phase = phase(time, np.median(TOOL.test_freqs))	#this is the median phase to test, not sure about this...
	q3, q1 = np.percentile(test_phase, [75 ,25])
	iqr = q3 - q1
	h = 2 * iqr * len(mag)**(-1/3)				#this is the bin width, ie the f-d rule
	CE_bins_phase = int((max(test_phase)-min(test_phase))/h)
	if CE_bins_phase < 5:					#often it tried to used 3 bins, this cant be right?
		CE_bins_phase = 5				#5 seems like a reasonable limit

	q3, q1 = np.percentile(mag, [75 ,25])			#interestingly, magnitude always gets more bins
	iqr = q3 - q1						#could this be fixed with better pre cleaning?
	h = 2 * iqr * len(mag)**(-1/3)				#also creates hypersensetive to outliers, big outlier will render CE useless
	CE_bins_mag = int((max(mag)-min(mag))/h)
	

	if TOOL.IO > 3:
		print('\t~~~~~~~~\n\tCE BINS\n\t',CE_bins_mag,'x',CE_bins_phase,'\n\t~~~~~~~~')
	#sort just in case
	sort = xp.argsort(time)
	mag = np.array(mag)[sort]
	magerr = np.array(magerr)[sort]
	time = np.array(time)[sort]
	data = [(time, mag, magerr)]

	#create CE class called proc with defined bins 
	proc = ConditionalEntropyAsyncProcess(phase_bins = CE_bins_phase, mag_bins = CE_bins_mag, weighted = True)#, block_size = 128, use_fast = True)

	#run CE, If data too big, run with limited memory
	results = proc.run(data, freqs = TOOL.test_freqs)
	#try:
	#except:
	#	results = proc.large_run(data, freqs = TOOL.test_freqs, max_memory = 1e8)
	#this command actualy runs it
	proc.finish()

	#extract period from results
	freqs, ce_spectrum = results[0]
	CE_OUT = [0,0]
	CE_OUT[0]=TOOL.test_freqs
	CE_OUT[1]=ce_spectrum


	if TOOL.IO > 20:

		f, ax_bin = plt.subplots()
		cplot = plot_ce_bins(ax_bin, time, mag, magerr, freqs[np.argmin(ce_spectrum)], proc)
		cbar = plt.colorbar(cplot)
		cbar.ax.set_title('$H(\\phi, m)$')
		ax_bin.set_xlabel('$\\phi$', fontsize = 15)
		ax_bin.set_ylabel('$m$', fontsize = 15)
		f.tight_layout()
		plt.savefig(TOOL.info_figure+TOOL.name+'_CE_INFO.png', format = 'png',dpi = image_dpi)###################################### == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == 

	#print('\t\tCE\tTime', round(timer()-tce, 4))
	return CE_OUT




###############
#
###############
def plot_ce_bins(ax, t, y, dy, freq, ce_proc):

	ax.set_xlim(0, 1)

	y0 = min(y)
	yrange = max(y) - y0

	# Phase-fold the data at the trial frequency
	phi = phase(t, freq)

	# Bin the data
	phi_bins = np.floor(phi * ce_proc.phase_bins).astype(np.int)

	yi = ce_proc.mag_bins * (y - y0)/yrange
	mag_bins = np.floor(yi).astype(np.int)

	bins = [[sum((phi_bins == i) & (mag_bins == j))
			 for j in range(ce_proc.mag_bins)]
			for i in range(ce_proc.phase_bins)]
	bins = np.array(bins).astype(np.float)

	# Convert to N(bin) / Ntotal
	bins /= np.sum(bins.ravel())

	# The fraction of data that fall within a given phase bin
	p_phi = [np.sum(bins[i]) for i in range(ce_proc.phase_bins)]

	# fractional width of the (magnitude) bins
	dm = float(1 + ce_proc.mag_overlap) / ce_proc.mag_bins
	dphi = float(1 + ce_proc.phase_overlap) / ce_proc.phase_bins
	dY = yrange * dm

	# Compute conditional entropy contribution from each of the bins
	dH = [[bins[i][j] * np.log(dm * p_phi[i] / bins[i][j])
		   if bins[i][j] > 0 else 0.
		   for j in range(ce_proc.mag_bins)]
		  for i in range(ce_proc.phase_bins)]

	dH = np.array(dH)

	extent = [0, 1, min(y), max(y)]

	# Mask out the unoccupied bins
	dH = np.ma.masked_where(dH == 0, dH)

	palette = copy(plt.cm.GnBu_r)
	palette.set_bad('w', 0.)

	# Draw gridlines
	for i in range(ce_proc.phase_bins + 1):
		ax.axvline(0 + i * dphi, ls = ':', color = 'k',
				   alpha = 0.5, zorder = 95)

	for i in range(ce_proc.mag_bins + 1):
		ax.axhline(min(y) + i * dY, ls = ':', color = 'k',
				   alpha = 0.5, zorder = 95)

	# Plot the conditional entropy
	cplot = ax.imshow(dH.T, cmap = palette, extent = extent,
					  aspect = 'auto', origin = 'lower',
					  alpha = 0.5, zorder = 90)

	# Plot the data
	ax.scatter(phi, y, c = 'k', s = 1, alpha = 1, zorder = 100)

	return cplot





class GPUAsyncProcess(object):
	def __init__(self, *args, **kwargs):
		self.reader = kwargs.get('reader', None)
		self.nstreams = kwargs.get('nstreams', None)
		self.function_kwargs = kwargs.get('function_kwargs', {})
		self.device = kwargs.get('device', 0)
		self.streams = []
		self.gpu_data = []
		self.results = []
		self._adjust_nstreams = self.nstreams is None
		if self.nstreams is not None:
				self._create_streams(self.nstreams)
		self.prepared_functions = {}

	def _create_streams(self, n):
		for i in range(n):
			self.streams.append(cuda.Stream())

	def _compile_and_prepare_functions(self):
		raise NotImplementedError()

	def run(self, *args, **kwargs):
		raise NotImplementedError()

	def finish(self):
		""" synchronize all active streams """
		for i, stream in enumerate(self.streams):
			stream.synchronize()

	def batched_run(self, data, batch_size=10, **kwargs):
		""" Run your data in batches (avoids memory problems) """
		nsubmit = 0
		results = []
		while nsubmit < len(data):
			batch = []
			while len(batch) < batch_size and nsubmit < len(data):
				batch.append(data[nsubmit])
				nsubmit += 1

			res = self.run(batch, **kwargs)
			self.finish()
			results.extend(res)

		return results










class ConditionalEntropyAsyncProcess(GPUAsyncProcess):
	"""
	GPUAsyncProcess for the Conditional Entropy period finder

	Parameters
	----------
	phase_bins: int, optional (default: 10)
		Number of phase bins to use.
	mag_bins: int, optional (default: 10)
		Number of mag bins to use.
	max_phi: float, optional (default: 3.)
		For weighted CE; skips contibutions to bins that are more than
		``max_phi`` sigma away.
	weighted: bool, optional (default: False)
		If true, uses the weighted version of the CE periodogram. Slower, but
		accounts for data uncertainties.
	block_size: int, optional (default: 256)
		Number of CUDA threads per CUDA block.
	phase_overlap: int, optional (default: 0)
		If > 0, the phase bins are overlapped with each other
	mag_overlap: int, optional (default: 0)
		If > 0, the mag bins are overlapped with each other
	use_fast: bool, optional (default: False)
		Use a somewhat experimental function to speed up
		computations. This is perfect for large Nfreqs and nobs <~ 2000.
		If True, use :func:`run` and not :func:`large_run` and set
		``nstreams = 1``.

	Example
	-------
	>>> proc = ConditionalEntropyAsyncProcess()
	>>> Ndata = 1000
	>>> t = np.sort(365 * np.random.rand(N))
	>>> y = 12 + 0.01 * np.cos(2 * np.pi * t / 5.0)
	>>> y += 0.01 * np.random.randn(len(t))
	>>> dy = 0.01 * np.ones_like(y)
	>>> results = proc.run([(t, y, dy)])
	>>> proc.finish()
	>>> ce_freqs, ce_powers = results[0]

	"""
	def __init__(self, *args, **kwargs):
		super(ConditionalEntropyAsyncProcess, self).__init__(*args, **kwargs)
		self.phase_bins = kwargs.get('phase_bins', 10)
		self.mag_bins = kwargs.get('mag_bins', 5)
		self.max_phi = kwargs.get('max_phi', 3.)
		self.weighted = kwargs.get('weighted', False)
		self.block_size = kwargs.get('block_size', 256)

		self.phase_overlap = kwargs.get('phase_overlap', 0)
		self.mag_overlap = kwargs.get('mag_overlap', 0)

		if self.mag_overlap > 0:
			if kwargs.get('balanced_magbins', False):
				raise Exception("mag_overlap must be zero "
								"if balanced_magbins is True")

		self.use_double = kwargs.get('use_double', False)

		self.real_type = np.float32
		if self.use_double:
			self.real_type = np.float64

		self.call_func = conditional_entropy
		if kwargs.get('use_fast', False):
			self.call_func = conditional_entropy_fast

		self.memory = kwargs.get('memory', None)
		self.shmem_lc = kwargs.get('shmem_lc', True)


	def _compile_and_prepare_functions(self, **kwargs):


		def _module_reader(fname, cpp_defs=None):
			txt = open(fname, 'r').read()

			if cpp_defs is None:
				return txt

			preamble = ['#define {key} {value}'.format(key=key,value=('' if value is None else value))
					for key, value in cpp_defs.items()]
			txt = txt.replace('//{CPP_DEFS}', '\n'.join(preamble))

			return txt

		cpp_defs = dict(NPHASE=self.phase_bins,
						NMAG=self.mag_bins,
						PHASE_OVERLAP=self.phase_overlap,
						MAG_OVERLAP=self.mag_overlap)

		if self.use_double:
			cpp_defs['DOUBLE_PRECISION'] = None

		# Read kernel & replace with
		kernel_txt = _module_reader(pkg_resources.resource_filename('cuvarbase', 'kernels/%s.cu' % ('ce')), cpp_defs=cpp_defs)

		# compile kernel
		self.module = SourceModule(kernel_txt, options=['--use_fast_math'])

		self.dtypes = dict(
			constdpdm_ce=[np.intp, np.int32, np.intp, np.intp],
			histogram_data_weighted=[np.intp, np.intp, np.intp, np.intp,
									 np.intp, np.uint32, np.uint32,
									 self.real_type],
			histogram_data_count=[np.intp, np.intp, np.intp, np.intp,
								  np.uint32, np.uint32],
			log_prob=[np.intp, np.uint32, np.intp, np.intp],
			standard_ce=[np.intp, np.uint32, np.intp],
			weighted_ce=[np.intp, np.uint32, np.intp],
			ce_classical_fast=[np.intp, np.intp, np.intp,
							   np.intp, np.uint32,
							   np.uint32, np.uint32, np.uint32,
							   np.uint32, np.uint32, np.uint32],
			ce_classical_faster=[np.intp, np.intp, np.intp,
								 np.intp, np.uint32,
								 np.uint32, np.uint32, np.uint32,
								 np.uint32, np.uint32, np.uint32]
		)
		for fname, dtype in self.dtypes.items():
			func = self.module.get_function(fname)
			self.prepared_functions[fname] = func.prepare(dtype)
		self.function_tuple = tuple(self.prepared_functions[fname]
									for fname in sorted(self.dtypes.keys()))

	def memory_requirement(self, data, **kwargs):
		"""
		Return an approximate GPU memory requirement in bytes.
		Will throw a ``NotImplementedError`` if called, so ... don't call it.
		"""
		raise NotImplementedError()

	def allocate_for_single_lc(self, t, y, freqs, dy=None,
							   stream=None, **kwargs):
		"""
		Allocate GPU (and possibly CPU) memory for single lightcurve

		Parameters
		----------
		t: array_like
			Observation times
		y: array_like
			Observations
		freqs: array_like
			frequencies
		dy: array_like, optional
			Observation uncertainties
		stream: pycuda.driver.Stream, optional
			CUDA stream you want this to run on
		**kwargs

		Returns
		-------
		mem: ConditionalEntropyMemory
			Memory object.
		"""

		kw = dict(phase_bins=self.phase_bins,
				  mag_bins=self.mag_bins,
				  mag_overlap=self.mag_overlap,
				  phase_overlap=self.phase_overlap,
				  max_phi=self.max_phi,
				  stream=stream,
				  weighted=self.weighted,
				  use_double=self.use_double)

		kw.update(kwargs)
		mem = ConditionalEntropyMemory(**kw)

		mem.fromdata(t, y, dy=dy, freqs=freqs, allocate=True, **kwargs)

		return mem

	def autofrequency(self, *args, **kwargs):
		""" calls :func:`cuvarbase.utils.autofrequency` """
		return utils_autofreq(*args, **kwargs)

	def _nfreqs(self, *args, **kwargs):
		return len(self.autofrequency(*args, **kwargs))

	def allocate(self, data, freqs=None, **kwargs):
		"""
		Allocate GPU memory for Conditional Entropy computations

		Parameters
		----------
		data: list of (t, y, dy) tuples
			List of data, ``[(t_1, y_1, w_1), ...]``
			* ``t``: Observation times
			* ``y``: Observations
			* ``dy``: Observation uncertainties
		freqs: list, optional
			Either a list of floats (same frequencies for all data),
			or a list of length ``n=len(data)``, with element ``i`` of the
			list being a list of frequencies for the ``i``-th lightcurve.
		**kwargs

		Returns
		-------
		allocated_memory: list of ``ConditionalEntropyMemory``
			list of allocated memory objects for each lightcurve

		"""

		if len(data) > len(self.streams):
			self._create_streams(len(data) - len(self.streams))

		allocated_memory = []

		frqs = freqs
		if frqs is None:
			frqs = [self.autofrequency(t, **kwargs) for (t, y, dy) in data]

		elif isinstance(freqs[0], float):
			frqs = [freqs] * len(data)

		for i, ((t, y, dy), f) in enumerate(zip(data, frqs)):
			mem = self.allocate_for_single_lc(t, y, dy=dy, freqs=f,
											  stream=self.streams[i],
											  **kwargs)
			allocated_memory.append(mem)

		return allocated_memory

	def preallocate(self, max_nobs, freqs,
					nlcs=1, streams=None, **kwargs):
		"""
		Preallocate memory for future runs.

		Parameters
		----------
		max_nobs: int
			Upper limit for the number of observations
		freqs: array_like
			Frequency array to be used by future ``run`` calls
		nlcs: int, optional (default: 1)
			Maximum batch size for ``run`` calls
		streams: list of ``pycuda.driver.Stream``
			Length of list must be ``>= nlcs``

		Returns
		-------
		self.memory: list
			List of ``ConditionalEntropyMemory`` objects
		"""
		kw = dict(phase_bins=self.phase_bins,
				  mag_bins=self.mag_bins,
				  mag_overlap=self.mag_overlap,
				  phase_overlap=self.phase_overlap,
				  max_phi=self.max_phi,
				  weighted=self.weighted,
				  use_double=self.use_double,
				  n0_buffer=max_nobs,
				  buffered_transfer=True,
				  allocate=True,
				  freqs=freqs)

		kw.update(kwargs)

		self.memory = []
		for i in range(nlcs):
			stream = None if streams is None else streams[i]
			kw.update(dict(stream=stream))
			mem = ConditionalEntropyMemory(**kw)
			mem.allocate(**kwargs)
			self.memory.append(mem)

		return self.memory

	def run(self, data,
			memory=None,
			freqs=None,
			set_data=True,
			**kwargs):

		"""
		Run Conditional Entropy on a batch of data.

		Parameters
		----------
		data: list of tuples
			list of [(t, y, dy), ...] containing
			* ``t``: observation times
			* ``y``: observations
			* ``dy``: observation uncertainties
		freqs: optional, list of ``np.ndarray`` frequencies
			List of custom frequencies. If not specified, calls
			``autofrequency`` with default arguments
		memory: optional, list of ``ConditionalEntropyMemory`` objects
			List of memory objects, length of list must be ``>= len(data)``
		set_data: boolean, optional (default: True)
			Transfers data to gpu if memory is provided
		**kwargs

		Returns
		-------
		results: list of lists
			list of (freqs, ce) corresponding to CE for each element of
			the ``data`` array

		"""
		# compile module if not compiled already
		if not hasattr(self, 'prepared_functions') or \
			not all([func in self.prepared_functions for func in
					 ['ce_wt']]):
			self._compile_and_prepare_functions(**kwargs)

		# create and/or check frequencies
		frqs = freqs
		if frqs is None:
			frqs = [self.autofrequency(d[0], **kwargs) for d in data]

		elif isinstance(frqs[0], float):
			frqs = [frqs] * len(data)

		assert(len(frqs) == len(data))

		memory = memory if memory is not None else self.memory

		if memory is None:
			memory = self.allocate(data, freqs=frqs,
								   **kwargs)
			for mem in memory:
				mem.transfer_freqs_to_gpu()
		elif set_data:
			for i, (t, y, dy) in enumerate(data):
				memory[i].set_gpu_arrays_to_zero(**kwargs)
				memory[i].setdata(t, y, dy=dy, **kwargs)

		kw = dict(block_size=self.block_size,
				  shmem_lc=self.shmem_lc)
		kw.update(kwargs)
		results = [self.call_func(memory[i], self.function_tuple, **kw)
				   for i in range(len(data))]

		results = [(f, r) for f, r in zip(frqs, results)]
		return results

def conditional_entropy(memory, functions, block_size=256,
						transfer_to_host=True,
						transfer_to_device=True,
						**kwargs):
	block = (block_size, 1, 1)
	grid = (int(np.ceil((memory.n0 * memory.nf) / float(block_size))), 1)
	fast_ce, faster_ce, ce_dpdm, hist_count, hist_weight,\
		ce_logp, ce_std, ce_wt = functions

	if transfer_to_device:
		memory.transfer_data_to_gpu()

	if memory.weighted:
		args = (grid, block, memory.stream)
		args += (memory.t_g.ptr, memory.y_g.ptr, memory.dy_g.ptr)
		args += (memory.bins_g.ptr, memory.freqs_g.ptr)
		args += (np.uint32(memory.nf), np.uint32(memory.n0))
		args += (memory.real_type(memory.max_phi),)
		hist_weight.prepared_async_call(*args)

		grid = (int(np.ceil(memory.nf / float(block_size))), 1)

		args = (grid, block, memory.stream)
		args += (memory.bins_g.ptr, np.uint32(memory.nf), memory.ce_g.ptr)
		ce_wt.prepared_async_call(*args)

		if transfer_to_host:
			memory.transfer_ce_to_cpu()
		return memory.ce_c

	args = (grid, block, memory.stream)
	args += (memory.t_g.ptr, memory.y_g.ptr)
	args += (memory.bins_g.ptr, memory.freqs_g.ptr)
	args += (np.uint32(memory.nf), np.uint32(memory.n0))
	hist_count.prepared_async_call(*args)

	grid = (int(np.ceil(memory.nf / float(block_size))), 1)
	args = (grid, block, memory.stream)
	args += (memory.bins_g.ptr, np.uint32(memory.nf), memory.ce_g.ptr)

	if memory.balanced_magbins:
		args += (memory.mag_bwf_g.ptr,)
		ce_dpdm.prepared_async_call(*args)
	elif memory.compute_log_prob:
		args += (memory.mag_bin_fracs_g.ptr,)
		ce_logp.prepared_async_call(*args)
	else:
		ce_std.prepared_async_call(*args)

	if transfer_to_host:
		memory.transfer_ce_to_cpu()

	return memory.ce_c





class ConditionalEntropyMemory(object):
	def __init__(self, **kwargs):
		self.phase_bins = kwargs.get('phase_bins', 10)
		self.mag_bins = kwargs.get('mag_bins', 5)
		self.phase_overlap = kwargs.get('phase_overlap', 0)
		self.mag_overlap = kwargs.get('mag_overlap', 0)

		self.max_phi = kwargs.get('max_phi', 3.)
		self.stream = kwargs.get('stream', None)
		self.weighted = kwargs.get('weighted', False)
		self.widen_mag_range = kwargs.get('widen_mag_range', False)
		self.n0 = kwargs.get('n0', None)
		self.nf = kwargs.get('nf', None)

		self.compute_log_prob = kwargs.get('compute_log_prob', False)

		self.balanced_magbins = kwargs.get('balanced_magbins', False)

		if self.weighted and self.balanced_magbins:
			raise Exception("simultaneous balanced_magbins and weighted"
							" options is not currently supported")

		if self.weighted and self.compute_log_prob:
			raise Exception("simultaneous compute_log_prob and weighted"
							" options is not currently supported")
		self.n0_buffer = kwargs.get('n0_buffer', None)
		self.buffered_transfer = kwargs.get('buffered_transfer', False)
		self.t = None
		self.y = None
		self.dy = None

		self.t_g = None
		self.y_g = None
		self.dy_g = None

		self.bins_g = None
		self.ce_c = None
		self.ce_g = None
		self.mag_bwf = None
		self.mag_bwf_g = None
		self.real_type = np.float32
		if kwargs.get('use_double', False):
			self.real_type = np.float64

		self.freqs = kwargs.get('freqs', None)
		self.freqs_g = None

		self.mag_bin_fracs = None
		self.mag_bin_fracs_g = None

		self.ytype = np.uint32 if not self.weighted else self.real_type

	def allocate_buffered_data_arrays(self, **kwargs):
		n0 = kwargs.get('n0', self.n0)
		if self.buffered_transfer:
			n0 = kwargs.get('n0_buffer', self.n0_buffer)
		assert(n0 is not None)

		kw = dict(dtype=self.real_type,
				  alignment=resource.getpagesize())

		self.t = cuda.aligned_zeros(shape=(n0,), **kw)
		self.t = cuda.register_host_memory(self.t)

		self.y = cuda.aligned_zeros(shape=(n0,),
									dtype=self.ytype,
									alignment=resource.getpagesize())

		self.y = cuda.register_host_memory(self.y)
		if self.weighted:
			self.dy = cuda.aligned_zeros(shape=(n0,), **kw)
			self.dy = cuda.register_host_memory(self.dy)

		if self.balanced_magbins:
			self.mag_bwf = cuda.aligned_zeros(shape=(self.mag_bins,), **kw)
			self.mag_bwf = cuda.register_host_memory(self.mag_bwf)

		if self.compute_log_prob:
			self.mag_bin_fracs = cuda.aligned_zeros(shape=(self.mag_bins,),
													**kw)
			self.mag_bin_fracs = cuda.register_host_memory(self.mag_bin_fracs)
		return self

	def allocate_pinned_cpu(self, **kwargs):
		nf = kwargs.get('nf', self.nf)
		assert(nf is not None)

		self.ce_c = cuda.aligned_zeros(shape=(nf,), dtype=self.real_type,
									   alignment=resource.getpagesize())
		self.ce_c = cuda.register_host_memory(self.ce_c)

		return self

	def allocate_data(self, **kwargs):
		n0 = kwargs.get('n0', self.n0)
		if self.buffered_transfer:
			n0 = kwargs.get('n0_buffer', self.n0_buffer)

		assert(n0 is not None)
		self.t_g = gpuarray.zeros(n0, dtype=self.real_type)
		self.y_g = gpuarray.zeros(n0, dtype=self.ytype)
		if self.weighted:
			self.dy_g = gpuarray.zeros(n0, dtype=self.real_type)

	def allocate_bins(self, **kwargs):
		nf = kwargs.get('nf', self.nf)
		assert(nf is not None)

		self.nbins = nf * self.phase_bins * self.mag_bins

		if self.weighted:
			self.bins_g = gpuarray.zeros(self.nbins, dtype=self.real_type)
		else:
			self.bins_g = gpuarray.zeros(self.nbins, dtype=np.uint32)

		if self.balanced_magbins:
			self.mag_bwf_g = gpuarray.zeros(self.mag_bins,dtype=self.real_type)
		if self.compute_log_prob:
			self.mag_bin_fracs_g = gpuarray.zeros(self.mag_bins,dtype=self.real_type)

	def allocate_freqs(self, **kwargs):
		nf = kwargs.get('nf', self.nf)
		assert(nf is not None)
		self.freqs_g = gpuarray.zeros(nf, dtype=self.real_type)
		if self.ce_g is None:
			self.ce_g = gpuarray.zeros(nf, dtype=self.real_type)

	def allocate(self, **kwargs):
		self.freqs = kwargs.get('freqs', self.freqs)
		self.nf = kwargs.get('nf', len(self.freqs))

		if self.freqs is not None:
			self.freqs = np.asarray(self.freqs).astype(self.real_type)

		assert(self.nf is not None)

		self.allocate_data(**kwargs)
		self.allocate_bins(**kwargs)
		self.allocate_freqs(**kwargs)
		self.allocate_pinned_cpu(**kwargs)

		if self.buffered_transfer:
			self.allocate_buffered_data_arrays(**kwargs)

		return self

	def transfer_data_to_gpu(self, **kwargs):
		assert(not any([x is None for x in [self.t, self.y]]))

		self.t_g.set_async(self.t, stream=self.stream)
		self.y_g.set_async(self.y, stream=self.stream)

		if self.weighted:
			assert(self.dy is not None)
			self.dy_g.set_async(self.dy, stream=self.stream)

		if self.balanced_magbins:
			self.mag_bwf_g.set_async(self.mag_bwf, stream=self.stream)

		if self.compute_log_prob:
			self.mag_bin_fracs_g.set_async(self.mag_bin_fracs,
										   stream=self.stream)

	def transfer_freqs_to_gpu(self, **kwargs):
		freqs = kwargs.get('freqs', self.freqs)
		assert(freqs is not None)

		self.freqs_g.set_async(freqs, stream=self.stream)

	def transfer_ce_to_cpu(self, **kwargs):
		self.ce_g.get_async(stream=self.stream, ary=self.ce_c)

	def compute_mag_bin_fracs(self, y, **kwargs):
		N = float(len(y))
		mbf = np.array([np.sum(y == i)/N for i in range(self.mag_bins)])

		if self.mag_bin_fracs is None:
			self.mag_bin_fracs = np.zeros(self.mag_bins, dtype=self.real_type)
		self.mag_bin_fracs[:self.mag_bins] = mbf[:]

	def balance_magbins(self, y, **kwargs):
		yinds = np.argsort(y)
		ybins = np.zeros(len(y))

		assert len(y) >= self.mag_bins

		di = len(y) / self.mag_bins
		mag_bwf = np.zeros(self.mag_bins)
		for i in range(self.mag_bins):
			imin = max([0, int(i * di)])
			imax = min([len(y), int((i + 1) * di)])

			inds = yinds[imin:imax]
			ybins[inds] = i

			mag_bwf[i] = y[inds[-1]] - y[inds[0]]

		mag_bwf /= (max(y) - min(y))

		return ybins, mag_bwf.astype(self.real_type)

	def setdata(self, t, y, **kwargs):
		dy = kwargs.get('dy', self.dy)

		self.n0 = kwargs.get('n0', len(t))

		t = np.asarray(t).astype(self.real_type)
		y = np.asarray(y).astype(self.real_type)

		yscale = max(y[:self.n0]) - min(y[:self.n0])
		y0 = min(y[:self.n0])
		if self.weighted:
			dy = np.asarray(dy).astype(self.real_type)
			if self.widen_mag_range:
				med_sigma = np.median(dy[:self.n0])
				yscale += 2 * self.max_phi * med_sigma
				y0 -= self.max_phi * med_sigma

			dy /= yscale
		y = (y - y0) / yscale
		if not self.weighted:
			if self.balanced_magbins:
				y, self.mag_bwf = self.balance_magbins(y)
				y = y.astype(self.ytype)

			else:
				y = np.floor(y * self.mag_bins).astype(self.ytype)

			if self.compute_log_prob:
				self.compute_mag_bin_fracs(y)

		if self.buffered_transfer:
			arrs = [self.t, self.y]
			if self.weighted:
				arrs.append(self.dy)

			if any([arr is None for arr in arrs]):
				if self.buffered_transfer:
					self.allocate_buffered_data_arrays(**kwargs)

			assert(self.n0 <= len(self.t))

			self.t[:self.n0] = t[:self.n0]
			self.y[:self.n0] = y[:self.n0]

			if self.weighted:
				self.dy[:self.n0] = dy[:self.n0]
		else:
			self.t = t
			self.y = y
			if self.weighted:
				self.dy = dy
		return self

	def set_gpu_arrays_to_zero(self, **kwargs):
		self.t_g.fill(self.real_type(0), stream=self.stream)
		self.y_g.fill(self.ytype(0), stream=self.stream)
		if self.weighted:
			self.bins_g.fill(self.real_type(0), stream=self.stream)
			self.dy_g.fill(self.real_type(0), stream=self.stream)
		else:
			self.bins_g.fill(np.uint32(0), stream=self.stream)

	def fromdata(self, t, y, **kwargs):
		self.setdata(t, y, **kwargs)

		if kwargs.get('allocate', True):
			self.allocate(**kwargs)

		return self






def conditional_entropy_fast(memory, functions, block_size=256,
							 transfer_to_host=True,
							 transfer_to_device=True,
							 freq_batch_size=None,
							 shmem_lc=True,
							 shmem_lim=None,
							 max_nblocks=200,
							 force_nblocks=None,
							 stream=None,
							 **kwargs):
	fast_ce, faster_ce, ce_dpdm, hist_count, hist_weight,\
		ce_logp, ce_std, ce_wt = functions

	if shmem_lim is None:
		dev = pycuda.autoinit.device
		att = cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK
		shmem_lim = pycuda.autoinit.device.get_attribute(att)

	if transfer_to_device:
		memory.transfer_data_to_gpu()

	if freq_batch_size is None:
		freq_batch_size = int(memory.nf)

	block = (block_size, 1, 1)

	# Get the shared memory requirement
	r = memory.real_type(1).nbytes
	u = np.uint32(1).nbytes
	shmem = (r + u) * memory.phase_bins * memory.mag_bins
	shmem += u * memory.phase_bins
	data_mem = (r + u) * len(memory.t)

	func = fast_ce

	# Decide whether or not to use shared memory for
	# loading the lightcurve. Only if the user
	# wants and we have enough memory
	data_in_shared_mem = False
	if shmem_lc:
		data_in_shared_mem = shmem + data_mem < shmem_lim

	if data_in_shared_mem:
		shmem += data_mem
		func = faster_ce

	# Make sure we have extra memory for alignment
	shmem += shmem % r

	i_freq = 0
	while (i_freq < memory.nf):
		j_freq = min([i_freq + freq_batch_size, memory.nf])

		grid = (min([int(np.ceil((j_freq - i_freq) / block_size)),
					 max_nblocks]), 1)
		if data_in_shared_mem:
			grid = (int(np.floor(2 * float(shmem_lim) / shmem)), 1)
		if force_nblocks is not None:
			grid = (force_nblocks, 1)

		assert(grid[0] > 0)

		args = (grid, block, stream)
		args += (memory.t_g.ptr, memory.y_g.ptr)
		args += (memory.freqs_g.ptr, memory.ce_g.ptr)
		args += (np.uint32(j_freq - i_freq), np.uint32(i_freq),
				 np.uint32(memory.n0))
		args += (np.uint32(memory.phase_bins), np.uint32(memory.mag_bins))
		args += (np.uint32(memory.phase_overlap),
				 np.uint32(memory.mag_overlap))

		func.prepared_async_call(*args, shared_size=shmem)

		i_freq += j_freq - i_freq

	if transfer_to_host:
		memory.transfer_ce_to_cpu()

	return memory.ce_c










