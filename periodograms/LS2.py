
import numpy as np
from time import time as timer
try:
	import cupy as xp
except:
	import numpy as xp
from joblib import Parallel, delayed
#from .mle import design_matrix



def LS(mag,magerr,time, F_start = None, F_stop = 10, df = 0.0005, nterms = 1):

	if F_start == None:
		F_start = 2/(max(time) - min(time))

	F_N = int((F_stop - F_start)/df)+1

	test_freqs = np.logspace(np.log10(F_start), np.log10(F_stop), F_N)

	#do the lombscargle, set frequeny grid to automatic but define min and max to give better unification amongst other methods
	ls_spectrum = lombscargle(time, mag, magerr, test_freqs, normalization='standard', fit_mean = True, center_data = False, nterms = nterms)
	LS_OUT = [0,0]
	LS_OUT[0]=test_freqs
	LS_OUT[1]=ls_spectrum

	return LS_OUT





def lombscargle(t, y, dy, frequency, normalization='standard', fit_mean=True, center_data=True, nterms=1):

	"""Lomb-Scargle Periodogram

	This implements a chi-squared-based periodogram, which is relatively slow
	but useful for validating the faster algorithms in the package.

	Parameters
	----------
	t, y, dy : array_like (NOT astropy.Quantities)
	times, values, and errors of the data points. These should be
	broadcastable to the same shape.
	frequency : array_like
	frequencies (not angular frequencies) at which to calculate periodogram
	normalization : str, optional
	Normalization to use for the periodogram.
	Options are 'standard', 'model', 'log', or 'psd'.
	fit_mean : bool, optional
	if True, include a constant offset as part of the model at each
	frequency. This can lead to more accurate results, especially in the
	case of incomplete phase coverage.
	center_data : bool, optional
	if True, pre-center the data by subtracting the weighted mean
	of the input data. This is especially important if ``fit_mean = False``
	nterms : int, optional
	Number of Fourier terms in the fit

	Returns
	-------
	power : array_like
	Lomb-Scargle power associated with each frequency.
	Units of the result depend on the normalization.

	References
	----------
	.. [1] M. Zechmeister and M. Kurster, A&A 496, 577-584 (2009)
	.. [2] W. Press et al, Numerical Recipes in C (2002)
	.. [3] Scargle, J.D. 1982, ApJ 263:835-853
	"""
	if dy is None:
		dy = 1

	t, y, dy = np.broadcast_arrays(t, y, dy)
	frequency = np.asarray(frequency)

	if t.ndim != 1:
		raise ValueError("t, y, dy should be one dimensional")
	if frequency.ndim != 1:
		raise ValueError("frequency should be one-dimensional")

	w = dy ** -2.0
	w /= w.sum()

	# if fit_mean is true, centering the data now simplifies the math below.
	if center_data or fit_mean:
		yw = (y - np.dot(w, y)) / dy
	else:
		yw = y / dy

	if normalization == 'psd':
		pass
	else:
		chi2_ref = np.dot(yw, yw)

	# compute the unnormalized model chi2 at each frequency
	def compute_power(f):
		X = design_matrix(t, f, dy=dy, bias=fit_mean, nterms=nterms)
		XTX = np.dot(X.T, X)
		XTy = np.dot(X.T, yw)
		return np.dot(XTy.T, np.linalg.solve(XTX, XTy))


	p = np.array([compute_power(f) for f in frequency])
	#p = np.array(Parallel(n_jobs=-1)(delayed(compute_power)(f) for f in frequency))

	if normalization == 'psd':
		p *= 0.5
	elif normalization == 'model':
		p /= (chi2_ref - p)
	elif normalization == 'log':
		p = -np.log(1 - p / chi2_ref)
	elif normalization == 'standard':
		p /= chi2_ref
	else:
		raise ValueError("normalization='{}' "
						 "not recognized".format(normalization))
	return p



def design_matrix(t, frequency, dy=None, bias=True, nterms=1):
	"""Compute the Lomb-Scargle design matrix at the given frequency

	This is the matrix X such that the periodic model at the given frequency
	can be expressed :math:`\\hat{y} = X \\theta`.

	Parameters
	----------
	t : array_like, shape=(n_times,)
		times at which to compute the design matrix
	frequency : float
		frequency for the design matrix
	dy : float or array_like, optional
		data uncertainties: should be broadcastable with `t`
	bias : bool (default=True)
		If true, include a bias column in the matrix
	nterms : int (default=1)
		Number of Fourier terms to include in the model

	Returns
	-------
	X : ndarray, shape=(n_times, n_parameters)
		The design matrix, where n_parameters = bool(bias) + 2 * nterms
	"""
	t = np.asarray(t)
	frequency = np.asarray(frequency)

	if t.ndim != 1:
		raise ValueError("t should be one dimensional")
	if frequency.ndim != 0:
		raise ValueError("frequency must be a scalar")

	if nterms == 0 and not bias:
		raise ValueError("cannot have nterms=0 and no bias")

	if bias:
		cols = [np.ones_like(t)]
	else:
		cols = []

	for i in range(1, nterms + 1):
		cols.append(np.sin(2 * np.pi * i * frequency * t))
		cols.append(np.cos(2 * np.pi * i * frequency * t))
	XT = np.vstack(cols)

	if dy is not None:
		XT /= dy

	return np.transpose(XT)

