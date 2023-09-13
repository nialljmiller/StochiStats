import numpy as np
from scipy.optimize import minimize
import emcee
#import celerite
#from celerite import terms#
import celerite2
from celerite2 import terms

import matplotlib.pyplot as plt


def create_kernel(F_start, F_stop):

    # Quasi-periodic term
    term1 = terms.SHOTerm(sigma=1.0, rho=1.0, tau=10.0)

    # Non-periodic component
    term2 = terms.SHOTerm(sigma=1.0, rho=5.0, Q=0.25)
    kernel = term1 + term2

    # Setup the GP
    gp = celerite2.GaussianProcess(kernel, mean=0.0)
    # Create the celerite2 GP model
    gp = celerite2.GaussianProcess(kernel, mean=0.0)
    return gp






def gp_compute(x, y, yerr, gp, nwalkers=32, nsteps=500, initial_params = None):
    gp, params = gp_scipy_compute(x, y, yerr, gp)
    #print(params)
    gp, params = gp_emcee_compute(x, y, yerr, gp, nwalkers=32, nsteps=500, initial_params = params)
    return gp, params    


def gp_scipy_compute(x, y, yerr, gp):

    def set_params(params, gp):
        gp.mean = params[0]
        theta = np.exp(params[1:])
        gp.kernel = terms.SHOTerm(
        sigma=theta[0], rho=theta[1], tau=theta[2]
        ) + terms.SHOTerm(sigma=theta[3], rho=theta[4], Q=0.25)
        gp.compute(x, diag=yerr**2 + theta[5], quiet=True)
        return gp


    def neg_log_like(params, gp):
        gp = set_params(params, gp)
        return -gp.log_likelihood(y)


    initial_params = [0.0, 0.0, 0.0, np.log(10.0), 0.0, np.log(5.0), np.log(0.01)]
    soln = minimize(neg_log_like, initial_params, method="L-BFGS-B", args=(gp,))
    opt_gp = set_params(soln.x, gp)
    return gp, soln.x    


def gp_emcee_compute(x, y, yerr, gp, nwalkers=32, nsteps=500, initial_params = None):
    def log_prior(params):
        # Implement your prior here (if any)
        # Return 0.0 for a flat prior
        return 0.0

    def log_likelihood(params, gp, x, y, yerr):
        gp = set_params(params, gp)
        return gp.log_likelihood(y)

    def log_probability(params, gp, x, y, yerr):
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(params, gp, x, y, yerr)

    def set_params(params, gp):
        gp.mean = params[0]
        theta = np.exp(params[1:])
        gp.kernel = (
            terms.SHOTerm(rho=theta[0], S0=theta[1], Q=theta[2])
            + terms.SHOTerm(rho=theta[3], S0=theta[4], Q=theta[5])
        )
        gp.compute(x, diag=yerr**2 + theta[5], quiet=True)
        return gp

    if initial_params is None:
        initial_params = [np.log(np.mean(y)), 0.0, 0.0, np.log(10.0), 0.0, np.log(5.0), np.log(0.01), 0.0, 0.0]
    
    # Initialize walkers
    ndim = len(initial_params)
    pos = [initial_params + 1e-4 * np.random.randn(ndim) for _ in range(nwalkers)]
    
    # Create the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(gp, x, y, yerr))
    
    # Run the MCMC sampling
    pos, _, _ = sampler.run_mcmc(pos, nsteps)
    
    # Get the best-fit parameters
    best_params = pos[np.argmax(sampler.flatlnprobability)]
    
    # Set GP model with the best-fit parameters
    gp = set_params(best_params, gp)
    
    return gp, best_params


def predict(y, yerr, x, gp, t1 = 1, t2 = 1, N = 500):
    # Make predictions at new points
    t_pred = np.linspace(min(x)*t1, max(x)*t2, N)
    y_pred, yerr_pred = gp.predict(y, t_pred, return_var=True)
    return y_pred, y_err_pred, t_pred


def GP(y, yerr, x, F_start = None, F_stop = 10):

    def plot_psd(gp):
        for n, term in enumerate(gp.kernel.terms):
            plt.loglog(freq, term.get_psd(omega), label="term {0}".format(n + 1))
        plt.loglog(freq, gp.kernel.get_psd(omega), ":k", label="full model")
        plt.xlim(freq.min(), freq.max())
        plt.legend()
        plt.xlabel("frequency [1 / day]")
        plt.ylabel("power [day ppt$^2$]")
        plt.show()

    def plot_prediction(gp):

        plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0, label="truth")

        mu, variance = gp.predict(y, t=true_x, return_var=True)
        sigma = np.sqrt(variance)
        plt.plot(true_x, mu, label="prediction")
        plt.fill_between(true_x, mu - sigma, mu + sigma, color="C0", alpha=0.2)

        plt.xlabel("x [day]")
        plt.ylabel("y [ppm]")
        plt.xlim(0, 10)
        plt.ylim(-2.5, 2.5)
        plt.legend()
        plt.show()

    if F_start is None:
        F_start = 2/(max(x) - min(x))

    gp = create_kernel(F_start, F_stop)
    gp, params = gp_scipy_compute(x, y, yerr, gp)

    true_x = np.linspace(min(x),max(x),1000)
    freq = np.linspace(F_start, F_stop, 10000)
    omega = 2 * np.pi * freq



    plot_psd(gp)
    plot_prediction(gp)


    gpmean = params[0]
    theta = np.exp(params[1:])#params[1:]#np.log(params[1:])#
    gprho1=theta[0]
    gpS01=theta[1]
    gpQ1=theta[2]
    gprho2=theta[3]
    gpS02=theta[4]
    gpQ2=theta[5]
    #gpa=theta[6]
    #gpc=theta[7]

    return gpmean, gprho1, gprho2





