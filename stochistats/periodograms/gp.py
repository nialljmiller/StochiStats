"""
Gaussian Process quasi-periodic period finding.

Uses a quasi-periodic kernel (celerite / celerite2) optimised via
scipy.optimize.minimize, with optional MCMC refinement via emcee.

The GP kernel (Equation A.3 in PRIMVS) combines a radial basis function
with a periodic component to model quasi-periodic variability.

Requires: celerite2 (preferred) or celerite, scipy.
Optional:  emcee (for MCMC posterior sampling).
"""

import numpy as np
from typing import Optional, Tuple, Dict
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


def GP(
    mag: np.ndarray,
    magerr: np.ndarray,
    time: np.ndarray,
    F_start: Optional[float] = None,
    F_stop: float = 10.0,
    run_mcmc: bool = False,
    mcmc_nwalkers: int = 32,
    mcmc_nsteps: int = 500,
) -> Dict[str, float]:
    """
    Find the best-fit quasi-periodic period using Gaussian Processes.

    Parameters
    ----------
    mag : array_like
        Magnitude values.
    magerr : array_like
        Magnitude uncertainties.
    time : array_like
        Observation times.
    F_start : float, optional
        Minimum frequency (default: 2/baseline).
    F_stop : float, optional
        Maximum frequency (default: 10).
    run_mcmc : bool, optional
        If True, refine with emcee MCMC after scipy (default: False).
    mcmc_nwalkers : int, optional
        Number of MCMC walkers (default: 32).
    mcmc_nsteps : int, optional
        Number of MCMC steps (default: 500).

    Returns
    -------
    dict
        Keys: 'log_likelihood', 'b', 'c', 'period', and optionally
        'b_err', 'c_err', 'period_err' if MCMC was run.
    """
    return gp_period(
        mag, magerr, time,
        F_start=F_start, F_stop=F_stop,
        run_mcmc=run_mcmc,
        mcmc_nwalkers=mcmc_nwalkers,
        mcmc_nsteps=mcmc_nsteps,
    )


def gp_period(
    mag: np.ndarray,
    magerr: np.ndarray,
    time: np.ndarray,
    F_start: Optional[float] = None,
    F_stop: float = 10.0,
    run_mcmc: bool = False,
    mcmc_nwalkers: int = 32,
    mcmc_nsteps: int = 500,
) -> Dict[str, float]:
    """
    Core GP period-finding implementation.

    See :func:`GP` for full documentation.
    """
    time = np.asarray(time, dtype=float)
    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)

    # Sort by time
    sort_idx = np.argsort(time)
    t = time[sort_idx]
    y = mag[sort_idx]
    yerr = magerr[sort_idx]

    # Normalise magnitudes to [0, 1]
    y_range = y.max() - y.min()
    if y_range == 0:
        return {"log_likelihood": np.nan, "b": np.nan, "c": np.nan, "period": np.nan}
    y_norm = (y - y.min()) / y_range
    yerr_norm = yerr / y_range

    baseline = t.max() - t.min()
    if F_start is None:
        F_start = 2.0 / baseline

    # Try celerite2 first, then celerite
    try:
        result = _gp_celerite2(t, y_norm, yerr_norm, F_start, F_stop)
    except ImportError:
        try:
            result = _gp_celerite1(t, y_norm, yerr_norm, F_start, F_stop)
        except ImportError:
            logger.error("Neither celerite2 nor celerite is installed")
            return {"log_likelihood": np.nan, "b": np.nan, "c": np.nan, "period": np.nan}

    if run_mcmc:
        try:
            result = _refine_with_mcmc(t, y_norm, yerr_norm, result,
                                       F_start, F_stop,
                                       mcmc_nwalkers, mcmc_nsteps)
        except ImportError:
            logger.warning("emcee not installed, skipping MCMC refinement")

    return result


def _gp_celerite2(t, y, yerr, F_start, F_stop):
    """GP period finding using celerite2."""
    import celerite2
    from celerite2 import terms

    # Quasi-periodic kernel: SHO + SHO (non-periodic component)
    term1 = terms.SHOTerm(sigma=1.0, rho=1.0, tau=10.0)
    term2 = terms.SHOTerm(sigma=1.0, rho=5.0, Q=0.25)
    kernel = term1 + term2

    gp = celerite2.GaussianProcess(kernel, mean=0.0)

    def set_params(params):
        gp.mean = params[0]
        theta = np.exp(params[1:])
        gp.kernel = terms.SHOTerm(
            sigma=theta[0], rho=theta[1], tau=theta[2]
        ) + terms.SHOTerm(sigma=theta[3], rho=theta[4], Q=0.25)
        gp.compute(t, diag=yerr**2 + theta[5], quiet=True)
        return gp

    def neg_log_like(params):
        try:
            set_params(params)
            return -gp.log_likelihood(y)
        except Exception:
            return 1e25

    initial_params = [0.0, 0.0, 0.0, np.log(10.0), 0.0, np.log(5.0), np.log(0.01)]

    soln = minimize(neg_log_like, initial_params, method="L-BFGS-B")
    set_params(soln.x)

    theta = np.exp(soln.x[1:])
    return {
        "log_likelihood": float(-soln.fun),
        "b": float(theta[0]),
        "c": float(theta[1]),
        "period": float(theta[1]),  # rho parameter ~ characteristic period
    }


def _gp_celerite1(t, y, yerr, F_start, F_stop):
    """GP period finding using celerite (v1) with the original CustomTerm."""
    import celerite
    from celerite import terms as cterms

    class CustomTerm(cterms.Term):
        parameter_names = ("log_b", "log_c", "log_xx", "log_P")

        def get_real_coefficients(self, params):
            log_b, log_c, log_xx, log_P = params
            c = np.exp(log_c)
            return (
                np.exp(log_c) * (1.0 + c) / (2.0 + c),
                np.exp(log_xx),
            )

        def get_complex_coefficients(self, params):
            log_b, log_c, log_xx, log_P = params
            c = np.exp(log_c)
            return (
                np.exp(log_b) / (2.0 + c), 0.0,
                np.exp(log_xx), 2 * np.pi * np.exp(-log_P),
            )

    # Bounds
    b_bounds = (np.log(0.01), np.log(2.0))
    c_bounds = (np.log(0.1), np.log(100.0))
    xx_bounds = (np.log(0.0001), np.log(10.0))
    P_bounds = (np.log(1.0 / F_stop), np.log(1.0 / F_start))

    kernel = CustomTerm(
        log_b=0.0, log_c=0.0, log_xx=0.0, log_P=np.log(10),
        bounds=dict(
            log_b=b_bounds, log_c=c_bounds,
            log_xx=xx_bounds, log_P=P_bounds,
        ),
    )
    gp = celerite.GP(kernel, mean=0.0)
    gp.compute(t, yerr)

    def nll(p):
        gp.set_parameter_vector(p)
        ll = gp.log_likelihood(y)
        return -ll if np.isfinite(ll) else 1e25

    def grad_nll(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y)[1]

    p0 = gp.get_parameter_vector()
    bounds = [b_bounds, c_bounds, xx_bounds, P_bounds]

    soln = minimize(nll, p0, method="L-BFGS-B", jac=grad_nll, bounds=bounds)
    gp.set_parameter_vector(soln.x)

    b_val = np.exp(soln.x[0])
    c_val = np.exp(soln.x[1])
    period_val = np.exp(soln.x[3])

    return {
        "log_likelihood": float(gp.log_likelihood(y)),
        "b": float(b_val),
        "c": float(c_val),
        "period": float(period_val),
    }


def _refine_with_mcmc(t, y, yerr, scipy_result, F_start, F_stop,
                      nwalkers, nsteps):
    """Refine GP parameters with emcee MCMC."""
    import emcee
    import celerite
    from celerite import terms as cterms

    class CustomTerm(cterms.Term):
        parameter_names = ("log_b", "log_c", "log_xx", "log_P")

        def get_real_coefficients(self, params):
            log_b, log_c, log_xx, log_P = params
            c = np.exp(log_c)
            return (np.exp(log_c) * (1.0 + c) / (2.0 + c), np.exp(log_xx))

        def get_complex_coefficients(self, params):
            log_b, log_c, log_xx, log_P = params
            c = np.exp(log_c)
            return (np.exp(log_b) / (2.0 + c), 0.0,
                    np.exp(log_xx), 2 * np.pi * np.exp(-log_P))

    kernel = CustomTerm(log_b=0.0, log_c=0.0, log_xx=0.0, log_P=np.log(10))
    gp = celerite.GP(kernel, mean=0.0)
    gp.compute(t, yerr)

    def log_prob(p):
        gp.set_parameter_vector(p)
        lp = gp.log_prior()
        if not np.isfinite(lp):
            return -np.inf
        ll = gp.log_likelihood(y)
        return lp + ll if np.isfinite(ll) else -np.inf

    ndim = len(gp.get_parameter_vector())
    p0 = gp.get_parameter_vector()
    pos = p0 + 1e-4 * np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(pos, nsteps, progress=False)

    # Use last 50 steps
    samples = sampler.get_chain(discard=max(0, nsteps - 50), flat=True)
    b_samples = np.exp(samples[:, 0])
    c_samples = np.exp(samples[:, 1])
    p_samples = np.exp(samples[:, 3])

    return {
        "log_likelihood": scipy_result["log_likelihood"],
        "b": float(np.median(b_samples)),
        "c": float(np.median(c_samples)),
        "period": float(np.median(p_samples)),
        "b_err": float(np.std(b_samples)),
        "c_err": float(np.std(c_samples)),
        "period_err": float(np.std(p_samples)),
    }
