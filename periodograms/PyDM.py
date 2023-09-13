import numpy as np


# Function for phase dispersion minimization to find optimal period
def PDM_single(periods, time, flux, num_phase_bins = None, num_flux_bins = None):

    total_dispersion = []
    for period in periods:
        dispersion = phase_dispersion_minimization(period, time, flux, num_phase_bins, num_flux_bins)
        total_dispersion.append(dispersion)

    return periods, total_dispersion 



def PDM_pool(periods, time, flux, num_phase_bins=None, num_flux_bins=None):
    import multiprocessing

    with multiprocessing.Pool() as pool:
        total_dispersion = pool.map(phase_dispersion_minimization, [(period, time, flux, num_phase_bins, num_flux_bins) for period in periods])

    return periods, total_dispersion



# Function for phase dispersion minimization with binning in both axes
def phase_dispersion_minimization(args):
    period, time, flux, num_phase_bins, num_flux_bins = args
    phase = (time / period) % 1
    phase_min = phase.min()
    phase_max = phase.max()
    flux_min = flux.min()
    flux_max = flux.max()

    phase_bins = np.linspace(phase_min, phase_max, num_phase_bins + 1)
    flux_bins = np.linspace(flux_min, flux_max, num_flux_bins + 1)

    dispersion = np.zeros((num_phase_bins, num_flux_bins))  # Initialize a 2D array for dispersions

    for i in range(num_phase_bins):
        for j in range(num_flux_bins):
            phase_bin_start, phase_bin_end = phase_bins[i], phase_bins[i + 1]
            flux_bin_start, flux_bin_end = flux_bins[j], flux_bins[j + 1]

            in_bin = (phase >= phase_bin_start) & (phase < phase_bin_end) & (flux >= flux_bin_start) & (flux < flux_bin_end)

            if np.any(in_bin):
                phase_bin = phase[in_bin]
                flux_bin = flux[in_bin]
                dispersion[i, j] = np.sum(np.square(flux_bin - flux_bin.mean()))

    return np.sum(dispersion)



