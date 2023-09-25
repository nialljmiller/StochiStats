import StochiStats as ss
import numpy as np
import Virac
import NN_FAP
import pandas as pd
import csv

df = pd.read_csv('/beegfs/car/njm/OUTPUT/P_high_amp.csv')

viracids = df['Virac2b_SID']

def get_best_fap(peaks):
	topfap = 1
	topperiod = 1
	for peak_info in peaks:
		period = peak_info['Period']
		fap = NN_FAP.inference(period, mag, time)
		if fap < topfap:
			topfap = fap
			topperiod = period
	return topfap, topperiod


def lc_plot(mag, magerr, phase, time, period, amplitude, best_fap, outputfp):
    plt.clf()               
    norm = mplcol.Normalize(vmin=min(time), vmax=max(time), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='brg')
    date_color = np.array(mapper.to_rgba(time))
    line_widths = 1
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle(r'$Period$:{:.2f}  $\Delta T:${:.2f}  $\Delta m:${:.2f}  $Amplitude:${:.2f}  $FAP:${:.2f}'.format(round(period, 2), round(max(time)-min(time), 2), round(max(mag)-min(mag), 2), round(amplitude, 2), round(best_fap, 2)), fontsize=8, y=0.99)
    ax1.vlines(x = min(time) + period, ymin=min(mag), ymax=max(mag), color = 'g', ls='--', lw=1, alpha = 0.5)
    ax1.scatter(time, mag, c = 'k', s = 1)
    for x, y, e, colour in zip(time, mag, magerr, date_color):
        ax1.errorbar(x, y, yerr = e, fmt = 'o', lw=0.5, capsize=0.8, color=colour, markersize = 3)
        ax1.errorbar(x+1, y, yerr = e, fmt = 'o', lw=0.5, capsize=0.8, color=colour, markersize = 3)
    ax1.set_xlabel(r"$mjd$")
    ax1.set_xlim(min(time)-1,max(time)+1)
    ax1.xaxis.tick_top()
    ax1.invert_yaxis()
    ax2.scatter(phase, mag, c = 'k', s = 1)
    ax2.scatter(phase+1, mag, c = 'k', s = 1)
    for x, y, e, colour in zip(phase, mag, magerr, date_color):
        ax2.errorbar(x, y, yerr = e, fmt = 'o', lw=0.5, capsize=0.8, color=colour, markersize = 3)
        ax2.errorbar(x+1, y, yerr = e, fmt = 'o', lw=0.5, capsize=0.8, color=colour, markersize = 3)
    ax2.set_xlabel(r"$\phi$")
    ax2.set_xlim(0,2)
    ax2.invert_yaxis()
    plt.ylabel(r"    $Magnitude [Ks]$")
    plt.savefig(outputfp, dpi=300, bbox_inches='tight')
    plt.clf()

for viracid in viracids:#viracid = viracids[3]#

    lightcurve = Virac.run_sourceid(viracid)

    filters = (lightcurve['filter'].astype(str) == 'Ks')
    mag_gt_0 = (lightcurve['hfad_mag'].astype(float) > 0)
    emag_gt_0 = (lightcurve['hfad_emag'].astype(float) > 0)
    ast_res_chisq_lt_20 = (lightcurve['ast_res_chisq'].astype(float) < 20)
    chi_lt_10 = (lightcurve['chi'].astype(float) < 10)
    filtered_indices = np.where(filters & mag_gt_0 & emag_gt_0 & ast_res_chisq_lt_20 & chi_lt_10)[0]

    lightcurve = lightcurve[filtered_indices]
    mag, magerr, time, chi, astchi = lightcurve['hfad_mag'], lightcurve['hfad_emag'], lightcurve['mjdobs'], lightcurve['chi'], lightcurve['ast_res_chisq']

    sigma = np.std(magerr)
    filtered_indices = np.where(magerr <= 4 * sigma)[0]
    mag, magerr, time, chi, astchi = mag[filtered_indices], magerr[filtered_indices], time[filtered_indices], chi[filtered_indices], astchi[filtered_indices]


    y_fit, polyparams, polyr2 = ss.polyn_fit(mag, time, magerr, 2)

    mag = mag - y_fit

    LS_OUT = ss.LS(mag,magerr,time, F_start = None, F_stop = 10, df = 0.0005)
    PDM_OUT = ss.PDM(mag,magerr,time, F_start = None, F_stop = 10, df = 0.0005)
    LS_peak0, LS_peak1, LS_peak2, LS_xystats = ss.peak_analysis(LS_OUT[0], LS_OUT[1], peak_min = 0)
    PDM_peak0, PDM_peak1, PDM_peak2, PDM_xystats = ss.peak_analysis(PDM_OUT[0],PDM_OUT[1], peak_min = 1)
    best_fap, best_period = get_best_fap([LS_peak0, LS_peak1, LS_peak2, PDM_peak0, PDM_peak1, PDM_peak2])

    q1, q50, q99 = np.percentile(mag, [1, 50, 99])
    cut_mag_n = len(mag)
    cut_mag_avg = q50
    cut_magerr_avg = np.median(magerr) 
    cut_max_slope = ss.MaxSlope(mag, time)
    cut_MAD = ss.MedianAbsDev(mag, magerr)
    cut_true_amplitude = abs(q99-q1)
    cut_weight_mean = ss.weighted_mean(mag,magerr)
    cut_weight_std = ss.weighted_variance(mag,magerr)
    cut_weight_skew = ss.weighted_skew(mag,magerr)
    cut_weight_kurt = ss.weighted_kurtosis(mag,magerr)
    cut_mean = ss.mu(mag)
    cut_std = ss.sigma(mag)
    cut_skew = ss.skewness(mag)
    cut_kurt = ss.kurtosis(mag)

    phase = ss.phaser(time, best_period)
    output_fp  = '/home/njm/Period/phil/AAAAAAAAA'
    lc_plot(mag, magerr, phase, time, best_period, amplitude, best_fap, outputfp)


    y_fit, sineparams, siner2 =  ss.sine_fit(mag, time, best_period)

    mag = mag - y_fit

    

    LS_OUT = ss.LS(mag,magerr,time, F_start = None, F_stop = 10, df = 0.0005)
    PDM_OUT = ss.PDM(mag,magerr,time, F_start = None, F_stop = 10, df = 0.0005)
    LS_peak3, LS_peak4, LS_peak5, LS_xystats = ss.peak_analysis(LS_OUT[0], LS_OUT[1], peak_min = 0)
    PDM_peak3, PDM_peak4, PDM_peak5, PDM_xystats = ss.peak_analysis(PDM_OUT[0],PDM_OUT[1], peak_min = 1)
    best_fap2, best_period2 = get_best_fap([LS_peak3, LS_peak4, LS_peak5, PDM_peak3, PDM_peak4, PDM_peak5])

    q1, q50, q99 = np.percentile(mag, [1,50,99])
    post_mag_avg = q50
    post_true_amplitude = abs(q99-q1)
    post_max_slope = ss.MaxSlope(mag, time)
    post_MAD = ss.MedianAbsDev(mag, magerr) 
    post_weight_std = ss.weighted_variance(mag,magerr)
    post_weight_skew = ss.weighted_skew(mag, magerr)
    post_weight_kurt = ss.weighted_skew(mag, magerr)
    post_weight_mean = ss.weighted_mean(mag, magerr)
    post_mean = ss.mu(mag)
    post_std = ss.sigma(mag)
    post_skew = ss.skewness(mag)
    post_kurt = ss.kurtosis(mag)


    OUTPUT =[viracid,polyparams[0],polyparams[1],polyparams[2],polyr2,best_fap,best_period,cut_mag_n,cut_mag_avg,cut_magerr_avg,cut_max_slope ,cut_MAD,cut_true_amplitude,cut_weight_mean,cut_weight_std ,cut_weight_skew ,cut_weight_kurt,cut_mean,cut_std ,cut_skew,cut_kurt ,sineparams[0],sineparams[1],sineparams[2],siner2,best_fap2,best_period2,post_mag_avg,post_max_slope,post_MAD,post_true_amplitude,post_weight_mean,post_weight_std ,post_weight_skew,post_weight_kurt,post_mean,post_std ,post_skew,post_kurt]

    with open('phil_high_var_output_linefit.csv', 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(OUTPUT)





