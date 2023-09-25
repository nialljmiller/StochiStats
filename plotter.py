import os
from astropy.io import fits
import numpy as np
import Virac as Virac
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.colors as mplcol
import matplotlib.cm as cm
import matplotlib.ticker as ticker





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



def lc_debug_plot(mag, magerr, time, chi, ast_chi, outputfp):
  
    plt.clf()               
    norm = mplcol.Normalize(vmin=min(time), vmax=max(time), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='brg')
    date_color = np.array(mapper.to_rgba(time))
    line_widths = 1
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.scatter(time, chi, c = 'k', s = 1)
    for x, y, colour in zip(time, chi, date_color):
        ax1.scatter(x, y, color=colour,s=1)
        ax1.scatter(x+1, y, color=colour,s=1)
    ax1.set_xlabel(r"$mjd$")
    ax1.set_ylabel(r"$chi$")
    ax1.set_xlim(min(time)-1,max(time)+1)
    ax1.xaxis.tick_top()

    for x, y, colour in zip(ast_chi, chi, date_color):
        ax2.scatter(x, y, color=colour,s=1)
        ax2.scatter(x+1, y, color=colour,s=1)
    ax2.set_xlabel(r"$astrometric residual chi$")
    ax2.set_ylabel(r"$chi$")
    plt.savefig(outputfp, dpi=300, bbox_inches='tight')
    plt.clf()


