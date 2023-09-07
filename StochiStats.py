import numpy as np
import random
from sklearn.neighbors import KNeighborsRegressor
from scipy.signal import savgol_filter
import warnings
# Suppress RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

def spline_fit(mag, magerr, time, do_it_anyway = 0):
    def sl(x, A, B): # this is your 'straight line' y=f(x)
        return A*x + B
            
    y = np.array(mag)        # to fix order when called, (better to always to mag, time) CONSISTENCY!!!!!!!!!)
    yerr = np.array(magerr)
    x = np.array(time)

    res = 10
    rq50 = np.empty(res)
    rq25 = np.empty(res)
    rq75 = np.empty(res)
    Q75, Q25 = np.percentile(y, [75, 25])
    rx = np.linspace(min(x), max(x), res)
    rdelta = (max(x) - min(x))/(2*res)

    ##bin need to have X points
    for i in range(res):
        check = []
        rdelta_temp = rdelta                        
        while len(check) < 1:
            check = np.where((x < rx[i]+rdelta_temp) & (x > rx[i]-rdelta_temp))[0]
            rdelta_temp = rdelta_temp + 0.2*rdelta
        rq50[i] = np.median(y[check])
        try:
            rq75[i], rq25[i] = np.percentile(y[check], [75, 25])
        except:
            rq75[i], rq25[i] = rq75[i-1], rq25[i-1]


    RQ75, RQ25 = np.percentile(rq50, [75, 25])
    RIQR = abs(RQ75 - RQ25)

    
    #if the range of IQR of binned data changes alot when a single bin is removed, its probably transient
    IQRs = []
    for i in range(1,res):
        tq75, tq25 = np.percentile(np.delete(rq50,i), [75, 25])
        IQRs.append(abs(tq75-tq25))
    
    if abs(max(IQRs)-min(IQRs)) > 0.1 * RIQR:
        trans_flag = 1 

    popt, pcov = curve_fit(sl, rx, rq50) # your data x, y to fit
    grad = popt[0]
    intercept = popt[1]
    #generate fit

    y_fit = sl(x, popt[0], popt[1])
    #compare
    y_diff = y - y_fit
    #residual sum of squares
    ss_res = np.sum((y_diff) ** 2)
    #total sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    # r-squared
    r2 = 1 - (ss_res / ss_tot)
        
 


def sine_fit(mag, time, period, name):
        def sinus(x, A, B, C): # this is your 'straight line' y=f(x)
            return (A * np.sin((2.*np.pi*x)+C))+B
        y = np.array(mag)        # to fix order when called, (better to always to mag, time) CONSISTENCY!!!!!!!!!)
        x = phaser(time, period)
        popt, pcov = curve_fit(sinus, x, y, bounds=((true_amplitude*0.3, mag_avg*0.3, -2), (true_amplitude*3.0, mag_avg*3, 2)))#, method = 'lm') # your data x, y to fit
        
        y_fit = sinus(x, popt[0], popt[1], popt[2])
        #compare
        y_diff = y - y_fit
        #residual sum of squares
        ss_res = np.sum((y_diff) ** 2)
        #total sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        # r-squared
        r2 = 1 - (ss_res / ss_tot)        #coefficient of determination
       
        return y_diff+np.median(y), x



#normalise list 'x' 
def normalise(x):
    x_list = []
    x_list = (x-min(x))/(max(x)-min(x))
    return x_list


def round_sig(x, sig=3):
    return round(x, sig-int(floor(log10(np.abs(x))))-1)

def average_seperation(x):
    diff_list = []
    x.sort()
    for i, xi in enumerate(x):
        try:
            diff_list.append(x[i+1]-xi)
        except:
            pass
    return diff_list



def dtw(s, t):
    n, m = len(s), len(t)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s[i - 1] - t[j - 1])
            last_min = min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])
            dtw_matrix[i, j] = cost + last_min

    return dtw_matrix


def lc_model(cat_type, x):
    if 'EB0' in cat_type:
        A2 = 0.3
        A1 = 0.7
        m = (A1*np.sin(2*np.pi*x)**2 - A2*np.sin(np.pi*x)**2)

    if 'EB1' in cat_type:
        A2 = 0.7    
        A1 = 0.3
        m = (A1*np.sin(2*np.pi*x)**2 - A2*np.sin(np.pi*x)**2)

    if 'EB2' in cat_type:
        A1 = 0.01
        A2 = 0.99
        m = (A1*np.sin(2*np.pi*x)**2 - A2*np.sin(np.pi*x)**2)

    if 'EB3' in cat_type:
        A1 = 0.99
        A2 = 0.01
        m = (A1*np.sin(2*np.pi*x)**2 - A2*np.sin(np.pi*x)**2)

    if 'Ceph' in cat_type:
        m = 0.5*np.sin(2*np.pi*x) - 0.15*random.uniform(0.5,1.5)*np.sin(2*2*np.pi*x) - 0.05*random.uniform(0.5,1.5)*np.sin(3*2*np.pi*x)

    if 'WUma' in cat_type:
        m = abs(np.sin(np.pi*x))

    if 'WUma1' in cat_type:
        m = (-1*abs(np.sin(np.pi*x))) + 1
        
    if 'YSO0' in cat_type:
        m = np.sin(2*np.pi*x)

    if 'YSO1' in cat_type:
        m = np.sin(2*np.pi*x) + np.sin((2*np.pi*x)/np.random.uniform(0.01,0.1)) * np.random.uniform(0.005,0.05)

    return m


    
    

def dtw_classify(time, period, mag, knn_neighbors=5, N=200):
    def phaser(time, period):
        return (time / period) % 1

    def norm_data(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def smooth(data, window_size):
        return savgol_filter(data, window_size * 2 + 1, 3)

    cat_types = ['EB0', 'EB1', 'EB2', 'EB3', 'Ceph', 'WUma', 'WUma1', 'YSO0', 'YSO1']

    phase = phaser(time, period)
    mag = norm_data(mag)

    even_phase = np.linspace(0, 1, 200)
    knn_m = KNeighborsRegressor(n_neighbors=knn_neighbors).fit(phase[:, np.newaxis], mag).predict(sorted(even_phase[:, np.newaxis]))
    knn_m = smooth(knn_m, int(N / 5))

    minima_index = np.argmin(knn_m)
    phase = even_phase - even_phase[minima_index]
    phase_mask = np.where(phase < 0)[0]
    phase[phase_mask] = phase[phase_mask] + 1

    sort = np.argsort(phase)
    knn_m = np.array(knn_m)[sort]
    phase = np.array(phase)[sort]

    cat_prob = []
    for cat_type in cat_types:
        class_m = lc_model(cat_type, phase)
        cat_prob.append(dtw(norm_data(class_m), norm_data(knn_m))[-1, -1])

    return cat_types[np.argmin(cat_prob)]






def numbins(x):
    q3, q1 = np.percentile(x, [75 ,25])
    iqr = q3 - q1            
    h = 2 * iqr * len(x)**(-1/3)    
    return int((max(x)-min(x))/h)


def err_var(mag, magerr):
    amp = abs(max(mag) - min(mag))
    err = np.median(magerr)
    return amp/err




def running_phase_statistics(mag, magerr, time, period, nbins = None):

    def phaser(time, period):
        return (time / period) % 1

    phase = phaser(time, period)
    
    if nbins == None:
        nbins = numbins(phase)

    bins = np.linspace(0, 1, nbins)
    digi = np.digitize(phase, bins)

    rstd = np.empty(nbins)
    rmvar = np.empty(nbins)
    rmad = np.empty(nbins)
    rpa = np.empty(nbins)
    rrange = np.empty(nbins)
    riqr = np.empty(nbins)

    for i in range(1, max(digi) + 1):
        idx = np.where(digi == i)[0]
        if len(idx) > 1:
            MAG = mag[idx]
            MAGERR = magerr[idx]

            rstd[i-1] = np.std(MAG) / MAGERR
            rmvar[i-1] = np.mean(MAG ** 2) / MAGERR
            rmad[i-1] = np.median(np.abs(MAG - np.median(MAG))) / MAGERR
            rpa[i-1] = np.max(MAG) / MAGERR
            rrange[i-1] = (np.max(MAG) - np.min(MAG)) / MAGERR
            riqr[i-1] = np.subtract(*np.percentile(MAG, [75, 25])) / MAGERR

    return np.array([rstd, rmvar, rmad, rpa, rrange, riqr])





def weighted_mean(var, wts):
    """Calculates the weighted mean"""
    return np.average(var, weights=wts)


def weighted_variance(var, wts):
    """Calculates the weighted variance"""
    return np.average((var - weighted_mean(var, wts))**2, weights=wts)


def weighted_skew(var, wts):
    """Calculates the weighted skewness"""
    return (np.average((var - weighted_mean(var, wts))**3, weights=wts) / weighted_variance(var, wts)**(1.5))

def weighted_kurtosis(var, wts):
    """Calculates the weighted skewness"""
    return (np.average((var - weighted_mean(var, wts))**4, weights=wts) / weighted_variance(var, wts)**(2))


def mu(y):
    return np.mean(y)

def sigma(y):
    return np.std(y)
    
    
def skewness(mag):
    n = len(mag)
    mean = np.mean(mag)
    std = np.std(mag)
    return np.sum(((mag - mean) / std) ** 3) * n / ((n - 1) * (n - 2))

def kurtosis(mag):
    n = len(mag)
    mean = np.mean(mag)
    std = np.std(mag)
    return np.sum(((mag - mean) / std) ** 4) * n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) - 3 * ((n - 1) ** 2) / ((n - 2) * (n - 3))

def Meanvariance(mag):
    return np.std(mag) / np.mean(mag)

def PercentAmplitude(mag):
    median_data = np.median(mag)
    distance_median = np.abs(mag - median_data)
    max_distance = np.max(distance_median)
    percent_amplitude = max_distance / median_data
    return percent_amplitude

def AndersonDarling(mag):
    ander = np.sort(mag)
    n = len(ander)
    k = np.arange(1, n + 1)
    t = -((2 * k - 1) / n - 1)
    z = (np.log(1 / (1 - t)) - np.log(t)) / n
    A2 = -n - np.sum((2 * k - 1) * (np.log(ander) + np.log(1 - ander[::-1])))
    return 1 / (1.0 + np.exp(-10 * (A2 - 0.3)))


def mann_whitney_u_test(data1, data2):
    combined_data = np.concatenate((data1, data2))
    ranked_data = np.argsort(combined_data)
    u1 = np.sum(ranked_data[:len(data1)])
    u2 = np.sum(ranked_data[len(data1):])
    min_u = min(u1, u2)
    max_u = max(u1, u2)
    n1 = len(data1)
    n2 = len(data2)
    expected_u = (n1 * n2 / 2)
    std_error = np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
    z = (min_u - expected_u) / std_error
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z)))
    return min_u, max_u, p_value

def anderson_darling_test(data1, data2):
    combined_data = np.sort(np.concatenate([data1, data2]))
    ecdf = np.arange(1, len(combined_data) + 1) / len(combined_data)
    A2 = -len(data1) - np.sum((2 * np.arange(1, len(data1) + 1) - 1) * (np.log(ecdf[:len(data1)]) + np.log(1 - ecdf[-len(data1):][::-1])))
    critical_values = np.array([0.576, 0.656, 0.787, 0.918, 1.092])  # Critical values for significance levels 15%, 10%, 5%, 2.5%, 1%
    p_value = np.interp(A2, [0.2, 0.6, 1.0, 1.5, 2.0], critical_values)
    return A2, p_value

def cohens_d(data1, data2):
    mean1, mean2 = np.mean(data1), np.mean(data2)
    std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    pooled_std = np.sqrt(((len(data1) - 1) * std1**2 + (len(data2) - 1) * std2**2) / (len(data1) + len(data2) - 2))
    cohens_d = np.abs(mean1 - mean2) / pooled_std
    return cohens_d


def emp_cramer_von_mises(data1, data2):
    combined_data = np.sort(np.concatenate([data1, data2]))
    ecdf1 = np.searchsorted(data1, combined_data, side='right') / len(data1)
    ecdf2 = np.searchsorted(data2, combined_data, side='right') / len(data2)
    ecvm_statistic = np.sum((ecdf1 - ecdf2)**2)
    return ecvm_statistic



def IQR(y):
    y = list(y)
    q50 = np.median(y)
    q001, q01, q1, q25, q75, q99, q999, q9999 = np.percentile(y, [0.01, 0.1, 1, 25, 75, 99, 99.9, 99.99])
    return q001, q01, q1, q25, q75, q99, q999, q9999

def RoMS(y, yerr):
    #Robust median statistic (RoMS)
    len_part = (1/(len(y) - 1)) 
    sum_part = np.sum(abs(y-np.median(y))/yerr)
    return len_part*sum_part 

def stdnxs(y, yerr):
    #Normalized excess variance
    len_part = 1/(len(y)*np.mean(y)**2)
    sum_part = np.sum(((y - np.mean(y))**2 - yerr**2))
    return len_part*sum_part

def ptop_var(y, yerr):
    #Peak-to-peak variability
    m_std_diffs = y - yerr
    v = (np.max(m_std_diffs) - np.min(m_std_diffs)) / (np.max(m_std_diffs) + np.min(m_std_diffs)) 
    return v

def lagauto(y):
    #Lag-1 autocorrelation
    numerator = []
    for i, yy in enumerate(y[:-1]):
        numerator.append((yy - np.mean(y))*(y[i+1] - np.mean(y)))
    denominator = np.sum((y - np.mean(y))**2)
    return np.sum(numerator) / denominator


def cody_Q(mag, time, period, name):
    #calculates AM cody's Q value https://arxiv.org/pdf/1401.6582.pdf (page 25)
    mag_resid, time = sine_fit(mag, time, period, name)
    Q_value = (np.mean(mag_resid**2) - np.std(mag)**2)/(np.mean(mag**2) - np.std(mag)**2)
    return Q_value


def cody_M(mag, time):
    #calculatews AM cody's M value https://arxiv.org/pdf/1401.6582.pdf (page 26)
    mag = np.array(mag)
    q90, q10 = np.percentile(mag, [90, 10])
    #this is a silly and slow way of doing this but it helps me visualise. feel free to make faster
    percentile_mags = []
    for m in mag:
        if m > q90 or m < q10:
            percentile_mags.append(m)
    percentile_mean = np.mean(percentile_mags)
    M_value = (percentile_mean - np.median(mag))/(np.sqrt(np.mean(mag**2)))
    return M_value

def medianBRP(mag, magerr):
    median = np.median(mag)
    amplitude = (np.max(mag) - np.min(mag)) / 10
    n = len(mag)
    count = np.sum(np.logical_and(mag < median + amplitude, mag > median - amplitude))
    return float(count) / n

def RangeCumSum(mag):
    #Rcs approaches 0 for a symmetric 
    sigma = np.std(mag)
    N = len(mag)
    m = np.mean(mag)
    s = np.cumsum(mag - m) * 1.0 / (N * sigma)
    Rcs = np.max(s) - np.min(s)
    return Rcs

def MaxSlope(mag, time):
    slope = np.abs(mag[1:] - mag[:-1]) / (time[1:] - time[:-1])
    return np.max(slope)

def MedianAbsDev(mag, magerr):
    median = np.median(mag)
    devs = (abs(mag - median))
    MAD = np.median(devs)
    return MAD

def Stetson_K(mag, magerr):
    mean_mag = (np.sum(mag/(magerr*magerr)) / np.sum(1.0 / (magerr * magerr)))
    N = len(mag)
    sigmap = (np.sqrt(N * 1.0 / (N - 1)) * (mag - mean_mag) / magerr)
    K = (1 / np.sqrt(N * 1.0) * np.sum(np.abs(sigmap)) / np.sqrt(np.sum(sigmap ** 2)))
    return K








def Eta_e(mag, time):
    w = 1.0 / np.power(np.subtract(time[1:], time[:-1]), 2)
    w_mean = np.mean(w)
    N = len(time)
    sigma2 = np.var(mag)
    S1 = sum(w * (mag[1:] - mag[:-1]) ** 2)
    S2 = sum(w)
    eta_e = (w_mean * np.power(time[N - 1] - time[0], 2) * S1 / (sigma2 * S2 * N ** 2))
    return eta_e


def Eta(mag, time):
    numerator = []
    denominator = []
    for i, m in enumerate(mag[:-1]):
        numerator.append(((mag[i+1] - m)**2) / (len(mag - 1)))
        denominator.append(((m-np.mean(mag))**2) /(len(mag - 1)))
    eta = np.sum(numerator)/np.sum(denominator)
    return eta



