import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.optimize import curve_fit

from k0_calc.miscK import get_FAS
from k0_calc.miscK import rotateGM
from k0_calc.miscK import ln_Omega2Model


def krBB_GMR(acc, dt, flow, fupp, R, rho, Vs):

    S_fas, S_ffi = get_FAS(acc, dt)
    # S_ffi = np.where(S_ffi <= 0, 1e-10, S_ffi)
    ln_S_fas = np.log(S_fas + 1e-20)

    # 
    mask = (S_ffi > flow) & (S_ffi < fupp)
    fit_freq = S_ffi[mask]
    fit_ln_spectrum = ln_S_fas[mask]
    
    p0 = [1e16, 0.03] # M0, kappa initial guesses
    bounds = [[1e10, 0.005], [1e30, 0.5]] # M0, kappa

    fcs = np.linspace(0.01, 100, 500)
    leastSquareDF = pd.DataFrame(fcs, columns=['fci'])
    leastSquareDF[['M0', 'kappa', 'Chi']] = np.nan

    for i in range(leastSquareDF.shape[0]):
        fci = leastSquareDF.loc[i, 'fci']
        fit_funct = lambda fi, M0, kappa: ln_Omega2Model(fi, fci, M0, rho, Vs, R, kappa)

        # popt, pcov = curve_fit(fit_funct, fit_freq, fit_ln_spectrum, p0=p0, bounds=bounds)
        popt, pcov = curve_fit(fit_funct, fit_freq, fit_ln_spectrum)
        
        leastSquareDF.loc[i, ['M0', 'kappa']] = popt

        M_fi = ln_Omega2Model(fit_freq, fci, popt[0], rho, Vs, R, popt[1])

        leastSquareDF.loc[i, 'Chi'] = np.sum((fit_ln_spectrum - M_fi)**2) / fit_freq.shape[0]

    print(leastSquareDF)

    minChi = leastSquareDF['Chi'].idxmin()
    fc_fit = leastSquareDF.loc[minChi, 'fci']
    M0_fit = leastSquareDF.loc[minChi, 'M0']
    kappa_fit = leastSquareDF.loc[minChi, 'kappa']

    
    
    fas_Omega = np.exp(ln_Omega2Model(fit_freq, fc_fit, M0_fit, rho, Vs, R, kappa_fit))
    plt.plot(fit_freq, np.exp(fit_ln_spectrum), color='tab:blue', linewidth=0.5)
    plt.plot(fit_freq, fas_Omega, color='tab:red', linewidth=0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    return [fc_fit, M0_fit, kappa_fit]

def krBB_FAS():
    return None


def kappa_1GMR(acc, dt, f1, f2):
    # Create a DF for storing coefficients and stderrs
    dfCols = ['Kappa', 'Intercept']
    dfRows = ['Mean', 'Stderr']
    linRegDF = pd.DataFrame(None, index=dfRows, columns=dfCols)
    
    # Conduct the linear regression
    fas, ff = get_FAS(acc, dt)
    idx = np.where((f1<=ff) & (ff<=f2))
    results = st.linregress(ff[idx], np.log(fas[idx]))
    
    # Storing results in the DataFrame
    # Remember that the direct slope lambda=-kappa*pi
    linRegDF.loc['Mean', 'Kappa'] = - results.slope/np.pi
    linRegDF.loc['Stderr', 'Kappa'] = results.stderr/np.pi
    linRegDF.loc['Mean', 'Intercept'] = results.intercept
    linRegDF.loc['Stderr', 'Intercept'] = results.intercept_stderr

    return linRegDF

def kappa_1FAS(fas, ff, f1, f2):
    '''
    Computes kappa for a single record.
    INPUT: 
        - acc:
        - dt:
        - npoints: 
        - f1:
        - f2:
    OUTPUT:
        - kappa_i:
        - intercept_i
    '''

    # Create a DF for storing coefficients and stderrs
    dfCols = ['Kappa', 'Intercept']
    dfRows = ['Mean', 'Stderr']
    linRegDF = pd.DataFrame(None, index=dfRows, columns=dfCols)

    idx = np.where((f1<=ff) & (ff<=f2))
    results = st.linregress(ff[idx], np.log(fas[idx]))

    # Storing results in the DataFrame
    # Remember that the direct slope lambda=-kappa*pi
    linRegDF.loc['Mean', 'Kappa'] = - results.slope/np.pi
    linRegDF.loc['Stderr', 'Kappa'] = results.stderr/np.pi
    linRegDF.loc['Mean', 'Intercept'] = results.intercept
    linRegDF.loc['Stderr', 'Intercept'] = results.intercept_stderr

    return linRegDF





def kappa_VH14(acc1, acc2, dt, N, f1, f2):
    '''
    INPUT: 
        - acc000: ground-motion record, component 000 (north) 
        - acc090: ground-motion record, component 090 (east) 
        - dt    : ground-motion time step
        - N     : number of elements for the computation of rotd50 (rotations)
        - f1    : 
        - f2    :
    NOTE: 
        PLEASE INCLUDE A N EQUAL OR GREATER THAN 2. 
        - For a dStep of 01°, use N=181
        - For a dStep of 02°, use N=91
        - For a dStep of 05°, use N=37 
        - For a dStep of 10°, use N=19
        - For a dStep of 15°, use N=13
        - For a dStep of 20°, use N=10
        - For a dStep of 30°, use N=07
        - For a dStep of 45°, use N=05
    '''
    dfCols = ['Kappa', 'Intercept']
    dfRows = ['Mean', 'Stderr']
    avgLinRegDF = pd.DataFrame(None, index=dfRows, columns=dfCols)
    thetas = np.radians(np.linspace(0, 180, N))
    allLinRegDF = np.zeros((thetas.shape[0], 2, 2)) # N rotations, 2r, 2c

    for i in range(thetas.shape[0]):
        acc_r = rotateGM(acc1, acc2, thetas[i]) # get rotated ground motion
        allLinRegDF[i] = kappa_1GMR(acc_r, dt, f1, f2).values
    
    # Van Houtte et al. (2014) uses mean. Rotd50 is computed with median
    avgLinRegDF.loc['Mean', 'Kappa'] = allLinRegDF[:, 0, 0].mean()
    avgLinRegDF.loc['Mean', 'Intercept'] = allLinRegDF[:, 0, 1].mean()
    avgLinRegDF.loc['Stderr', 'Kappa'] = np.sqrt(1 / (1/(allLinRegDF[:, 1, 0]**2)).sum())
    avgLinRegDF.loc['Stderr', 'Intercept'] = np.sqrt(1 / (1/(allLinRegDF[:, 1, 1]**2)).sum())

    return avgLinRegDF


def kappa_RMS(acc1, acc2, dt, f1, f2):

    fas000, ff000 = get_FAS(acc1, dt)
    fas090, ff090 = get_FAS(acc2, dt)

    H_f = np.sqrt(0.5*(fas000**2 + fas090**2))

    linRegDF = kappa_1FAS(H_f, ff000, f1, f2)
    
    return linRegDF


def kappa_Mean(acc1, acc2, dt, f1, f2):

    dfCols = ['Kappa', 'Intercept']
    dfRows = ['Mean', 'Stderr']
    avgLinRegDF = pd.DataFrame(None, index=dfRows, columns=dfCols)
    dimens = 2 # We compute two values of kappa and intercept
    allLinRegDF = np.zeros((dimens, 2, 2)) # 2d, 2r, 2c

    allLinRegDF[0] = kappa_1GMR(acc1, dt, f1, f2).values
    allLinRegDF[1] = kappa_1GMR(acc2, dt, f1, f2).values

    avgLinRegDF.loc['Mean', 'Kappa'] = allLinRegDF[:, 0, 0].mean()
    avgLinRegDF.loc['Mean', 'Intercept'] = allLinRegDF[:, 0, 1].mean()
    avgLinRegDF.loc['Stderr', 'Kappa'] = np.sqrt(1 / (1/(allLinRegDF[:, 1, 0]**2)).sum())
    avgLinRegDF.loc['Stderr', 'Intercept'] = np.sqrt(1 / (1/(allLinRegDF[:, 1, 1]**2)).sum())

    return avgLinRegDF

if __name__=='__main__':

    print('JJSG: Hello World!')