import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.optimize import curve_fit

from k0_calc.miscK import get_FAS
from k0_calc.miscK import rotateGM
from k0_calc.miscK import ln_Omega2Model


def krBB_FAS(fas, ffi, flow, fupp, R, rho, Vs):
    '''
    Note: Careful with units, they are highly important. 

    Input params:
        acc: gm accelerations (cm/s2). Remember accs in Lee etal 22 are in g.
        dt:  time step (s)
        flow: lower frequency (Hz)
        flow: upper frequency (Hz)
        R: source-to-site distance (cm)
        rho: density (g/cm3)
        Vs: shear wave velocity (cm/s)
    Output: 
        list with fitted fc, M0, and kappa.
    '''

    ln_S_fas = np.log(fas + 1e-20) # I use 1e-20 to avoid FAS=0

    # Mask FAS. In other words, filter the frequencies we will fit
    mask = (ffi > flow) & (ffi < fupp)
    fit_freq = ffi[mask] # Frequencies to fit
    fit_ln_spectrum = ln_S_fas[mask]  # FAS to fit
    
    # Defining some params for curve_fit
    p0 = [10e17, 0.03] # M0, kappa initial guesses
    bounds = [[10e13, 0.005], [10e35, 0.1]] # M0, kappa bounds

    # Defining pandas DF for the iterations of fc
    fcs = np.logspace(np.log10(0.01), np.log10(50.0), 400) # fc values
    leastSquareDF = pd.DataFrame(fcs, columns=['fci']) 
    leastSquareDF[['M0', 'kappa', 'Chi']] = np.nan
    
    # Iterate and compute best parameters (M0 and kappa) for each fc
    for i in range(leastSquareDF.shape[0]):

        fci = leastSquareDF.loc[i, 'fci']

        # I needed to create a lambda function to fit the structure of curve_fit
        # Note that M0, rho, and Vs, come from the function definition
        fit_funct = lambda fi, M0, kappa: ln_Omega2Model(fi, fci, M0, rho, Vs, 
                                                         R, kappa)

        # Obtaining and assigning opt params (M0 and kappa)
        popt, pcov = curve_fit(fit_funct, fit_freq, fit_ln_spectrum, p0=p0, 
                               bounds=bounds)
        leastSquareDF.loc[i, ['M0', 'kappa']] = popt

        # Obtain theoretical spectrum (M_f) and compute residuals (Chi)
        M_fi = ln_Omega2Model(fit_freq, fci, popt[0], rho, Vs, R, popt[1])
        residuals = fit_ln_spectrum - M_fi
        leastSquareDF.loc[i, 'Chi'] = np.sum(residuals**2) / fit_freq.shape[0]

    # Obtaining idx of min Chi and define parameters in this row as opt params
    minChi = leastSquareDF['Chi'].idxmin()

    fc_fit = leastSquareDF.loc[minChi, 'fci']
    M0_fit = leastSquareDF.loc[minChi, 'M0']
    kappa_fit = leastSquareDF.loc[minChi, 'kappa']

    print(leastSquareDF.loc[minChi, 'Chi'])
    
    fas_Omega = ln_Omega2Model(fit_freq, fc_fit, M0_fit, rho, Vs, R, kappa_fit)
    plt.plot(fit_freq, np.exp(fit_ln_spectrum), color='tab:blue', linewidth=0.5)
    plt.plot(fit_freq, np.exp(fas_Omega), color='tab:red', linewidth=0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    return [fc_fit, M0_fit, kappa_fit]


def krBB_GMR(acc, dt, flow, fupp, R, rho, Vs):
    '''
    Note: Careful with units, they are highly important. 

    Input params:
        acc: gm accelerations (cm/s2). Remember accs in Lee_etal_22 are in g.
        dt:  time step (s)
        flow: lower frequency (Hz)
        flow: upper frequency (Hz)
        R: source-to-site distance (cm)
        rho: density (g/cm3)
        Vs: shear wave velocity (cm/s)
    Output: 
        list with fitted fc, M0, and kappa.
    '''

    # Get FAS from accs and dt
    fas, ffi = get_FAS(acc, dt)

    # Calling krBB_FAS(), which does all the work. It returns fitted params in 
    # this order [fc_fit, M0_fit, kappa_fit]
    optParams = krBB_FAS(fas, ffi, flow, fupp, R, rho, Vs)

    return optParams


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