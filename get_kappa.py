import numpy as np 
import scipy.stats as st
from k0_calc.miscK import get_FAS
from k0_calc.miscK import rotateGM


def kappa_1GMR(acc, dt, f1, f2):
    # Create a DF for storing coefficients and stderrs
    dfCols = ['Slope', 'Intercept']
    dfRows = ['Mean', 'Stderr']
    kappaDF = pd.dataframe(None, index=dfRows, columns=dfCols)

    # Conduct the linear regression
    fas, ff = get_FAS(acc, dt)
    idx = np.where((f1<=ff) & (ff<=f2))
    results = st.linregress(ff[idx], np.log(fas[idx]))

    # Storing results in the DataFrame
    # Remember that the direct slope is lambda, which is lambda=-kappa*pi
    kappaDF['Mean', 'Slope'] = - results.slope/np.pi
    kappaDF['Stderr', 'Slope'] = results.stderr/np.pi
    kappaDF['Mean', 'Intercept'] = results.intercept
    kappaDF['Stderr', 'Intercept'] = results.intercept_stderr

    return kappaDF

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
    dfCols = ['Slope', 'Intercept']
    dfRows = ['Mean', 'Stderr']
    kappaDF = pd.dataframe(None, index=dfRows, columns=dfCols)

    idx = np.where((f1<=ff) & (ff<=f2))
    results = st.linregress(ff[idx], np.log(fas[idx]))

    # Storing results in the DataFrame
    # Remember that the direct slope is lambda, which is lambda=-kappa*pi
    kappaDF['Mean', 'Slope'] = - results.slope/np.pi
    kappaDF['Stderr', 'Slope'] = results.stderr/np.pi
    kappaDF['Mean', 'Intercept'] = results.intercept
    kappaDF['Stderr', 'Intercept'] = results.intercept_stderr

    return kappaDF

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
    thetas = np.radians(np.linspace(0, 180, N))
    kappas = np.zeros((thetas.shape[0], 2))

    for i in range(thetas.shape[0]):
        acc_r = rotateGM(acc1, acc2, thetas[i]) # get rotated ground motion
        kappas[i, :], xxx = kappa_1GMR(acc_r, dt, f1, f2)
    
    # Van Houtte et al. (2014) uses mean. Rotd50 is computed with median
    kappaVH14 = np.mean(kappas, axis=0) 

    return kappaVH14

def kappa_RMS(acc1, acc2, dt, f1, f2):

    fas000, ff000 = get_FAS(acc1, dt)
    fas090, ff090 = get_FAS(acc2, dt)

    H_f = np.sqrt(0.5*(fas000**2 + fas090**2))

    kappaRMS, interceptRMS = kappa_1FAS(H_f, ff000, f1, f2)
    
    return kappaRMS


def kappa_GM(acc1, acc2, dt, f1, f2):
    '''
    Not recommended. It crashes when kappa is negative in one direction. 
    '''
    
    kappa000, intercept_000 = kappa_1GMR(acc1, dt, f1, f2)
    kappa090, intercept_090 = kappa_1GMR(acc2, dt, f1, f2)

    kappaGM = np.sqrt(kappa000*kappa090)

    return kappaGM

def kappa_Mean(acc1, acc2, dt, f1, f2):
    kappa000, intercept000 = kappa_1GMR(acc1, dt, f1, f2)
    kappa090, intercept090 = kappa_1GMR(acc2, dt, f1, f2)
    kappaAVG = np.mean([[kappa000, kappa090]])

    return kappaAVG

if __name__=='__main__':

    print('JJSG: Hello World!')