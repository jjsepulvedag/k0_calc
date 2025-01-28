import numpy as np 
import scipy.stats as st
from k0_calc.miscK import get_FAS

def kappa_Srcrd(acc, dt, npoints, f1, f2):
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

    fas, ff = get_FAS(acc, dt)
    idx = np.where((f1<=ff) & (ff<=f2))
    results = st.linregress(ff[idx], np.log(fas[idx]))
    lambda_i, intercept_i = results.slope, results.intercept
    kappa_i = - lambda_i/np.pi

    return kappa_i, intercept_i

def kappa_VH14(acc1, acc2, dt, npoints, f1, f2):

    return None 

def kappa_RMS(acc1, acc2, dt, npoints, f1, f2):
    
    return None

def kappa_AVG(acc1, acc2, dt, npoints, f1, f2):
    
    return None

if __name__=='__main__':

    print('Hello world JJSG')