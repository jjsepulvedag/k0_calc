r"""FILE TITLE

PURPOSE: 
# ------------------------------------- . ------------------------------------ #
FUNCTIONS:
funct1    : short description
# ------------------------------------- . ------------------------------------ #
AUTHORSHIP
Written by: Juan Jose Sepulveda Garcia
Mail      : jjs134@uclive.ac.nz / jjsepulvedag@unal.edu.co
Date      : March 2025
# ------------------------------------- . ------------------------------------ #
REFERENCES:
NA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def silva_etal_1998():
    '''
    Input
    Output:
    Scope: Vs range?
    Calibred from data in: 
    Calibred to: 
    Main reference: 
    '''
    
    return None

def chandler_etal_2006():

    return None

def vanHoutte_etal_2011():

    return None

def edwards_etal_2011():

    return None

def xu_etal_2021(Vs30):

    cond_list = [(100<=Vs30) & (Vs30<155), 
                 (155<Vs30) & (Vs30<=2000), 
                 (2000<Vs30) & (Vs30<=3000)
                 ]
    func_list = [-0.18*(np.log(155))**2 +1.816*np.log(155) - 7.38,
                 lambda Vs30: -0.18*(np.log(Vs30))**2+1.816*np.log(Vs30) - 7.38,
                 -0.18*(np.log(2000))**2+1.816*np.log(2000) - 7.38
                 ]

    ln_k0 = np.piecewise(Vs30, cond_list, func_list)

    return np.exp(ln_k0)




if __name__ == '__main__':

    x = np.linspace(100, 3000, 1000)
    y = xu_etal_2021(x)


    plt.plot(x, y)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0, 3000)
    plt.ylim(0.01, 0.1)
    plt.grid(which='both')
    plt.show()