import numpy as np


def Mw_to_M0(M_w):

    M0 = None
    return M0

def Ml_to_M0(Ml):

    M0 = None
    return M0

def get_FAS(y, dt):
    """
    Returns Fourier amplitude spectrum of a time series.
    INPUT:
        - y: ground motion time series
        - dt: time step in ground motion time series
    OUTPUT:
        - fa_spectrum: Fourier amplitude spectrum
        - fa_frequencies: frequency vector corresponding to fa_spectrum values
    """

    n = y.shape[0] # number of values in time series
    nfft = 2 ** int(np.ceil(np.log2(n)))
    fa_spectrum = np.fft.rfft(y, n=nfft, axis=0) * dt
    fa_spectrum = np.abs(fa_spectrum)
    fa_frequencies = np.fft.rfftfreq(nfft, dt)

    return fa_spectrum, fa_frequencies

def get_RSM(acc1, acc2, dt):

    fas000, ff000 = get_FAS(acc1, dt)
    fas090, ff090 = get_FAS(acc2, dt)

    H_f = np.sqrt(0.5*(fas000**2 + fas090**2))

    return H_f, ff000

def ln_Omega2Model(fi, fc, M0, rho, Vs, R, kappa):
    '''
    Model from Brune (1970) with the modification of Anderson (1976)
    Input: 
    rho: density (gm/cm3)
    beta: shear-wave velocity (km/sec)
    '''
    gamma = 2 # omega square model
    
    ln_A0 = np.log( 0.85 * M0 / (4 * np.pi * rho * Vs**3 * R) )
    ln_path = - np.pi * kappa * fi
    ln_source = np.log( (2 * np.pi * fi)**2 / (1 + (fi/fc)**gamma) )

    return ln_A0 + ln_path + ln_source


def rotateGM(gm000, gm090, theta):

    gm_r = gm000*np.cos(theta) + gm090*np.sin(theta) # rotated ground motion

    return gm_r


def readGP(loc, fname):
    """
    Convenience function for reading files in the Graves and Pitarka format
    INPUT:
        - loc: location of the ground-motion time series 
        - fname: file name
    OUTPUT:
        - data: ground-motion time series
        - num_pts: number of points of the waveform 
        - dt: 
        - shift:
    """

    with open("/".join([loc, fname]), 'r') as f:
        lines = f.readlines()

    data = []

    for line in lines[2:]:
        data.append([float(val) for val in line.split()])

    data=np.concatenate(data)

    line1=lines[1].split()
    num_pts=float(line1[0])
    dt=float(line1[1])
    shift=float(line1[4])

    return data, num_pts, dt, shift


if __name__=='__main__':



    print('hola Mundo')