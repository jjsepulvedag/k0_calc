import numpy as np


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