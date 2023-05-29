import numpy as np
import matplotlib.pyplot as plt
from .single_det_inspiral_snr import SNR
from .constants import *
from astropy.time import Time
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))


def Network_SNR(network,Mc,dL,z,iota,psi,RAdeg,DECdeg,gmst_deg):
    """
    Compute the network SNR for a compact binary inspiral.
    
    Parameters:
    - network: a list of instances of ::class:: detector
    - Mc: source chirp mass in Msun
    - dL: source distance in Mpc
    - z: source redshift
    - iota: source orbital plane inclination (radians)
    - psi: source polarization angle in geocentric frame (radians)
    - RAdeg: the right ascension of the source in degrees
    - DECdeg: the declination of the source in degrees
    - gmst: the greenwich mean sidereal time at coalescence time, in degrees
    
    Returns: SNR_net, (SNR_0, SNR_1, ...) [, (dt01, dt02, dt12, ...)]
    - SNR_net: the network SNR
    - (SNR_0, SNR_1, ...): a tuple of single-detector SNRs, with the same size as network
    
    """
        
    # for each detector, compute source position in detector frame
    snr = np.zeros(len(network))
    
    for i,detector in enumerate(network):
        # compute SNR
        snr[i] = SNR(2**0.2*Mc,2**0.2*Mc,dL,z,iota,RAdeg,DECdeg,gmst_deg,psi,detector)
    
    return np.sum(snr**2)**0.5,snr
        
    
