import math as m
import numpy as np
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
from .constants import *

"""
# Detector antenna pattern functions (as given in Schutz+11) 
def Fplus(theta, phi, psi):
    return 0.5*(1.+np.cos(theta)**2)*np.cos(2*phi)* \
            np.cos(2*psi)-np.cos(theta)*np.sin(2*phi)*np.sin(2*psi)

def Fcross(theta, phi, psi):
    return 0.5*(1+np.cos(theta)**2)*np.cos(2*phi)*np.sin(2*psi)+np.cos(theta)*np.sin(2*phi)*np.cos(2*psi)
"""

# fourier transformed inspiral strain as given in Maggiore's book
def Hft(freq,Mc,dl,iota,fplus,fcross):
    A = np.pi**(-2./3.)*(5./24.)**(1./2.)
    B = c/dl*(G*Mc/c**3)**(5./6.)*freq**(-7./6.)
    Psiplus = 2*np.pi*freq-np.pi/4.+(3./4.)*(G*Mc/c**3/8*np.pi*freq)**(-5./3.) #assumes zero initial phase
    Psicross = Psiplus + np.pi/2.
    hplusft = A*np.exp(1j*Psiplus)*B*(1+np.cos(iota)**2)/2.
    hcrossft = A*np.exp(1j*Psicross)*B*np.cos(iota)

    return fplus*hplusft+fcross*hcrossft

# SNR computation following Sesana & Colpi 2017
def SNR(mass1, mass2, dl, z, iota, RA_deg, Dec_deg, gmst_deg, psi, detector):
    """Compute the single detector SNR for the inspiral of a compact binary, with a cutoff frequency at the non-spinning ISCO.
    
    Arguments:
    - mass1, mass2: component masses in Msun
    - dl: luminosity distance in Mpc
    - z: redshift
    - iota: angle between orbital angular momentum and line of sight
    - RA_deg, Dec_deg: source sky position in degrees
    - psi: polarization angle in geocentric frame
    - gmst_deg: Greenwich mean sidereal time at coalescence, in degrees
        
    Keyword arguments:
    - detector: instance of detector class. Default: LIGO Livingston with design zero_det_high_P PSD (200 Mpc BNS range).
    
    Returns:
    - snr: single detector SNR for the given PSD (scalar)
    """
    
    # chirp mass
    Mc = (mass1*mass2)**(3./5.)/(mass1+mass2)**(1./5.)*(1+z)*M_sun_to_g
    # lum distance in cm
    dL = dl*Mpc_to_cm
    
    # compute antenna pattern functions
    fplus,fcross = detector.antenna_pattern(RA_deg,Dec_deg,gmst_deg,psi)
    
    # compute strain amplitude at the same frequencies as the ASD ones
    freq = detector.PSD[0]
    hft =  Hft(freq,Mc,dL,iota,fplus,fcross)
    
    # cutoff frequency at f_isco
    fisco = 2.*1./(6*np.sqrt(6)*2*np.pi)*(c**3/(G*(mass1+mass2)))*(1+z)
    hft[freq>fisco]=0.
    
    # perform the SNR integral as given in Sesana & Colpi 2017
    integrandum = np.abs(2*hft*np.sqrt(freq))**2/detector.PSD[1]**2
    snr = np.sqrt(np.trapz(integrandum,np.log(freq)))
    
    return snr
