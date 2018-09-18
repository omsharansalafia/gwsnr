import math as m
import numpy as np
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

# Load ASD
zdhP_f,zdhP_asd = np.loadtxt(os.path.join(dir_path,'zero_det_high_P.txt'),usecols=[0,1],unpack=True)

# Constants
c = 3e10
G = 6.67e-8
M_sun_to_g = 1.989e33 # Sun mass [g]
Mpc_to_cm = 3.08e24 # Megaparsec [cm]

# Detector antenna pattern functions (as given in Schutz+11) 
def Fplus(theta, phi, psi):
    return 0.5*(1.+np.cos(theta)**2)*np.cos(2*phi)* \
            np.cos(2*psi)-np.cos(theta)*np.sin(2*phi)*np.sin(2*psi)

def Fcross(theta, phi, psi):
    return 0.5*(1+np.cos(theta)**2)*np.cos(2*phi)*np.sin(2*psi)+np.cos(theta)*np.sin(2*phi)*np.cos(2*psi)


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
def SNR(mass1, mass2, dl, z, iota, theta, phi, psi, ASD=(zdhP_f,zdhP_asd)):
    """Compute the single detector SNR for the inspiral of a compact binary, with a cutoff frequency at the non-spinning ISCO.
    
    Arguments:
    - mass1, mass2: component masses in Msun
    - dl: luminosity distance in Mpc
    - z: redshift
    - iota: angle between orbital angular momentum and line of sight
    - theta, phi: spherical coordinates of the binary direction in the detector frame
    - psi: polarization angle in the detector frame
    
    Keyword arguments:
    - ASD: tuple (f,asd) containing an array of frequencies f and the corresponding detector ASD. Default: LIGO design zero_det_high_P ASD, with 200 Mpc BNS range.
    
    Returns:
    - snr: single detector SNR for the given ASD (scalar)
    """
    freq,asd = ASD # unpack ASD
    
    # chirp mass
    Mc = (mass1*mass2)**(3./5.)/(mass1+mass2)**(1./5.)*(1+z)*M_sun_to_g
    # lum distance in cm
    dL = dl*Mpc_to_cm
    
    # compute antenna pattern functions
    fplus = Fplus(theta, phi, psi)
    fcross = Fcross(theta, phi, psi)
    
    # compute strain amplitude at the same frequencies as the ASD ones
    hft =  Hft(freq,Mc,dL,iota,fplus,fcross)
    
    # cutoff frequency at f_isco
    fisco = 2.*1./(6*np.sqrt(6)*2*np.pi)*(c**3/(G*(mass1+mass2)))*(1+z)
    hft[freq>fisco]=0.
    
    # perform the SNR integral as given in Sesana & Colpi 2017
    integrandum = np.abs(2*hft*np.sqrt(freq))**2/asd**2
    snr = np.sqrt(np.trapz(integrandum,np.log(freq)))
    
    return snr
