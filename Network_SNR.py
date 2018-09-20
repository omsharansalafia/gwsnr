import numpy as np
import matplotlib.pyplot as plt
from single_det_inspiral_snr import SNR
from astropy.time import Time
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

# Load ASD
zdhP_f,zdhP_asd = np.loadtxt(os.path.join(dir_path,'zero_det_high_P.txt'),usecols=[0,1],unpack=True)

#cross product
def CrossP(v1, v2):
	
	v = np.zeros(3)
	
	for i in range(3):
		v[i] = v1[(i+1)%3]*v2[(i+2)%3] - v1[(i+2)%3]*v2[(i+1)%3]
	
	return(v)

#dot product
def DotP(v1,v2):
	
	return(v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])

# Constants
c = 3e10
G = 6.67e-8
M_sun_to_g = 1.989e33 # Sun mass [g]
Mpc_to_cm = 3.08e24 # Megaparsec [cm]

# detector position class
class detector:
    
    def __init__(s,pos,xarm,yarm,PSD=(zdhP_f,zdhP_asd),name='detectorname'):
        s.pos = pos
        s.xarm = xarm
        s.yarm = yarm
        s.z = CrossP(xarm,yarm)
        s.PSD = PSD
        s.name=name
        

# LIGO H1, LIGO L1 and Virgo positions
H1 = detector([-2.16141492636e+06,-3.83469517889e+06,4.60035022664e+06],[-0.22389266154,0.79983062746,0.55690487831],[-0.91397818574,0.02609403989,-0.40492342125],name='LIGO Hanford')
L1 = detector([-7.42760447238e+04,-5.49628371971e+06,3.22425701744e+06],[ -0.95457412153, -0.14158077340,-0.26218911324],[0.29774156894,-0.48791033647,-0.82054461286],name='LIGO Livingston')
Virgo = detector([4.54637409900e+06,8.42989697626e+05,4.37857696241e+06],[-0.70045821479,0.20848948619,0.68256166277],[-0.05379255368,-0.96908180549,0.24080451708],name='Virgo')

HLV_zerodethighP = [H1,L1,Virgo]


def Network_SNR(network,Mc,dL,z,iota,psi,RAdeg,DECdeg,gmst):
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
    
    # compute source spherical coordinates in geocentric frame
    thetaGC = np.pi/2-DECdeg/180.*np.pi
    phiGC = (RAdeg-gmst)/180.*np.pi
    source_pos = np.array([np.sin(thetaGC)*np.cos(phiGC),np.sin(thetaGC)*np.sin(phiGC),np.cos(thetaGC)]) # cartesian unit vector in geocentric frame
    
    # for each detector, compute source position in detector frame
    snr = np.zeros(len(network))
    
    for i,detector in enumerate(network):
        
        ## compute the projection of the source position onto the 
        ## detector frame cartesian basis
        xdet = DotP(source_pos,detector.xarm)
        ydet = DotP(source_pos,detector.yarm)
        zdet = DotP(source_pos,detector.z)
        
        ## transform to spherical coordinates
        theta = np.arccos(zdet)
        phi = np.arctan2(ydet,xdet)
        
        
        ## vector of zero psi in detector frame
        exR = np.array([-np.cos(theta)**2-np.sin(theta)**2*np.sin(phi)**2,np.sin(theta)**2*np.sin(phi)*np.cos(phi),np.cos(theta)*np.sin(theta)*np.cos(phi)])/np.sqrt(1+np.cos(phi)**2*(np.cos(theta)**2-1))
        
        ## vector of zero psi in GC frame
        psi0GC = np.array([-np.sin(phiGC),np.cos(phiGC),0.])
        
        ## third vector of the orthonormal triple
        psi0GC_orthogonal = CrossP(psi0GC,-source_pos)
        
        ## vecor along binary major axis
        va =  psi0GC*np.cos(psi) + psi0GC_orthogonal*np.sin(psi)
        
        ## psi in the detector frame
        psi_det = np.arccos(DotP(va,exR))
        
        # compute SNR
        snr[i] = SNR(2**0.2*Mc,2**0.2*Mc,dL,z,iota,theta,phi,psi_det,ASD=detector.PSD)
    
    return np.sum(snr**2)**0.5,snr
        
    
