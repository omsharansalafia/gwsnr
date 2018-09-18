import numpy as np
import matplotlib.pyplot as plt
from single_det_inspiral_snr import SNR
from astropy.time import Time

#cross product
def CrossP(v1, v2):
	
	v = np.zeros(3)
	
	for i in range(3):
		v[i] = v1[(i+1)%3]*v2[(i+2)%3] - v1[(i+2)%3]*v2[(i+1)%3]
	
	return(v)

#dot product
def DotP(v1,v2):
	
	return(np.sum(v1*v2))

# Constants
c = 3e10
G = 6.67e-8
M_sun_to_g = 1.989e33 # Sun mass [g]
Mpc_to_cm = 3.08e24 # Megaparsec [cm]

# LIGO H1 position
H1pos = np.array([-2.16141492636e+06,-3.83469517889e+06,4.60035022664e+06])
H1xarm = np.array([-0.22389266154,0.79983062746,0.55690487831])
H1yarm = np.array([-0.91397818574,0.02609403989,-0.40492342125])
H1z = CrossP(H1xarm,H1yarm)


# GW170817 parameters
dL = 41 # Mpc
z = 0.01
Mc = 1.1977 #Msun
RAdeg = 15*(13 + 9/60. + 48.068639/3600.) #deg
DECdeg= -23 - 22/60. - 53.3909/3600. #deg
tgw = Time("2017-08-17 12:41:04")
gmst =  tgw.sidereal_time('mean',longitude=0.).to('deg').value #deg


# GW170817 position in geocentric frame
thetaGC = np.pi/2-DECdeg/180*np.pi
phiGC = (RAdeg-gmst)/180*np.pi
source_pos = np.array([np.sin(thetaGC)*np.cos(phiGC),np.sin(thetaGC)*np.sin(phiGC),np.cos(thetaGC)]) # cartesian unit vector in geocentric frame

# GW170817 position in detector frame

## compute the projection onto the detector frame cartesian basis
xdet = DotP(source_pos,H1xarm)
ydet = DotP(source_pos,H1yarm)
zdet = DotP(source_pos,H1z)

## transform to spherical coordinates
theta = np.arccos(zdet)
phi = np.arctan2(ydet,xdet)

# polarization angle

## vector of zero psi in detector frame
exR = np.array([-np.cos(theta)**2-np.sin(theta)**2*np.sin(phi)**2,np.sin(theta)**2*np.sin(phi)*np.cos(phi),np.cos(theta)*np.sin(theta)*np.cos(phi)])/np.sqrt(1+np.cos(phi)**2*(np.cos(theta)**2-1))

## vector of zero psi in GC frame
psi0GC = np.array([-np.sin(phiGC),np.cos(phiGC),0.])

## third vector of the orthonormal triple
psi0GC_orthogonal = CrossP(psi0GC,-source_pos)

## vecor along binary major axis
va = lambda psi: psi0GC*np.cos(psi) + psi0GC_orthogonal*np.sin(psi)

## psi in the detector frame
psi_det = lambda psi: np.arccos(DotP(va(psi),exR))

# plot SNR contours in iota, psi space
iota = np.linspace(0/180*np.pi,180/180*np.pi,100)
psi = np.linspace(0/180*np.pi,360/180*np.pi,50)
snr = np.zeros([len(iota),len(psi)])

for i in range(len(iota)):
    for j in range(len(psi)):
        snr[i,j] = SNR(2**0.2*Mc,2**0.2*Mc,dL,0.,iota[i],theta,phi,psi_det(psi[j]))

plt.contour(iota/np.pi*180,psi/np.pi*180,snr.T)
plt.colorbar(label="Hanford Design Sensitivity SNR")
plt.xlabel(r'$\iota$ [deg]')
plt.ylabel(r'$\psi$ [deg]')
plt.show()
