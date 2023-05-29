import numpy as np
from .vectors import CrossP
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

# Load zero-det-high-power ASD
zdhP_f,zdhP_asd = np.loadtxt(os.path.join(dir_path,'zero_det_high_P.txt'),usecols=[0,1],unpack=True)

# detector position class
class detector:
    
    def __init__(s,pos,xarm,yarm,PSD=(zdhP_f,zdhP_asd),name='detectorname'):
        s.pos = pos
        s.xarm = np.array(xarm)
        s.yarm = np.array(yarm)
        s.z = CrossP(s.xarm,s.yarm)
        s.PSD = PSD
        s.name=name
        s.d_ab = 0.5*(s.xarm[:,None]*s.xarm[None,:]-s.yarm[:,None]*s.yarm[None,:])
    
    def antenna_pattern(s,RA_deg,Dec_deg,gmst_deg,psi):
        
        # following Whelan 2013
        
        a = (RA_deg-gmst_deg)/180.*np.pi
        d = Dec_deg/180.*np.pi
        
        i = np.array([np.sin(a),-np.cos(a),0.])
        j = np.array([-np.sin(d)*np.cos(a),-np.sin(d)*np.sin(a),np.cos(d)])
        
        l =  i*np.cos(psi)+j*np.sin(psi)
        m = -i*np.sin(psi)+j*np.cos(psi)
                
        eplus  = l[:,None]*l[None,:] - m[:,None]*m[None,:]
        ecross = l[:,None]*m[None,:] + m[:,None]*l[None,:]
        
        Fplus  = np.sum(s.d_ab*eplus)
        Fcross = np.sum(s.d_ab*ecross)
        
        return Fplus,Fcross
        
        

# LIGO H1, LIGO L1 and Virgo positions
H1 = detector([-2.16141492636e+06,-3.83469517889e+06,4.60035022664e+06],[-0.22389266154,0.79983062746,0.55690487831],[-0.91397818574,0.02609403989,-0.40492342125],name='LIGO Hanford')
L1 = detector([-7.42760447238e+04,-5.49628371971e+06,3.22425701744e+06],[ -0.95457412153, -0.14158077340,-0.26218911324],[0.29774156894,-0.48791033647,-0.82054461286],name='LIGO Livingston')
Virgo = detector([4.54637409900e+06,8.42989697626e+05,4.37857696241e+06],[-0.70045821479,0.20848948619,0.68256166277],[-0.05379255368,-0.96908180549,0.24080451708],name='Virgo')

HLV_zerodethighP = [H1,L1,Virgo]

