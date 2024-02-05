import numpy as np
from .vectors import CrossP
from .single_det_inspiral_snr import SNR
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

# Load zero-det-high-power ASD
zdhP_f,zdhP_asd = np.loadtxt(os.path.join(dir_path,'zero_det_high_P.txt'),usecols=[0,1],unpack=True)

# Load O4 LIGO ASD (from https://dcc.ligo.org/T2200043-v3/public)
O4_LIGO_f,O4_LIGO_asd = np.loadtxt(os.path.join(dir_path,'LIGO_O4.txt'),usecols=[0,1],unpack=True)

O4_Virgo_f,O4_Virgo_asd = np.loadtxt(os.path.join(dir_path,'Virgo_O4.txt'),usecols=[0,1],unpack=True)

# Load O5 LIGO ASD (from https://dcc.ligo.org/T2200043-v3/public)
O5_LIGO_f,O5_LIGO_asd = np.loadtxt(os.path.join(dir_path,'LIGO_O5.txt'),usecols=[0,1],unpack=True)

O5_Virgo_f,O5_Virgo_asd = np.loadtxt(os.path.join(dir_path,'Virgo_O5.txt'),usecols=[0,1],unpack=True)




# Load ET-D ASD
ETD_f,ETD_asd = np.loadtxt(os.path.join(dir_path,'ETDSensitivityCurveTxtFile.txt'),usecols=[0,3],unpack=True)
ETD_f = ETD_f[::3]
ETD_asd = ETD_asd[::3]

# detector position class
class detector:
    
    def __init__(s,pos,xarm,yarm,PSD=(zdhP_f,zdhP_asd),name='detectorname',thin=30):
        s.pos = pos
        s.xarm = np.array(xarm)
        s.yarm = np.array(yarm)
        s.z = CrossP(s.xarm,s.yarm)
        s.PSD = (PSD[0][::thin],PSD[1][::thin])
        s.name=name
        s.d_ab = 0.5*(s.xarm[:,None]*s.xarm[None,:]-s.yarm[:,None]*s.yarm[None,:])
        s.r0 = 1.
        #s.rho0 = s.inspiral_snr(1.4,1.4,s.r0,0.,0.,0.,
    
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
    
    def inspiral_snr(self,m1,m2,dL,z,iota,psi,RA_deg,Dec_deg,gmst_deg):
        return SNR(m1,m2,dL,z,iota,RA_deg,Dec_deg,gmst_deg,psi,self)
    
    
        
        

# LIGO H1, LIGO L1 and Virgo zerodethighP
H1_zerodethighP = detector([-2.16141492636e+06,-3.83469517889e+06,4.60035022664e+06],[-0.22389266154,0.79983062746,0.55690487831],[-0.91397818574,0.02609403989,-0.40492342125],name='LIGO Hanford')
L1_zerodethighP = detector([-7.42760447238e+04,-5.49628371971e+06,3.22425701744e+06],[ -0.95457412153, -0.14158077340,-0.26218911324],[0.29774156894,-0.48791033647,-0.82054461286],name='LIGO Livingston')
Virgo_zerodethighP = detector([4.54637409900e+06,8.42989697626e+05,4.37857696241e+06],[-0.70045821479,0.20848948619,0.68256166277],[-0.05379255368,-0.96908180549,0.24080451708],name='Virgo')

HLV_zerodethighP = [H1_zerodethighP,L1_zerodethighP,Virgo_zerodethighP]

# LIGO H1, LIGO L1 and Virgo O4
H1_O4 = detector([-2.16141492636e+06,-3.83469517889e+06,4.60035022664e+06],[-0.22389266154,0.79983062746,0.55690487831],[-0.91397818574,0.02609403989,-0.40492342125],name='LIGO Hanford',PSD=(O4_LIGO_f,O4_LIGO_asd))
L1_O4 = detector([-7.42760447238e+04,-5.49628371971e+06,3.22425701744e+06],[ -0.95457412153, -0.14158077340,-0.26218911324],[0.29774156894,-0.48791033647,-0.82054461286],name='LIGO Livingston',PSD=(O4_LIGO_f,O4_LIGO_asd))
Virgo_O4 = detector([4.54637409900e+06,8.42989697626e+05,4.37857696241e+06],[-0.70045821479,0.20848948619,0.68256166277],[-0.05379255368,-0.96908180549,0.24080451708],name='Virgo',PSD=(O4_Virgo_f,O4_Virgo_asd))

HLV_O4 = [H1_O4,L1_O4,Virgo_O4]
HL_O4 = [H1_O4,L1_O4]


# LIGO H1, LIGO L1 and Virgo O5
H1_O5 = detector([-2.16141492636e+06,-3.83469517889e+06,4.60035022664e+06],[-0.22389266154,0.79983062746,0.55690487831],[-0.91397818574,0.02609403989,-0.40492342125],name='LIGO Hanford',PSD=(O5_LIGO_f,O5_LIGO_asd))
L1_O5 = detector([-7.42760447238e+04,-5.49628371971e+06,3.22425701744e+06],[ -0.95457412153, -0.14158077340,-0.26218911324],[0.29774156894,-0.48791033647,-0.82054461286],name='LIGO Livingston',PSD=(O5_LIGO_f,O5_LIGO_asd))
Virgo_O5 = detector([4.54637409900e+06,8.42989697626e+05,4.37857696241e+06],[-0.70045821479,0.20848948619,0.68256166277],[-0.05379255368,-0.96908180549,0.24080451708],name='Virgo',PSD=(O5_Virgo_f,O5_Virgo_asd))

HLV_O5 = [H1_O5,L1_O5,Virgo_O5]


a1 = np.pi/12. + (1-1)*np.pi/3.
a2 = np.pi/12. + (2-1)*np.pi/3.
a3 = np.pi/12. + (3-1)*np.pi/3.

x1 = np.cos(a1)
x2 = np.cos(a2)
x3 = np.cos(a3)
y1 = np.sin(a1)
y2 = np.sin(a2)
y3 = np.sin(a3)
 
ETD1 = detector([0.,0.,0.],[x1,y1,0.],[x2,y2,0.],name='ETD1',PSD=(ETD_f,ETD_asd))
ETD2 = detector([0.,0.,0.],[x2,y2,0.],[x3,y3,0.],name='ETD2',PSD=(ETD_f,ETD_asd))
ETD3 = detector([0.,0.,0.],[x3,y3,0.],[x1,y1,0.],name='ETD3',PSD=(ETD_f,ETD_asd))

ETD = [ETD1,ETD2,ETD3]
