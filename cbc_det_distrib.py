import numpy as np
import matplotlib.pyplot as plt
from single_det_inspiral_snr import SNR

# Schutz+11 expected distribution of inclinations for detected inspirals
def P_iota_det(iota):
    return 7.6e-2*(1+6*np.cos(iota)**2+np.cos(iota)**4)**1.5*np.sin(iota)

# BNS masses
M1 = 1.4
M2 = 1.4

# SNR limit for detection
SNR_det = 8.

# generate extrinsic parameter distributions
N = 100000
dLmax = 500 #Mpc
th = np.arccos(np.random.uniform(-1.,1.,N))
phi = np.random.uniform(0.,2*np.pi,N)
psi = np.random.uniform(0.,2*np.pi,N)

iota = np.arccos(np.random.uniform(-1.,1.,N))
dL = dLmax*np.random.uniform(0.,1,N)**(1/3)

# plot input distributions
plt.figure("Input distance distribution")
plt.hist(dL,bins=40,histtype='step',lw=2,color='b',density=True)
dl = np.linspace(0.,dLmax,200)
plt.plot(dl,dl**2*3/dLmax**3,'--r')
plt.xlabel(r"$d_\mathrm{L}\,\mathrm{[Mpc]}$")

plt.figure("Input inclination distribution")
plt.hist(iota/np.pi*180,bins=40,histtype='step',lw=2,color='b',density=True)
io = np.linspace(0.,np.pi,100)
plt.plot(io/np.pi*180,0.5*np.sin(io)*np.pi/180,'--r')
plt.xlabel(r"$\iota\,\mathrm{[deg]}$")

plt.show()

# compute SNR's
snr = np.empty(N)
print("Number of simulated CBCs: {0:d}".format(N))
for i in range(N):
    snr[i] = SNR(M1,M2,dL[i],0.,iota[i],th[i],phi[i],psi[i])
    if (i%100)==0:
        print("Computing SNR for binary number {0:06d}...".format(i),end='\r')

print('Done.')


# plot distributions for detected binaries

det = snr>SNR_det # select only detected binaries

plt.figure("Detected inspiral inclination distribution")
plt.hist(iota[det]/np.pi*180,bins=40,histtype='step',lw=2,color='r',density=True)
io = np.linspace(0.,np.pi,100)
plt.plot(io/np.pi*180,P_iota_det(io)*np.pi/180,'--b')
plt.xlabel(r"$\iota\,\mathrm{[deg]}$")

plt.figure("Detected inspiral distance distribution")
h,b,p = plt.hist(dL[det],bins=40,histtype='step',lw=2,color='r',density=True)
plt.plot([np.mean(dL[det]),np.mean(dL[det])],[0.,np.max(h),],'--k')
#dl = np.linspace(0.,dLmax,200)
#plt.plot(dl,P_dL_det(dl,np.max(dL[det])),'--b')
plt.xlabel(r"$d_\mathrm{L}\,\mathrm{[Mpc]}$")

plt.show()
