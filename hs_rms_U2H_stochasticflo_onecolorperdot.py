#In this code, Nx can be unequal to Ny. Most other codes that takes x,y from Bia's code do not have this compability. (Aug.14, 2024)
#Draw random samples of currents with differnt alpha, and compute rms from U2H outcomes 
#No zero padding is done, as the velocities are periodic by construction
#The random seeds are varied in each snapshot, and the seeds are different for phit and psit.

import netCDF4 as nc
import numpy as np
import os
import matplotlib.pyplot as plt
import colorcet as cc
from scipy.special import comb
import sys
sys.path.insert(0, '../hw_pythonfuncs')
import hwffts 
import scipy.signal
from scipy import special
from matplotlib.markers import MarkerStyle
import itertools
from scipy import integrate
import matplotlib
import matplotlib.patches as mpatches
import random
 
def hwdetrend(A,SSH,X,Y):
        C,_,_,_=scipy.linalg.lstsq(A, SSH.flatten())
        SSH_detrended=SSH-(C[0]*X + C[1]*Y + C[2])
        return SSH_detrended
def hwdiffx(u,x2d): #Compute derivative in x with finite difference,
#and trim the outcome so that the axis in y also has -1 index.
#Has been verified by comparing the outcome with the one from ifft for periodic u.
    ux=np.diff(u,axis=0)/np.diff(x2d,axis=0)   
    ux=ux[:,:-1]
    return ux

def hwdiffy(u,y2d): #Compute derivative in y with finite difference
#and trim the outcome so that the axis in x also has -1 index.
#Has been verified by comparing the outcome with the one from ifft for periodic u.
    uy=np.diff(u,axis=1)/np.diff(y2d,axis=1)   
    uy=uy[:-1,:]
    return uy
def find_i_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx 
pi=np.pi
#datapath='C:/Users/han/OneDrive - University of Edinburgh/Project_surfacewaves/Biadata_JPO2020/synthetic/standard/gridded/'
g=9.806 #Gravity acceleration. 
Hs0=1 #Mean assumed for the CZ transform; have tested that changing Hs0 does not affect the outcome in terms of hs/Hs0
padportionx=0 #how much to pad in each direction 
padportiony=0

#Grid of varphi for the integration in varphi to get lphi, lpsi
Nvarphiint=64 #Change this for different levels of accuracy
varphiint=np.linspace(0,2*pi,Nvarphiint)

colorlist_line=['#b2e2e2','#66c2a4','#2ca25f','#006d2c']
colorlist_asymp=['#fdcc8a','#fc8d59','#e34a33','#b30000']

#************This slope should have negative sign in this context! 
KEslopeall=[-1,-2, -3] #Slope for the isotropic or along-wave spectra
#**********check that this is the case in the other codes that don't read WW3 outputs. In the codes
#that read WW3 outputs, change the variable name or something, so that this does not lead to confusions. 
#divfracall=np.linspace(0,1,11)#[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]divfracall=np.linspace(0,1,11)#[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
#divfracall=[0,0.5,1]
divfracall=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
#waveTall= [7.0, 10.3, 16.6]
waveTall= [10.3]
MKEall=[0.01] 
#MKEall=np.linspace(0.005,0.04,5)
#MKEall=[0.01]
s_all=[1,5,10]
#s_all=[10]

Nx=250 #Bia's JPO paper: 250
Ny=250 #Bia's JPO paper: 250
dx=2500 #Bia's JPO paper: 2500
dy=2500 #Bia's JPO paper: 2500
x1d=np.linspace(0,(Nx-1)*dx,Nx) 
y1d=np.linspace(0,(Ny-1)*dy,Ny) 
#Weird, to think about: decreasing Nx, Ny would lead the dots to be consistently lower
    
Nx=len(x1d)
Ny=len(y1d)
#Shift x1d and y1d so that they center at 0, for convenience of hwfft. 
xshift=x1d[Nx//2-1]
yshift=y1d[Ny//2-1]
x1d=x1d-xshift
y1d=y1d-yshift

Nxtopad=round(Nx*padportionx)
Nytopad=round(Ny*padportiony)
dx=x1d[1]-x1d[0]
dy=y1d[1]-y1d[0]

if (Nx % 2) == 0:
    Nxpad=Nx+2*Nxtopad
    xminpad=x1d[0]-dx*(Nxtopad)
    xmaxpad=x1d[-1]+dx*(Nxtopad)
    x1dpad=np.linspace(xminpad,xmaxpad,num=Nxpad)
else:
    Nxpad=Nx+2*Nxtopad+1
    xminpad=x1d[0]-dx*(Nxtopad)
    xmaxpad=x1d[-1]+dx*(Nxtopad+1) #Add an extra grid point to make x1d have even number of grid points after padding
    x1dpad=np.linspace(xminpad,xmaxpad,num=Nxpad)

if (Ny % 2) == 0:
    Nypad=Ny+2*Nytopad
    yminpad=y1d[0]-dy*(Nytopad)
    ymaxpad=y1d[-1]+dy*(Nytopad)
    y1dpad=np.linspace(yminpad,ymaxpad,num=Nypad)
else:
    Nypad=Ny+2*Nytopad+1
    yminpad=y1d[0]-dy*(Nytopad)
    ymaxpad=y1d[-1]+dy*(Nytopad+1)
    y1dpad=np.linspace(yminpad,ymaxpad,num=Nypad)

x1dpad=x1dpad-x1dpad[Nxpad//2-1]
y1dpad=y1dpad-y1dpad[Nypad//2-1] 

#recast everything, and the rest of the codes are the same as the version without padding. 
Nxo=Nx
Nyo=Ny
x1do=x1d
y1do=y1d
x1d=x1dpad
y1d=y1dpad
Nx=len(x1d)
Ny=len(y1d)

dx=x1d[2]-x1d[1]
dy=y1d[2]-y1d[1]
# #lx and Ly that are equal to 2*pi/(dq1) and 2*pi/(dq2) respectively
# Lx=max(x1d)-min(x1d)+dx #Lx and Ly have to be exact for the random phase approximation to work. Otherwise the random variables will not have zero expectation
# Ly=max(y1d)-min(y1d)+dy

#20240814:
# By experiments, Lx and Ly defined in this way would ensure that the reconstructed MKE has the right magnitude
Lx=max(x1d)-min(x1d)
Ly=max(y1d)-min(y1d)

#Wavenumber for the currents (or Hs)
q1_1d=hwffts.k_of_x(x1d)
q2_1d=hwffts.k_of_x(y1d)
dq1=q1_1d[1]-q1_1d[0]
dq2=q2_1d[1]-q2_1d[0]
ddq=dq2/10**10

q1_2d,q2_2d=np.meshgrid(q1_1d,q2_1d)
q1_2d=np.transpose(q1_2d) #Make indexing intuitive, i.e. F[i,j] would mean we want q1 at the ith and q2 at the jth.
q2_2d=np.transpose(q2_2d)
#polar coordinate of q
q=np.sqrt(q1_2d**2+q2_2d**2)#wavenumber magnitude of q in 2d
varphi = np.arctan2(q2_2d, q1_2d) #polar angle of q in 2d; in paper it is varphi

#Spectral cutoff scales
qcut=2*pi/6000#np.max(q1_1d)*0.8
iq1cut=find_i_nearest(q1_1d,qcut)
iq2cut=find_i_nearest(q2_1d,qcut)

Ccut=np.exp(-(q-qcut)**2/np.min([dq1,dq2])**2/100)
Ccut[q<qcut]=1
#Added Sep.3 2024: Make the spectral content zero at small wavenumbers, so that the spectra are not dominated by small-scale contents which
#are more affected by anisotropic effects of the rectangular grid.
#If Nx and Ny are increased, the error induced by rectangular grids should diminish. 
lambdamax=200*1000 #Largest spatial scale
qmin=2*pi/lambdamax
Ccut[q<qmin]=0

#spatial grids for the gradient computations
x2do,y2do=np.meshgrid(x1do,y1do)
x2do=np.transpose(x2do) #Make indexing intuitive, i.e. F[i,j] would mean we want q1 at the ith and q2 at the jth.
y2do=np.transpose(y2do)

# #For truncations when computing statistics; this should be on unpadded domain
# ixs=0#int(Nxo//10);
# ixe=Nxo-1;
# iys=0;
# iye=Nyo-1;


Ndir = 48*2 #number of grid points in theta
Nf = 32 #number of grid points in frequency 
lowf = 0.04118 #Lowest frequency resolved (in Hz)
theta = np.linspace(-pi, pi, Ndir) #vector for theta
ffactor = 1.1 #controling the grids in frequency
highf=lowf*(ffactor**32)
freq=np.linspace(lowf,highf,Nf)
k=(2*pi*freq)**2/g #Wavenumber magnitude of k in 1D, in meters. Note 2 pi is needed as f is not angular frequency
#grid in k, theta
k2D,theta2D=np.meshgrid(k,theta)
k2D=np.transpose(k2D) #Making indexing intuitive, i.e. E2D[i,j] would refer to E2D at k[i] and theta[j]
theta2D=np.transpose(theta2D)

#Also get how many seeds are there in each file. This is also assuming the number of seeds are the same in all files.
nseeddrawn=1

ncconfigcount=len(waveTall)*len(KEslopeall)*len(divfracall)*len(MKEall)*len(s_all)
#It's ok that some of the runs have random seeds less than neseed. We just want to creat arrays with sized big enough to hold the outcomes.
cgstar_i=np.zeros(ncconfigcount*nseeddrawn)*np.nan
vargradhs_i=np.zeros(ncconfigcount*nseeddrawn)*np.nan
KEslope_i=np.zeros(ncconfigcount*nseeddrawn)*np.nan
divfrac_i=np.zeros(ncconfigcount*nseeddrawn)*np.nan
waveT_i=np.zeros(ncconfigcount*nseeddrawn)*np.nan
MKEall_i=np.zeros(ncconfigcount*nseeddrawn)*np.nan
varhs_i=np.zeros(ncconfigcount*nseeddrawn)*np.nan
s_i=np.zeros(ncconfigcount*nseeddrawn)*np.nan
lphi_i=np.zeros(ncconfigcount*nseeddrawn)*np.nan
lpsi_i=np.zeros(ncconfigcount*nseeddrawn)*np.nan
#Debugging: check if MKE is reconstructed correctly
MKE_r=np.zeros(ncconfigcount*nseeddrawn)*np.nan
divfrac_r=np.zeros(ncconfigcount*nseeddrawn)*np.nan
#Debugging: record Uphi and Upsi to check the more straightforward identity
varUphi_i=np.zeros(ncconfigcount*nseeddrawn)*np.nan
varUpsi_i=np.zeros(ncconfigcount*nseeddrawn)*np.nan
varUphifromuv_i=np.zeros(ncconfigcount*nseeddrawn)*np.nan
varUpsifromuv_i=np.zeros(ncconfigcount*nseeddrawn)*np.nan
varvort_i=np.zeros(ncconfigcount*nseeddrawn)*np.nan
vardiv_i=np.zeros(ncconfigcount*nseeddrawn)*np.nan
vargradu_i=np.zeros(ncconfigcount*nseeddrawn)*np.nan
#The unperturbed wave action: the f(k) component
def gaussian_spec(freq, fp, sip):
    frrel = (freq - fp)/sip
    return np.exp(-1.25*(frrel**2))

sip = 0.01 #variance of frequency
lowf = 0.04118 #Lowest frequency resolved (in Hz)
ffactor = 1.1 #controling the grids in frequency
thetacenter=0

ivec=0 #element number of the arrays to record the ratio etc. 

irds=0
for waveT, s in itertools.product(waveTall, s_all):
    fp=1/waveT
    E2D=gaussian_spec(np.sqrt(g*k2D)/(2*pi), fp, sip)*np.cos((theta2D-thetacenter)/2)**(2*s)/2*np.sqrt(g)*k2D**(-1/2)/k2D
    #Normalize so that the mean Hs is Hs0
    Hs0_r=4*np.sqrt(integrate.trapezoid(integrate.trapezoid(E2D*k2D,x=k,axis=0),x=theta)/g)
    E2D=E2D*Hs0**2/Hs0_r**2
    #Wave action
    A2D=E2D/np.sqrt(g*k2D)
       
    #Get L using the general formula that doesn't require separability
    #mathcal{P}(theta), equation (3.8)  
    Pcal=integrate.trapezoid(A2D*k2D**2,x=k,axis=0)
    #pn
    if s>0:
        ncutoff=s #p_n=0 at n>s for LHCS 
    else:
        ncutoff=1
    #Decide the n to cut off the computation of pn
    if 'ncutoff' not in globals():
        if Ndir>80:
            ncutoff=40 #40 corresponds to directional spreading at s=40, already extremely swell-like for oce an 
        else:
            ncutoff=Ndir//2-1 #Nyquist frequency in case Ndir is too small to accurately compute s up to 40.
        print('Parameter ncutoff is not given in the code by user. Setting ncutoff to be %i' % ncutoff)
    if ncutoff > Ndir//2-1:
        raise ValueError("Ndir must be at least twice as big as ncutoff")
    
    pn=np.zeros(2*ncutoff+1,dtype=complex)
    nvec=np.arange(-ncutoff,ncutoff+1,1)
    for i in np.arange(len(nvec)):
        ni=nvec[i]
        pn[i]=1/(2*pi)*integrate.trapezoid(Pcal*np.exp(-1j*ni*theta),x=theta)        
    #Get L using equation (3.11)
    #The term along P
    P1=np.real(2*pi*pn[nvec==1])
    P2=-np.imag(2*pi*pn[nvec==1])
    LtP1=-32/(g*Hs0**2)*P1
    LtP2=-32/(g*Hs0**2)*P2
    #The term along eqperp
    Lt2=np.zeros([Nx,Ny],dtype=complex)
    for iq1 in np.arange(Nx):
        for iq2 in np.arange(Ny):
            varphii=varphi[iq1,iq2]
            Lt2[iq1,iq2]=16/g/(Hs0**2)*np.sum(nvec*(-1j)**np.abs(nvec)*2*pi*pn*(np.exp(1j*nvec*varphii)))
    Lt21=-Lt2*np.sin(varphi)
    Lt22=Lt2*np.cos(varphi)        
    
    #Sum the two terms up
    Lhat1=(LtP1+Lt21)#$\hat{\bm{L}}(\varphi)$ along $q_1$
    Lhat2=(LtP2+Lt22) #$\hat{\bm{L}}(\varphi)$ along $q_2$
    
   
      
    #Also get peak group velocity, which is a term in the ratio.
    kstar=(fp*2*pi)**2/g #peak wavenumber
    cgstar=np.sqrt(g*kstar)/kstar/2 #peak group velo        
    
    Lhatqperp=Lhat1*(-np.sin(varphi))+Lhat2*np.cos(varphi)
    Lhatqpar=Lhat1*np.cos(varphi)+Lhat2*np.sin(varphi)
    
    #Aug.20/2024: Lt evaluated on the grid of varphi, which gives better accuracy in the angular integration that leads to lphi, lpsi
    Lt2varphi=np.zeros(Nvarphiint,dtype=complex)
    for ivarphi, varphii in enumerate(varphiint):
        Lt2varphi[ivarphi]=16/g/(Hs0**2)*np.sum(nvec*(-1j)**np.abs(nvec)*2*pi*pn*(np.exp(1j*nvec*varphii)))
    
    Lt21varphi=-Lt2varphi*np.sin(varphiint)
    Lt22varphi=Lt2varphi*np.cos(varphiint)        

    #Sum the two terms up
    Lhat1varphi=(LtP1+Lt21varphi)#$\hat{\bm{L}}(\varphi)$ along $q_1$
    Lhat2varphi=(LtP2+Lt22varphi) #$\hat{\bm{L}}(\varphi)$ along $q_2$

    
    #Alternatively, we decompost L into q and qperp directions
    Lhatqparvarphi=Lhat1varphi*np.cos(varphiint)+Lhat2varphi*np.sin(varphiint)
    Lhatqperpvarphi=Lhat1varphi*(-np.sin(varphiint))+Lhat2varphi*np.cos(varphiint)

    lphi=integrate.trapezoid((np.abs(Lhatqparvarphi))**2, x=varphiint)/(2*pi)
    lpsi=integrate.trapezoid((np.abs(Lhatqperpvarphi))**2, x=varphiint)/(2*pi)

    # # print('s:')
    # # print(s)
    # print('lphi:')
    # print(lphi)
    # print('lpsi:')
    # print(lpsi)
    for KEslope, divfrac, MKE in itertools.product(KEslopeall, divfracall, MKEall):
        SKErot=KEslope
        SKEdiv=KEslope
        #Overall mean KE (including both rot and div)       
        #power laws in psi and phi
        Cpsit = (q+ddq)**(-1-2+SKErot)*Ccut*(1-divfrac) #in the power, -1 comes from Abel inverse, and -2 comes from conversion to KE
        Cphit = (q+ddq)**(-1-2+SKEdiv)*Ccut*divfrac
        Cpsit[-1,:]=0 #Set the last rows/columns to be zero (on the q grid, they don't have -q conterparts)
        Cpsit[:,-1]=0
        Cphit[-1,:]=0
        Cphit[:,-1]=0

        # #Make them decay away from the crossover scales
        # qdecayphi=qcrossover*3
        # Ccut_Cphit=np.exp(-(q-qdecayphi)**2/dq**2/100) #Dividing the argument by a larger number results in a gentler decay
        # Ccut_Cphit[q>qdecayphi]=1
        # Cphit = Cphit * Ccut_Cphit
        # qdecaypsi=qcrossover
        # Ccut_Cpsit=np.exp(-(q-qdecaypsi)**2/dq**2/100)
        # Ccut_Cpsit[q<qdecaypsi]=1
        # Cpsit=Cpsit*Ccut_Cpsit
        
        #Normalize Cpsit and Cphit according to mean KE computed numerically 
        Kpsit = Cpsit*q**2/2
        Kphit = Cphit*q**2/2
        # #mean KE (averaged over space) computed from the current values
        # mKE_div = np.sum(Kphit)*dq*dq/(2*pi)**2/(Lx*Ly)
        # mKE_rot = np.sum(Kpsit)*dq*dq/(2*pi)**2/(Lx*Ly)
        # Mdiv = MKE*rr/mKE_div
        # Mrot = MKE*(1-rr)/mKE_rot
        # #Normalize
        # Kpsit = Mrot * Kpsit
        # Kphit = Mdiv * Kphit
        # Cpsit = Mrot * Cpsit
        # Cphit = Mdiv * Cphit
        KEtot=Kpsit+Kphit
        #mKE_r=np.sum(KEtot)*dq*dq/(2*pi)**2/(Lx*Ly) #I seem to get the normalization wrong here
        mKE_r=np.trapz(np.trapz(KEtot, q2_1d, axis=1), q1_1d, axis=0)/(2*pi)**2   #np.sum(KEtot)*dq1*dq2/(2*pi)**2
        M = MKE/mKE_r
        
        Kpsit = M * Kpsit
        Kphit = M * Kphit
        Cpsit=M*Cpsit
        Cphit=M*Cphit
        KEtot=M*KEtot
  
        Chst_reconstr=0*q1_2d
        KE_reconstr=0*q1_2d
        Ut_reconstr=0*q1_2d
        Vt_reconstr=0*q1_2d
    
        for iseed in np.arange(nseeddrawn):
            rds=np.random.RandomState(random.randint(0, 1000))
            #rds = np.random.RandomState(irds)    
            irds=irds+1 #08.14.2024: making the random states different in each configuration
            
            #Draw a sample of psit and phit, and comput ut, vt.
            #Use the usual random phase sampling. Similar to Bia, but we force reality by enforcing the hermitianilty when we draw the Fourier
            #samples, not by taking the real part of the ifft. This way, we can force the distributions of the phases to be truly uniformly 
            #distributed.
       
            #Make phit and psit have different random phases. 
            psit=0*q1_2d+1j
            #psit_part=np.sqrt(Lx*Ly*Cpsit[Nx//2-1:,:])*np.exp(1j*2*pi*(2*np.random.uniform(size=(Nx//2+1,Ny))-1))
            psit[0:Nx//2-1,0:Ny//2-1]=np.sqrt(Lx*Ly*Cpsit[0:Nx//2-1,0:Ny//2-1])*np.exp(1j*2*pi*rds.rand(Nx//2-1,Ny//2-1))
            psit[Nx//2:Nx-1,0:Ny//2-1]=np.sqrt(Lx*Ly*Cpsit[Nx//2:Nx-1,0:Ny//2-1])*np.exp(1j*2*pi*rds.rand(Nx//2-1,Ny//2-1))
            #C.C.
            psit[0:Nx//2-1,Ny//2:Ny-1]=np.conjugate(np.fliplr(np.flipud(psit[Nx//2:Nx-1,0:Ny//2-1])))
            psit[Nx//2:Nx-1,Ny//2:Ny-1]=np.conjugate(np.fliplr(np.flipud(psit[0:Nx//2-1,0:Ny//2-1])))
            #q1=0, q2=0, and last rows/columns
            #Outside the origin point
            psit[Nx//2-1,0:Ny//2-1]=np.sqrt(Lx*Ly*Cpsit[Nx//2-1,0:Ny//2-1])*np.exp(1j*2*pi*rds.rand(Ny//2-1))
            psit[Nx//2-1,Ny//2:Ny-1]=np.conjugate(np.flipud(psit[Nx//2-1,0:Ny//2-1]))
            psit[0:Nx//2-1,Ny//2-1]=np.sqrt(Lx*Ly*Cpsit[0:Nx//2-1,Ny//2-1])*np.exp(1j*2*pi*rds.rand(Nx//2-1))
            psit[Nx//2:Nx-1,Ny//2-1]=np.conjugate(np.flipud(psit[0:Nx//2-1,Ny//2-1]))
            #Origin point; note that it has to be real, for the reality condition to hold. We choose to make it deterministic; it will not affect velocity samples, as they are timed by q1 or q2 at these origin points.
            psit[Nx//2-1,Ny//2-1]=np.sqrt(Lx*Ly*Cpsit[Nx//2-1,Ny//2-1])
            #Set the last rows/columns to be zero (on the q grid, they don't have -q conterparts)
            psit[Nx-1,:]=0
            psit[:,Ny-1]=0
            
            phit=0*q1_2d+1j
            #phit_part=np.sqrt(Lx*Ly*Cphit[Nx//2-1:,:])*np.exp(1j*2*pi*(2*np.random.uniform(size=(Nx//2+1,Ny))-1))
            phit[0:Nx//2-1,0:Ny//2-1]=np.sqrt(Lx*Ly*Cphit[0:Nx//2-1,0:Ny//2-1])*np.exp(1j*2*pi*rds.rand(Nx//2-1,Ny//2-1))
            phit[Nx//2:Nx-1,0:Ny//2-1]=np.sqrt(Lx*Ly*Cphit[Nx//2:Nx-1,0:Ny//2-1])*np.exp(1j*2*pi*rds.rand(Nx//2-1,Ny//2-1))
            #C.C.
            phit[0:Nx//2-1,Ny//2:Ny-1]=np.conjugate(np.fliplr(np.flipud(phit[Nx//2:Nx-1,0:Ny//2-1])))
            phit[Nx//2:Nx-1,Ny//2:Ny-1]=np.conjugate(np.fliplr(np.flipud(phit[0:Nx//2-1,0:Ny//2-1])))
            #q1=0, q2=0, and last rows/columns
            #Outside the origin point
            phit[Nx//2-1,0:Ny//2-1]=np.sqrt(Lx*Ly*Cphit[Nx//2-1,0:Ny//2-1])*np.exp(1j*2*pi*rds.rand(Ny//2-1))
            phit[Nx//2-1,Ny//2:Ny-1]=np.conjugate(np.flipud(phit[Nx//2-1,0:Ny//2-1]))
            phit[0:Nx//2-1,Ny//2-1]=np.sqrt(Lx*Ly*Cphit[0:Nx//2-1,Ny//2-1])*np.exp(1j*2*pi*rds.rand(Nx//2-1))
            phit[Nx//2:Nx-1,Ny//2-1]=np.conjugate(np.flipud(phit[0:Nx//2-1,Ny//2-1]))
            #Origin point; note that it has to be real, for the reality condition to hold. We choose to make it deterministic; it will not affect velocity samples, as they are timed by q1 or q2 at these origin points.
            phit[Nx//2-1,Ny//2-1]=np.sqrt(Lx*Ly*Cphit[Nx//2-1,Ny//2-1])
            #Set the last rows/columns to be zero (on the q grid, they don't have -q conterparts)
            phit[Nx-1,:]=0
            phit[:,Ny-1]=0
            
            # #Make phit and psit have the same random phases; this would fix the patterns of speed, but lead to gaps in the Chs spectra when one of the div or rot components dominate.
            # Cpt=Cpsit+Cphit 
            # pt=0*q1_2d+1j
            # #pt_part=np.sqrt(Lx*Ly*Cpt[Nx//2-1:,:])*np.exp(1j*2*pi*(2*np.random.uniform(size=(Nx//2+1,Ny))-1))
            # pt[0:Nx//2-1,0:Ny//2-1]=np.sqrt(Lx*Ly*Cpt[0:Nx//2-1,0:Ny//2-1])*np.exp(1j*2*pi*rds.rand(Nx//2-1,Ny//2-1))
            # pt[Nx//2:Nx-1,0:Ny//2-1]=np.sqrt(Lx*Ly*Cpt[Nx//2:Nx-1,0:Ny//2-1])*np.exp(1j*2*pi*rds.rand(Nx//2-1,Ny//2-1))
            # #C.C.
            # pt[0:Nx//2-1,Ny//2:Ny-1]=np.conjugate(np.fliplr(np.flipud(pt[Nx//2:Nx-1,0:Ny//2-1])))
            # pt[Nx//2:Nx-1,Ny//2:Ny-1]=np.conjugate(np.fliplr(np.flipud(pt[0:Nx//2-1,0:Ny//2-1])))
            # #q1=0, q2=0, and last rows/columns
            # #Outside the origin point
            # pt[Nx//2-1,0:Ny//2-1]=np.sqrt(Lx*Ly*Cpt[Nx//2-1,0:Ny//2-1])*np.exp(1j*2*pi*rds.rand(Ny//2-1))
            # pt[Nx//2-1,Ny//2:Ny-1]=np.conjugate(np.flipud(pt[Nx//2-1,0:Ny//2-1]))
            # pt[0:Nx//2-1,Ny//2-1]=np.sqrt(Lx*Ly*Cpt[0:Nx//2-1,Ny//2-1])*np.exp(1j*2*pi*rds.rand(Nx//2-1))
            # pt[Nx//2:Nx-1,Ny//2-1]=np.conjugate(np.flipud(pt[0:Nx//2-1,Ny//2-1]))
            # #Origin point; note that it has to be real, for the reality condition to hold. We choose to make it deterministic; it will not affect velocity samples, as they are timed by q1 or q2 at these origin points.
            # pt[Nx//2-1,Ny//2-1]=np.sqrt(Lx*Ly*Cpt[Nx//2-1,Ny//2-1])
            # #Set the last rows/columns to be zero (on the q grid, they don't have -q conterparts)
            # pt[Nx-1,:]=0
            # pt[:,Ny-1]=0
            
            # phit=pt*np.sqrt(divfrac)
            # psit=pt*np.sqrt(1-divfrac)
        
            ut=1j*(q1_2d*phit-q2_2d*psit)
            vt=1j*(q2_2d*phit+q1_2d*psit)  

            hst=Lhat1*ut+Lhat2*vt #hs means the change of SWH relative to Hs0. 
            
            #Note that taking the reality component here should only be changing the numerical format; the imaginary parts should have already been zero.
            hs=np.real(hwffts.hwifft2(q1_1d, q2_1d, hst))
            u=np.real(hwffts.hwifft2(q1_1d, q2_1d, ut))
            v=np.real(hwffts.hwifft2(q1_1d, q2_1d, vt))
            # print('mean (real[hs])^2 with ifft:')
            # print(np.mean(hs**2))
            # print('mean (imag[hs])^2 with ifft:')
            # print(np.mean((np.imag(hwffts.hwifft2(q1_1d, q2_1d, hst)))**2))
            
            #Truncate back into unpadded domains
            hs=hs[Nxtopad:Nxtopad+Nxo,Nytopad:Nytopad+Nyo]
            u=u[Nxtopad:Nxtopad+Nxo,Nytopad:Nytopad+Nyo]
            v=v[Nxtopad:Nxtopad+Nxo,Nytopad:Nytopad+Nyo]
            
            #IMPORTANT: make the mean of hs to be zero over the unpadded domain.  
            hs=hs-np.mean(hs)
            
            #changed Sep.3, 2024: Compute grad hs spectrally
            d1hst=1j*hst*q1_2d
            d2hst=1j*hst*q2_2d
            d1hs=np.real(hwffts.hwifft2(q1_1d, q2_1d, d1hst))
            d2hs=np.real(hwffts.hwifft2(q1_1d, q2_1d, d2hst))
            gradhsnorm=np.sqrt(d1hs**2+d2hs**2)
            #Compute the finite differences: this leads to siginificant error at small spatial scales (shows up at shallow spectral slopes)
            #gradhsnorm=np.sqrt(hwdiffx(hs,x2do)**2+hwdiffy(hs,y2do)**2)
            
            ux=np.real(hwffts.hwifft2(q1_1d, q2_1d, 1j*ut*q1_2d))
            uy=np.real(hwffts.hwifft2(q1_1d, q2_1d, 1j*ut*q2_2d))
            vx=np.real(hwffts.hwifft2(q1_1d, q2_1d, 1j*vt*q1_2d))
            vy=np.real(hwffts.hwifft2(q1_1d, q2_1d, 1j*vt*q2_2d))
            vargradu_i[ivec]=np.mean(ux**2+vx**2+uy**2+vy**2)
            
            #Vort and Div
            vortt=1j*(vt*q1_2d-ut*q2_2d)
            divt=1j*(ut*q1_2d+vt*q2_2d)
            vort=(hwffts.hwifft2(q1_1d, q2_1d, vortt))
            div=(hwffts.hwifft2(q1_1d, q2_1d, divt))
            varvort_i[ivec]=np.mean(np.abs(vort)**2)
            vardiv_i[ivec]=np.mean(np.abs(div)**2)
            
            
            cgstar_i[ivec]=cgstar
            vargradhs_i[ivec]=np.mean(gradhsnorm**2)
            KEslope_i[ivec]=KEslope
            divfrac_i[ivec]=divfrac           
            waveT_i[ivec]=waveT           
            MKEall_i[ivec]=MKE
            s_i[ivec]=s
            
            varhs_i[ivec]=np.mean(np.abs(hs)**2)
            #rmshs_CZ_i[ivec]=np.sqrt(np.mean(hs[ixs:ixe,iys:iye]**2))
            lphi_i[ivec]=lphi
            lpsi_i[ivec]=lpsi
            
            
            
            #DebuggingL checking if MKE is reconstructed correctly
            MKE_r[ivec]=0.5*np.mean(u**2+v**2)
            uphit=1j*(q1_2d*phit)
            vphit=1j*(q2_2d*phit)  
            uphi=np.real(hwffts.hwifft2(q1_1d, q2_1d, uphit))
            vphi=np.real(hwffts.hwifft2(q1_1d, q2_1d, vphit))
            #upsit=1j*(-q2_2d*psit)
            #vpsit=1j*(q1_2d*psit)  
            divfrac_r[ivec]=np.mean(uphi**2+vphi**2)/np.mean(u**2+v**2)


            #Debugging: record Uphi and Upsi
            Uphit=1j*q*phit
            Upsit=1j*q*psit
            Uphi=(hwffts.hwifft2(q1_1d, q2_1d, Uphit))
            Upsi=(hwffts.hwifft2(q1_1d, q2_1d, Upsit))
            varUphi_i[ivec]=np.mean(np.abs(Uphi)**2)
            varUpsi_i[ivec]=np.mean(np.abs(Upsi)**2)
            varUphifromuv_i[ivec]=np.mean(uphi**2+vphi**2)
            upsit=-1j*(q2_2d*psit)
            vpsit=1j*(q1_2d*psit)  
            upsi=np.real(hwffts.hwifft2(q1_1d, q2_1d, upsit))
            vpsi=np.real(hwffts.hwifft2(q1_1d, q2_1d, vpsit))
            varUpsifromuv_i[ivec]=np.mean(upsi**2+vpsi**2)

            ivec=ivec+1
            Chst_reconstr=Chst_reconstr+np.abs(hst)**2/(Lx*Ly)
            KE_reconstr=KE_reconstr+(np.abs(ut)**2+np.abs(vt)**2)/(Lx*Ly)
            Ut_reconstr=Ut_reconstr+(np.abs(ut))/(Lx*Ly)
            Vt_reconstr=Vt_reconstr+(np.abs(vt))/(Lx*Ly)
        
        Chst_reconstr=Chst_reconstr/nseeddrawn
        KE_reconstr=KE_reconstr/nseeddrawn
        Ut_reconstr=Ut_reconstr/nseeddrawn
        Vt_reconstr=Vt_reconstr/nseeddrawn
        #Chst expected analytically
        Chst=np.abs(Lhatqpar)**2*q**2*Cphit+np.abs(Lhatqperp)**2*q**2*Cpsit
        
        #make small values of Chst and Chst_reconstr zero, so that they are not shown in plots
        Chstplot=Chst
        Chstplot[Chst<np.mean(Chst)*10**(-15)]=0        
        Chst_reconstrplot=Chst_reconstr
        Chst_reconstrplot[Chst_reconstr<np.mean(Chst_reconstr)*10**(-15)]=0
        
        print('mean of Chst from samples:')
        print(np.mean(Chst_reconstr))
        print('mean of Chst from power spectral relations:')
        print(np.mean(Chst))
        
        
        # fcut=10 #Setting how much of the axis I want to display. e.g. fcut=10 would should only about 1/10 of the axis in either direction
      
        
        # fig2d,ax2d=plt.subplots()
        # fig2d.set_dpi(256)
        # im1=ax2d.pcolor(q1_1d[(Nx//2-round(Nx//2/fcut)):(Nx//2+round(Nx//2/fcut))]*1000,q2_1d[(Ny//2-round(Ny//2/fcut)):(round(Ny//2/fcut)+Ny//2)]*1000,np.transpose((Chst_reconstrplot[(Nx//2-round(Nx//2/fcut)):(Nx//2+round(Nx//2/fcut)),(Ny//2-round(Ny//2/fcut)):(round(Ny//2/fcut)+Ny//2)])), norm=matplotlib
        #                     .colors.LogNorm(),cmap=cc.cm.bgy)
        # ax2d.set_aspect(1)
        # ax2d.set_title(r'$\hat{C}^{h_s}, from sample$ $[m^4]$, U2H')
        # ax2d.set_xlabel(r'$q_1$ [1/km]')
        # ax2d.set_ylabel(r'$q_2$ [1/km]')
        
        # fig2d.colorbar(im1, ax=ax2d,shrink=0.5)
        
        # fig2d.suptitle(r'2D spectra (zoomed in), $s=%i$' % s)
        
        # fig2d,ax2d=plt.subplots()
        # fig2d.set_dpi(256)
        # im1=ax2d.pcolor(q1_1d[(Nx//2-round(Nx//2/fcut)):(Nx//2+round(Nx//2/fcut))]*1000,q2_1d[(Ny//2-round(Ny//2/fcut)):(round(Ny//2/fcut)+Ny//2)]*1000,np.transpose((Chstplot[(Nx//2-round(Nx//2/fcut)):(Nx//2+round(Nx//2/fcut)),(Ny//2-round(Ny//2/fcut)):(round(Ny//2/fcut)+Ny//2)])), norm=matplotlib
        #                     .colors.LogNorm(),cmap=cc.cm.bgy)
        # ax2d.set_aspect(1)
        # ax2d.set_title(r'$\hat{C}^{h_s}, from analy $ $[m^4]$, U2H')
        # ax2d.set_xlabel(r'$q_1$ [1/km]')
        # ax2d.set_ylabel(r'$q_2$ [1/km]')
        
        # fig2d.colorbar(im1, ax=ax2d,shrink=0.5)
        
        # fig2d.suptitle(r'2D spectra (zoomed in), $s=%i$' % s)
        
        # fig2d,ax2d=plt.subplots()
        # fig2d.set_dpi(256)
        # im1=ax2d.pcolor(q1_1d[(Nx//2-round(Nx//2/fcut)):(Nx//2+round(Nx//2/fcut))]*1000,q2_1d[(Ny//2-round(Ny//2/fcut)):(round(Ny//2/fcut)+Ny//2)]*1000,np.transpose((KE_reconstr[(Nx//2-round(Nx//2/fcut)):(Nx//2+round(Nx//2/fcut)),(Ny//2-round(Ny//2/fcut)):(round(Ny//2/fcut)+Ny//2)])), norm=matplotlib
        #                    .colors.LogNorm(),cmap=cc.cm.bgy)
        # ax2d.set_aspect(1)
        # ax2d.set_title(r'$KE$ reconstructed')
        # ax2d.set_xlabel(r'$q_1$ [1/km]')
        # ax2d.set_ylabel(r'$q_2$ [1/km]')
        
        # fig2d.colorbar(im1, ax=ax2d,shrink=0.5)
        
        # fig2d.suptitle(r'2D spectra (zoomed in), $s=%i$' % s)
        
        # fig2d,ax2d=plt.subplots()
        # fig2d.set_dpi(256)
        # im1=ax2d.pcolor(q1_1d[(Nx//2-round(Nx//2/fcut)):(Nx//2+round(Nx//2/fcut))]*1000,q2_1d[(Ny//2-round(Ny//2/fcut)):(round(Ny//2/fcut)+Ny//2)]*1000,np.transpose((Ut_reconstr[(Nx//2-round(Nx//2/fcut)):(Nx//2+round(Nx//2/fcut)),(Ny//2-round(Ny//2/fcut)):(round(Ny//2/fcut)+Ny//2)])), norm=matplotlib
        #                     .colors.LogNorm(),cmap=cc.cm.bgy)
        # ax2d.set_aspect(1)
        # ax2d.set_title(r'$|\hat u|^2$')
        # ax2d.set_xlabel(r'$q_1$ [1/km]')
        # ax2d.set_ylabel(r'$q_2$ [1/km]')
        # fig2d.colorbar(im1, ax=ax2d,shrink=0.5) 
        # fig2d.suptitle(r'2D spectra (zoomed in), $s=%i$' % s)
        
        # fig2d,ax2d=plt.subplots()
        # fig2d.set_dpi(256)
        # im1=ax2d.pcolor(q1_1d[(Nx//2-round(Nx//2/fcut)):(Nx//2+round(Nx//2/fcut))]*1000,q2_1d[(Ny//2-round(Ny//2/fcut)):(round(Ny//2/fcut)+Ny//2)]*1000,np.transpose((Vt_reconstr[(Nx//2-round(Nx//2/fcut)):(Nx//2+round(Nx//2/fcut)),(Ny//2-round(Ny//2/fcut)):(round(Ny//2/fcut)+Ny//2)])), norm=matplotlib
        #                     .colors.LogNorm(),cmap=cc.cm.bgy)
        # ax2d.set_aspect(1)
        # ax2d.set_title(r'$|\hat v|^2$')
        # ax2d.set_xlabel(r'$q_1$ [1/km]')
        # ax2d.set_ylabel(r'$q_2$ [1/km]')
        # fig2d.colorbar(im1, ax=ax2d,shrink=0.5)
        # fig2d.suptitle(r'2D spectra (zoomed in), $s=%i$' % s)
                    

#Expected rms of hs from U2H statistical relations
hsvar_pred=2*MKEall_i*(divfrac_i*lphi_i+(1-divfrac_i)*lpsi_i)
#hsvar_pred=lphi_i*varUphi_i+lpsi_i*varUpsi_i

fighsratio_sub,axhsratio_subhs=plt.subplots(figsize=[5,5])
fighsratio_sub.set_dpi(256)
#To get colors fram color maps
cmapKEslope = matplotlib.cm.get_cmap('Paired')
cnormKEslope = matplotlib.colors.Normalize(vmin=np.min(KEslopeall), vmax=np.max(KEslopeall))

#Plot scatter plot

imag1=axhsratio_subhs.scatter(divfrac_i,varhs_i*10**4, marker=MarkerStyle("o"), color=cmapKEslope(cnormKEslope(KEslope_i)))


# #Distinguish different slopes by markers (dots for one and dashes for the other)
# for slopeii in slopeall:
#     isubset= slope_i == slopeii
#     if slopeii == slopeall[0]:
#         imag1=axhsratio_subhs.scatter(divfrac_i[isubset],varhs_i[isubset], marker=MarkerStyle("o", fillstyle="right"), color=cmapMKE(cnormMKE(MKEall_i[isubset])))
#         imag2=axhsratio_subhs.scatter(divfrac_i[isubset],varhs_i[isubset], marker=MarkerStyle("o", fillstyle="left"), color=cmaps(cnorms(s_i[isubset])))
#     else:
#         axhsratio_subhs.scatter(divfrac_i[isubset],varhs_i[isubset], marker=MarkerStyle(1), color=cmapMKE(cnormMKE(MKEall_i[isubset])))
#         axhsratio_subhs.scatter(divfrac_i[isubset],varhs_i[isubset], marker=MarkerStyle(0), color=cmaps(cnorms(s_i[isubset])))
#Expected power law
for s, waveT, KEslope in itertools.product(s_all, waveTall, KEslopeall):
    isubset = (s_i == s) & (waveT_i == waveT) & (KEslope_i == KEslope)
    axhsratio_subhs.plot(divfrac_i[isubset],hsvar_pred[isubset]*10**4,color="0.4",alpha=0.5)

#Add legends
gammalegends=[]
gammalegends.append(mpatches.Patch(color=(0,0,0,0), label=r'$\gamma$'))
for KEslopei in KEslopeall:
    gammai=KEslopei #Slope for the 2D spectra
    gammalegends.append(mpatches.Patch(color=cmapKEslope(cnormKEslope(gammai)) , label=f'    {gammai:.0f}'))


legend_patches =  gammalegends
# Creating a two-column legend 
axhsratio_subhs.legend(handles=legend_patches, loc='upper right', fancybox=True, fontsize='small', ncol=1, framealpha=1,handlelength=0.7,handletextpad=-0.7) #handlelength sets the length of the color patch, and handletextpad sets the gap between text and patch
axhsratio_subhs.set_xlabel(r'$\alpha$')
axhsratio_subhs.set_ylabel(r'$\left<h_s^2\right>/\bar {H}_s^2 \times 10^4$')

#-------------------------
axhsratio_subhs.annotate(r'$s=1$',
            xy=(0, 2.2), xycoords='data',
            horizontalalignment='left', verticalalignment='top',
            fontsize=12)
axhsratio_subhs.annotate(r'$s=5$',
            xy=(0.1, 6.1), xycoords='data',
            horizontalalignment='left', verticalalignment='top',
            fontsize=12)
axhsratio_subhs.annotate(r'$s=10$',
            xy=(0.2, 10.1), xycoords='data',
            horizontalalignment='left', verticalalignment='top',
            fontsize=12)
