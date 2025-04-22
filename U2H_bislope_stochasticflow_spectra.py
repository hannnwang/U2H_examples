#Compute U2H map from velocity samples where Cpsit and Cphit follow different spectral slopesimport numpy as np
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
from hwffts import hwfft2,hwifft2,k_of_x
import scipy.signal
from scipy import special
import matplotlib
from scipy import integrate
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

#colors=['#4477AA','#66CCEE','#228833','#CCBB44','#EE6677','#AA3377','#BBBBBB']
colors=['#004488','#DDAA33','#BB5566']
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
 
def halflist(a_list): #Take the latter half of an 1D array (for plotting)
    ihalf = len(a_list)//2
    return a_list[ihalf:]

def alias1D(Cut,q1_1d):
    dq=q1_1d[1]-q1_1d[0]
    Cut1D=np.sum(Cut,axis=0)*dq-1/2*dq*(Cut[0,:]+Cut[-1,:])
    return Cut1D

def find_i_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
pi=np.pi
g=9.806 #Gravity acceleration. 
Hs0=1 #Mean assumed for the CZ transform; have tested that changing Hs0 does not affect the outcome in terms of hs/Hs0

waveT=10.3
s=10 #Set s=1 or s=10 to get the figures in the paper. You can change s to be other integers too. 
thetacenter=0

sip = 0.01 #variance of frequency
lowf = 0.04118 #Lowest frequency resolved (in Hz)




# #parameter controling how much divergent energy is taking up the total KE
# #so that KE= rr KE^{phi} + (1-rr) KE^{psi}
# rr=0.5

#Slopes for the rotational and divergent spectra
#They are typed in terms of the slope of the 1-D along-track KE spectra integrated over q1 or q2, assuming isotropic psi and phi spectra. 
SKErot=-3
SKEdiv=-1 #5/3
#q at which rot and div KE are equipartationed
qcrossover = 2*pi/(100*1000) 

# Another set of values more realistic for oceanic flows. qmax probably needs to be increased to show any trend in hs.
# SKErot=-3
# SKEdiv=-2
# #q at which rot and div KE are equipartationed
# qcrossover = 2*pi/(10*1000) 


#Overall mean KE (including both rot and div)
MKE=0.01 #unit:(m/s)^2 

x1d=np.linspace(-310000, 312500, num=250, endpoint=True)
y1d=np.linspace(-310000, 312500, num=250, endpoint=True)
Nx=len(x1d)
Ny=len(y1d)

#Shift x1d and y1d so that they center at 0, for convenience of hwfft. 
xshift=x1d[Nx//2-1]
yshift=y1d[Ny//2-1]
x1d=x1d-xshift
y1d=y1d-yshift

#Wavenumber for the currents (or Hs)
q1_1d=k_of_x(x1d)
q2_1d=k_of_x(y1d)

dq=q2_1d[1]-q2_1d[0]
ddq=dq/10**8

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

dq1=q1_1d[1]-q1_1d[0]
dq2=q2_1d[1]-q2_1d[0]
Ccut=np.exp(-(q-qcut)**2/np.min([dq1,dq2])**2/100)
Ccut[q<qcut]=1
#Added Sep.3 2024: Make the spectral content zero at small wavenumbers, so that the spectra are not dominated by small-scale contents which
#are more affected by anisotropic effects of the rectangular grid.
#If Nx and Ny are increased, the error induced by rectangular grids should diminish. 
lambdamax=200*1000 #Largest spatial scale
qmin=2*pi/lambdamax
Ccut[q<qmin]=0

# figccut,axccut=plt.subplots()
# axccut.loglog(q1_1d,Ccut[:,Ny//2])
# axccut.set_title('Roll off at high wavenumbers')

dx=x1d[2]-x1d[1]
dy=y1d[2]-y1d[1]
Lx=max(x1d)-min(x1d)+dx #Lx and Ly have to be exact for the random phase approximation to work. Otherwise the random variables will not have zero expectation
Ly=max(y1d)-min(y1d)+dy


#power laws in psi and phi. make it so that their maxima are the same
Cpsit = (q+ddq)**(-1-2+SKErot)*Ccut #in the power, -1 comes from Abel inverse, and -2 comes from conversion to KE
Cphit = (q+ddq)**(-1-2+SKEdiv)*Ccut
Cpsit[-1,:]=0 #Set the last rows/columns to be zero (on the q grid, they don't have -q conterparts)
Cpsit[:,-1]=0
#Make them decay away from the crossover scales
# qdecayphi=qcrossover*3
# Ccut_Cphit=np.exp(-(q-qdecayphi)**2/dq**2/100) #Dividing the argument by a larger number results in a gentler decay
# Ccut_Cphit[q>qdecayphi]=1
# Cphit = Cphit * Ccut_Cphit
# qdecaypsi=qcrossover
# Ccut_Cpsit=np.exp(-(q-qdecaypsi)**2/dq**2/100)
# Ccut_Cpsit[q<qdecaypsi]=1
# Cpsit=Cpsit*Ccut_Cpsit

#Make it so that they are equal at a certain crossover scale
iqcrossover = find_i_nearest(q1_1d,qcrossover)
Cphit = Cphit * Cpsit[iqcrossover,iqcrossover]/Cphit[iqcrossover,iqcrossover] 

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
mKE_r=np.sum(KEtot)*dq*dq/(2*pi)**2
M = MKE/mKE_r

Kpsit = M * Kpsit
Kphit = M * Kphit
Cpsit=M*Cpsit
Cphit=M*Cphit
KEtot=M*KEtot

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

#The unperturbed wave action: the f(k) component
def gaussian_spec(freq, fp, sip):
    frrel = (freq - fp)/sip
    return np.exp(-1.25*(frrel**2))

sip = 0.01 #variance of frequency
lowf = 0.04118 #Lowest frequency resolved (in Hz)
ffactor = 1.1 #controling the grids in frequency

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
        ncutoff=40 #40 corresponds to directional spreading at s=40, already extremely swell-like for ocean 
    else:
        ncutoff=Ndir//2-1 #Nyquist frequency in case Ndir is too small to accurately compute s up to 40.
    print('Parameter ncutoff is not given in the code by user. Setting ncutoff to be %i' % ncutoff)
if ncutoff > Ndir//2-1:
    raise ValueError("Ndir must be at least twice as big as ncutoff")

pn=np.zeros(2*ncutoff+1)+1j
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

#iseedvec=[20]#indices for Random seeds
iseedvec=np.arange(10)
nseed=len(iseedvec)
Chst=0*q1_2d
KEtot_r=0*q1_2d
KEpsit_r=0*q1_2d
KEphit_r=0*q1_2d
for iseed in iseedvec:
        rds = np.random.RandomState(iseed)    
        #Draw a sample of psit and phit, and comput ut, vt.
        #Note that we just need one sample in order to determine the spectrum of hs
        #Use the usual random phase sampling. Similar to Bia, but we force reality by enforcing the hermitianilty when we draw the Fourier
        #samples, not by taking the real part of the ifft. This way, we can force the distributions of the phases to be truly uniformly 
        #distributed.
        
        # #Make phit and psit have the same random phases; this would fix the patterns of speed, but lead to gaps in the Chs spectra when one of the div or rot components dominate.
        # Cpt=Cpsit+Cphit #For random sampling
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
        # phit=pt*np.sqrt(rr)
        # psit=pt*np.sqrt(1-rr)
        
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
       
       
        ut=1j*(q1_2d*phit-q2_2d*psit)
        vt=1j*(q2_2d*phit+q1_2d*psit)
        
        upsit=1j*(-q2_2d*psit)
        vpsit=1j*(q1_2d*psit)
        uphit=1j*(q1_2d*phit)
        vphit=1j*(q2_2d*phit)

        #Check if ut, vt reconstructs the KE spectra
        KEtot_r = KEtot_r+1/2*(ut*np.conjugate(ut)+vt*np.conjugate(vt))/(Lx*Ly)
        KEpsit_r =KEpsit_r+1/2*(upsit*np.conjugate(upsit)+vpsit*np.conjugate(vpsit))/(Lx*Ly)
        KEphit_r =KEphit_r+1/2*(uphit*np.conjugate(uphit)+vphit*np.conjugate(vphit))/(Lx*Ly)

        hst=Lhat1*ut+Lhat2*vt #hs means the change of SWH relative to Hs0. 
        Chst=Chst+np.abs(hst)**2/(Lx*Ly)
        
Chst=Chst/nseed
KEtot_r=KEtot_r/nseed
KEpsit_r=KEpsit_r/nseed
KEphit_r=KEphit_r/nseed

#Note that taking the reality component here should only be changing the numerical format; the imaginary parts should have already been zero.
hs=np.real(hwifft2(q1_1d, q2_1d, hst))
u=np.real(hwifft2(q1_1d, q2_1d, ut))
v=np.real(hwifft2(q1_1d, q2_1d, vt))
# #Check that the imag part is indeed zero
# print('max |hs.real|:')
# print(np.max(np.abs(hs.real)))
# print('max |hs.imag|:')
# print(np.max(np.abs(hs.imag)))
# print('mean |hs.real|:')
# print(np.mean(np.abs(hs.real)))
# print('mean |hs.imag|:')
# print(np.mean(np.abs(hs.imag)))



#Plotting
#Indices for isotropic spectra plots
iq1min=find_i_nearest(q1_1d,qmin)
iq1max=iq1cut
iq1plot=np.arange(iq1min+1,iq1max)

#Plot the isotropic spectra along with hs
Cpsitiso=Cpsit[:,Ny//2-1]
Cphitiso=Cphit[:,Ny//2-1]
Nvarphiint=256 #Change this for different levels of accuracy
varphiint=np.linspace(0,2*pi,Nvarphiint)
Lt2varphi=np.zeros(Nvarphiint,dtype=complex)
for ivarphi, varphii in enumerate(varphiint):
    Lt2varphi[ivarphi]=16/g/(Hs0**2)*np.sum(nvec*(-1j)**np.abs(nvec)*2*pi*pn*(np.exp(1j*nvec*varphii)))

Lt21varphi=-Lt2varphi*np.sin(varphiint)
Lt22varphi=Lt2varphi*np.cos(varphiint)        

#Sum the two terms up
Lhat1varphi=(LtP1+Lt21varphi)#$\hat{\bm{L}}(\varphi)$ along $q_1$
Lhat2varphi=(LtP2+Lt22varphi) #$\hat{\bm{L}}(\varphi)$ along $q_2$

#Alternatively, we decompost L into q and qperp directions
Lhatqperpvarphi=Lhat1varphi*(-np.sin(varphiint))+Lhat2varphi*np.cos(varphiint)
Lhatqparvarphi=Lhat1varphi*np.cos(varphiint)+Lhat2varphi*np.sin(varphiint)
   
Lper2int=integrate.trapezoid((np.abs(Lhatqperpvarphi))**2, x=varphiint)/(2*pi)
Lpar2int=integrate.trapezoid((np.abs(Lhatqparvarphi))**2, x=varphiint)/(2*pi)

Chstiso=Lpar2int*q1_1d**2*Cphitiso+Lper2int*q1_1d**2*Cpsitiso
pltiso,ax1Diso=plt.subplots()
pltiso.set_dpi(256)
ax1Diso.loglog(q1_1d[iq1plot]*1000,q1_1d[iq1plot]*Chstiso[iq1plot],color=colors[0],label=r'$\hat{C}^{h_s}(q)$')
axKEiso = ax1Diso.twinx() #sharing the x axis
axKEiso.loglog(q1_1d[iq1plot]*1000,q1_1d[iq1plot]*KEtot[iq1plot,Ny//2-1],'-',color=colors[2],label=r'KE$(q)$') #Timed with the Jacobian q
axKEiso.loglog(q1_1d[iq1plot]*1000,q1_1d[iq1plot]*Kphit[iq1plot,Ny//2-1],':',color=colors[2],label=r'KE$_{\phi}(q)$')
axKEiso.loglog(q1_1d[iq1plot]*1000,q1_1d[iq1plot]*Kpsit[iq1plot,Ny//2-1],'-.',color=colors[2],label=r'KE$_{\psi}(q)$')
hsylim_0, hsylim_1 = ax1Diso.get_ylim()
yscale= np.max(q1_1d[iq1plot]*KEtot[iq1plot,Ny//2-1])/np.max(q1_1d[iq1plot]*Chstiso[iq1plot])
axKEiso.set_ylim([hsylim_0*yscale,hsylim_1*yscale]) #It's important that the axis limits of axKE scale with the axis for hs. Otherwise the slopes change. 
ax1Diso.legend(loc='lower left')
axKEiso.legend(loc='upper right')
pltiso.suptitle('Isotropic spectra, $s=%i$' % s)
ax1Diso.set_xlabel(r'$q$ [1/km]')
ax1Diso.set_ylabel(r'[m$^3$]',color=colors[0]) #Units after timing the Jacobian q
axKEiso.set_ylabel(r'[m$^3$/s$^2$]',color=colors[2]) #Units after timing the Jacobian q
ax1Diso.set_box_aspect(1)
ax1Diso.tick_params(axis='y', labelcolor=colors[0])
axKEiso.tick_params(axis='y', labelcolor=colors[2])
ax1Diso.grid(True, which="both", linewidth=0.5,c='#D3D3D3')



iq1alongtrackmax=find_i_nearest(q1_1d,q1_1d[iq1cut]/np.sqrt(2))
iq1plotalongtrack=np.arange(Nx//2,iq1alongtrackmax)
iq2plotalongtrack=np.arange(Ny//2,iq1alongtrackmax)

#Plot along/across wave spectra of hs and KE on the same panel
plt1D,ax1D=plt.subplots()
plt1D.set_dpi(256)
#Plot the along-track spectra along q1
ax1D.loglog(q1_1d[iq1plotalongtrack]*1000,np.sum(Chst[iq1plotalongtrack,:],axis=1)*dq/(2*pi),color=colors[0],label=r'$\hat{C}^{h_s}(q_1)$')
#Plot the along-track spectra along q2
ax1D.loglog(q2_1d[iq2plotalongtrack]*1000,np.sum(Chst[:,iq2plotalongtrack],axis=0)*dq/(2*pi),'--',color=colors[0],label=r'$\hat{C}^{h_s}(q_2)$')
#Mark the kinetic energy spectrum
#Use dual axis
axKE = ax1D.twinx() #sharing the x axis
axKE.loglog(q1_1d[iq1plotalongtrack]*1000,np.sum(KEtot_r[iq1plotalongtrack,:],axis=1)*dq/(2*pi),'-',color=colors[2],label=r'KE$(q_1)$')
#Helmholtz 
axKE.loglog(q1_1d[iq1plotalongtrack]*1000,np.sum(KEphit_r[iq1plotalongtrack,:],axis=1)*dq/(2*pi),':',color=colors[2],label=r'KE$_{\phi}(q_1)$')
axKE.loglog(q1_1d[iq1plotalongtrack]*1000,np.sum(KEpsit_r[iq1plotalongtrack,:],axis=1)*dq/(2*pi),'-.',color=colors[2],label=r'KE$_{\psi}(q_1)$')

hsylim_0, hsylim_1 = ax1D.get_ylim()
ax1D.set_ylim([hsylim_0,hsylim_1*5])
hsylim_0, hsylim_1 = ax1D.get_ylim()
yscale= np.max((np.sum(Chst[iq1plotalongtrack,:],axis=1),np.sum(Chst[:,iq2plotalongtrack],axis=0)))/np.max(np.sum(KEtot_r[iq1plotalongtrack,:],axis=1))*5
axKE.set_ylim([hsylim_0/yscale,hsylim_1/yscale]) #It's important that the axis limits of axKE scale with the axis for hs. Otherwise the slopes change. 
ax1D.legend(loc='lower left')
axKE.legend(loc='upper right')
ax1D.set_xlabel(r'$q_1$ [1/km]')
#ax1D.set_ylim([np.amin(np.sum(Chst[:,iq2plotalongtrack],axis=0)*dq/(2*pi))/10, np.amax(np.sum(Chst[:,iq2plotalongtrack],axis=0)*dq/(2*pi))*2])
plt1D.suptitle(r'Along/across-wave spectra, $s=%i$' % s)
ax1D.set_ylabel(r'[m$^3$]',color=colors[0]) #Units after timing the Jacobian q
axKE.set_ylabel(r'[m$^3$/s$^2$]',color=colors[2]) #Units after timing the Jacobian q
ax1D.set_box_aspect(1)
ax1D.tick_params(axis='y', labelcolor=colors[0])
axKE.tick_params(axis='y', labelcolor=colors[2])
ax1D.grid(True, which="both", linewidth=0.5,c='#D3D3D3')
plt.tight_layout()

#Plotting in log scale
# fig2d,ax2ds=plt.subplots(2)
# fig2d.set_dpi(256)
# ax2d=ax2ds.ravel()
# ax2d[0].pcolor(np.log(q1_1d[iq1plot]),np.log(q2_1d[iq2plot]),np.transpose(np.log(KEtot[Nx//2+1:iq1cut,Ny//2+1:iq2cut])))
# ax2d[1].pcolor(np.log(q1_1d[iq1plot]),np.log(q2_1d[iq2plot]),np.transpose(np.log(Chst[Nx//2+1:iq1cut,Ny//2+1:iq2cut])))
# ax2d[0].set_aspect(1)
# ax2d[1].set_aspect(1)
# fig2d.suptitle(r'2D spectra, 1st quadrant, log scale, $s=%i$' % s)


#Plotting only the upper half plane
# fig2d,ax2ds=plt.subplots(2,2)
# fig2d.set_dpi(256)
# ax2d=ax2ds.ravel()
# fcut=15 #Setting how much of the axis I want to display. e.g. fcut=10 would should only about 1/10 of the axis in either direction
# #[0].contourf(q1_1d,q2_1d[iq2plot],np.transpose(np.log(KEtot[:,Ny//2+1:iq2cut])))
# #ax2d[1].contourf(q1_1d,q2_1d[iq2plot],np.transpose(np.log(Chst[:,Ny//2+1:iq2cut])))
# ax2d[0].pcolor(q1_1d[(Nx//2-round(Nx//2/fcut)):(Nx//2+round(Nx//2/fcut))]*1000,q2_1d[Ny//2:(round(Ny//2/fcut)+Ny//2)]*1000,np.transpose(np.log(KEtot[(Nx//2-round(Nx//2/fcut)):(Nx//2+round(Nx//2/fcut)),Ny//2:(round(Ny//2/fcut)+Ny//2)])))
# ax2d[1].pcolor(q1_1d[(Nx//2-round(Nx//2/fcut)):(Nx//2+round(Nx//2/fcut))]*1000,q2_1d[Ny//2:(round(Ny//2/fcut)+Ny//2)]*1000,np.transpose(np.log(Chst[(Nx//2-round(Nx//2/fcut)):(Nx//2+round(Nx//2/fcut)),Ny//2:(round(Ny//2/fcut)+Ny//2)])))
# ax2d[0].set_aspect(1)
# ax2d[1].set_aspect(1)
# ax2d[0].set_title(r'KE')
# ax2d[1].set_title(r'$h_s$')
# ax2d[0].set_xlabel(r'$q_1$ [1/km]')
# ax2d[0].set_ylabel(r'$q_2$ [1/km]')
# ax2d[1].set_xlabel(r'$q_1$ [1/km]')
# ax2d[1].set_ylabel(r'$q_2$ [1/km]')
# ax2d[2].set_xlabel(r'$q_1$ [1/km]')
# ax2d[2].set_ylabel(r'$q_2$ [1/km]')

# ax2d[2].pcolor(q1_1d[(Nx//2-round(Nx//2/fcut)):(Nx//2+round(Nx//2/fcut))]*1000,q2_1d[Ny//2:(round(Ny//2/fcut)+Ny//2)]*1000,np.transpose(np.log(Chst_U2H[(Nx//2-round(Nx//2/fcut)):(Nx//2+round(Nx//2/fcut)),Ny//2:(round(Ny//2/fcut)+Ny//2)])))
# ax2d[2].set_aspect(1)
# ax2d[2].set_title(r'$h_s$, from U2H directly')

# fig2d.suptitle(r'log[2D spectra], $s=%i$' % s)
# plt.tight_layout()


#Ploting all the four quadrants
fcut=1/(qcut/np.max(q1_1d))*np.sqrt(2) #Setting how much of the axis I want to display. e.g. fcut=10 would should only about 1/10 of the axis in either direction

# fig2d,ax2ds=plt.subplots(1,2)
# fig2d.set_dpi(256)
# ax2d=ax2ds.ravel()

# im0=ax2d[0].pcolor(q1_1d[(Nx//2-round(Nx//2/fcut)):(Nx//2+round(Nx//2/fcut))]*1000,q2_1d[(Ny//2-round(Ny//2/fcut)):(round(Ny//2/fcut)+Ny//2)]*1000,np.transpose((KEtot[(Nx//2-round(Nx//2/fcut)):(Nx//2+round(Nx//2/fcut)),(Ny//2-round(Ny//2/fcut)):(round(Ny//2/fcut)+Ny//2)])),norm=matplotlib.colors.LogNorm(),cmap=cc.cm.bmw)
# im1=ax2d[1].pcolor(q1_1d[(Nx//2-round(Nx//2/fcut)):(Nx//2+round(Nx//2/fcut))]*1000,q2_1d[(Ny//2-round(Ny//2/fcut)):(round(Ny//2/fcut)+Ny//2)]*1000,np.transpose((Chst[(Nx//2-round(Nx//2/fcut)):(Nx//2+round(Nx//2/fcut)),(Ny//2-round(Ny//2/fcut)):(round(Ny//2/fcut)+Ny//2)])), norm=matplotlib
#                    .colors.LogNorm(),cmap=cc.cm.bgy)
# ax2d[0].set_aspect(1)
# ax2d[1].set_aspect(1)
# ax2d[0].set_title(r'KE, $[m^4/t^2]$')
# ax2d[1].set_title(r'$\hat{C}^{h_s}$ $[m^4]$, U2H')
# ax2d[0].set_xlabel(r'$q_1$ [1/km]')
# ax2d[0].set_ylabel(r'$q_2$ [1/km]')
# ax2d[1].set_xlabel(r'$q_1$ [1/km]')

# ax2d[1].set_yticks([])

# fig2d.colorbar(im0, ax=ax2d[0],shrink=0.5)
# fig2d.colorbar(im1, ax=ax2d[1],shrink=0.5)


# fig2d.suptitle(r'2D spectra, $s=%i$' % s)
# plt.tight_layout()




#Ploting all the four quadrants
fcut=10 #Setting how much of the axis I want to display. e.g. fcut=10 would should only about 1/10 of the axis in either direction
fig2d,ax2ds=plt.subplots(1,2)
fig2d.set_dpi(256)
ax2d=ax2ds.ravel()

im0=ax2d[0].pcolor(q1_1d[(Nx//2-round(Nx//2/fcut)):(Nx//2+round(Nx//2/fcut))]*1000,q2_1d[(Ny//2-round(Ny//2/fcut)):(round(Ny//2/fcut)+Ny//2)]*1000,np.transpose((KEtot[(Nx//2-round(Nx//2/fcut)):(Nx//2+round(Nx//2/fcut)),(Ny//2-round(Ny//2/fcut)):(round(Ny//2/fcut)+Ny//2)])),norm=matplotlib.colors.LogNorm(),cmap=cc.cm.bmw)
im1=ax2d[1].pcolor(q1_1d[(Nx//2-round(Nx//2/fcut)):(Nx//2+round(Nx//2/fcut))]*1000,q2_1d[(Ny//2-round(Ny//2/fcut)):(round(Ny//2/fcut)+Ny//2)]*1000,np.transpose((Chst[(Nx//2-round(Nx//2/fcut)):(Nx//2+round(Nx//2/fcut)),(Ny//2-round(Ny//2/fcut)):(round(Ny//2/fcut)+Ny//2)])), norm=matplotlib
                   .colors.LogNorm(),cmap=cc.cm.bgy)
ax2d[0].set_aspect(1)
ax2d[1].set_aspect(1)
ax2d[0].set_title(r'KE, $[m^4/t^2]$')
ax2d[1].set_title(r'$\hat{C}^{h_s}$ $[m^4]$, U2H')
ax2d[0].set_xlabel(r'$q_1$ [1/km]')
ax2d[0].set_ylabel(r'$q_2$ [1/km]')
ax2d[1].set_xlabel(r'$q_1$ [1/km]')

ax2d[1].set_yticks([])

fig2d.colorbar(im0, ax=ax2d[0],shrink=0.5)
fig2d.colorbar(im1, ax=ax2d[1],shrink=0.5)


fig2d.suptitle(r'2D spectra (zoomed in), $s=%i$' % s)



figr,axrs=plt.subplots(1,2)
figr.set_dpi(256)
axr=axrs.ravel()
im0=axr[0].pcolor(x1d/1000,y1d/1000,np.transpose(np.sqrt(u**2+v**2)),cmap=cc.cm.fire)
axr[0].set_aspect(1)

im1=axr[1].pcolor(x1d/1000,y1d/1000,np.transpose(hs)*100,cmap=cc.cm.gwv)
axr[1].set_aspect(1)

axr[0].set_title(r'$U$ [m/s]')
axr[1].set_title(r'$h_s/\bar{H}_s$, U2H')
axr[0].set_xlabel(r'$x$ [km]')
axr[0].set_ylabel(r'$y$ [km]')
axr[1].set_xlabel(r'$x$ [km]')
axr[1].set_yticks([])

figr.colorbar(im0, ax=axr[0],shrink=0.5)
figr.colorbar(im1, ax=axr[1],shrink=0.5,format ='%.0f%%')


figr.suptitle(r'$s=%i$' % s)

plt.tight_layout()

