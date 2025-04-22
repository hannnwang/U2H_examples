#Update Aug.20/2024: change the way the anglur integrals in varphi are done, so that they do not rely on the gridded 2-D L operators any more.
#Compare the integration of |Lpar|^2, |Lper|^2 over varphi under different s, and 
#check if the integration of Lpar*Lper is zero.
#Note: lphi is not supposed to be a constant right after s=1
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import colorcet as cc
from scipy.special import comb
from scipy import special
from scipy import integrate
pi=np.pi
colorlist_line=['#DDAA33','#BB5566','#004488']

ncoslist=np.arange(0,100,2)
#ncoslist=np.arange(0,100,2)

#Grid of varphi for the integration in varphi to get lphi, lpsi
Nvarphiint=256 #Change this for different levels of accuracy
varphiint=np.linspace(0,2*pi,Nvarphiint)

s_all=np.zeros(len(ncoslist))
Lpar2int=np.zeros(len(ncoslist))
Lper2int=np.zeros(len(ncoslist))
Lcrossint=np.zeros(len(ncoslist),dtype=complex)

#Parameters for the incoming wave spectra. Default values correspond to the snapshot shown in Figure 1 of our paper. 
g=9.806 #Gravity acceleration
Hs0=1 #Unperturbed significant wave height (unit: m). This parameter does not affect the outcomes presented in terms of hs/Hs0


Nx=50#Bia has Nx=Ny=250
Ny=Nx-2
nq1=Nx 
nq2=Ny
qmax=0.1
qvec1=np.linspace(-qmax,qmax,nq1)
qvec2=qvec1[1:-1]

q1_2D,q2_2D=np.meshgrid(qvec1,qvec2)
q1_2D=np.transpose(q1_2D) #Make indexing intuitive, i.e. F[i,j] would mean we want q1 at the ith and q2 at the jth.
q2_2D=np.transpose(q2_2D)
#polar coordinate of q
q=np.sqrt(q1_2D**2+q2_2D**2)#wavenumber magnitude of q in 2D
varphi = np.arctan2(q2_2D, q1_2D) #polar angle $\varphi$.

#Ndir = 48*2 #number of grid points in theta
Ndir = 120 #number of grid points in theta

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

waveT=10.3
thetacenter=0

fp=1/waveT


for incos in np.arange(len(ncoslist)):
    ncos=ncoslist[incos]
    s=ncos//2
    s_all[incos]=s

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
    ncutoff=s #p_n=0 at n>s for LHCS model
    #Decide the n to cut off the computation of pn
    if 'ncutoff' not in globals():
        if Ndir>80:
            ncutoff=40 #40 corresponds to directional spreading at s=40, already extremely swell-like for ocean 
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
    if s == 0: #Added Aug.6/2024
        P1=0
        P2=0
    LtP1=-32/(g*Hs0**2)*P1
    LtP2=-32/(g*Hs0**2)*P2
    #The term along eqperp; changed Aug.20/2024 to be the angular version
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
   
    Lper2int[incos]=integrate.trapezoid((np.abs(Lhatqperpvarphi))**2, x=varphiint)/(2*pi)
    Lpar2int[incos]=integrate.trapezoid((np.abs(Lhatqparvarphi))**2, x=varphiint)/(2*pi)
    Lcrossint[incos]=integrate.trapezoid(Lhatqparvarphi*np.conjugate(Lhatqperpvarphi), x=varphiint)/(2*pi)
    
figLcoef,axLcoef=plt.subplots()
figLcoef.set_dpi(256)

axLcoef.plot(s_all, Lpar2int, color=colorlist_line[0], marker='o',label=r'$\ell_\phi$',markersize=3)
axLcoef.plot(s_all, 4*Lpar2int, color=colorlist_line[0], linestyle='--',label='$4\ell_\phi$')

axLcoef.plot(s_all, Lper2int, color=colorlist_line[1], marker='o',label='$\ell_\psi$',markersize=3)
#axLcoef.plot(s_all, np.abs(Lcrossint), color=colorlist_line[2], marker='o',label='int Lcross')
axLcoef.legend(loc='upper left',prop={'size': 15})
axLcoef.set_xlabel(r'$s$')
axLcoef.set_ylabel(r'[s$^2$/m$^2$]')
axLcoef.set_xticks(s_all)
# Label every 5th number while keeping all tick marks
tick_labels = [str(num) if num % 5 == 1 or num == 1 else '' for num in range(1, int(1+np.max(s_all)))]
axLcoef.set_xticks(np.arange(1, int(1+np.max(s_all)), 1), tick_labels)  # Apply custom labels
#Plot the power law
plaw=3/2
plnorm=(Lper2int[-1]-Lper2int[10])/((s_all[-1])**plaw-(s_all[10])**plaw)
poffset=0.01
axLcoef.plot(s_all[10:-1],(s_all[10:-1])**plaw*plnorm+poffset,color='g',linestyle='-') 

axLcoef.annotate(r'$\propto s^{3/2}$',
            xy=(25, 0.16), xycoords='data',
            horizontalalignment='left', verticalalignment='top',
            fontsize=14,color='g')

# figLcoef,axLcoef=plt.subplots()
# figLcoef.set_dpi(256)
# axLcoef.plot(s_all**(4/3), Lper2int, color=colorlist_line[1], marker='o',label='$\ell_\psi$',markersize=3)
# axLcoef.legend(loc='upper left',prop={'size': 15})
# plnorm=Lper2int[30]/(s_all[30])**(4/3)
# axLcoef.plot((s_all[10:-1])**(4/3),(s_all[10:-1])**(4/3)*plnorm,color='g',linestyle='--')
# axLcoef.set_xlabel('s^(4/3)')
# axLcoef.set_ylabel(r'[sec$^2$/m$^2$]')
# axLcoef.set_xticks(s_all)



# figLcoef,axLcoef=plt.subplots()
# figLcoef.set_dpi(256)
# axLcoef.plot(s_all**(3/2), Lper2int, color=colorlist_line[1], marker='o',label='$\ell_\psi$')
# plnorm=Lper2int[30]/(s_all[30])**(3/2)
# axLcoef.plot((s_all[10:-1])**(3/2),(s_all[10:-1])**(3/2)*plnorm,color='g',linestyle='--')
# axLcoef.legend(loc='upper left',prop={'size': 15})
# axLcoef.set_xlabel('s^(3/2)')
# axLcoef.set_ylabel(r'[sec$^2$/m$^2$]')
# axLcoef.set_xticks(s_all)


# figLcoef,axLcoef=plt.subplots()
# figLcoef.set_dpi(256)
# axLcoef.loglog(s_all, Lper2int, color=colorlist_line[1], marker='o',label='$\ell_\psi$')
# plnorm=Lper2int[30]/(s_all[30])**(3/2)
# axLcoef.loglog((s_all[10:-1]),(s_all[10:-1])**(3/2)*plnorm,color='g',linestyle='--')
# axLcoef.legend(loc='upper left',prop={'size': 15})
# axLcoef.set_xlabel('s')
# axLcoef.set_ylabel(r'[sec$^2$/m$^2$]')
# axLcoef.set_title('with s^(3/2) slope')


# figLcoef,axLcoef=plt.subplots()
# figLcoef.set_dpi(256) 
# axLcoef.loglog(s_all, Lper2int, color=colorlist_line[1], marker='o',label='$\ell_\psi$')
# plnorm=Lper2int[30]/(s_all[30])**(4/3)
# axLcoef.loglog((s_all[10:-1]),(s_all[10:-1])**(4/3)*plnorm,color='g',linestyle='--')
# axLcoef.legend(loc='upper left',prop={'size': 15})
# axLcoef.set_xlabel('s')
# axLcoef.set_ylabel(r'[sec$^2$/m$^2$]')
# axLcoef.set_title('with s^(4/3) slope')

# figLcoef,axLcoef=plt.subplots()
# figLcoef.set_dpi(256) 
# axLcoef.loglog(s_all, Lper2int, color=colorlist_line[1], marker='o',label='$\ell_\psi$')
# plnorm=Lper2int[30]/(s_all[30])**(5/4)
# axLcoef.loglog((s_all[10:-1]),(s_all[10:-1])**(5/4)*plnorm,color='g',linestyle='--')
# axLcoef.legend(loc='upper left',prop={'size': 15})
# axLcoef.set_xlabel('s')
# axLcoef.set_ylabel(r'[sec$^2$/m$^2$]')
# axLcoef.set_title('with s^(5/4) slope')

# plawis=15
# plaw_data=(np.log(Lper2int[-1])-np.log(Lper2int[plawis]))/(np.log(s_all[-1])-np.log(s_all[plawis]))
# print(plaw_data)