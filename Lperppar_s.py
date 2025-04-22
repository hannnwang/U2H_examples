import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import colorcet as cc
from scipy.special import comb
from scipy import special
from scipy import integrate
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
pi=np.pi
colorlist_line=['#DDAA33','#BB5566','#004488']
colors_more=['#EECC66','#EE99AA','#6699CC','#997700','#994455','#004488']

def halflist(a_list): #Take the latter half of an 1D array (for plotting)
    ihalf = len(a_list)//2
    return a_list[ihalf:]

#ncoslist=[2,10,20,80]
ncoslist=[2,20,80]
s_all=np.zeros(len(ncoslist))
Lpar2int=np.zeros(len(ncoslist))
Lper2int=np.zeros(len(ncoslist))
Lcrossint=np.zeros(len(ncoslist))

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

waveT=10.3
thetacenter=0

fp=1/waveT

#*******************Mapping onto eta 1D ******************************
#Go around a rectangular in the 2D grid to get the L components along a 1D vector. 
iqslice=np.min([nq1,nq2])//2-1
varphislice=np.zeros(iqslice*8)
#varphislice starts at -pi and ends at pi
idvarphi1=np.arange(0,iqslice)
idfun11=nq1//2-iqslice
idfun12=np.arange(nq2//2,nq2//2-iqslice,-1)
varphislice[idvarphi1]=varphi[idfun11,idfun12] 

idvarphi2=np.arange(iqslice,3*iqslice)
idfun21=np.arange((nq1//2-iqslice),(nq1//2+iqslice))
idfun22=nq2//2-iqslice
varphislice[idvarphi2]=varphi[idfun21,idfun22]

idvarphi3=np.arange(3*iqslice,5*iqslice)
idfun31=nq1//2+iqslice
idfun32=np.arange((nq2//2-iqslice),(nq2//2+iqslice))
varphislice[idvarphi3]=varphi[idfun31,idfun32]

idvarphi4=np.arange(5*iqslice,7*iqslice)
idfun41=np.arange((nq1//2+iqslice),(nq1//2-iqslice),(-1))
idfun42=nq2//2+iqslice
varphislice[idvarphi4]=varphi[idfun41,idfun42]

idvarphi5=np.arange(7*iqslice,8*iqslice)
idfun51=nq1//2-iqslice
idfun52=np.arange((nq2//2+iqslice),nq2//2,(-1))
varphislice[idvarphi5]=varphi[idfun51,idfun52]

varphislice[0]=varphislice[0]-2*pi#By python's convention, -pi was mapped to pi. We make it -pi for continuity.

def slicerect(L2):
    L2slice=np.zeros(iqslice*8,dtype=complex)
    L2slice[idvarphi1]=L2[idfun11,idfun12] 
    L2slice[idvarphi2]=L2[idfun21,idfun22]
    L2slice[idvarphi3]=L2[idfun31,idfun32]
    L2slice[idvarphi4]=L2[idfun41,idfun42]
    L2slice[idvarphi5]=L2[idfun51,idfun52]
    return L2slice
    
    
figLpp,axLpp=plt.subplots()
figLpp.set_dpi(256)


# Add inset for zoom-in view
axins = inset_axes(axLpp, width='30%', height='30%', loc='upper right', borderpad=0.5)

# Specify the limits of the zoomed-in section
x1, x2, y1, y2 = 0, pi/4, 0, 0.2
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
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

    #Alternatively, we decompost L into q and qperp directions
    Lhatqperp=Lhat1*(-np.sin(varphi))+Lhat2*np.cos(varphi)
    Lhatqpar=Lhat1*np.cos(varphi)+Lhat2*np.sin(varphi)
    
    Lhatqperpslice=slicerect(Lhatqperp)
    Lhatqparslice=slicerect(Lhatqpar)
    #axLpp.plot(halflist(varphislice),halflist(np.abs(Lhatqperpslice)),color=colorlist_line[incos],label=r'$|\hat{L}_{\psi}|$,s= '+str(s))
    axLpp.plot(halflist(varphislice),halflist(np.abs(Lhatqperpslice)),color=colorlist_line[incos],label=r's= '+str(s))
    marker_interval = 5  # Distance between markers
    axLpp.plot(halflist(varphislice),halflist(np.abs(Lhatqparslice)),color=colorlist_line[incos],linestyle='--')#,label=r'$|\hat{L}_{\phi}|$,s= '+str(s)) #,,marker='o',markevery=marker_interval,

    # Plot the same data in the inset
    axins.plot(halflist(varphislice), halflist(np.abs(Lhatqperpslice)), color=colorlist_line[incos])
    axins.plot(halflist(varphislice), halflist(np.abs(Lhatqparslice)), color=colorlist_line[incos], linestyle='--')




axLpp.legend(loc='upper left',prop={'size': 11})
axLpp.set_xlabel(r'$\varphi$',fontsize=12)
axLpp.set_xticks([0,pi/4,pi/2,3*pi/4,pi])
axLpp.set_xticklabels(['0',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',r'$\pi$'])
plt.subplots_adjust(wspace=0.4)  # adjust spacing between subplots if necessary
axLpp.set_ylabel('[s/m]',fontsize=12)
axLpp.minorticks_on()
axLpp.tick_params(axis='both', which='major', labelsize=12)

# Add grid, labels, and ticks
axins.set_xticks([0, pi/8, pi/4])
axins.set_xticklabels(['0', r'$\frac{\pi}{8}$​', r'$\frac{\pi}{4}$​'])
axins.set_yticks([0, 0.1, 0.2])
#axins.set_xlabel(r'$\varphi$')
#axins.set_ylabel('[s/m]')
axins.minorticks_on()
axins.tick_params(axis='both', which='major', labelsize=12)

# Add a frame to the inset
axins.set_facecolor('none')
mark_inset(axLpp, axins, loc1=2, loc2=3, fc="none", ec="0.5")
