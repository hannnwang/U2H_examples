#Plot lphi and lpsi as functions of peak period and s
#Plots all dots first, and then use transparency to imply the coverage by data
#20250121: Try the computation with non-integer s. Cutting of fourier modes at a limited number.
#Plot for paper: ditch the alpha setting, but adding a contour that includes most of data. Add a pdf plot 
#of data.
#20250213: chop off the s values by a little so that smaller s are not shown. you can change scut_min to adjust it.
import netCDF4 as nc
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../hw_pythonfuncs')
import scipy.signal
from scipy import integrate
from matplotlib.cm import get_cmap
import glob
from matplotlib.colors import LogNorm
import colorcet as cc #required to access the colormaps within
import matplotlib.ticker as tkr
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

colorlist_line=['#DDAA33','#BB5566','#004488']
colors_more=['#EECC66','#EE99AA','#6699CC','#997700','#994455','#004488']

pi=np.pi
g=9.806 #Gravity acceleration. 
Hs0=1 #Mean assumed for the CZ transform; have tested that changing Hs0 does not affect the outcome in terms of hs/Hs0

scut_min=1.6#Cut off data whose s are smaller than this so that the contour maps  will look nicer. Only do this if you are confident that there is not much data there.
sigma_max=np.sqrt(2/scut_min) #directional spreading; not frequency
scut_max=60
sigma_min=np.sqrt(2/scut_max)
#bin numbers in directional spreading and peak freq. Adjust this for better-looking histograms
nbinsigma=30
nbinfreq=30
#Cut off data whose values of peak frequency sigma_p are extreme
sigma_p_max=1.7
sigma_p_min=0.05 #Seems like setting it to be <0.2 has no effect. We just set it to be nonzero so that the division (next step) works
waveT_max=2*pi/sigma_p_min
waveT_min=2*pi/sigma_p_max

#Grid of varphi for the integration in varphi to get lphi, lpsi
Nvarphiint=64 #Change this for different level of accuracy
varphiint=np.linspace(0,2*pi,Nvarphiint)

#Read sigma and Tp from data file
datapath='./buoy_LHCS'
#ncfiles = glob.glob(os.path.join(datapath, '*.nc'))
#Get data from just one buoy
ncfiles = glob.glob(os.path.join(datapath, '*_166.nc'))

# Initialize accumulators for results
all_sigma_data = []
all_waveTdata = []
noriginalrecord=0
for ncfilename in ncfiles:
    ncfile = ncfilename
    ncds = nc.Dataset(ncfile)
    
    sigma_data_original=ncds['mean_sigma_theta'][:].data*pi/180 #Directional spreading. sigma is in degrees in this file originally, so we convert to radians here
    waveTdata_original=ncds['mean_period'][:].data # Tm is the mean period, and it's a reasonable estimate of the "peak period" parameter 
    
    noriginalrecord=noriginalrecord+len(waveTdata_original)

    waveTdata=waveTdata_original[(sigma_data_original>sigma_min) & (sigma_data_original < sigma_max) &
    (waveTdata_original>waveT_min) & (waveTdata_original< waveT_max)]
    sigma_data=sigma_data_original[(sigma_data_original>sigma_min) & (sigma_data_original < sigma_max) &
    (waveTdata_original>waveT_min) & (waveTdata_original< waveT_max)]

    s_data=2/sigma_data**2
    all_sigma_data.extend(sigma_data)
    all_waveTdata.extend(waveTdata)
#Convert lists to np arrays
all_sigma_data = np.array(all_sigma_data)
all_waveTdata = np.array(all_waveTdata)

all_frequencyp_data=2*pi/all_waveTdata
ndata_chopped=len(all_waveTdata)
print('The plots cover this proportion of data:')
print(ndata_chopped/noriginalrecord)

#Bins for the histograms and for the lphi/lpsi computations
waveTmin=np.min(all_waveTdata)
waveTmax=np.max(all_waveTdata)
sigmamin=np.min(all_sigma_data)
sigmamax=np.max(all_sigma_data)
sigma_all=np.linspace(sigmamin,sigmamax,nbinsigma)

#evaluate ellphi, ellpsi on the bin centers (04/2025)
frequencypall=np.linspace(2*pi/waveTmax,2*pi/waveTmin,nbinfreq)
freqgridcenter=(frequencypall[1:]+frequencypall[:-1])/2
siggridcenter=(sigma_all[1:]+sigma_all[:-1])/2

s_gridcenter=2/siggridcenter**2
#The unperturbed wave action: the f(k) component
def gaussian_spec(freq, fp, sip):
    frrel = (freq - fp)/sip
    return np.exp(-1.25*(frrel**2))

#Assume a delsig that's smaller than 50% of the peak sigma in data
sip = 0.01 #variance of frequency
delsig = sip * 2*pi/np.sqrt(2*1.25)
thetacenter=0

lphi_i=np.zeros([len(siggridcenter),len(freqgridcenter)])*np.nan
lpsi_i=np.zeros([len(siggridcenter),len(freqgridcenter)])*np.nan

ncutoff=50 #Tested that ncutoff=50 and ncutoff=100 doesn't make a noticeable difference in the contour plot of lphi, lpsi
Ndir = 2*ncutoff+2


for i_s, s in enumerate(s_gridcenter):
    for ifp, fp in enumerate(freqgridcenter):
        
        Nf = 32*4 #number of grid points in frequency 
        #lowf = fp-3*sip #0.04118 #Lowest frequency resolved (in Hz)
        lowf = 0.04118 #Lowest frequency resolved (in Hz)
        highf=fp+3*sip#lowf*(ffactor**32)
        freq=np.linspace(lowf,highf,Nf)
        k=(2*pi*freq)**2/g #Wavenumber magnitude of k in 1D, in meters. Note 2 pi is needed as f is not angular frequency
    
        # #Debugging: check if the frequency distribution is sufficiently resolved
        # if iwaveT%5 == 0:
        #     figcheckf,axcheckf=plt.subplots()
        #     axcheckf.plot(k, gaussian_spec(np.sqrt(g*k)/(2*pi), fp, sip))
        
        #Ndir = int(2*s+2)# 48*2 #number of grid points in theta
        
        theta = np.linspace(-pi, pi, Ndir) #vector for theta
        #grid in k, theta
        k2D,theta2D=np.meshgrid(k,theta)
        k2D=np.transpose(k2D) #Making indexing intuitive, i.e. E2D[i,j] would refer to E2D at k[i] and theta[j]
        theta2D=np.transpose(theta2D)
        
        E2D=gaussian_spec(np.sqrt(g*k2D)/(2*pi), fp, sip)*np.cos((theta2D-thetacenter)/2)**(2*s)/2*np.sqrt(g)*k2D**(-1/2)/k2D
        #Normalize so that the mean Hs is Hs0
        Hs0_r=4*np.sqrt(integrate.trapezoid(integrate.trapezoid(E2D*k2D,x=k,axis=0),x=theta)/g)
        if Hs0_r == 0:
            print("Hs0_r=0. Check what goes wrong.")
            sys.exit(1) 
        E2D=E2D*Hs0**2/Hs0_r**2
        #Wave action
        A2D=E2D/np.sqrt(g*k2D)
           
        #Get L using the general formula that doesn't require separability
        #mathcal{P}(theta), equation (3.8)  
        Pcal=integrate.trapezoid(A2D*k2D**2,x=k,axis=0)
        #pn
        # if s>0:
        #     ncutoff=int(s) #p_n=0 at n>s for LHCS 
        # else:
        #     ncutoff=1
        # #Decide the n to cut off the computation of pn
        # if 'ncutoff' not in globals():
        #     if Ndir>80:
        #         ncutoff=40 #40 corresponds to directional spreading at s=40, already extremely swell-like for oce an 
        #     else:
        #         ncutoff=Ndir//2-1 #Nyquist frequency in case Ndir is too small to accurately compute s up to 40.
        #     print('Parameter ncutoff is not given in the code by user. Setting ncutoff to be %i' % ncutoff)
        # if ncutoff > Ndir//2-1:
        #     raise ValueError("Ndir must be at least twice as big as ncutoff")
        
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
    
        lphi_i[i_s,ifp]=integrate.trapezoid((np.abs(Lhatqparvarphi))**2, x=varphiint)/(2*pi)
        lpsi_i[i_s,ifp]=integrate.trapezoid((np.abs(Lhatqperpvarphi))**2, x=varphiint)/(2*pi)


# #Contour map where the colorbars are fixed the same between the two panels
# figl, axl = plt.subplots(1, 3, figsize=(8.5, 6))
# figl.set_dpi(256)
# # Define a single colormap to simplify visualization
# cmap = get_cmap("cet_rainbow_bgyrm_35_85_c71")

# # LogNorm normalizations w
# norm_lboth = LogNorm(vmin=np.nanmin([lphi_i,lpsi_i]), vmax=np.nanmax([lphi_i,lpsi_i]))

# im0 = axl[1].pcolor(frequencypall, sigma_all, lphi_i,
#                       cmap=cmap, norm=norm_lboth)

# im1 = axl[2].pcolor(frequencypall, sigma_all, lpsi_i, cmap=cmap, norm=norm_lboth)

    
# #  Calculate the frequency of occurrence
# counts, xedges, yedges = np.histogram2d(all_frequencyp_data, all_sigma_data, bins=[frequencypall, sigma_all])
# counts = np.transpose(counts)
# axhisto=axl[0]

# # Make ones with zero counts Nan, so that they are presented as white colors, as we choose the colormap to start from non-white colors
# counts[counts==0]=np.nan

# # Make a contour plot with the level specified
# level = [100]
# print('data points with >= contour level account for this percentage of all data:')
# print(np.nansum(counts[counts>=100])/noriginalrecord*100)
# freqgridcenter=(frequencypall[1:]+frequencypall[:-1])/2
# siggridcenter=(sigma_all[1:]+sigma_all[:-1])/2
# cmap = get_cmap("cet_linear_blue_95_50_c20")
# im=axhisto.pcolor(freqgridcenter,siggridcenter,counts,cmap=cmap)
# axhisto.contour(freqgridcenter,siggridcenter,counts, level, colors='black') 
# axl[1].contour(freqgridcenter,siggridcenter,counts, level, colors='black') 
# axl[2].contour(freqgridcenter,siggridcenter,counts, level, colors='black')  

# axl[0].set_ylabel(r'$s$', fontsize=15)
# axl[0].set_title(r'Histogram', fontsize=15)
# axl[1].set_title(r'$\ell_\phi$', fontsize=15)
# axl[2].set_title(r'$\ell_\psi$', fontsize=15)

# s_at_yticks= np.array([2, 4, 8, 16, 32]) 
# yticklocations=np.sqrt(2/s_at_yticks)
# axhisto.set_yticks(yticklocations, labels=s_at_yticks)
# axl[1].set_yticks(yticklocations, labels=[])
# axl[2].set_yticks(yticklocations, labels=[])

# axl[1].set_aspect((np.max(frequencypall) - np.min(frequencypall)) / (np.max(sigma_all) - np.min(sigma_all)))
# axl[2].set_aspect((np.max(frequencypall) - np.min(frequencypall)) / (np.max(sigma_all) - np.min(sigma_all)))
# axhisto.set_aspect((np.max(frequencypall) - np.min(frequencypall)) / (np.max(sigma_all) - np.min(sigma_all)))

# for ax in axl:
#     ax.tick_params(axis='both', which='major', labelsize=14)
#     ax.set_xlabel(r'$\sigma_p$ [s$^{-1}$]', fontsize=15)
#     ax.xaxis.set_tick_params(length=7)
#     ax.yaxis.set_tick_params(length=7)

# #Color bars
# cbarax_histo = figl.add_axes([0.1, 0.15, 0.25, 0.015]) #tuple (left, bottom, width, height).
# cbarax_histo.set_title('Bin Count', y=-5.5)
# cbarhisto=plt.colorbar(im,cax=cbarax_histo,orientation='horizontal')

# cbarax_l = figl.add_axes([0.41, 0.15, 0.55, 0.015]) #tuple (left, bottom, width, height).
# cbarax_l.set_title(r'[s$^2$/m$^2$]', y=-5.5)
# cbarl=plt.colorbar(im0,cax=cbarax_l,orientation='horizontal')

# plt.tight_layout()
# plt.subplots_adjust(wspace=0.07)

# #Add a curve denoting the linear regime
# Utyp = 1
# ylim0, ylim1 = axl[0].get_ylim()
# slinregime = (g*frequencypall/(2*Utyp))**2
# sigmalinregime=np.sqrt(2/slinregime)
# axl[0].plot(frequencypall,sigmalinregime,'--',c='green')
# axl[0].set_ylim(ylim0, ylim1)


#Re-plot, replacing the histogram with pdf

#Contour map where the colorbars are fixed the same between the two panels
figl, axl = plt.subplots(1, 3, figsize=(8.5, 6))
figl.set_dpi(256)
# Define a single colormap to simplify visualization
cmap = get_cmap("cet_rainbow_bgyrm_35_85_c71")

# LogNorm normalizations w
norm_lboth = LogNorm(vmin=np.nanmin([lphi_i,lpsi_i]), vmax=np.nanmax([lphi_i,lpsi_i]))

im0 = axl[1].pcolor(freqgridcenter,siggridcenter, lphi_i,
                      cmap=cmap, norm=norm_lboth)

im1 = axl[2].pcolor(freqgridcenter,siggridcenter, lpsi_i, cmap=cmap, norm=norm_lboth)

    
#  Calculate the frequency of occurrence
counts, xedges, yedges = np.histogram2d(all_frequencyp_data, all_sigma_data, bins=[frequencypall, sigma_all])
counts = np.transpose(counts)
axhisto=axl[0]

# Make ones with zero counts Nan, so that they are presented as white colors, as we choose the colormap to start from non-white colors
counts[counts==0]=np.nan

#Probability density function at each point. Note that we are plotting the pdf with regard to variables (directional spreading, frequency), not (s, frequency)
pdf = counts/np.nansum(counts)/((np.max(freqgridcenter)-np.min(freqgridcenter))*(np.max(siggridcenter)-np.min(siggridcenter)))

# Make a contour plot with the level specified
level = 0.001
print('data points with >= contour level account for this percentage of all data:')
print(np.nansum(counts[pdf>=level])/noriginalrecord*100)

cmap = get_cmap("cet_linear_blue_95_50_c20")
im=axhisto.pcolor(freqgridcenter,siggridcenter,pdf,cmap=cmap)
axhisto.contour(freqgridcenter,siggridcenter,pdf, [level], colors='black') 
axl[1].contour(freqgridcenter,siggridcenter,pdf, [level], colors='black') 
axl[2].contour(freqgridcenter,siggridcenter,pdf, [level], colors='black')  
axhisto.set_aspect((np.max(frequencypall) - np.min(frequencypall)) / (np.max(sigma_all) - np.min(sigma_all)))
axl[1].set_aspect((np.max(frequencypall) - np.min(frequencypall)) / (np.max(sigma_all) - np.min(sigma_all)))
axl[2].set_aspect((np.max(frequencypall) - np.min(frequencypall)) / (np.max(sigma_all) - np.min(sigma_all)))


axl[0].set_ylabel(r'$\delta$ [rad]', fontsize=15)
axl[0].set_title(r'Probability density', fontsize=15)
axl[1].set_title(r'$\ell_\phi$', fontsize=15)
axl[2].set_title(r'$\ell_\psi$', fontsize=15)

# Mark the values of s on the right y aixs in the first figure panel
# Seen on matplotlib documentation: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/secondary_axis.html
def new2old(x):
    return np.sqrt(2/x)
def old2new(x):
    return 2/x**2

secax = axhisto.secondary_yaxis('right', functions=(old2new,new2old))
s_at_yticks= np.array([2, 4, 8, 16, 32]) 
yticklocations=np.sqrt(2/s_at_yticks)
# axhistotwin.set_yticks(yticklocations, labels=s_at_yticks)
secax.set_yticks(s_at_yticks,fontsize=13)


secax1 = axl[1].secondary_yaxis('right', functions=(old2new,new2old))
axl[1].set_yticks([])
secax1.set_yticks(s_at_yticks,fontsize=13)

secax2 = axl[2].secondary_yaxis('right', functions=(old2new,new2old))
axl[2].set_yticks([])
secax2.set_yticks(s_at_yticks,fontsize=13)
secax2.set_ylabel(r'$s$',fontsize=15)


for ax in axl:
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.set_xlabel(r'$\sigma_p$ [s$^{-1}$]', fontsize=15)
    #ax.xaxis.set_tick_params(length=5)
    #ax.yaxis.set_tick_params(length=5)

#Color bars
cbarax_histo = figl.add_axes([0.1, 0.15, 0.25, 0.015]) #tuple (left, bottom, width, height).
cbarax_histo.set_title(r'[s]', y=-5.5)
cbarhisto=plt.colorbar(im,cax=cbarax_histo,orientation='horizontal',format=tkr.FormatStrFormatter('%.2g'))

cbarax_l = figl.add_axes([0.41, 0.15, 0.55, 0.015]) #tuple (left, bottom, width, height).
cbarax_l.set_title(r'[s$^2$/m$^2$]', y=-5.5)
cbarl=plt.colorbar(im0,cax=cbarax_l,orientation='horizontal')

plt.tight_layout()
plt.subplots_adjust(wspace=0.2)

#Add a curve denoting the linear regime
Utyp = 1
ylim0, ylim1 = axl[0].get_ylim()
slinregime = (g/(2*Utyp*frequencypall))**2
sigmalinregime=np.sqrt(2/slinregime)
axl[0].plot(frequencypall,sigmalinregime,'--',c='green')
axl[0].set_ylim(ylim0, ylim1)