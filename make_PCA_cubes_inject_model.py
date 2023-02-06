import matplotlib as mpl
from matplotlib.pyplot import *
#from fm import *
import time
import pickle
from scipy import constants
import numpy as np
from matplotlib.pyplot import cm
import pdb
from numba import jit
import math
import numpy as np
import scipy as sp
from array import *
from scipy import interpolate
from scipy import signal
from scipy import special
from scipy import interp
from scipy import ndimage
from astropy.io import fits
import pdb
import datetime
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
import matplotlib.pyplot as plt

__all__ = ['injection_testing']

#injects dopper shifted model into actual data given a specified Kp and Vsys. When running
#CCF.py on the PCA cubes here, should see "fake blob" from your injected signal


def injection_testing(wl_data, data, Rvel, ph, model_fn, scale,
                      Vsys, Kp, npca, output_fn):
    """
    wl_data : wavelength array
    data : data array
    Rvel : barycentric velocity array
    ph : phi array
    model_fn : filename of the model txt file
    scale : scaling factor
    Vsys : systemic velocity
    Kp : planet orbital velocity
    npca : number of principal components
    """
    ##loading in real data
    #wl_data, data=pickle.load(open('wavelengthcalibrated.pic','rb')) #calibrated data cube
    Ndet, Nphi,Npix = data.shape
    #Rvel=pickle.load(open('Vbary.pic','rb'))
    #ph = pickle.load(open('phi.pic','rb'))      # Time-resolved phases

    #loading in model
    wl_model, Depth=np.loadtxt(model_fn).T

    #injection parameters
    #scale=0.3*np.sqrt(3.) # change this to "emulate" systems with different signal to noises or to "strecth" the model
    #Vsys=-50 #system velocity
    #Kp=107.6  #orbital velocity of planet
    RVp=Vsys+Rvel+Kp*np.sin(2.*np.pi*ph) #radial velocity of planet relative to earth at phase "phi"


    #instrumental profile convolution and setting up model interpolation
    xker = np.arange(41)-20
    sigma = 5.5/(2.* np.sqrt(2.0*np.log(2.0)))  #nominal
    yker = np.exp(-0.5 * (xker / sigma)**2.0)
    yker /= yker.sum()
    Fp_conv = np.convolve(Depth,yker,mode='same')*scale
    cs_p = interpolate.splrep(wl_model,Fp_conv,s=0.0)


    #building fake RV spectra--THIS IS THE MODEL INJECTION PROCESS RIGHT HERE------------
    Shifted_model_arr=np.ones((Ndet, Nphi, Npix))  #note, this is 1's not zeros

    for j in range(Ndet): #looping over detectors
        for i in range(Nphi):  #looping over phase

            #shifting model
            Vp=RVp[i]  #just grabbing velocity from RV array
            #Interpolating to doppler shifted wavelength grid
            wl = wl_data[j,] * (1.0 - Vp * 1E3 / constants.c) #dopppler shifting
            Dp = interpolate.splev(wl,cs_p,der=0)  #interpolating hi-res model to doppler shifted data wl grid
            Shifted_model_arr[j,i,:] -= Dp     #this is 1-depth, or 1-(Rp/Rs)**2


    #injecting model onto data right here....
    Obs_arr_mod=Shifted_model_arr*data
    data=Obs_arr_mod
    #END MODEL INJECTION PROCESS-----------------------------------------------------



    #SVD/PCA method (same as make_PCA_cubes.py)-------------------------------------

    data_arr1=np.zeros(data.shape)
    data_arr=np.zeros(data.shape)

    for i in range(Ndet):

        '''
        #normalize each frame/spectrum by it's median
        for j in range(Nphi): data[i,j,]=data[i,j,]/np.median(data[i,j,])
        #A=data[i,]
        #imshow(A,extent=[wl_data[i,].min(),wl_data[i,].max(), 0, A.shape[0]],aspect='auto',cmap='gray')
        #show()

        #subtract off mean from each column
        for k in range(Npix): data[i,:,k]=data[i,:,k]-np.mean(data[i,:,k])
        #A=data[i,]
        #imshow(A,extent=[wl_data[i,].min(),wl_data[i,].max(), 0, A.shape[0]],aspect='auto',cmap='gray')
        #show()
        '''


        #taking only first four vectors, reconstructiong, and saving
        u,s,vh=np.linalg.svd(data[i,],full_matrices=False)  #decompose
        s[npca:]=0.
        W=np.diag(s)
        A=np.dot(u,np.dot(W,vh))
        data_arr1[i,]=A

        #pdb.set_trace()
        #removing first four vectors...this is the 'processesed data'
        u,s,vh=np.linalg.svd(data[i,],full_matrices=False)  #decompose
        s[0:npca]=0.
        W=np.diag(s)
        A=np.dot(u,np.dot(W,vh))
        #sigma clipping
        #'''
        sig=np.std(A)
        med=np.median(A)
        loc=np.where(A > 3.*sig+med)
        A[loc]=0#*0.+20*sig
        loc=np.where(A < -3.*sig+med)
        A[loc]=0#*0.+20*sig
        #'''
        #
        data_arr[i,]=A

    ###CHANGE THESE TO CHANGE THE OUTPUT FILE NAMES
    pickle.dump([wl_data,data_arr1],open('injection_results_{}.pic'.format(output_fn),
                                         'wb'))

    pickle.dump([wl_data,data_arr],open('injection_results_noise_{}.pic'.format(output_fn),
                                        'wb'))  #saving includes telluric contamination


    ###plotting...
    i=25
    A=data[i,]

    imshow(A,extent=[wl_data[i,].min(),wl_data[i,].max(), 0, A.shape[0]],aspect='auto',cmap='gray')
    #title('RAW SEQUENCE')
    xlabel('Wavelength [$\mu$m]')
    ylabel('Frame #')
    subplots_adjust(left=0.1, right=0.9, top=0.65, bottom=0.4)
    savefig('RAW_SEQUENCE_ORDER_'+str(i)+'.pdf',fmt='pdf')
    show()
    close()

    A=data_arr[i,]

    imshow(A,extent=[wl_data[i,].min(),wl_data[i,].max(), 0, A.shape[0]],aspect='auto',cmap='gray')
    #title("TELLURIC'S REMOVED")
    xlabel('Wavelength [$\mu$m]')
    ylabel('Frame #')
    subplots_adjust(left=0.1, right=0.9, top=0.65, bottom=0.4)

    savefig('PCA_SEQUENCE_ORDER'+str(i)+'.pdf',fmt='pdf')
    show()
    close()
