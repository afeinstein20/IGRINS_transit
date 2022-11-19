#import matplotlib as mpl
#mpl.use('TkAgg')
from matplotlib.pyplot import *
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
import glob
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
import matplotlib.pyplot as plt


from read import *
params = read_input()

path = "."
name='TEST'

filearr=sorted(glob.glob(path+'/test_PCA_*.pic'))


Vsys0=-50 #system velocity
Kp0=params['Kp']

i = 0
#for i in range(len(filearr)):
Vsysarr, Kparr, CCF,logL=pickle.load(open(filearr[i],'rb'))
Vsys = Vsysarr

CCFarr1=CCF-np.mean(CCF)
Vsysarr=Vsys


rc('axes',linewidth=2)
fig,ax=subplots()

xlimm=-1
ylimm=10

##sigma clipping
from astropy.stats import sigma_clipped_stats, sigma_clip
mean, med, sc = sigma_clipped_stats(CCFarr1,sigma_lower=3.,sigma_upper=3.)
CCFarr1=(CCFarr1-med)/sc
CCFarr1[ylimm:-20,140:xlimm]=0


CCFarr1=CCFarr1[::-1,:]
CCFarr1=CCFarr1-np.mean(CCFarr1)
stdev1=np.std(CCFarr1[ylimm:-20,140:xlimm])
maxx1=(CCFarr1/stdev1).max()
print(maxx1)
loc1=np.where(CCFarr1/stdev1 == maxx1)


cax=ax.imshow(CCFarr1[:,100:]/stdev1, extent=[-50,50, Kparr.min(),Kparr.max()],
aspect="auto",interpolation='none')
cbar=colorbar(cax,ticks=[-4,-2, 0, 2, 4, 6, 8,10, 12, 14, 16, 18])
axvline(x=0,color='white',ls='--',lw=2)
axhline(y=Kp0,color='white',ls='--',lw=2)
xlabel('$\Delta$V$_{sys}$ [km/s]',fontsize=20)
ylabel('K$_{p}$ [km/s]',fontsize=20)
cbar.set_label('Cross-correlation SNR',fontsize=15)
cbar.ax.tick_params(labelsize=15,width=2,length=6)
plt.tick_params(labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
plt.tight_layout()
savefig('Kp_Vsys_CO_0.5.pdf',fmt='pdf')



pdb.set_trace()
##slice at Kp
slicee=CCFarr1[loc1[0][0],]

plot(Vsysarr[100:]+50,slicee[100:]+50,color='r',label='C/O=0.8',linewidth=2)

xlim(-50,50)
xlabel('$\Delta$V$_{sys}$ [km/s]',fontsize=15)
ylabel('CCF',fontsize=15)
axvline(x=0.0,ls='--',color='black')
subplots_adjust(left=0.1, right=0.9, top=0.7, bottom=0.3)
plt.legend(fontsize=15)
plt.tick_params(labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
plt.tight_layout()
savefig('CCF_slice.pdf',fmt='pdf')
show()
close()




pdb.set_trace()



pdb.set_trace()
