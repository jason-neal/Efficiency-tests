
# coding: utf-8

# # Testing Python Efficiency:
# 
# To determine fasted methods to apply sections of code.
# 
# I.e. test using numpy arrays indexing, broadcasting etc comapred to my written functions and compreshnsion lists.
# 
# Also test overhead of converting to and from lists.

# In[1]:

import numpy as np


# ## Wavelength selection codes
# Fastest way to select wavelength and flux from a spectra.
# 
# Experimenting with np array mask for quicker wavelenght selection than mine or pedros codes. Created fast_wav_selector 29/5/2015

# In[2]:


def wav_selector(wav, flux, wav_min, wav_max):
    """
    function that returns wavelength and flux withn a giving range
    """    
    wav_sel = np.array([value for value in wav if(wav_min < value < wav_max)], dtype="float64")
    flux_sel = np.array([value[1] for value in zip(wav,flux) if(wav_min < value[0] < wav_max)], dtype="float64")
    
    return [wav_sel, flux_sel]

def slice_spectra(wl, spectrum, low, high):
    """ Extract a section of a spectrum between wavelength bounds.
    This was faster than wav_selector but it only works on numpy arrays.
        """
    #print("lower bound", low)
    #print("upper bound", high)
    map1 = wl > low
    map2 = wl < high
    wl_sec = wl[map1*map2]
    spectrum_sec = spectrum[map1*map2]   
    return wl_sec, spectrum_sec 


def fast_wav_selector(wav, flux, wav_min, wav_max):
    """ Faster Wavelenght selector
    
    If passed lists it will return lists.
    If passed np arrays it will return arrays
    
    Fastest is using np.ndarrays
    fast_wav_selector ~1000-2000 * quicker than wav_selector
    """
    
    if isinstance(wav, list): # if passed lists
          wav_sel = [value for value in wav if(wav_min < value < wav_max)]
          flux_sel = [value[1] for value in zip(wav,flux) if(wav_min < value[0] < wav_max)]
    elif isinstance(wav, np.ndarray):
        # Super Fast masking with numpy
        mask = (wav > wav_min) & (wav < wav_max)
        wav_sel = wav[mask]
        flux_sel = flux[mask]
    else:
          raise TypeError("Unsupported input wav type")
    return [wav_sel, flux_sel]


# In[3]:

# Create some data to test with
wl = list(range(2000, 30000))
wl_np = np.arange(2000, 30000)

flux = [x**2 +5 for x in wl] 
flux_np = wl_np**2 +5


# In[95]:

# Make some data as lists
print("Make list array")
get_ipython().magic('timeit wl = list(range(2000, 30000))')
print("Make np array")
get_ipython().magic('timeit wl_np = np.arange(2000, 30000)')


# In[98]:

get_ipython().magic('timeit list(wl_np)')

get_ipython().magic('timeit np.array(wl)')

get_ipython().magic('timeit np.array(wl_np)   # very quick as does nothing*')


# In[ ]:

print("list comprehension")
get_ipython().magic('timeit flux = [x**2 +5 for x in wl]')
print("np calculation")
get_ipython().magic('timeit flux_np = wl_np**2 +5')
# comprehension list much slower then numpy arrary but want to start with lists

flux = [x**2 +5 for x in wl] 
flux_np = wl_np**2 +5


# In[ ]:

print(" min of list")
get_ipython().magic('timeit min(wl)')
print("np min of list")
get_ipython().magic('timeit np.min(wl)')

print(" min of np array")
get_ipython().magic('timeit min(wl_np)')
print("np min of np array")
get_ipython().magic('timeit np.min(wl_np)')
# Timing time to turn list 


# In[ ]:

# Time to convert between list and array
print("list to array")
get_ipython().magic('timeit np.array(wl)')
print("list to array")
get_ipython().magic('timeit x = np.array(wl)')
print("array to list")
get_ipython().magic('timeit list(wl_np)')
get_ipython().magic('timeit y = list(wl_np)')


# In[48]:

# time to turn an array into and array
get_ipython().magic('timeit np.array(wl_np)')
wl_np2 = np.array(wl_np)
print(type(wl_np2))


# In[ ]:

# Copied from H20 Scaling
get_ipython().magic('timeit slice_spectra(tapas_h20_data[0], tapas_h20_data[1], np.min(wl2), np.max(wl2))')
get_ipython().magic('timeit wav_selector(tapas_h20_data[0], tapas_h20_data[1], np.min(wl2), np.max(wl2))')
get_ipython().magic('timeit fast_wav_selector(tapas_h20_data[0], tapas_h20_data[1], np.min(wl2), np.max(wl2))')
# fast_wav_selector ~1000-2000 * quicker
#1 loop, best of 3: 1.88 s per loop
#1000 loops, best of 3: 1.25 ms per loop
#1 loop, best of 3: 1.9 s per loop
#1000 loops, best of 3: 831 µs per loop
(1.88*1000)/(1.25)  # 


# In[7]:

# Simliar time to split or not split out tuple/list returned
get_ipython().magic('timeit t1 = fast_wav_selector(wl_np, flux_np, 6000, 24000)')

get_ipython().magic('timeit t2, t3 = fast_wav_selector(wl_np, flux_np, 6000, 24000)')


# In[15]:

## Time difference between my slice spectra and pedros wave selector
#print("Slice_spectra with lists")
#%timeit slice_spectra(wl, flux, 6000, 24000)
print("Wav_selector with lists")
get_ipython().magic('timeit wav_selector(wl, flux, 6000, 24000)')
print("fast Wav_selecor with list")
get_ipython().magic('timeit fast_wav_selector(wl, flux, 6000, 24000)')
print("slice_spectra with array")
get_ipython().magic('timeit slice_spectra(wl_np, flux_np, 6000, 24000)')
print("Wav_selecor with array")
get_ipython().magic('timeit wav_selector(wl_np, flux_np, 6000, 24000)')
print("fast Wav_selecor_np with array")
get_ipython().magic('timeit fast_wav_selector(wl_np, flux_np, 6000, 24000)')


# In[ ]:




# In[10]:

# Testing line profiling

get_ipython().magic('load_ext line_profiler')



# In[85]:

get_ipython().magic('lprun -f wav_selector wav_selector(wl_np, flux_np, 6000, 24000)')


# In[13]:

get_ipython().magic('lprun -f fast_wav_selector fast_wav_selector(wl_np, flux_np, 6000, 24000)')


# In[11]:

get_ipython().magic('lprun -f fast_wav_selector fast_wav_selector(wl, flux, 6000, 24000)')


# In[12]:

get_ipython().magic('lprun -f slice_spectra slice_spectra(wl_np, flux_np, 6000, 24000)')


# # Unitary Gauss Fucntion

# In[1]:


def unitary_Gauss(x, center, FWHM):
    """
    Gaussian_function of area=1
	
	p[0] = A;
	p[1] = mean;
	p[2] = FWHM;
    """
    
    sigma = np.abs(FWHM) /( 2 * np.sqrt(2 * np.log(2)) );
    Amp = 1.0 / (sigma*np.sqrt(2*np.pi))
    tau = -((x - center)**2) / (2*(sigma**2))
    result = Amp * np.exp( tau );
    
    return result


# In[ ]:


x = np.arange(-50,50)


# Unitatry Gauss function doesn't need speed up as it is all in numpy.

# In[4]:

get_ipython().magic('timeit unitary_Gauss(np.arange(-50,50), 1, 5)')
#%timeit unitary_Gauss(xlist, 1, 5) # List does not work


# In[ ]:




# In[4]:

# Main convolution loop

def convolve(wav, R, wav_extended, flux_extended, FWHM_lim):
        # select all values such that they are within the FWHM limits
        FWHM = wav/R
        indexes = [ i for i in range(len(wav_extended)) if ((wav - FWHM_lim*FWHM) < wav_extended[i] < (wav + FWHM_lim*FWHM))]
        flux_2convolve = flux_extended[indexes[0]:indexes[-1]+1]
        IP = unitary_Gauss(wav_extended[indexes[0]:indexes[-1]+1], wav, FWHM)
        val = np.sum(IP*flux_2convolve) 
        unitary_val = np.sum(IP*np.ones_like(flux_2convolve))  # Effect of convolution onUnitary. For changing number of points
        return val/unitary_val

def fast_convolve(wav, R, wav_extended, flux_extended, FWHM_lim):
    FWHM = wav/R
    
    index_mask = (wav_extended > (wav - FWHM_lim*FWHM)) &  (wav_extended < (wav + FWHM_lim*FWHM))
    
    flux_2convolve = flux_extended[index_mask]
    IP = unitary_Gauss(wav_extended[index_mask], wav, FWHM)
    
    val = np.sum(IP*flux_2convolve) 
    unitary_val = np.sum(IP*np.ones_like(flux_2convolve))  # Effect of convolution onUnitary. For changing number of points
        
    return val/unitary_val


# In[5]:

wav_extended = np.arange(-2000,2000)
flux_extended = np.ones_like(wave_extended)
fwhm_lim=5

wav = 60
R=50000


# In[6]:

get_ipython().magic('timeit convolve(wav, R, wav_extended, flux_extended, fwhm_lim)')
get_ipython().magic('timeit fast_convolve(wav, R, wav_extended, flux_extended, fwhm_lim)')


# In[7]:

49/.2
# ~250 x increase


# In[8]:

a = convolve(wav, R, wav_extended, flux_extended, fwhm_lim)
b = fast_convolve(wav, R, wav_extended, flux_extended, fwhm_lim)
np.isclose(a,b)


# # Test convolution fix

# In[9]:

import numpy as np
import time
import datetime

def wav_selector(wav, flux, wav_min, wav_max):
    """
    function that returns wavelength and flux withn a giving range
    """    
    wav_sel = np.array([value for value in wav if(wav_min < value < wav_max)], dtype="float64")
    flux_sel = np.array([value[1] for value in zip(wav,flux) if(wav_min < value[0] < wav_max)], dtype="float64")
    
    return [wav_sel, flux_sel]



def unitary_Gauss(x, center, FWHM):
    """
    Gaussian_function of area=1
	
	p[0] = A;
	p[1] = mean;
	p[2] = FWHM;
    """
    
    sigma = np.abs(FWHM) /( 2 * np.sqrt(2 * np.log(2)) );
    Amp = 1.0 / (sigma*np.sqrt(2*np.pi))
    tau = -((x - center)**2) / (2*(sigma**2))
    result = Amp * np.exp( tau );
    
    return result


def chip_selector(wav, flux, chip):
    chip = str(chip)
    if(chip in ["ALL", "all", "","0"]):
        chipmin = float(hdr1["HIERARCH ESO INS WLEN STRT1"])  # Wavelength start on detector [nm]
        chipmax = float(hdr1["HIERARCH ESO INS WLEN END4"])   # Wavelength end on detector [nm]
        #return [wav, flux]
    elif(chip == "1"):
        chipmin = float(hdr1["HIERARCH ESO INS WLEN STRT1"])  # Wavelength start on detector [nm]
        chipmax = float(hdr1["HIERARCH ESO INS WLEN END1"])   # Wavelength end on detector [nm]
    elif(chip == "2"):
        chipmin = float(hdr1["HIERARCH ESO INS WLEN STRT2"])  # Wavelength start on detector [nm]
        chipmax = float(hdr1["HIERARCH ESO INS WLEN END2"])   # Wavelength end on detector [nm]
    elif(chip == "3"):   
        chipmin = float(hdr1["HIERARCH ESO INS WLEN STRT3"])  # Wavelength start on detector [nm]
        chipmax = float(hdr1["HIERARCH ESO INS WLEN END3"])   # Wavelength end on detector [nm]
    elif(chip == "4"):   
        chipmin = float(hdr1["HIERARCH ESO INS WLEN STRT4"])  # Wavelength start on detector [nm]
        chipmax = float(hdr1["HIERARCH ESO INS WLEN END4"])   # Wavelength end on detector [nm]
    elif(chip == "Joblib_small"):   
        chipmin = float(2118)  # Wavelength start on detector [nm]
        chipmax = float(2119)  # Wavelength end on detector [nm]
    elif(chip == "Joblib_large"):   
        chipmin = float(2149)  # Wavelength start on detector [nm]
        chipmax = float(2157)  # Wavelength end on detector [nm]
    else:
        print("Unrecognized chip tag.")
        exit()
    
    #select values form the chip  
    wav_chip, flux_chip = wav_selector(wav, flux, chipmin, chipmax)
    
    return [wav_chip, flux_chip]


# In[18]:

def fast_convolution(wav, flux, chip, R, FWHM_lim=5.0, n_jobs=-1):
    """Convolution code adapted from pedros code"""
    
    wav_chip, flux_chip = chip_selector(wav, flux, chip)
    #we need to calculate the FWHM at this value in order to set the starting point for the convolution
    
    #print(wav_chip)
    #print(flux_chip)
    FWHM_min = wav_chip[0]/R    #FWHM at the extremes of vector
    FWHM_max = wav_chip[-1]/R       
    
    #wide wavelength bin for the resolution_convolution
    wav_extended, flux_extended = wav_selector(wav, flux, wav_chip[0]-FWHM_lim*FWHM_min, wav_chip[-1]+FWHM_lim*FWHM_max) 
    wav_extended = np.array(wav_extended, dtype="float64")
    flux_extended = np.array(flux_extended, dtype="float64")
    
    print("Starting the fast Resolution convolution...")
    # Pre allocate space
    #flux_conv_res = []
    flux_conv_res = np.empty_like(wav_chip, dtype="float64")
    counter = 0 
    for n, wav in enumerate(wav_chip):
        result = fast_convolve(wav, R, wav_extended, flux_extended, FWHM_lim)
        #print(result)
        #if(len(flux_conv_res)%(len(wav_chip)//100 ) == 0):
        if(n%(len(wav_chip)//100 ) == 0):
            counter = counter+1
            print("Resolution Convolution at {}%%...".format(counter))
        flux_conv_res[n] = result
    flux_conv_res = np.array(result, dtype="float64")
    print("Done.\n")
    

    return [wav_chip, flux_conv_res ]

def slow_convolution(wav, flux, chip, R, FWHM_lim=5.0, n_jobs=-1):
    """Convolution code adapted from pedros code"""
    
    wav_chip, flux_chip = chip_selector(wav, flux, chip)
    #we need to calculate the FWHM at this value in order to set the starting point for the convolution
    
    #print(wav_chip)
    #print(flux_chip)
    FWHM_min = wav_chip[0]/R    #FWHM at the extremes of vector
    FWHM_max = wav_chip[-1]/R       
    
    #wide wavelength bin for the resolution_convolution
    wav_extended, flux_extended = wav_selector(wav, flux, wav_chip[0]-FWHM_lim*FWHM_min, wav_chip[-1]+FWHM_lim*FWHM_max) 
    wav_extended = np.array(wav_extended, dtype="float64")
    flux_extended = np.array(flux_extended, dtype="float64")
    
    print("Starting the fast Resolution convolution...")
    # Pre allocate space
    #flux_conv_res = []
    flux_conv_res = np.empty_like(wav_chip, dtype="float64")
    counter = 0 
    for n, wav in enumerate(wav_chip):
        result = convolve(wav, R, wav_extended, flux_extended, FWHM_lim)
        #print(result)
        #if(len(flux_conv_res)%(len(wav_chip)//100 ) == 0):
        if(n%(len(wav_chip)//100 ) == 0):
            counter = counter+1
            print("Resolution Convolution at {}%%...".format(counter))
        flux_conv_res[n] = result
    flux_conv_res = np.array(result, dtype="float64")
    print("Done.\n")
    

    return [wav_chip, flux_conv_res ]


# In[16]:

### Test h20 convolution
from astropy.io import fits
chip1 = "H20_scaling/CRIRE.2012-04-07T00-08-29.976_1.nod.ms.norm.sum.wavecal.fits"

Obs1 = fits.getdata(chip1) 
hdr1 = fits.getheader(chip1) 

import Obtain_Telluric as obt
tapas_h20 = "H20_scaling/tapas_2012-04-07T00-24-03_ReqId_12_No_Ifunction_barydone-NO.ipac"

tapas_h20_data, tapas_h20_hdr = obt.load_telluric("", tapas_h20)
tapas_h20_airmass = float(tapas_h20_hdr["airmass"])

print("Telluric Airmass ", tapas_h20_airmass)
try:
    tapas_h20_respower = int(float((tapas_h20_hdr["respower"])))
except:
    tapas_h20_respower = "Nan"
print("Telluric Resolution Power =", tapas_h20_respower)






# In[20]:


start = time.time()

fast_x,fast_y = fast_convolution(tapas_h20_data[0], tapas_h20_data[1], "1", 50000, FWHM_lim=5.0)
  
done = time.time()
elapsed = done - start
print("Convolution time for fast convolution = ", elapsed)



# In[19]:


start = time.time()

slow_x, slow_y = slow_convolution(tapas_h20_data[0], tapas_h20_data[1], "1", 50000, FWHM_lim=5.0)
  
done = time.time()
elapsed = done - start
print("Convolution time for fast convolution = ", elapsed)


# In[21]:

937/8


# In[62]:

#%%timeit
# Test affect of list append verse np assignment with enumerate
list_store = []
for n, i in enumerate(range(10000000)):
    list_store.append(i)



# In[63]:

len(list_store)


# In[41]:

get_ipython().run_cell_magic('timeit', '', 'np_store = np.empty(10000000)\nfor n, i in enumerate(range(10000000)):\n    np_store[n] = i\n\n')


# In[48]:

get_ipython().run_cell_magic('timeit', '', 'np_store = np.empty_like(list_store)\nfor n, i in enumerate(range(10000000)):\n    np_store[n] = i')


# In[56]:

# np empty is faster than 
# np empty_like is faster on arrays than lists
# np empty_like on an arary is faster than generating a new empty array of same size
get_ipython().magic('timeit np.empty_like(list_store)')
get_ipython().magic('timeit np.arange(1000000)')
a=np.arange(1000000)
get_ipython().magic('timeit np.empty_like(a)')



# In[65]:

# Get length of list verse ndarray
get_ipython().magic('timeit len(list_store)')
get_ipython().magic('timeit len(a)')
print((len(a)))
print((len(list_store)))
get_ipython().magic('timeit np.empty(len(a))')
get_ipython().magic('timeit np.empty(len(list_store))')

# For an unknown reason np.empty is 6* slower when given len(list) rather than len(ndarray)? The len() times are similar


# In general I think when working with data it is best to transform to ndarrays and compute np functions on them for faster results.
# 

# # Insturment profile calculaltion
# 
#  cache values?
# 
#  calculate in own loop?
# 
#  broadcasting of some form?

# In[ ]:







