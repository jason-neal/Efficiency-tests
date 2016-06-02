
# coding: utf-8

# # Masks not Comprehension lists for Numpy Arrays
# 
# Masks are are more efficient to use numpy arrays than list comprehension.
# Here I will show two places I have improved speed in inherited code.
# 
# The first example is a function that selects a section out of a spectra specified by wavelength range. 

# In[86]:

import numpy as np
from __future__ import division


# In[28]:

def wav_selector(wav, flux, wav_min, wav_max):
    """
    function that returns wavelength and flux within a given range
    """    
    wav_sel = np.array([value for value in wav if(wav_min < value < wav_max)], dtype="float64")
    flux_sel = np.array([value[1] for value in zip(wav,flux) if(wav_min < value[0] < wav_max)], dtype="float64")
    
    return [wav_sel, flux_sel]


# In[29]:

Test_wav = np.linspace(2000,2200,300000)  # nm
Test_flux = np.random.random(len(Test_wav)) # nm
min_wav = 2050
max_wav = 2170

# If you had lists instead
Test_list_wav = list(Test_wav)
Test_list_flux = list(Test_flux)


# In[30]:

# Timing it running
print("wav_selector with numpy inputs")
get_ipython().magic(u'timeit wav_selector(Test_wav, Test_flux, min_wav, max_wav)')
print("\nwav_selector list inputs")
get_ipython().magic(u'timeit wav_selector(Test_list_wav, Test_list_flux, min_wav, max_wav)')


# This is intesting that inputing lists is faster in this function than numpy arrays. Numpy arrays are built to be faster.
# 
# Looking at the code in wav_selector we see it preforms a comphension list and turns it back into a numpy array. 
# 
# It also actively turns the input values into an array even though you could pass it lists.
# 
# I have changed the function to use masks on the numpy arrays and to do list comprehension on lists. This avoids any time spent changing between the different data types.
# 

# In[31]:

# wav_selector using 
def fast_wav_selector(wav, flux, wav_min, wav_max):
    """ Faster Wavelength selector
    If passed lists it will return lists.
    If passed np arrays it will return arrays
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


# In[32]:

print("fast_wav_selector with numpy inputs")
get_ipython().magic(u'timeit fast_wav_selector(Test_wav, Test_flux, min_wav, max_wav)')

print("\nfast_wav_selector list inputs")
get_ipython().magic(u'timeit fast_wav_selector(Test_list_wav, Test_list_flux, min_wav, max_wav)')


# We can see here that the numpy mask is about >100 X faster than the old version.
# 
# Also when you input lists it runs slightly faster ~15% for this case.

# # Convolution Example 

# Here is another example that is part of a convolution to a desired Resolution(R). 
# It gets ran for every wavelength value (wav) in the spectra which can be over 100,000 values. It was a large bottleneck in my research due to the comprehension lists. 
# 
# Lets see the preformance increase with masks.

# In[87]:

# Other function we need
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


# In[96]:

# This is the inner loop 
def convolve(wav, R, wav_extended, flux_extended, FWHM_lim):
        FWHM = wav/R
        
        indexes = [ i for i in range(len(wav_extended)) if ((wav - FWHM_lim*FWHM) < wav_extended[i] < (wav + FWHM_lim*FWHM))]

        flux_2convolve = flux_extended[indexes[0]:indexes[-1]+1]
        IP = unitary_Gauss(wav_extended[indexes[0]:indexes[-1]+1], wav, FWHM)
        
        val = np.sum(IP*flux_2convolve) 
        unitary_val = np.sum(IP*np.ones_like(flux_2convolve))  # Effect of convolution onUnitary. For changing number of points
        
        return val/unitary_val


# In[97]:

# This is the improved version with masks
def fast_convolve(wav, R, wav_extended, flux_extended, FWHM_lim):
    FWHM = wav/R
    # Numpy mask
    index_mask = (wav_extended > (wav - FWHM_lim*FWHM)) &  (wav_extended < (wav + FWHM_lim*FWHM))
    
    flux_2convolve = flux_extended[index_mask]
    IP = unitary_Gauss(wav_extended[index_mask], wav, FWHM)
    
    val = np.sum(IP*flux_2convolve) 
    unitary_val = np.sum(IP*np.ones_like(flux_2convolve))  # Effect of convolution onUnitary. For changing number of points
        
    return val/unitary_val


# In[109]:

# some Random test spectra
wav_extended = np.linspace(2020,2040,10000)
FWHM_lim=5
R = 500000

flux_extended = np.random.random(len(wav_extended)) + np.ones_like(wav_extended)

wav = 2029. # one wave length value


# In[110]:

print("Convolution with comprehension list and indices")
get_ipython().magic(u'timeit convolve(wav, R, wav_extended, flux_extended, FWHM_lim)')
print("Faster convolution with masks")
get_ipython().magic(u'timeit fast_convolve(wav, R, wav_extended, flux_extended, FWHM_lim)')


# That was for one time though the loop with a large differnce in speed. 
# 
# Now lets loop this over all the wavelength values and see the change in time of result
# 
# I will also compare the affect of saving the result from each loop as a list or in a pre allocated numpy array.

# In[112]:

get_ipython().run_cell_magic(u'timeit', u'', u'# Preallocating a numpy array is also faster than appending to a list\nflux_conv_res = np.empty_like(wav_extended)\n#print("Convolution with comprehension list and indices")\nfor i, wav in enumerate(wav_extended):\n    flux_conv_res[i] = convolve(wav, R, wav_extended, flux_extended, FWHM_lim)')


# In[113]:

get_ipython().run_cell_magic(u'timeit', u'', u'# Preallocating a numpy array is also faster than appending to a list \nflux_conv_res = np.empty_like(wav_extended)\n#print("Faster convolution with masks")\nfor i, wav in enumerate(wav_extended):\n    flux_conv_res = fast_convolve(wav, R, wav_extended, flux_extended, FWHM_lim)')


# 47s for comprehension verse 300ms for masks is around 150 times faster
# 
# Shown below it is also much faster to preallocate the numpy array and then fill it up rather than to appending to the list every time.

# In[ ]:

get_ipython().run_cell_magic(u'timeit', u'', u'# Preallocating a numpy array is also faster than appending to a list in each loop but I won\'t show that now.\nflux_conv_res = []\n#print("Convolution with comprehension list and indices")\nfor i, wav in enumerate(wav_extended):\n    flux_conv_res.append(convolve(wav, R, wav_extended, flux_extended, FWHM_lim)) ')


# In[ ]:

get_ipython().run_cell_magic(u'timeit', u'', u'flux_conv_res = []\n#print("Convolution with comprehension list and indices")\nfor i, wav in enumerate(wav_extended):\n    flux_conv_res.append(fast_convolve(wav, R, wav_extended, flux_extended, FWHM_lim)) ')


# In[ ]:



