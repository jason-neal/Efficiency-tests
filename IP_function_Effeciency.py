
# coding: utf-8

# # Gaussian IP Efficencies:
# 
# Invesitgating improving the unitary gauss function to be faster
# 
# 

# In[1]:

import numpy as np
def unitary_Gauss(x, center, FWHM):
    """
    Gaussian_function of area=1
	
	p[0] = A;
	p[1] = mean;
	p[2] = FWHM;
    """
    
    sigma = np.abs(FWHM) /( 2 * np.sqrt(2 * np.log(2)) )
    Amp = 1.0 / (sigma*np.sqrt(2*np.pi))
    tau = -((x - center)**2) / (2*(sigma**2))
    result = Amp * np.exp( tau )
    
    return result


# In[12]:

x = np.arange(10, 100000)
center = 20400
FWHM = 50 


# In[13]:

get_ipython().run_cell_magic(u'timeit', u'', u'for wav in x:\n   result = unitary_Gauss(x, wav, FWHM)\n    ')


# In[ ]:

get_ipython().run_cell_magic(u'timeit', u'', u'for center in x:\n    # in this case doing the sigma calulation twice\n    result = (1.0 / ((np.abs(FWHM) /( 2 * np.sqrt(2 * np.log(2)) ))*np.sqrt(2*np.pi))) * np.exp( -((x - center)**2) / (2*((np.abs(FWHM) /( 2 * np.sqrt(2 * np.log(2)) ))**2)) )')


# In[8]:

get_ipython().magic(u'load_ext line_profiler')
get_ipython().magic(u'lprun -f unitary_Gauss unitary_Gauss(x, c, fwhm)')


# In[17]:

get_ipython().run_cell_magic(u'timeit', u' ', u'# single line gaussian\nfor center in x:\n    (1.0 / ((np.abs(FWHM) /( 2 * np.sqrt(2 * np.log(2)) ))*np.sqrt(2*np.pi))) * np.exp( -((x - center)**2) / (2*((np.abs(FWHM) /( 2 * np.sqrt(2 * np.log(2)) ))**2)) )')


# In[ ]:

get_ipython().run_cell_magic(u'timeit', u' ', u'# single line gaussian\nsigma = np.abs(FWHM) /( 2 * np.sqrt(2 * np.log(2)) );\nfor center in x:\n    result = np.exp(-((x - center)**2) / (2*(sigma**2)))  / (sigma*np.sqrt(2*np.pi))')


# # Split up gausian calculation
# 
# Amplitude and sigma outside of loop to stop repition of calculations

# In[14]:

def unitary_gaussian_constants(FWHM):
    sigma = np.abs(FWHM) / ( 2 * np.sqrt(2 * np.log(2)) )
    Amp = 1.0 / (sigma * np.sqrt(2 * np.pi))
    two_sigma_squared = 2 * (sigma**2)
    return Amp, two_sigma_squared

def unitary_gaussian_exp_part(x, center, Amp, two_sigma_squared):
    tau = -((x - center)**2) / two_sigma_squared
    return Amp * np.exp( tau )


# In[ ]:




# In[16]:

get_ipython().run_cell_magic(u'timeit', u'', u'A, sigma22 = unitary_gaussian_constants(FWHM)\nfor center in x:\n    result = unitary_gaussian_exp_part(x, center, A, sigma22)')


# In[17]:

get_ipython().run_cell_magic(u'timeit', u'', u'for center in x:\n    A, sigma22 = unitary_gaussian_constants(FWHM)\n    result = unitary_gaussian_exp_part(x, center, A, sigma22)')


# In[18]:

len(x)


# In[24]:

get_ipython().magic(u'timeit A, sigma22 = unitary_gaussian_constants(FWHM)   # cached result ?')
A, sigma22 = unitary_gaussian_constants(FWHM)
get_ipython().magic(u'timeit result = unitary_gaussian_exp_part(x, center, A, sigma22)')


# In[21]:

6.1 * 99990 /1e6
#seconds for constant


# In[25]:

1.7 *99990/1e3
   # seconds for exp


# In[ ]:



