
# coding: utf-8

# # Isinstance() before Numpy arraying:
# 
# This is to test if it is more efficient to np.array() things all the time or to first check if they are already np.ndarrays.
# 
# Jason Neal
# 31/5/2016

# In[32]:

import numpy as np

# Make up some random data
large_test_numpy = np.array(np.random.randn(100000), dtype="float64")
small_test_numpy = np.array(np.random.randn(10), dtype="float64")

large_test_list = list(large_test_numpy)
small_test_list = list(small_test_numpy)


# ### Timing calls
# From the following cells an isinstance call is around an order of magnitude faster than a np.array call on a small np.ndarray and around 3 orders of magnitude for a large numpy array. A np.array call on a np.ndarray is order of magnitudes faster then a np.array call on a large list.
# 
# Strangely a negative isinstace response is twice as long as a positive response.
# 

# In[33]:

# Time Small lists and arrays
print("Positive isinstance call:")
get_ipython().magic('timeit isinstance(small_test_numpy, np.ndarray)')
print("\nNegative isinstance call:")
get_ipython().magic('timeit isinstance(small_test_list, np.ndarray)')

print("\nSmall numpy array -> numpy array:")
get_ipython().magic('timeit np.array(small_test_numpy, dtype="float64")')
print("\nSmall list  -> numpy array:")
get_ipython().magic('timeit np.array(small_test_list, dtype="float64")')


# In[34]:

# Time larger lsits and arrays
print("Positive isinstance call:")
get_ipython().magic('timeit isinstance(large_test_numpy, np.ndarray)')
print("\nNegative isinstance call:")
get_ipython().magic('timeit isinstance(large_test_list, np.ndarray)')

print("\nLarge numpy array -> numpy array:")
get_ipython().magic('timeit np.array(large_test_numpy, dtype="float64")')
print("\nLarge list  -> numpy array:")
get_ipython().magic('timeit np.array(large_test_list, dtype="float64")')



# In[35]:

get_ipython().run_cell_magic('timeit', '', 'if not isinstance(small_test_numpy, np.ndarray):\n    np.array(small_test_numpy, dtype="float64") ')


# In[36]:

get_ipython().run_cell_magic('timeit', '', 'if not isinstance(large_test_numpy, np.ndarray):\n    np.array(large_test_numpy, dtype="float64") ')


# In[37]:

get_ipython().run_cell_magic('timeit', '', 'if type(small_test_list) is not np.ndarray:\n    np.array(small_test_list, dtype="float64") ')


# In[38]:

get_ipython().run_cell_magic('timeit', '', 'if type(large_test_list) is not np.ndarray:\n    np.array(large_test_list, dtype="float64") ')


# ## Conclusion:
# 
# If want to make sure that you value is a numpy array instead of just calling np.array on the object first do an isinstance call to check that it is not an numpy array already for a slight efficiency gain 10-100 microseconds per instance.
# 
# #### Note:
# Not suitable if you are wanting to change the type of numpy array. I.e float32, float64, int. This should probably be done in the begining.
