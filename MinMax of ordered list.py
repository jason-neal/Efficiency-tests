
# coding: utf-8

# # Slicing Ends verse min/max
# 
# If you have an order list of numbers how best to return the min and max values.

# In[ ]:

list min max
list []

np min max
np []


# In[ ]:

# Slicing the first and last elements is faster than np.min(), np.max() if you know that the list is ordered
#min_dwl = wl2[1:]-wl2[:-1]

#%timeit np.min(wl2)                                     ## min/max ~15 µs 
#%timeit np.max(wl2)                                     ## min/max ~15 µs 

#%timeit wl2[0]                                          ## [] ~300ns
#%timeit wl2[-1]                                         ## [] ~300ns

#%timeit [np.min(wl2)-2*min_dwl, np.max(wl2)+2*min_dwl]  ## ~ 70 µs 

#%timeit [wl2[0]-2*min_dwl, wl2[-1]+2*min_dwl]           ## ~ 40 µs 
    


# In[ ]:




# In[ ]:



