#!/usr/bin/env python
# coding: utf-8

# In[3]:


import bifacialvf


# In[7]:


import numpy as np


# In[4]:


# Lats 20 to 60
# Bellyham highest city in continental US : 48.7519° N, 122.4787° W
# Hawwaii: 19.8987° N, 155.6659° W
# Point Barrow is the northernmost point of Alaska at 71° 23' 25"


# In[19]:


lats = [0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
tilts = [20, 20, 20, 20, 25, 30, 35, 40, 40, 40, 40, 40]
lon = -105.2211 # ° W


# In[12]:


y = 2


# In[23]:


for ii in range (0, len(lats)):
    DD = bifacialvf.vf.rowSpacing(beta = tilts[ii],
                                  sazm=180, lat = lats[ii],
                                  lng = lon,
                                  tz = -7,
                                  hour = 9,
                                  minute = 0.0)

    if (DD <= 0) or (DD > 3.725):
        DD = 3.725
        print("Cannot find ideal pitch for location, setting D to 3.725")

    normalized_pitch = DD + np.cos(np.round(tilts[ii]) / 180.0 * np.pi)
    pitch_unnorm = np.round(normalized_pitch*y,2)
    DD_unnorm = np.round(DD*y, 2)
    GCR = np.round(2/pitch_unnorm,2)
    print(lats[ii], tilts[ii], pitch_unnorm, DD_unnorm, GCR)


# For reference, BARN HSAT has a GCR of 0.35
