# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 15:31:43 2021

@author: Mohammad
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

plt.figure(1)

plt.subplot(3,1,1)
data = pd.read_csv("loss_history_test.csv")
plt.plot(data[0:])
plt.grid()


plt.subplot(3,1,2)
data = pd.read_csv("loss_history.csv") 
plt.plot(data[0:])
# plt.ylim([0.008,0.04])
plt.grid()


plt.subplot(3,1,3)
data = pd.read_csv("loss_counting_test.csv") 
plt.plot(data[0:])
# plt.ylim([0.008,0.04])
plt.grid()