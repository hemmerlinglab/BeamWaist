#Libraries
import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt, sin, cos, pi
import matplotlib.pylab as pylab
from scipy.optimize import curve_fit
from scipy import special
from scipy.special import erf
import pandas as pd
import math


#formatting graphs 
pylab.rcParams['figure.figsize'] = 7,7/1.62
pylab.rcParams['figure.autolayout'] = False
pylab.rcParams.update({'axes.labelsize': 20})
pylab.rcParams.update({'xtick.labelsize': 15})
pylab.rcParams.update({'ytick.labelsize': 15})
pylab.rcParams.update({'lines.linewidth': 1.0})
pylab.rcParams.update({'axes.titlesize': 20.0})

pylab.rcParams.update({'ytick.direction': 'in'}) 
pylab.rcParams.update({'xtick.major.size': 7})   
pylab.rcParams.update({'xtick.direction': 'in'}) 
pylab.rcParams.update({'xtick.top': True}) 
pylab.rcParams.update({'xtick.minor.bottom': True}) 
plt.xlabel('Microns',fontsize=16)
plt.ylabel('Voltage', fontsize=16)

#read in beam waist data
# put your csv data file path here
data = pd.read_csv("/home/molecules/Desktop/BeamWaist//03292019_inputBeam8.csv",header=0)

microns = np.array(data['Microns'])
voltage = np.array(data['Voltage'])

# defining the function to fit to
# x is the knife edge distance acrosss the beam
# w is the waist parameter
# p is the total power without the knife edge
# c is the x offset
# d is the voltage offset from background lights
def func(x,w,c):
	return max(voltage)/2*(1-erf(sqrt(2)*(x-c)/w))

# assigning two variables in one return statement
# curve fit function return parameter optimization (popt) and parameter covariance (pcov)
popt, pcov = curve_fit(func,microns,voltage,bounds=([10,10],[500,2000]))
print(popt)
print(pcov)
print('waist (in microns):')
print(popt[0])
print('center')
print(popt[1])

plt.plot(microns,voltage,'b.')
plt.plot(microns, func(microns,*popt),'r')
plt.show()

