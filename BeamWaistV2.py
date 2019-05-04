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
import scipy.optimize as optimization

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
data = pd.read_csv("/home/molecules/Desktop/BeamWaist//03292019_inputBeam.csv",header=0)

microns = np.array(data['Microns'])
voltage = np.array(data['Voltage'])
x0 =[0.0,0.0,0.0,0.0,0.0]
#sigma = numpy.array([1.0,1.0,1.0,1.0,1.0,1.0])

# defining the function to fit to
# x is the knife edge distance acrosss the beam
# w is the waist parameter
# p is the total power without the knife edge
# c is the x offset
def func(x,w,p,c):
	return p/2*(1-erf(sqrt(2)*(x-c)/w))



# linspace params
lowX    = microns[0]
highX   = microns[len(microns)-1]
datapts = 2000
newMicrons = np.linspace(lowX,highX,datapts)
#newVoltage = optimization.curve_fit(poly, microns, voltage, x0)


popt, pcov = curve_fit(func,newMicrons,voltage)#,bounds=([10,0,10],[60,5,800]))
print('waist (in microns):')
print(popt[0])

#print(newVoltage)

# Find the Maximum Gradient, this way we can shift it to have this be at zero
#gradx = np.array([])
#grady = np.array([])
#for i in range(0,len(newMicrons)):
#	gx = newMicrons[i+1]-newMicrons[i-1]
#	gy = newVoltage[i+1]-newVoltage[i-1]
#	gradx = np.append(gradx,gx)
#	grady = np.append(grady,gy)

#grad      = grady/gradx
#maxgrad   = max(grad)
#max_index = grad.index(maxgrad)
#print(max_index)

# assigning two variables in one return statement
# curve fit function return parameter optimization (popt) and parameter covariance (pcov)
#popt, pcov = curve_fit(func,newMicrons,voltage,bounds=([10,0,10],[60,5,800]))
#print(popt)
#print(pcov)
#print('waist (in microns):')
#print(popt[0])

plt.plot(microns,voltage, 'b.')
plt.plot(newMicrons, func(microns,*popt),'r')
plt.show()


