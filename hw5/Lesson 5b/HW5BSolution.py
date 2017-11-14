# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:15:59 2017

@author: Matthew Bowers
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# use %reset -f to clear variables
# Read in data
data = np.loadtxt("LIDAR_100ms_Wander.csv", delimiter=",")

# Append all raw data measurements over 40m to be 40m
for i in range(len(data)):
    if data[i] >= 4000:
        data[i] = 4000

#set cj's and other initial parameters
c=[0.25, 0.25, 0.25, 0.25]
#for i in range(len(c)):
#    print (c[i])
delta=5
lamb=1/2586.876986
zmin=5
zmax=4000
mean=3216.194
stdv=12.62869

#Expectation and Maximization steps
#initialize r vectors
r0 = np.zeros(data.size)
r1 = np.zeros(data.size)
r2 = np.zeros(data.size)
r3 = np.zeros(data.size)
cnt=0
#iteration of finding parameters
while True:
    #keeping a count
    cnt=cnt+1
    
    #set current paramters as old parameters
    cold=c
    lambold=lamb
    stdvold=stdv
    
    #Find rij's
    for i in range(len(data)):
        if data[i] >= zmax:
            # above max range
            r0[i]=c[0]*0
            r1[i]=c[1]*0
            r2[i]=c[2]*0
            r3[i]=(c[3]/delta)/(c[3]/delta)
        elif data[i] <= zmin:
            # below min range
            r0[i]=(c[0]/delta)/((c[0]/delta)+c[1]*0+c[2]*0+c[3]*0)
            r1[i]=c[1]*0
            r2[i]=c[2]*0
            r3[i]=c[3]*0
        elif zmin < data[i] < mean:
            # if the objects and wall
            r0[i]=0
            r1[i]=c[1]*(np.power((2*math.pi*stdv),-0.5)*(np.exp((np.square(data[i]-mean))*(-0.5/stdv))))/(c[1]*(np.power((2*math.pi*stdv),-0.5)*(np.exp((np.square(data[i]-mean))*(-0.5/stdv))))+c[2]*lamb*(np.exp(-1*lamb*data[i])))
            r2[i]=c[2]*lamb*(np.exp(-1*lamb*data[i]))/(c[2]*lamb*(np.exp(-1*lamb*data[i]))+c[1]*(np.power((2*math.pi*stdv),-0.5)*(np.exp((np.square(data[i]-mean))*(-0.5/stdv)))))
            r3[i]=0
        elif mean <= data[i] < (zmax-delta):
            # if the wall
            r0[i]=0
            r1[i]=1
            r2[i]=0
            r3[i]=0
              
    #Set new cj's          
    c[0]=(np.sum(r0))/(len(data))
    c[1]=(np.sum(r1))/(len(data))
    c[2]=(np.sum(r2))/(len(data))
    c[3]=(np.sum(r3))/(len(data))
        
    #Set new stdv
    step1=0          
    for i in range(len(data)):
        step1=step1+r1[i]*(data[i]-mean)**2
    stdv=step1/(np.sum(r1))                  
        
    #Set new lamb
    step2=0           
    for i in range(len(data)):
        step2=step2+r2[i]*data[i]
    lamb=1/(step2/(np.sum(r2)))    
    
    # breaking loop if too many iterations
    if cnt>1000:
        print ('error')
        break
    #breaking loop if done bye criteria
    elif abs(c[0]-cold[0])<0.01:
        if abs(c[1]-cold[1])<0.01:
            if abs(c[2]-cold[2])<0.01:
                if abs(c[3]-cold[3])<0.01:
                    if abs(stdv-stdvold)<0.01:
                        if abs(lamb-lambold)<0.01:
                            break
                        
# Print final c_j's,σ^2,and λ 
print (lamb)
print (stdv)
print (c[0])
print (c[1])
print (c[2])
print (c[3])

#create z vector
z = np.zeros(4001)
for i in range(len(z)):
    z[i]=i    

#create final mixed distribution model plot vector p
p = np.zeros(4001)
for i in range(len(z)):
    if z[i] >= zmax:
        # above max range
        p[i]=c[3]/delta
    elif z[i] <= zmin:
        # below min range
        p[i]=c[0]/delta
    elif zmin < z[i] < mean:
        # if the objects and wall
        p[i]=c[1]*(np.power((2*math.pi*stdv),-0.5)*(np.exp((np.square(z[i]-mean))*(-0.5/stdv))))+c[2]*lamb*(np.exp(-1*lamb*z[i]))
    elif mean <= z[i] < (zmax-delta):
        # if the wall
        p[i]=c[1]*(np.power((2*math.pi*stdv),-0.5)*(np.exp((np.square(z[i]-mean))*(-0.5/stdv))))
    elif (zmax-delta) <= z[i] < zmax:
        # if the wall and out of range
        p[i]=c[1]*(np.power((2*math.pi*stdv),-0.5)*(np.exp((np.square(z[i]-mean))*(-0.5/stdv))))+c[3]*(1/delta)   

#plot graph
plt.plot(z,  p, 'r--')
plt.show()