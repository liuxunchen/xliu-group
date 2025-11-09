# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from scipy import constants
import math




cBolts = constants.Boltzmann*1E7 # CGS Boltzmann constant unit is erg/K, CGS, c is cm/s 
#print cBolts
cc = constants.speed_of_light*1E2 # CGS
#print cc
cNA = constants.N_A
#print cNA

cGammaD = math.sqrt(2*cBolts*cNA*math.log(2))/cc

for T 
GammaD = cGammaD*math.sqrt(T/Mass)*nu# + laser # eqn (5), 


def load_plif(filename,size):
    plif = open(filename)
    firstline = plif.readline().split(' ')
    x_num,y_num = firstline[3:5]
    delta_x,x0 = firstline[6:8]
    delta_y,y0 = firstline[10:12]
    plif.close()
    x_num = int(x_num)
    y_num = int(y_num)
    x0 = float(x0)
    y0 = float(y0)
    delta_x = float(delta_x)
    delta_y = float(delta_y)
    X = np.arange(x0, x0+(x_num)*delta_x, delta_x)
    Y = np.arange(y0, y0+(y_num)*delta_y, delta_y)   
    
    plif_txt = pd.read_csv(filename, decimal=',', sep='\t', skiprows=1, header=None)
    plif = plif_txt.values
    return x_num,y_num,X,Y,plif

import cv2 as cv2
plif = cv2.imread('Acetone PLIF-PIV-test_20191013_204733_C002H001S0001000001.tif.tif')


# %%
for i in range(10):
    plt.scatter(i,1)
    plt.savefig(str(i)+'.png')
