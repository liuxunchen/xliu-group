# def hitran: Calculate Absorption Coefficient and Absorption Spectrum using HITRAN/HITEMP database
# input:  database, partition function file 
# output: coef
# lineshape: voigt use w(z) function special.wofz

#warnings.simplefilter('error')
import time as time
import numpy as np
import matplotlib.pyplot as plt
import math as math
from scipy import special
from scipy import constants
import warnings as warnings



##################################################################################

partition_function = np.loadtxt('/home/xliu/Doc/2-prj/210-TDLAS/3-simulation/voigt_simulation/q7.txt')[:,1]

##################################################################################
# physical constants

T_ref = 296.0
cBolts = constants.Boltzmann*1E7 # CGS Boltzmann constant unit is erg/K, CGS, c is cm/s 
cc = constants.speed_of_light*1E2 # CGS
cNA = constants.N_A
c2 = constants.physical_constants['second radiation constant'][0]*100 # CGS cm K
cP = constants.physical_constants['standard atmosphere'][0]*10 # pressure unit in CGS is Ba. covert to 0.1 Pa 

##################################################################################
# Voigt lineshape from Faddeeva function

cGammaD = math.sqrt(2*cBolts*cNA*math.log(2))/cc
def profile_voigt(nu,gamma_air,n_air,delta_air,T,p,wavenumber): # return voigt profile on wavenumber, function of (T,p)
    gamma = gamma_air*p*((T_ref/T)**n_air)#+gamma_self*p*c*((T_ref/T)**n_self) # Eqn 6
    #gamma = gamma_air*gamma_air_ratio*p*(1-c)*((T_ref/T)**n_air)+gamma_self*p*c*((T_ref/T)**n_self) # Eqn 6
    Mass = 43.98983
    GammaD = cGammaD*math.sqrt(T/Mass)*nu# + laser # eqn (5), 
    sigma = GammaD/(math.sqrt(2*math.log(2)))
    variable = (wavenumber- nu - delta_air*p+gamma*1j)/(sigma*math.sqrt(2)) # air shift
    voigt = (special.wofz(variable)).real/(sigma*math.sqrt(2*constants.pi))
    return voigt

##################################################################################
# line-by-line coef calculation
def coef(T,p,database,wavenumber):#,color): # k  absorption coef 
    coef = np.zeros(len(wavenumber))
    for line in database:
        nu          = line[0]
        S           = line[1]
        gamma_air   = line[3]
        E           = line[5]
        n_air       = line[6]
        delta_air   = line[7]
        profile = profile_voigt(nu,gamma_air,n_air,delta_air,T,p,wavenumber)
        intensity = S * partition_function[int(T_ref-1)]/partition_function[int(T-1)] * math.exp(-c2*E/T)/math.exp(-c2*E/T_ref)*(1-math.exp(-c2*nu/T))/(1-math.exp(-c2*nu/T_ref)) # Eqn 4, T is rounded
        #        ax1.bar(nu,peak,align='center',width=0.001,color='b')
        coef += profile*intensity
    return coef

def OD(T,p,c,l,database,wavenumber): # optical density # k*[X]*l   absorption coef times volume density, assuming total pressure p and c is pressure fraction. c needs to times length in unit cm to get dimensionless optical depth \tao, bkg is added at the end
    acoef = coef(T,p,database,wavenumber)
    density = p*c*cP/(cBolts) /T # volumn density /cm^3 from p/kT  # check if concentration change
    OD = acoef*density*l
    Tr = np.exp(-OD)
    Ab = 1-Tr
    return OD,Ab,Tr

########################################################################
# How to use
def main():
    
    hitran = np.array([
     [ 2396.020721, 8.492E-26, 2.173E+02, 0.0568, 0.062, 4084.9341, 0.66, -0.004384 ],#    0 0 0 11       0 0 0 01                    R102e     3377541827 5 4 5 7   207.0  205.0
     [ 2396.263738, 3.968E-26, 2.172E+02, 0.0564, 0.061, 4245.3037, 0.66, -0.004439 ],#    0 0 0 11       0 0 0 01                    R104e     3377541827 5 4 5 7   211.0  209.0
     [ 2396.481639, 1.827E-26, 2.173E+02, 0.0561, 0.061, 4408.7251, 0.66, -0.004489 ],#    0 0 0 11       0 0 0 01                    R106e     3377541827 5 4 5 7   215.0  213.0
     [ 2396.672735, 8.280E-27, 2.171E+02, 0.0558, 0.061, 4575.1958, 0.66, -0.004540 ],#    0 0 0 11       0 0 0 01                    R108e     3377541827 5 4 5 7   219.0  217.0
     [ 2396.839384, 3.697E-27, 2.171E+02, 0.0555, 0.061, 4744.7134, 0.66, -0.004595 ],#    0 0 0 11       0 0 0 01                    R110e     3377541927 5 4 5 7   223.0  221.0
     [ 2396.979410, 1.626E-27, 2.170E+02, 0.0551, 0.060, 4917.2744, 0.67, -0.004653 ],#    0 0 0 11       0 0 0 01                    R112e     3377541827 5 4 5 7   227.0  225.0
     [ 2397.094125, 7.045E-28, 2.170E+02, 0.0548, 0.060, 5092.8770, 0.67, -0.004710 ],#    0 0 0 11       0 0 0 01                    R114e     3377541827 5 4 5 7   231.0  229.0
     [ 2397.182454, 3.006E-28, 2.169E+02, 0.0546, 0.060, 5271.5171, 0.67, -0.004760 ],#    0 0 0 11       0 0 0 01                    R116e     3377541827 5 4 5 7   235.0  233.0
     [ 2397.245227, 1.264E-28, 2.169E+02, 0.0543, 0.060, 5453.1924, 0.68, -0.004805 ],#    0 0 0 11       0 0 0 01                    R118e     3377541927 5 4 5 7   239.0  237.0
     [ 2397.282072, 5.233E-29, 2.167E+02, 0.0540, 0.060, 5637.9004, 0.68, -0.004847 ],#    0 0 0 11       0 0 0 01                    R120e     3377541927 5 4 5 7   243.0  241.0
     [ 2397.293041, 2.135E-29, 2.167E+02, 0.0537, 0.059, 5825.6372, 0.68, -0.004891 ],#    0 0 0 11       0 0 0 01                    R122e     3377541927 5 4 5 7   247.0  245.0
     [ 2397.278121, 8.577E-30, 2.165E+02, 0.0535, 0.059, 6016.3999, 0.68, -0.004939 ],#    0 0 0 11       0 0 0 01                    R124e     3377541927 5 4 5 7   251.0  249.0
     [ 2397.237297, 3.396E-30, 2.164E+02, 0.0532, 0.059, 6210.1855, 0.69, -0.004986 ],#    0 0 0 11       0 0 0 01                    R126e     3377541927 5 4 5 7   255.0  253.0
     [ 2397.170557, 1.324E-30, 2.162E+02, 0.0530, 0.059, 6406.9912, 0.69, -0.005032 ],#    0 0 0 11       0 0 0 01                    R128e     3377541927 5 4 5 7   259.0  257.0
        ])
    
    
    ########################################################################
    # variables
    start = 2396
    end = 2399
    resolution = 0.0001
    wavenumber = np.arange(start,end,resolution)
    database = hitran
    T= 600
    p = 10
    c = 0.1
    l = 5
    od,ab,tr = OD(T,p,c,l,database,wavenumber)
    plt.plot(wavenumber,ab,label='hitran') #/np.max(hitemp[(2397-start)/resolution:])
    plt.xlim(start,end)
    plt.legend(loc='best')
    plt.show()
    
    
if __name__ == "__main__":
    main()
