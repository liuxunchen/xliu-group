# def hitran: Calculate Absorption Coefficient and Absorption Spectrum using HITRAN/HITEMP database
# input:  database .par file, partition function file 
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

#fig = plt.figure()
#ax1 = fig.add_subplot(211)
#ax2 = fig.add_subplot(212)

##################################################################################
# prerequisite data of HITRAN calculation



##################################################################################
# physical constants

cBolts = constants.Boltzmann*1E7 # CGS Boltzmann constant unit is erg/K, CGS, c is cm/s 
cc = constants.speed_of_light*1E2 # CGS
cNA = constants.N_A
cP = constants.physical_constants['standard atmosphere'][0]*10 # pressure unit in CGS is Ba. covert to 0.1 Pa 

##################################################################################
# Voigt lineshape from Faddeeva function

def profile_voigt(M,I,nu,gamma_air,n_air,delta_air,T,p,wavenumber): # return voigt profile on wavenumber, function of (T,p), laser is the instrumetnal linewidth, gamma_air_ratio is the modification factor to the database value
    cGammaD = math.sqrt(2*cBolts*cNA*math.log(2))/cc
    T_ref = 296.0

    ISO_INDEX = {
        'id':0,
        'iso_name':1,
        'abundance':2,
        'mass':3,
        'mol_name':4
    }

    ISO = {
        ( 2, 1 ): [   7,  '(12C)(16O)2',      0.9842,             43.98983,   'CO2' ],
        ( 2, 2 ): [   8,  '(13C)(16O)2',      0.01106,            44.993185,  'CO2' ],
        ( 2, 3 ): [   9,  '(16O)(12C)(18O)',  0.0039471,          45.994076,  'CO2' ],
        ( 2, 4 ): [  10,  '(16O)(12C)(17O)',  0.000734,           44.994045,  'CO2' ],
        ( 2, 5 ): [  11,  '(16O)(13C)(18O)',  0.00004434,         46.997431,  'CO2' ],
        ( 2, 6 ): [  12,  '(16O)(13C)(17O)',  0.00000825,         45.9974,    'CO2' ],
        ( 2, 7 ): [  13,  '(12C)(18O)2',      0.0000039573,       47.998322,  'CO2' ],
        ( 2, 8 ): [  14,  '(17O)(12C)(18O)',  0.00000147,         46.998291,  'CO2' ],
        ( 2, 0 ): [  15,  '(13C)(18O)2',      0.000000044967,     49.001675,  'CO2' ],
        ( 2,11 ): [ 120,  '(18O)(13C)(17O)',  0.00000001654,      48.00165,   'CO2' ],
        ( 2, 9 ): [ 121,  '(12C)(17O)2',      0.0000001368,       45.998262,  'CO2' ],
    }
    ###################

    gamma = gamma_air*p*((T_ref/T)**n_air)#+gamma_self*p*c*((T_ref/T)**n_self) # Eqn 6
    #gamma = gamma_air*gamma_air_ratio*p*(1-c)*((T_ref/T)**n_air)+gamma_self*p*c*((T_ref/T)**n_self) # Eqn 6
    Mass = ISO[(M,I)][ISO_INDEX['mass']]
    GammaD = cGammaD*math.sqrt(T/Mass)*nu# + laser # eqn (5), 
    sigma = GammaD/(math.sqrt(2*math.log(2)))
    variable = (wavenumber- nu - delta_air*p*0+gamma*1j)/(sigma*math.sqrt(2)) # air shift # xliu modify
    voigt = (special.wofz(variable)).real/(sigma*math.sqrt(2*constants.pi))

#        if CDSD :
#            n_self      = float(params[10])
#            profile = profile_voigt(M,I,nu,gamma_air,gamma_self,n_air,n_self,delta_air,T,p,c,wavenumber,laser,gamma_air_ratio)
#        else:
    return voigt

##################################################################################
# line-by-line coef calculation
def coef(T,p,database,wavenumber): #color): # k
    database='/home/xliu/ownCloud/prj4um/3_spectra_fit_code/4.2um_band_database/hitemp_2380_2400.par'
    T_ref = 296.0
    c2 = constants.physical_constants['second radiation constant'][0]*100 # CGS cm K
    partition_function = {}
    partition_function[(2,1)]  = np.loadtxt('/home/xliu/ownCloud/prj4um/3_spectra_fit_code/4.2um_band_database/partition_function/q7.txt')[:,1]
    partition_function[(2,2)]  = np.loadtxt('/home/xliu/ownCloud/prj4um/3_spectra_fit_code/4.2um_band_database/partition_function/q8.txt')[:,1]
    partition_function[(2,3)]  = np.loadtxt('/home/xliu/ownCloud/prj4um/3_spectra_fit_code/4.2um_band_database/partition_function/q9.txt')[:,1]
    partition_function[(2,4)]  = np.loadtxt('/home/xliu/ownCloud/prj4um/3_spectra_fit_code/4.2um_band_database/partition_function/q10.txt')[:,1]
    partition_function[(2,5)]  = np.loadtxt('/home/xliu/ownCloud/prj4um/3_spectra_fit_code/4.2um_band_database/partition_function/q11.txt')[:,1]
    partition_function[(2,6)]  = np.loadtxt('/home/xliu/ownCloud/prj4um/3_spectra_fit_code/4.2um_band_database/partition_function/q12.txt')[:,1]
    partition_function[(2,7)]  = np.loadtxt('/home/xliu/ownCloud/prj4um/3_spectra_fit_code/4.2um_band_database/partition_function/q13.txt')[:,1]
    partition_function[(2,8)]  = np.loadtxt('/home/xliu/ownCloud/prj4um/3_spectra_fit_code/4.2um_band_database/partition_function/q14.txt')[:,1]
    partition_function[(2,9)]  = np.loadtxt('/home/xliu/ownCloud/prj4um/3_spectra_fit_code/4.2um_band_database/partition_function/q121.txt')[:,1]
    partition_function[(2,0)]  = np.loadtxt('/home/xliu/ownCloud/prj4um/3_spectra_fit_code/4.2um_band_database/partition_function/q9.txt')[:,1]
    partition_function[(2,11)] = np.loadtxt('/home/xliu/ownCloud/prj4um/3_spectra_fit_code/4.2um_band_database/partition_function/q120.txt')[:,1]
    ###################
    data = open(database,'r') 
    coef = np.zeros(len(wavenumber))
    ###################
    # save the cut-off selected lines
    #intensity_file = open('CDSD-2390-2405-E-5-E-6','w')
    #coef_select = np.zeros(len(wavenumber))
    ###################
    # need to modify .par file 
    for line in data:
        params = line.split()
        M        = int(params[0])
        I        = int(params[1])
        nu       = float(params[2])
        gamma_air= float(params[5])
        n_air    = float(params[8])
        delta_air= float(params[9])*0.0 # xliu
        S        = float(params[3])
        E        = float(params[7])
        profile = profile_voigt(M,I,nu,gamma_air,n_air,delta_air,T,p,wavenumber)
        intensity = S * partition_function[(M,I)][int(T_ref-1)]/partition_function[(M,I)][int(T-1)] * math.exp(-c2*E/T)/math.exp(-c2*E/T_ref)*(1-math.exp(-c2*nu/T))/(1-math.exp(-c2*nu/T_ref)) # Eqn 4, T is rounded
        #plt.bar(nu+delta_air*p,np.max(profile*intensity),align='center',width = 0.008,color=color)
        #        ax1.bar(nu,peak,align='center',width=0.001,color='b')
        coef += profile*intensity
        ######################################
        # plot bar at center peak
        # get  the index of nu in the wavenumber
        #if wavenumber[0] < nu < wavenumber [-1]:
        #    peak = (profile*intensity)[int((nu - wavenumber[0])*1000)]
        #    if 10E5 > peak > 10E-24:
        #        print nu
                #plt.bar(nu,peak,align='center',width=0.001,color='b')
        #        intensity_file.write(line)
        #        coef_select += profile*intensity*density
        ######################################
        '''
        # plot nu-J
        J   = int(params[26])
        ax2.plot(J,nu,'bo')
    ax2.set_xlim(70,170)
    ax2.set_ylim(2380,2410)
        '''
    data.close()
    #intensity_file.close()
    #ax1.plot(wavenumber,coef,'b')
    return coef#,coef_select+bkg

def OD(T,p,c,l,database,wavenumber): # optical density
    acoef = coef(T,p,database,wavenumber)
    density = p*c*cP/(cBolts) /T # volumn density /cm^3 from p/kT  # check if concentration change
    OD = acoef*density*l
    Tr = np.exp(-OD)
    Ab = 1-Tr
    return OD,Ab,Tr


def main():

    wn = np.load('../../4_calibration_results/high_T_tube/low-pressure-10cm/171-1106.npy')

    acoef = coef(T=900,p=1,database='../4.2um_band_database/2390-2405-E-6',wavenumber=wn)

    plt.plot(wn,acoef)
    plt.show()


    
if __name__ == "__main__":
    main()
