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

# physical constants

cBolts = constants.Boltzmann*1E7 # CGS Boltzmann constant unit is erg/K, CGS, c is cm/s 
#print cBolts
cc = constants.speed_of_light*1E2 # CGS
#print cc
cNA = constants.N_A
#print cNA
cP = constants.physical_constants['standard atmosphere'][0]*10 # pressure unit in CGS is Ba. covert to 0.1 Pa 

# Voigt lineshape from Faddeeva function
def profile_dicke(M,I,nu,gamma_air,n_air,delta_air,T,p,beta,wavenumber): # return voigt profile on wavenumber, function of (T,p), laser is the instrumetnal linewidth, gamma_air_ratio is the modification factor to the database value
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

    #c = 0.1
    #gamma = gamma_air*p*(1-c)*((T_ref/T)**n_air)+gamma_self*p*c*((T_ref/T)**n_self) # Eqn 6
    gamma = gamma_air*p*((T_ref/T)**n_air)
    #print gamma
    Mass = ISO[(M,I)][ISO_INDEX['mass']]
    GammaD = cGammaD*math.sqrt(T/Mass)*nu# + laser # eqn (5), 
    #print 'GammaD'
    #print GammaD
    sigma = GammaD/(math.sqrt(2*math.log(2)))
    variable = (wavenumber- nu - delta_air*p+(gamma+beta*p)*1j)/(sigma*math.sqrt(2)) # air shift # xliu modify
    z = beta*p/(sigma*math.sqrt(2))
    Dicke = (special.wofz(variable)/(1-math.sqrt(math.pi)*z*special.wofz(variable))).real/(sigma*math.sqrt(2*constants.pi))
    return Dicke

# line-by-line coef calculation
def coef(T,p,database,loadfromdatabase,beta,wavenumber): #color): # k 
    CDSD= np.array([
        [2, 1, 2390.522470, 4.080E-22, 0.0664,  0.0649,    2278.9960,   0.71,  -0.004338,   0.56], 
        [2, 1, 2391.098970, 2.330E-22, 0.0660,  0.0645,    2399.4676,   0.71,  -0.004379,   0.56], 
        [2, 1, 2391.649940, 1.310E-22, 0.0656,  0.0641,    2523.0215,   0.70,  -0.004419,   0.56], 
        [2, 1, 2392.175380, 7.240E-23, 0.0651,  0.0636,    2649.6558,   0.70,  -0.004457,   0.56], 
        [2, 1, 2392.675270, 3.940E-23, 0.0645,  0.0633,    2779.3683,   0.70,  -0.004495,   0.56], 
        [2, 1, 2393.149580, 2.110E-23, 0.0639,  0.0631,    2912.1569,   0.69,  -0.004531,   0.57], 
        [2, 1, 2393.598290, 1.120E-23, 0.0634,  0.0629,    3048.0194,   0.69,  -0.004567,   0.57], 
        [2, 1, 2394.021400, 5.810E-24, 0.0629,  0.0629,    3186.9537,   0.68,  -0.004602,   0.58], 
        [2, 1, 2394.418870, 2.980E-24, 0.0622,  0.0628,    3328.9574,   0.68,  -0.004635,   0.58], 
        [2, 1, 2394.790690, 1.500E-24, 0.0616,  0.0629,    3474.0282,   0.67,  -0.004668,   0.59], 
        [2, 1, 2395.136850, 7.450E-25, 0.0610,  0.0629,    3622.1636,   0.67,  -0.004701,   0.59], 
        [2, 1, 2395.457320, 3.640E-25, 0.0604,  0.0631,    3773.3614,   0.66,  -0.004732,   0.60], 
        [2, 1, 2395.752090, 1.760E-25, 0.0597,  0.0632,    3927.6189,   0.65,  -0.004763,   0.60], 
        [2, 1, 2396.021140, 8.330E-26, 0.0589,  0.0634,    4084.9337,   0.65,  -0.004792,   0.61], 
        [2, 1, 2396.264460, 3.890E-26, 0.0583,  0.0636,    4245.3032,   0.64,  -0.004822,   0.62], 
        [2, 1, 2396.482010, 1.790E-26, 0.0577,  0.0638,    4408.7247,   0.64,  -0.004850,   0.62], 
        [2, 1, 2396.673800, 8.110E-27, 0.0571,  0.0640,    4575.1955,   0.63,  -0.004878,   0.63], 
        [2, 1, 2396.839800, 3.620E-27, 0.0565,  0.0643,    4744.7130,   0.63,  -0.004905,   0.64], 
        [2, 1, 2396.980000, 1.590E-27, 0.0558,  0.0646,    4917.2743,   0.62,  -0.004931,   0.64], 
        [2, 1, 2397.094380, 6.900E-28, 0.0551,  0.0648,    5092.8765,   0.62,  -0.004957,   0.65], 
        [2, 1, 2397.182920, 2.940E-28, 0.0546,  0.0651,    5271.5169,   0.61,  -0.004983,   0.65], 
        [2, 1, 2397.245620, 1.240E-28, 0.0539,  0.0654,    5453.1924,   0.61,  -0.005007,   0.66], 
        [2, 1, 2397.282440, 5.120E-29, 0.0533,  0.0656,    5637.9001,   0.60,  -0.005032,   0.66], 
        [2, 1, 2397.293390, 2.090E-29, 0.0528,  0.0659,    5825.6369,   0.60,  -0.005055,   0.67], 
        [2, 1, 2397.278440, 8.380E-30, 0.0523,  0.0662,    6016.3998,   0.59,  -0.005078,   0.67], 
        [2, 1, 2397.237590, 3.320E-30, 0.0518,  0.0663,    6210.1856,   0.59,  -0.005101,   0.68], 
        [2, 1, 2397.170810, 1.290E-30, 0.0512,  0.0665,    6406.9911,   0.58,  -0.005123,   0.68],
        [2, 1, 2397.048650, 7.890E-25, 0.0691,  0.0902,    1805.2522,   0.68,  -0.003167,   0.68]  # 36e  
    ]) 


    database = np.array([ #HITRAN
        [2,1, 2390.522224, 4.140E-22,  0.0620, 0.065, 2278.9963, 0.65,0.0, 0.65],#-0.003664 ,
        [2,1, 2391.098651, 2.364E-22,  0.0616, 0.065, 2399.4680, 0.65,0.0, 0.65],#-0.003720 ,
        [2,1, 2391.649608, 1.329E-22,  0.0611, 0.064, 2523.0217, 0.65,0.0, 0.65],#-0.003768 ,
        [2,1, 2392.175077, 7.358E-23,  0.0607, 0.064, 2649.6560, 0.65,0.0, 0.65],#-0.003824 ,
        [2,1, 2392.675004, 4.010E-23,  0.0603, 0.064, 2779.3687, 0.65,0.0, 0.65],#-0.003888 ,
        [2,1, 2393.149224, 2.152E-23,  0.0599, 0.064, 2912.1572, 0.65,0.0, 0.65],#-0.003946 ,
        [2,1, 2393.597874, 1.137E-23,  0.0595, 0.063, 3048.0198, 0.65,0.0, 0.65],#-0.004005 ,
        [2,1, 2394.020948, 5.914E-24,  0.0590, 0.063, 3186.9541, 0.65,0.0, 0.65],#-0.004062 ,
        [2,1, 2394.418060, 3.030E-24,  0.0586, 0.063, 3328.9578, 0.65,0.0, 0.65],#-0.004122 ,
        [2,1, 2394.790248, 1.528E-24,  0.0583, 0.063, 3474.0286, 0.65,0.0, 0.65],#-0.004184 ,
        [2,1, 2395.136206, 7.592E-25,  0.0579, 0.062, 3622.1641, 0.65,0.0, 0.65],#-0.004238 ,
        [2,1, 2395.456616, 3.714E-25,  0.0575, 0.062, 3773.3618, 0.65,0.0, 0.65],#-0.004285 ,
        [2,1, 2395.751585, 1.790E-25,  0.0571, 0.062, 3927.6194, 0.65,0.0, 0.65],#-0.004331 ,
        [2,1, 2396.020721, 8.492E-26,  0.0568, 0.062, 4084.9341, 0.66,0.0, 0.66],#-0.004384 ,
        [2,1, 2396.263738, 3.968E-26,  0.0564, 0.061, 4245.3037, 0.66,0.0, 0.66],#-0.004439 ,
        [2,1, 2396.481639, 1.827E-26,  0.0561, 0.061, 4408.7251, 0.66,0.0, 0.66],#-0.004489 ,
        [2,1, 2396.672735, 8.280E-27,  0.0558, 0.061, 4575.1958, 0.66,0.0, 0.66],#-0.004540 ,
        [2,1, 2396.839384, 3.697E-27,  0.0555, 0.061, 4744.7134, 0.66,0.0, 0.66],#-0.004595 ,
        [2,1, 2396.979410, 1.626E-27,  0.0551, 0.060, 4917.2744, 0.67,0.0, 0.67],#-0.004653 ,
        [2,1, 2397.094125, 7.045E-28,  0.0548, 0.060, 5092.8770, 0.67,0.0, 0.67],#-0.004710 ,
        [2,1, 2397.182454, 3.006E-28,  0.0546, 0.060, 5271.5171, 0.67,0.0, 0.67],#-0.004760 ,
        [2,1, 2397.245227, 1.264E-28,  0.0543, 0.060, 5453.1924, 0.68,0.0, 0.68],#-0.004805 ,
        [2,1, 2397.282072, 5.233E-29,  0.0540, 0.060, 5637.9004, 0.68,0.0, 0.68],#-0.004847 ,
        [2,1, 2397.293041, 2.135E-29,  0.0537, 0.059, 5825.6372, 0.68,0.0, 0.68],#-0.004891 ,
        [2,1, 2397.278121, 8.577E-30,  0.0535, 0.059, 6016.3999, 0.68,0.0, 0.68],#-0.004939 , #8.577E-30
        [2,1, 2397.237297, 3.396E-30,  0.0532, 0.059, 6210.1855, 0.69,0.0, 0.69],#-0.004986 ,
        [2,1, 2397.170557, 1.324E-30,  0.0530, 0.059, 6406.9912, 0.69,0.0, 0.69],#-0.005032 ,
        [2,1, 2397.078100, 4.968E-31,  0.0508, 0.058, 6606.8131, 0.58,0.0, 0.69], 
        [2,1, 2396.959430, 1.878E-31,  0.0502, 0.057, 6809.6483, 0.58,0.0, 0.69],
        [2,1, 2397.048650, 7.890E-25,  0.0691, 0.0902,1805.2522, 0.68,0.0, 0.68] #-0.003167 , # 36e  
    ])


    database = np.array([
        [2,1, 2390.522224, 4.140E-22,  0.0620, 0.065, 2278.9963, 0.65,0.0, 0.65],#-0.003664 ,
        [2,1, 2391.098651, 2.364E-22,  0.0616, 0.065, 2399.4680, 0.65,0.0, 0.65],#-0.003720 ,
        [2,1, 2391.649608, 1.329E-22,  0.0611, 0.064, 2523.0217, 0.65,0.0, 0.65],#-0.003768 ,
        [2,1, 2392.175077, 7.358E-23,  0.0607, 0.064, 2649.6560, 0.65,0.0, 0.65],#-0.003824 ,
        [2,1, 2392.675004, 4.010E-23,  0.0603, 0.064, 2779.3687, 0.65,0.0, 0.65],#-0.003888 ,
        [2,1, 2393.149224, 2.152E-23,  0.0599, 0.064, 2912.1572, 0.65,0.0, 0.65],#-0.003946 ,
        [2,1, 2393.597874, 1.137E-23,  0.0595, 0.063, 3048.0198, 0.65,0.0, 0.65],#-0.004005 ,
        [2,1, 2394.020948, 5.914E-24,  0.0590, 0.063, 3186.9541, 0.65,0.0, 0.65],#-0.004062 ,
        [2,1, 2394.418060, 3.030E-24,  0.0586, 0.063, 3328.9578, 0.65,0.0, 0.65],#-0.004122 ,
        [2,1, 2394.790248, 1.528E-24,  0.0583, 0.063, 3474.0286, 0.65,0.0, 0.65],#-0.004184 ,
        [2,1, 2395.136206, 7.592E-25,  0.0579, 0.062, 3622.1641, 0.65,0.0, 0.65],#-0.004238 ,
        [2,1, 2395.456616, 3.714E-25,  0.0575, 0.062, 3773.3618, 0.65,0.0, 0.65],#-0.004285 ,
        [2,1, 2395.751585, 1.790E-25,  0.0571, 0.062, 3927.6194, 0.65,0.0, 0.65],#-0.004331 ,
        [2,1, 2396.020721, 8.492E-26,  0.0568, 0.062, 4084.9341, 0.66,0.0, 0.66],#-0.004384 ,
        [2,1, 2396.263738, 3.968E-26,  0.0564, 0.061, 4245.3037, 0.66,0.0, 0.66],#-0.004439 ,
        [2,1, 2396.481639, 1.827E-26,  0.0561, 0.061, 4408.7251, 0.66,0.0, 0.66],#-0.004489 ,
        [2,1, 2396.672735, 8.280E-27,  0.0558, 0.061, 4575.1958, 0.66,0.0, 0.66],#-0.004540 ,
        [2,1, 2396.839384, 3.697E-27,  0.0555, 0.061, 4744.7134, 0.66,0.0, 0.66],#-0.004595 ,
        [2,1, 2396.979410, 1.626E-27,  0.0599, 0.060, 4917.2744, 0.67,0.0, 0.67],#-0.004653
        [2,1, 2397.094125, 7.045E-28,  0.0600, 0.060, 5092.8770, 0.67,0.0, 0.67],#-0.004710
        [2,1, 2397.182454, 3.006E-28,  0.05632,0.060, 5271.5171, 0.67,0.0017, 0.67],#-0.004760
        [2,1, 2397.245227, 1.264E-28,  0.0543, 0.060, 5453.1924, 0.68,0.00, 0.68],#0.001794-0.004805   0.0466
        [2,1, 2397.282072, 5.233E-29,  0.0540, 0.060, 5637.9004, 0.68,0.0004, 0.68],# -0.0029-0.004847   0.0332
        [2,1, 2397.293041, 2.135E-29,  0.0537, 0.059, 5825.6372, 0.68,-0.0135, 0.68],# 0.00348-0.004891  0.0638
        [2,1, 2397.278121, 8.577E-30,  0.0535, 0.059, 6016.3999, 0.68,0.0, 0.68],#-0.00276-0.004939  0.0412
        [2,1, 2397.237297, 3.396E-30,  0.0532, 0.059, 6210.1855, 0.69,0.0, 0.69],#-0.00363-0.004986  0.0532
        [2,1, 2397.170557, 1.324E-30,  0.0548, 0.059, 6406.9912, 0.69,0.0, 0.69],#-0.005032
        [2,1, 2397.078100, 4.968E-31,  0.0508, 0.058, 6606.8131, 0.58,0.0, 0.69], 
        [2,1, 2396.959430, 1.878E-31,  0.0502, 0.057, 6809.6483, 0.58,0.0, 0.69],
        [2,1, 2397.048650, 7.890E-25,  0.0691, 0.0902,1805.2522, 0.68,0.0, 0.68] #-0.003167 , # 36e  
    ])
        
    T_ref = 296.0
    c2 = constants.physical_constants['second radiation constant'][0]*100 # CGS cm K
    #print 'c2'
    #print c2
    coef = np.zeros(len(wavenumber))
    if loadfromdatabase:
        partition_function = {}
        partition_function[(2,1)]  = np.loadtxt('/home/xliu/data/prj4um/3_spectra_fit_code/4.2um_band_database/partition_function/q7.txt')[:,1]
        partition_function[(2,2)]  = np.loadtxt('/home/xliu/data/prj4um/3_spectra_fit_code/4.2um_band_database/partition_function/q8.txt')[:,1]
        partition_function[(2,3)]  = np.loadtxt('/home/xliu/data/prj4um/3_spectra_fit_code/4.2um_band_database/partition_function/q9.txt')[:,1]
        partition_function[(2,4)]  = np.loadtxt('/home/xliu/data/prj4um/3_spectra_fit_code/4.2um_band_database/partition_function/q10.txt')[:,1]
        partition_function[(2,5)]  = np.loadtxt('/home/xliu/data/prj4um/3_spectra_fit_code/4.2um_band_database/partition_function/q11.txt')[:,1]
        partition_function[(2,6)]  = np.loadtxt('/home/xliu/data/prj4um/3_spectra_fit_code/4.2um_band_database/partition_function/q12.txt')[:,1]
        partition_function[(2,7)]  = np.loadtxt('/home/xliu/data/prj4um/3_spectra_fit_code/4.2um_band_database/partition_function/q13.txt')[:,1]
        partition_function[(2,8)]  = np.loadtxt('/home/xliu/data/prj4um/3_spectra_fit_code/4.2um_band_database/partition_function/q14.txt')[:,1]
        partition_function[(2,9)]  = np.loadtxt('/home/xliu/data/prj4um/3_spectra_fit_code/4.2um_band_database/partition_function/q121.txt')[:,1]
        partition_function[(2,0)]  = np.loadtxt('/home/xliu/data/prj4um/3_spectra_fit_code/4.2um_band_database/partition_function/q9.txt')[:,1]
        partition_function[(2,11)] = np.loadtxt('/home/xliu/data/prj4um/3_spectra_fit_code/4.2um_band_database/partition_function/q120.txt')[:,1]
        data = open(database,'r') 
        for line in data:
            params = line.split()
            M        = int(params[0])
            I        = int(params[1])
            nu       = float(params[2])
            gamma_air= float(params[5])
            n_air    = float(params[8])
            delta_air= float(params[9]) # xliu
            S        = float(params[3])
            E        = float(params[7])
            profile = profile_dicke(M,I,nu,gamma_air,n_air,delta_air,T,p,beta,wavenumber)
            intensity = S * partition_function[(M,I)][int(T_ref-1)]/partition_function[(M,I)][int(T-1)] * math.exp(-c2*E/T)/math.exp(-c2*E/T_ref)*(1-math.exp(-c2*nu/T))/(1-math.exp(-c2*nu/T_ref)) # Eqn 4, T is rounded
            #plt.bar(nu+delta_air*p,np.max(profile*intensity),align='center',width = 0.008,color=color)
            #        ax1.bar(nu,peak,align='center',width=0.001,color='b')
            coef += profile*intensity
        data.close()
    else:
        partition_function = np.loadtxt('q7.txt')[:,1]
        for line in database:
            M           = line[0]
            I           = line[1]
            nu          = line[2]
            S           = line[3]
            gamma_air   = line[4]
            E           = line[6]
            n_air       = line[7]
            delta_air   = line[8]
            profile = profile_dicke(M,I,nu,gamma_air,n_air,delta_air,T,p,beta,wavenumber)
            #np.savetxt('profile',profile)
            #print 'Q'
            #print partition_function[int(T_ref-1)]/partition_function[int(T-1)]
            intensity = S * partition_function[int(T_ref-1)]/partition_function[int(T-1)] * math.exp(-c2*E/T)/math.exp(-c2*E/T_ref)*(1-math.exp(-c2*nu/T))/(1-math.exp(-c2*nu/T_ref)) # Eqn 4, T is rounded
            #print 'intensity'
            #print intensity
            #plt.bar(nu+delta_air*p,np.max(profile*intensity),align='center',width = 0.008,color=color)
            #        ax1.bar(nu,peak,align='center',width=0.001,color='b')
            coef += profile*intensity
    return coef#/max(coef[700:])

def OD(T,p,c,l,database,wavenumber): # optical density
    acoef = coef(T,p,database,False,wavenumber)
    #np.savetxt('coef',acoef)
    density = p*c*cP/(cBolts) /T # volumn density /cm^3 from p/kT  # check if concentration change
    #print density
    OD = acoef*density*l
    Tr = np.exp(-OD)
    Ab = 1-Tr
    return OD,Ab,Tr

def f_k(x,Temp,conc):
    output = OD(Temp,0.99,conc,1,'NA',x)[0]
    return output

def f_coef(x,Temp): # define the function, must define x variable
    output= coef(T=Temp,p=0.99,database='NA',loadfromdatabase=False,wavenumber=x)
    return output

def main():

    # calculate coef of single molecule
    #wn = np.load('/home/xliu/data/prj4um/4_calibration_results/high_T_tube/low-pressure-10cm/171-1106.npy')
    wn = np.arange(2396.9,2397.4,0.001)
    acoef = coef(T=1764,p=1,database='NA',loadfromdatabase=False,beta=0.01,wavenumber=wn) # if loadfromdatabase = False, database='' can write any string does not matter

    # test wgq 20180501
    #wn = np.arange(2397.000,2397.402,0.001)  #test 20180501 check with wgq
    #od,AB,Tr = OD(T=900,p=1,c=0.05,l=1,database='database',wavenumber=wn) # optical density

    #np.savetxt('wgq-output-wn',wn)
    #np.savetxt('wgq-output',AB)

    plt.plot(wn,acoef)
    plt.show()

if __name__ == "__main__":
    main()
