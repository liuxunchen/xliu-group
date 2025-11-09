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
import pathlib 

##################################################################################
# HITRAN 160位格式定义
HITRAN_FORMAT_160 = {
   'M'          : {'pos' :   1,   'len' :  2,   'format' : '%2d' },
   'I'          : {'pos' :   3,   'len' :  1,   'format' : '%1d' },
   'nu'         : {'pos' :   4,   'len' : 12,   'format' : '%12f'},
   'S'          : {'pos' :  16,   'len' : 10,   'format' : '%10f'},
   'R'          : {'pos' :  26,   'len' :  0,   'format' : '%0f' },
   'A'          : {'pos' :  26,   'len' : 10,   'format' : '%10f'},
   'gamma_air'  : {'pos' :  36,   'len' :  5,   'format' : '%5f' },
   'gamma_self' : {'pos' :  41,   'len' :  5,   'format' : '%5f' },
   'E_'         : {'pos' :  46,   'len' : 10,   'format' : '%10f'},
   'n_air'      : {'pos' :  56,   'len' :  4,   'format' : '%4f' },
   'delta_air'  : {'pos' :  60,   'len' :  8,   'format' : '%8f' },
   'V'          : {'pos' :  68,   'len' : 15,   'format' : '%15s'},
   'V_'         : {'pos' :  83,   'len' : 15,   'format' : '%15s'},
   'Q'          : {'pos' :  98,   'len' : 15,   'format' : '%15s'},
   'Q_'         : {'pos' : 113,   'len' : 15,   'format' : '%15s'},
   'Ierr'       : {'pos' : 128,   'len' :  6,   'format' : '%6s' },
   'Iref'       : {'pos' : 134,   'len' : 12,   'format' : '%12s'},
   'flag'       : {'pos' : 146,   'len' :  1,   'format' : '%1s' },
   'g'          : {'pos' : 147,   'len' :  7,   'format' : '%7f' },
   'g_'         : {'pos' : 154,   'len' :  7,   'format' : '%7f' }
}

# HITRAN 160位格式读取函数
def read_hitran_par(filename):
    """
    读取HITRAN 160位格式的par文件
    返回格式: [波数, 线强, 爱因斯坦A系数, 空气加宽半宽, 自加宽半宽, 低态能量, 温度依赖系数, 压力位移]
    """
    database = []
    
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.rstrip('\n')  # 只去掉行尾换行符，保留其他空白
            if len(line) < 160:  # 确保行长度足够
                print(f"警告: 第 {line_num} 行长度不足160字符，已跳过")
                continue
                
            try:
                # 根据HITRAN格式定义解析各个字段
                nu = float(line[3:15])           # 波数
                S = float(line[15:25])           # 线强
                A = float(line[25:35])           # 爱因斯坦A系数
                gamma_air = float(line[35:40])   # 空气加宽半宽
                gamma_self = float(line[40:45])  # 自加宽半宽
                E = float(line[45:55])           # 低态能量
                n_air = float(line[55:59])       # 温度依赖系数
                delta_air = float(line[59:67])   # 压力位移
                
                # 将解析的数据添加到数据库
                database.append([nu, S, A, gamma_air, gamma_self, E, n_air, delta_air])
                
            except ValueError as e:
                print(f"解析第 {line_num} 行时出错: {repr(line)}")
                print(f"错误信息: {e}")
                continue
    
    print(f"成功读取 {len(database)} 条谱线")
    return np.array(database)


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
def profile_voigt(Mass,nu,gamma_air,gamma_self,n_air,delta_air,T,p,c,wavenumber): # return voigt profile on wavenumber, function of (T,p)
    #gamma = gamma_air*p*((T_ref/T)**n_air)#+gamma_self*p*c*((T_ref/T)**n_self) # Eqn 6. omit gamma_self so profile is only a function of T,p
    gamma = gamma_air*gamma_air*p*(1-c)*((T_ref/T)**n_air)+gamma_self*p*c*((T_ref/T)**n_air)# Eqn 6
    GammaD = cGammaD*math.sqrt(T/Mass)*nu# + laser # eqn (5), 
    sigma = GammaD/(math.sqrt(2*math.log(2)))
    variable = (wavenumber- nu - delta_air*p+gamma*1j)/(sigma*math.sqrt(2)) # air shift
    voigt = (special.wofz(variable)).real/(sigma*math.sqrt(2*constants.pi))
    return voigt

##################################################################################
# line-by-line coef calculation
def coef(Mass,T,p,c,database,partition_function,wavenumber):#,color): # k  absorption coef 
    coef = np.zeros(len(wavenumber))
    for line in database:
        nu          = line[0]
        S           = line[1]
        gamma_air   = line[3]
        gamma_self  = line[4]
        E           = line[5]
        n_air       = line[6]
        delta_air   = line[7]
        profile = profile_voigt(Mass,nu,gamma_air,gamma_self,n_air,delta_air,T,p,c,wavenumber)
        intensity = S * partition_function[int(T_ref-1)]/partition_function[int(T-1)] * math.exp(-c2*E/T)/math.exp(-c2*E/T_ref)*(1-math.exp(-c2*nu/T))/(1-math.exp(-c2*nu/T_ref)) # Eqn 4, T is rounded
        #        ax1.bar(nu,peak,align='center',width=0.001,color='b')
        coef += profile*intensity
    return coef

def OD(Mass,T,p,c,l,database,partition_function,wavenumber): # optical density # k*[X]*l   absorption coef times volume density, assuming total pressure p and c is pressure fraction. c needs to times length in unit cm to get dimensionless optical depth \tao, bkg is added at the end
    acoef = coef(Mass,T,p,c,database,partition_function,wavenumber)
    density = p*c*cP/(cBolts) /T # volumn density /cm^3 from p/kT  # check if concentration change
    OD = acoef*density*l
    Tr = np.exp(-OD)
    Ab = 1-Tr
    return OD,Ab,Tr

########################################################################
# How to use

def main():
    
    # 从par文件读取数据
    database = read_hitran_par('CO_1416.par')
    partition_function_path = pathlib.Path("/home/xliu/prj/210-TDLAS/3-simulation/hitran_database/Q/")
    partition_function = np.loadtxt(partition_function_path/ 'q5.txt')[:,1]
    Mass = 27.994915 # 从molparam.txt 文件读取 
    
    # 显示一些基本信息
    if len(database) > 0:
        print(f"波数范围: {np.min(database[:,0]):.4f} - {np.max(database[:,0]):.4f} cm⁻¹")
        print(f"线强范围: {np.min(database[:,1]):.2e} - {np.max(database[:,1]):.2e} cm⁻¹/(molecule·cm⁻²)")
        print(f"前5条谱线数据:")
        for i in range(min(5, len(database))):
            print(f"  波数: {database[i,0]:.6f}, 线强: {database[i,1]:.2e}")
    
    ########################################################################
    # variables
    start = 1870
    end = 2310
    resolution = 0.001
    wavenumber = np.arange(start,end,resolution)
    
    T= 600
    p = 1
    c = 0.00001
    l = 100
    od,ab,tr = OD(Mass,T,p,c,l,database,partition_function,wavenumber)
    plt.plot(wavenumber,ab,label='hitran') #/np.max(hitemp[(2397-start)/resolution:])
    plt.xlim(start,end)
    plt.legend(loc='best')
    plt.show()
    
    
if __name__ == "__main__":
    main()
