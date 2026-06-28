# hitran_spectrum.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Source Han Sans CN']  # 思源黑体
rcParams['axes.unicode_minus'] = False
rcParams['text.usetex'] = False 
rcParams['mathtext.fontset'] = 'stix'

import math
from scipy import special
from scipy import constants
import pathlib
import os

class HitranSpectrum:
    """
    HITRAN光谱仿真类
    用于读取HITRAN数据库并计算吸收光谱
    支持多分子混合光谱计算
    """
    
    # HITRAN 160位格式定义
    HITRAN_FORMAT_160 = {
        'M':          {'pos': 1,   'len': 2,  'format': '%2d'},   # 分子ID
        'I':          {'pos': 3,   'len': 1,  'format': '%1d'},   # 同位素ID
        'nu':         {'pos': 4,   'len': 12, 'format': '%12f'},  # 波数
        'S':          {'pos': 16,  'len': 10, 'format': '%10f'},  # 线强
        'A':          {'pos': 26,  'len': 10, 'format': '%10f'},  # 爱因斯坦A系数
        'gamma_air':  {'pos': 36,  'len': 5,  'format': '%5f'},   # 空气加宽半宽
        'gamma_self': {'pos': 41,  'len': 5,  'format': '%5f'},   # 自加宽半宽
        'E':          {'pos': 46,  'len': 10, 'format': '%10f'},  # 低态能量
        'n_air':      {'pos': 56,  'len': 4,  'format': '%4f'},   # 温度依赖系数
        'delta_air':  {'pos': 60,  'len': 8,  'format': '%8f'},   # 压力位移
        'V':          {'pos': 68,  'len': 15, 'format': '%15s'},  # 空位
        'V_':         {'pos': 83,  'len': 15, 'format': '%15s'},  # 空位
        'Q':          {'pos': 98,  'len': 15, 'format': '%15s'},  # 空位
        'Q_':         {'pos': 113, 'len': 15, 'format': '%15s'},  # 空位
        'Ierr':       {'pos': 128, 'len': 6,  'format': '%6s'},   # 空位
        'Iref':       {'pos': 134, 'len': 12, 'format': '%12s'},  # 空位
        'flag':       {'pos': 146, 'len': 1,  'format': '%1s'},   # 空位
        'g':          {'pos': 147, 'len': 7,  'format': '%7f'},   # 空位
        'g_':         {'pos': 154, 'len': 7,  'format': '%7f'}    # 空位
    }
    
    # ISO字典定义 
    ISO = {
        (  1,  1 ):    [      1,  'H2(16O)',                 9.973173E-01,  1.801056E+01,  'H2O'         ], 
        (  1,  2 ):    [      2,  'H2(18O)',                 1.999827E-03,  2.001481E+01,  'H2O'         ], 
        (  1,  3 ):    [      3,  'H2(17O)',                 3.718841E-04,  1.901478E+01,  'H2O'         ], 
        (  1,  4 ):    [      4,  'HD(16O)',                 3.106928E-04,  1.901674E+01,  'H2O'         ], 
        (  1,  5 ):    [      5,  'HD(18O)',                 6.230031E-07,  2.102098E+01,  'H2O'         ], 
        (  1,  6 ):    [      6,  'HD(17O)',                 1.158526E-07,  2.002096E+01,  'H2O'         ], 
        (  1,  7 ):    [    129,  'D2(16O)',                 2.419741E-08,  2.002292E+01,  'H2O'         ], 
        (  2,  1 ):    [      7,  '(12C)(16O)2',             9.842043E-01,  4.398983E+01,  'CO2'         ], 
        (  2,  2 ):    [      8,  '(13C)(16O)2',             1.105736E-02,  4.499318E+01,  'CO2'         ], 
        (  2,  3 ):    [      9,  '(16O)(12C)(18O)',         3.947066E-03,  4.599408E+01,  'CO2'         ], 
        (  2,  4 ):    [     10,  '(16O)(12C)(17O)',         7.339890E-04,  4.499404E+01,  'CO2'         ], 
        (  2,  5 ):    [     11,  '(16O)(13C)(18O)',         4.434456E-05,  4.699743E+01,  'CO2'         ], 
        (  2,  6 ):    [     12,  '(16O)(13C)(17O)',         8.246233E-06,  4.599740E+01,  'CO2'         ], 
        (  2,  7 ):    [     13,  '(12C)(18O)2',             3.957340E-06,  4.799832E+01,  'CO2'         ], 
        (  2,  8 ):    [     14,  '(17O)(12C)(18O)',         1.471799E-06,  4.699829E+01,  'CO2'         ], 
        (  2,  9 ):    [    121,  '(12C)(17O)2',             1.368466E-07,  4.599826E+01,  'CO2'         ], 
        (  2, 10 ):    [     15,  '(13C)(18O)2',             4.446000E-08,  4.900167E+01,  'CO2'         ], 
        (  2, 11 ):    [    120,  '(18O)(13C)(17O)',         1.653540E-08,  4.800165E+01,  'CO2'         ], 
        (  2, 12 ):    [    122,  '(13C)(17O)2',             1.537446E-09,  4.700162E+01,  'CO2'         ], 
        (  3,  1 ):    [     16,  '(16O)3',                  9.929009E-01,  4.798474E+01,  'O3'          ], 
        (  3,  2 ):    [     17,  '(16O)(16O)(18O)',         3.981942E-03,  4.998899E+01,  'O3'          ], 
        (  3,  3 ):    [     18,  '(16O)(18O)(16O)',         1.990971E-03,  4.998899E+01,  'O3'          ], 
        (  3,  4 ):    [     19,  '(16O)(16O)(17O)',         7.404746E-04,  4.898896E+01,  'O3'          ], 
        (  3,  5 ):    [     20,  '(16O)(17O)(16O)',         3.702373E-04,  4.898896E+01,  'O3'          ], 
        (  4,  1 ):    [     21,  '(14N)2(16O)',             9.903328E-01,  4.400106E+01,  'N2O'         ], 
        (  4,  2 ):    [     22,  '(14N)(15N)(16O)',         3.640926E-03,  4.499810E+01,  'N2O'         ], 
        (  4,  3 ):    [     23,  '(15N)(14N)(16O)',         3.640926E-03,  4.499810E+01,  'N2O'         ], 
        (  4,  4 ):    [     24,  '(14N)2(18O)',             1.985822E-03,  4.600531E+01,  'N2O'         ], 
        (  4,  5 ):    [     25,  '(14N)2(17O)',             3.692797E-04,  4.500528E+01,  'N2O'         ], 
        (  5,  1 ):    [     26,  '(12C)(16O)',              9.865444E-01,  2.799491E+01,  'CO'          ], 
        (  5,  2 ):    [     27,  '(13C)(16O)',              1.108364E-02,  2.899827E+01,  'CO'          ], 
        (  5,  3 ):    [     28,  '(12C)(18O)',              1.978224E-03,  2.999916E+01,  'CO'          ], 
        (  5,  4 ):    [     29,  '(12C)(17O)',              3.678671E-04,  2.899913E+01,  'CO'          ], 
        (  5,  5 ):    [     30,  '(13C)(18O)',              2.222500E-05,  3.100252E+01,  'CO'          ], 
        (  5,  6 ):    [     31,  '(13C)(17O)',              4.132920E-06,  3.000249E+01,  'CO'          ], 
        (  6,  1 ):    [     32,  '(12C)H4',                 9.882741E-01,  1.603130E+01,  'CH4'         ], 
        (  6,  2 ):    [     33,  '(13C)H4',                 1.110308E-02,  1.703466E+01,  'CH4'         ], 
        (  6,  3 ):    [     34,  '(12C)H3D',                6.157511E-04,  1.703748E+01,  'CH4'         ], 
        (  6,  4 ):    [     35,  '(13C)H3D',                6.917852E-06,  1.804083E+01,  'CH4'         ], 
        (  7,  1 ):    [     36,  '(16O)2',                  9.952616E-01,  3.198983E+01,  'O2'          ], 
        (  7,  2 ):    [     37,  '(16O)(18O)',              3.991410E-03,  3.399408E+01,  'O2'          ], 
        (  7,  3 ):    [     38,  '(16O)(17O)',              7.422352E-04,  3.299404E+01,  'O2'          ], 
        (  8,  1 ):    [     39,  '(14N)(16O)',              9.939737E-01,  2.999799E+01,  'NO'          ], 
        (  8,  2 ):    [     40,  '(15N)(16O)',              3.654311E-03,  3.099502E+01,  'NO'          ], 
        (  8,  3 ):    [     41,  '(14N)(18O)',              1.993122E-03,  3.200223E+01,  'NO'          ], 
        (  9,  1 ):    [     42,  '(32S)(16O)2',             9.456777E-01,  6.396190E+01,  'SO2'         ], 
        (  9,  2 ):    [     43,  '(34S)(16O)2',             4.195028E-02,  6.595770E+01,  'SO2'         ], 
        (  9,  3 ):    [    137,  '(33S)(16O)2',             7.464462E-03,  6.496129E+01,  'SO2'         ], 
        (  9,  4 ):    [    138,  '(16O)(32S)(18O)',         3.792558E-03,  6.596615E+01,  'SO2'         ], 
        ( 10,  1 ):    [     44,  '(14N)(16O)2',             9.916160E-01,  4.599290E+01,  'NO2'         ], 
        ( 10,  2 ):    [    130,  '(15N)(16O)2',             3.645643E-03,  4.698994E+01,  'NO2'         ], 
        ( 11,  1 ):    [     45,  '(14N)H3',                 9.958716E-01,  1.702655E+01,  'NH3'         ], 
        ( 11,  2 ):    [     46,  '(15N)H3',                 3.661289E-03,  1.802358E+01,  'NH3'         ], 
        ( 12,  1 ):    [     47,  'H(14N)(16O)3',            9.891098E-01,  6.299564E+01,  'HNO3'        ], 
        ( 12,  2 ):    [    117,  'H(15N)(16O)3',            3.636429E-03,  6.399268E+01,  'HNO3'        ], 
        ( 13,  1 ):    [     48,  '(16O)H',                  9.974726E-01,  1.700274E+01,  'OH'          ], 
        ( 13,  2 ):    [     49,  '(18O)H',                  2.000138E-03,  1.900699E+01,  'OH'          ], 
        ( 13,  3 ):    [     50,  '(16O)D',                  1.553706E-04,  1.800891E+01,  'OH'          ], 
        ( 14,  1 ):    [     51,  'H(19F)',                  9.998443E-01,  2.000623E+01,  'HF'          ], 
        ( 14,  2 ):    [    110,  'D(19F)',                  1.557410E-04,  2.101240E+01,  'HF'          ], 
        ( 15,  1 ):    [     52,  'H(35Cl)',                 7.575870E-01,  3.597668E+01,  'HCl'         ], 
        ( 15,  2 ):    [     53,  'H(37Cl)',                 2.422573E-01,  3.797373E+01,  'HCl'         ], 
        ( 15,  3 ):    [    107,  'D(35Cl)',                 1.180050E-04,  3.698285E+01,  'HCl'         ], 
        ( 15,  4 ):    [    108,  'D(37Cl)',                 3.773502E-05,  3.897990E+01,  'HCl'         ], 
        ( 16,  1 ):    [     54,  'H(79Br)',                 5.067811E-01,  7.992616E+01,  'HBr'         ], 
        ( 16,  2 ):    [     55,  'H(81Br)',                 4.930632E-01,  8.192412E+01,  'HBr'         ], 
        ( 16,  3 ):    [    111,  'D(79Br)',                 7.893838E-05,  8.093234E+01,  'HBr'         ], 
        ( 16,  4 ):    [    112,  'D(81Br)',                 7.680162E-05,  8.293029E+01,  'HBr'         ], 
        ( 17,  1 ):    [     56,  'H(127I)',                 9.998443E-01,  1.279123E+02,  'HI'          ], 
        ( 17,  2 ):    [    113,  'D(127I)',                 1.557410E-04,  1.289185E+02,  'HI'          ], 
        ( 18,  1 ):    [     57,  '(35Cl)(16O)',             7.559077E-01,  5.096377E+01,  'ClO'         ], 
        ( 18,  2 ):    [     58,  '(37Cl)(16O)',             2.417203E-01,  5.296082E+01,  'ClO'         ], 
        ( 19,  1 ):    [     59,  '(16O)(12C)(32S)',         9.373947E-01,  5.996699E+01,  'OCS'         ], 
        ( 19,  2 ):    [     60,  '(16O)(12C)(34S)',         4.158284E-02,  6.196278E+01,  'OCS'         ], 
        ( 19,  3 ):    [     61,  '(16O)(13C)(32S)',         1.053146E-02,  6.097034E+01,  'OCS'         ], 
        ( 19,  4 ):    [     62,  '(16O)(12C)(33S)',         7.399083E-03,  6.096637E+01,  'OCS'         ], 
        ( 19,  5 ):    [     63,  '(18O)(12C)(32S)',         1.879670E-03,  6.197123E+01,  'OCS'         ], 
        ( 19,  6 ):    [    135,  '(16O)(13C)(34S)',         4.671757E-04,  6.296614E+01,  'OCS'         ], 
        ( 20,  1 ):    [     64,  'H2(12C)(16O)',            9.862371E-01,  3.001056E+01,  'H2CO'        ], 
        ( 20,  2 ):    [     65,  'H2(13C)(16O)',            1.108020E-02,  3.101392E+01,  'H2CO'        ], 
        ( 20,  3 ):    [     66,  'H2(12C)(18O)',            1.977609E-03,  3.201481E+01,  'H2CO'        ], 
        ( 21,  1 ):    [     67,  'H(16O)(35Cl)',            7.557900E-01,  5.197159E+01,  'HOCl'        ], 
        ( 21,  2 ):    [     68,  'H(16O)(37Cl)',            2.416826E-01,  5.396864E+01,  'HOCl'        ], 
        ( 22,  1 ):    [     69,  '(14N)2',                  9.926874E-01,  2.800615E+01,  'N2'          ], 
        ( 22,  2 ):    [    118,  '(14N)(15N)',              7.299165E-03,  2.900318E+01,  'N2'          ], 
        ( 23,  1 ):    [     70,  'H(12C)(14N)',             9.851143E-01,  2.701090E+01,  'HCN'         ], 
        ( 23,  2 ):    [     71,  'H(13C)(14N)',             1.106758E-02,  2.801425E+01,  'HCN'         ], 
        ( 23,  3 ):    [     72,  'H(12C)(15N)',             3.621740E-03,  2.800793E+01,  'HCN'         ], 
        ( 24,  1 ):    [     73,  '(12C)H3(35Cl)',           7.489369E-01,  4.999233E+01,  'CH3Cl'       ], 
        ( 24,  2 ):    [     74,  '(12C)H3(37Cl)',           2.394912E-01,  5.198938E+01,  'CH3Cl'       ], 
        ( 25,  1 ):    [     75,  'H2(16O)2',                9.949516E-01,  3.400548E+01,  'H2O2'        ], 
        ( 26,  1 ):    [     76,  '(12C)2H2',                9.775989E-01,  2.601565E+01,  'C2H2'        ], 
        ( 26,  2 ):    [     77,  '(12C)(13C)H2',            2.196629E-02,  2.701900E+01,  'C2H2'        ], 
        ( 26,  3 ):    [    105,  '(12C)2HD',                3.045499E-04,  2.702182E+01,  'C2H2'        ], 
        ( 27,  1 ):    [     78,  '(12C)2H6',                9.769900E-01,  3.004695E+01,  'C2H6'        ], 
        ( 27,  2 ):    [    106,  '(12C)H3(13C)H3',          2.195261E-02,  3.105031E+01,  'C2H6'        ], 
        ( 28,  1 ):    [     79,  '(31P)H3',                 9.995329E-01,  3.399724E+01,  'PH3'         ], 
        ( 29,  1 ):    [     80,  '(12C)(16O)(19F)2',        9.865444E-01,  6.599172E+01,  'COF2'        ], 
        ( 29,  2 ):    [    119,  '(13C)(16O)(19F)2',        1.108366E-02,  6.699508E+01,  'COF2'        ], 
        ( 30,  1 ):    [    126,  '(32S)(19F)6',             9.501800E-01,  1.459625E+02,  'SF6'         ], 
        ( 31,  1 ):    [     81,  'H2(32S)',                 9.498841E-01,  3.398772E+01,  'H2S'         ], 
        ( 31,  2 ):    [     82,  'H2(34S)',                 4.213687E-02,  3.598351E+01,  'H2S'         ], 
        ( 31,  3 ):    [     83,  'H2(33S)',                 7.497664E-03,  3.498710E+01,  'H2S'         ], 
        ( 32,  1 ):    [     84,  'H(12C)(16O)(16O)H',       9.838977E-01,  4.600548E+01,  'HCOOH'       ], 
        ( 33,  1 ):    [     85,  'H(16O)2',                 9.951066E-01,  3.299766E+01,  'HO2'         ], 
        ( 34,  1 ):    [     86,  '(16O)',                   9.976280E-01,  1.599492E+01,  'O'           ], 
        ( 35,  1 ):    [    127,  '(35Cl)(16O)(14N)(16O)2',  7.495702E-01,  9.695667E+01,  'ClONO2'      ], 
        ( 35,  2 ):    [    128,  '(37Cl)(16O)(14N)(16O)2',  2.396937E-01,  9.895372E+01,  'ClONO2'      ], 
        ( 36,  1 ):    [     87,  '(14N)(16O)+',             9.939737E-01,  2.999799E+01,  'NOp'         ], 
        ( 37,  1 ):    [     88,  'H(16O)(79Br)',            5.055790E-01,  9.592108E+01,  'HOBr'        ], 
        ( 37,  2 ):    [     89,  'H(16O)(81Br)',            4.918937E-01,  9.791903E+01,  'HOBr'        ], 
        ( 38,  1 ):    [     90,  '(12C)2H4',                9.772944E-01,  2.803130E+01,  'C2H4'        ], 
        ( 38,  2 ):    [     91,  '(12C)H2(13C)H2',          2.195946E-02,  2.903466E+01,  'C2H4'        ], 
        ( 39,  1 ):    [     92,  '(12C)H3(16O)H',           9.859299E-01,  3.202622E+01,  'CH3OH'       ], 
        ( 40,  1 ):    [     93,  '(12C)H3(79Br)',           5.009946E-01,  9.394181E+01,  'CH3Br'       ], 
        ( 40,  2 ):    [     94,  '(12C)H3(81Br)',           4.874334E-01,  9.593976E+01,  'CH3Br'       ], 
        ( 41,  1 ):    [     95,  '(12C)H3(12C)(14N)',       9.738662E-01,  4.102655E+01,  'CH3CN'       ], 
        ( 42,  1 ):    [     96,  '(12C)(19F)4',             9.888900E-01,  8.799362E+01,  'CF4'         ], 
        ( 43,  1 ):    [    116,  '(12C)4H2',                9.559980E-01,  5.001565E+01,  'C4H2'        ], 
        ( 44,  1 ):    [    109,  'H(12C)3(14N)',            9.633460E-01,  5.101090E+01,  'HC3N'        ], 
        ( 45,  1 ):    [    103,  'H2',                      9.996885E-01,  2.015650E+00,  'H2'          ], 
        ( 45,  2 ):    [    115,  'HD',                      3.114316E-04,  3.021825E+00,  'H2'          ], 
        ( 46,  1 ):    [     97,  '(12C)(32S)',              9.396236E-01,  4.397207E+01,  'CS'          ], 
        ( 46,  2 ):    [     98,  '(12C)(34S)',              4.168171E-02,  4.596787E+01,  'CS'          ], 
        ( 46,  3 ):    [     99,  '(13C)(32S)',              1.055650E-02,  4.497543E+01,  'CS'          ], 
        ( 46,  4 ):    [    100,  '(12C)(33S)',              7.416675E-03,  4.497146E+01,  'CS'          ], 
        ( 47,  1 ):    [    114,  '(32S)(16O)3',             9.434345E-01,  7.995682E+01,  'SO3'         ], 
        ( 48,  1 ):    [    123,  '(12C)2(14N)2',            9.707524E-01,  5.200615E+01,  'C2N2'        ], 
        ( 49,  1 ):    [    124,  '(12C)(16O)(35Cl)2',       5.663918E-01,  9.793262E+01,  'COCl2'       ], 
        ( 49,  2 ):    [    125,  '(12C)(16O)(35Cl)(37Cl)',  3.622350E-01,  9.992967E+01,  'COCl2'       ], 
        ( 50,  1 ):    [    146,  '(32S)(16O)',              9.479262E-01,  4.796699E+01,  'SO'          ], 
        ( 50,  2 ):    [    147,  '(34S)(16O)',              4.205002E-02,  4.996278E+01,  'SO'          ], 
        ( 50,  3 ):    [    148,  '(32S)(18O)',              1.900788E-03,  4.997123E+01,  'SO'          ], 
        ( 51,  1 ):    [    144,  '(12C)H3(19F)',            9.884280E-01,  3.402188E+01,  'CH3F'        ], 
        ( 52,  1 ):    [    139,  '(74Ge)H4',                3.651724E-01,  7.795248E+01,  'GeH4'        ], 
        ( 52,  2 ):    [    140,  '(72Ge)H4',                2.741292E-01,  7.595338E+01,  'GeH4'        ], 
        ( 52,  3 ):    [    141,  '(70Ge)H4',                2.050722E-01,  7.395555E+01,  'GeH4'        ], 
        ( 52,  4 ):    [    142,  '(73Ge)H4',                7.755167E-02,  7.695476E+01,  'GeH4'        ], 
        ( 52,  5 ):    [    143,  '(76Ge)H4',                7.755167E-02,  7.995270E+01,  'GeH4'        ], 
        ( 53,  1 ):    [    131,  '(12C)(32S)2',             8.928115E-01,  7.594414E+01,  'CS2'         ], 
        ( 53,  2 ):    [    132,  '(32S)(12C)(34S)',         7.921026E-02,  7.793994E+01,  'CS2'         ], 
        ( 53,  3 ):    [    133,  '(32S)(12C)(33S)',         1.409435E-02,  7.694353E+01,  'CS2'         ], 
        ( 53,  4 ):    [    134,  '(13C)(32S)2',             1.003057E-02,  7.694750E+01,  'CS2'         ], 
        ( 54,  1 ):    [    145,  '(12C)H3(127I)',           9.884280E-01,  1.419279E+02,  'CH3I'        ], 
        ( 55,  1 ):    [    136,  '(14N)(19F)3',             9.963370E-01,  7.099829E+01,  'NF3'         ], 
    }
    
    
    # 物理常数
    T_ref = 296.0
    cBolts = constants.Boltzmann * 1E7  # CGS Boltzmann常数，单位erg/K
    cc = constants.speed_of_light * 1E2  # CGS，cm/s
    cNA = constants.N_A
    c2 = constants.physical_constants['second radiation constant'][0] * 100  # CGS cm K
    cP = constants.physical_constants['standard atmosphere'][0] * 10  # 压力单位转换为CGS的Ba
    cGammaD = math.sqrt(2 * cBolts * cNA * math.log(2)) / cc  # 多普勒展宽常数
    
    def __init__(self, q_folder=None):
        """
        初始化HitranSpectrum类
        
        参数:
        q_folder: 配分函数文件夹路径
        """
        self.q_folder = q_folder
        self.molecules = {}  # 存储多个分子的数据
        self.molecule_list = []  # 分子名称列表，保持顺序
        
    def add_molecule(self, par_file, concentration=1.0, molecule_name=None):
        """
        添加分子数据
        
        参数:
        par_file: HITRAN数据库文件路径
        concentration: 该分子的体积分数（默认1.0）
        molecule_name: 自定义分子名称（可选）
        """
        if self.q_folder is None:
            raise ValueError("请先设置配分函数文件夹路径")
            
        # 读取数据库
        database, molecule_info = self.read_hitran_par(par_file)
        
        if molecule_info is None:
            raise ValueError(f"无法从par文件 {par_file} 中识别分子信息")
        
        # 自动获取分子质量
        molecule_id = molecule_info['molecule_id']
        isotope_id = molecule_info['isotope_id']
        mass = self.get_molar_mass(molecule_id, isotope_id)
        
        if mass is None:
            # 如果找不到精确的同位素质量，尝试使用主要同位素的质量
            print(f"警告: 未找到分子ID {molecule_id} 同位素ID {isotope_id} 的质量信息，尝试使用主要同位素")
            mass = self.get_molar_mass(molecule_id, 1)  # 尝试使用同位素1
            if mass is None:
                raise ValueError(f"无法找到分子ID {molecule_id} 的质量信息")
        
        # 自动读取配分函数文件
        q_file = os.path.join(self.q_folder, f'q{molecule_id}.txt')
        
        if not os.path.exists(q_file):
            # 尝试其他可能的文件名格式
            q_file_alt = os.path.join(self.q_folder, f'Q{molecule_id}.txt')
            if os.path.exists(q_file_alt):
                q_file = q_file_alt
            else:
                raise FileNotFoundError(f"找不到配分函数文件: {q_file}")
        
        partition_function_data = self.read_partition_function(q_file)
        
        # 设置默认的波数范围
        if len(database) > 0:
            default_start = np.min(database[:, 0])
            default_end = np.max(database[:, 0])
        else:
            default_start = 0
            default_end = 0
        
        # 获取分子名称
        if molecule_name is None:
            molecule_name = self.get_molecule_name(molecule_id, isotope_id)
        
        # 存储分子数据
        molecule_data = {
            'database': database,
            'molecule_info': molecule_info,
            'concentration': concentration,
            'molar_mass': mass,
            'partition_function_data': partition_function_data,
            'default_start': default_start,
            'default_end': default_end,
            'q_file': q_file,
            'par_file': par_file
        }
        
        # 使用分子名称作为键
        self.molecules[molecule_name] = molecule_data
        self.molecule_list.append(molecule_name)
        
        print(f"成功添加分子: {molecule_name}")
        print(f"  谱线数量: {len(database)}")
        print(f"  体积分数: {concentration}")
        print(f"  波数范围: {default_start:.4f} - {default_end:.4f} cm⁻¹")
        print(f"  分子质量: {mass:.6f} g/mol")
        print(f"  分子ID: {molecule_id}, 同位素ID: {isotope_id}")
    
    def get_molar_mass(self, molecule_id, isotope_id):
        """根据分子ID和同位素ID获取分子质量"""
        key = (molecule_id, isotope_id)
        if key in self.ISO:
            return self.ISO[key][3]  # 第4列是质量
        print(f"警告: 未找到分子ID {molecule_id} 同位素ID {isotope_id} 的质量信息")
        return None
    
    def get_molecule_name(self, molecule_id, isotope_id):
        """根据分子ID和同位素ID获取分子名称"""
        key = (molecule_id, isotope_id)
        if key in self.ISO:
            return self.ISO[key][4]  # 第5列是分子名称
        return f"未知分子 ({molecule_id}-{isotope_id})"
    
    def read_hitran_par(self, filename):
        """
        读取HITRAN 160位格式的par文件，自动识别分子和同位素
        
        返回:
        database: 数据库数组
        molecule_info: 分子信息字典
        """
        database = []
        molecule_ids = set()
        isotope_ids = set()
        
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.rstrip('\n')
                if len(line) < 160:
                    print(f"警告: 第 {line_num} 行长度不足160字符，已跳过")
                    continue
                    
                try:
                    # 使用HITRAN格式定义解析各个字段
                    fields = {}
                    
                    # 解析每个字段
                    for field_name, field_info in self.HITRAN_FORMAT_160.items():
                        start_pos = field_info['pos'] - 1  # 转换为0-based索引
                        end_pos = start_pos + field_info['len']
                        field_str = line[start_pos:end_pos].strip()
                        
                        if field_str:  # 只处理非空字段
                            if field_info['format'].endswith('d'):
                                try:
                                    fields[field_name] = int(field_str)
                                except ValueError:
                                    # 如果无法转换为整数，记录警告并使用默认值
                                    if field_name == 'I':  # 同位素ID
                                        print(f"警告: 第 {line_num} 行同位素ID '{field_str}' 不是有效整数，使用默认值1")
                                        fields[field_name] = 1
                                    else:
                                        raise
                            elif field_info['format'].endswith('f'):
                                fields[field_name] = float(field_str)
                            else:
                                fields[field_name] = field_str
                        else:
                            # 对于空字段，设置默认值
                            if field_name == 'I':  # 同位素ID不能为空
                                fields[field_name] = 1
                    
                    # 提取主要字段
                    molec_id = fields.get('M')
                    local_iso_id = fields.get('I')  # 默认同位素ID为1
                    nu = fields.get('nu')
                    S = fields.get('S')
                    A = fields.get('A')
                    gamma_air = fields.get('gamma_air')
                    gamma_self = fields.get('gamma_self')
                    E = fields.get('E')
                    n_air = fields.get('n_air')
                    delta_air = fields.get('delta_air')
                    
                    # 验证必需字段
                    if None in [molec_id, local_iso_id, nu, S, A, gamma_air, gamma_self, E, n_air, delta_air]:
                        print(f"警告: 第 {line_num} 行缺少必需字段，已跳过")
                        print(f"  字段状态: M={molec_id}, I={local_iso_id}, nu={nu}, S={S}, A={A}")
                        continue
                    
                    # 验证同位素ID的合理性
                    if local_iso_id == 0:
                        print(f"警告: 第 {line_num} 行同位素ID为00，自动修正为1")
                        local_iso_id = 1
                    
                    molecule_ids.add(molec_id)
                    isotope_ids.add(local_iso_id)
                    
                    database.append([nu, S, A, gamma_air, gamma_self, E, n_air, delta_air])
                    
                except (ValueError, IndexError) as e:
                    print(f"解析第 {line_num} 行时出错: {repr(line)}")
                    print(f"错误信息: {e}")
                    continue
        
        if len(molecule_ids) == 0:
            raise ValueError("未找到有效的分子ID")
        
        # 获取主要分子ID（假设文件中只有一个分子）
        main_molecule_id = list(molecule_ids)[0]
        
        # 选择最常出现的同位素ID作为主要同位素
        if isotope_ids:
            # 统计每个同位素ID的出现次数
            iso_counts = {}
            for iso_id in isotope_ids:
                iso_counts[iso_id] = sum(1 for line in database if True)  # 这里简化处理
            
            main_isotope_id = max(iso_counts, key=iso_counts.get)
        else:
            main_isotope_id = 1
        
        molecule_info = {
            'molecule_id': main_molecule_id,
            'isotope_id': main_isotope_id,
            'all_molecules': list(molecule_ids),
            'all_isotopes': list(isotope_ids)
        }
        
        print(f"识别到分子ID: {main_molecule_id}, 主要同位素ID: {main_isotope_id}")
        print(f"文件中包含的同位素: {list(isotope_ids)}")
        
        return np.array(database), molecule_info
    
    def read_partition_function(self, filename):
        """
        读取配分函数文件
        返回: 字典，包含温度数组和配分函数值数组
        """
        try:
            data = np.loadtxt(filename)
            
            if data.ndim == 2 and data.shape[1] >= 2:
                temperatures = data[:, 0]  # 第一列是温度，保持为浮点数
                q_values = data[:, 1]  # 第二列是配分函数值
                
                return {
                    'temperatures': temperatures,
                    'values': q_values
                }
            else:
                raise ValueError("配分函数文件格式不正确，应该是两列数据")
        except Exception as e:
            print(f"读取配分函数文件错误: {e}")
            print(f"文件路径: {filename}")
            raise

    def get_partition_function(self, molecule_name, T):
        """
        获取指定分子在指定温度下的配分函数值
        使用线性插值
        """
        if molecule_name not in self.molecules:
            raise ValueError(f"分子 {molecule_name} 未加载")
        
        partition_function_data = self.molecules[molecule_name]['partition_function_data']
        temperatures = partition_function_data['temperatures']
        q_values = partition_function_data['values']
        
        # 直接使用线性插值
        return np.interp(T, temperatures, q_values)
    
    def profile_voigt(self, nu, gamma_air, gamma_self, n_air, delta_air, T, p, c, wavenumber, mass):
        """
        使用Faddeeva函数计算Voigt线型
        """
        # 计算压力展宽
        gamma = gamma_air * p * (1 - c) * ((self.T_ref / T) ** n_air) + \
                gamma_self * p * c * ((self.T_ref / T) ** n_air)
        
        # 计算多普勒展宽
        GammaD = self.cGammaD * math.sqrt(T / mass) * nu
        sigma = GammaD / (math.sqrt(2 * math.log(2)))
        
        # 计算Voigt线型
        variable = (wavenumber - nu - delta_air * p + gamma * 1j) / (sigma * math.sqrt(2))
        voigt = (special.wofz(variable)).real / (sigma * math.sqrt(2 * constants.pi))
        
        return voigt
    
    def coef_single(self, molecule_name, T, p, wavenumber=None, start=None, end=None, resolution=0.001, omega_wing=10):
        """
        计算单个分子的吸收系数
        
        参数:
        molecule_name: 分子名称
        T: 温度 (K)
        p: 压力 (atm)
        wavenumber: 波数数组 (可选)
        start: 起始波数 (可选)
        end: 结束波数 (可选)
        resolution: 分辨率 (cm⁻¹)
        omega_wing: 谱线计算域的倍数
        
        返回:
        coef_array: 吸收系数数组
        wavenumber: 波数数组
        """
        if molecule_name not in self.molecules:
            raise ValueError(f"分子 {molecule_name} 未加载")
        
        molecule_data = self.molecules[molecule_name]
        database = molecule_data['database']
        concentration = molecule_data['concentration']
        mass = molecule_data['molar_mass']
        
        # 确定波数范围
        if wavenumber is None:
            if start is None:
                start = molecule_data['default_start']
            if end is None:
                end = molecule_data['default_end']
            wavenumber = np.arange(start, end, resolution)
        
        coef_array = np.zeros(len(wavenumber))
        
        # 获取当前温度和参考温度的配分函数值
        q_T = self.get_partition_function(molecule_name, int(T))
        q_T_ref = self.get_partition_function(molecule_name, int(self.T_ref))
        
        for line in database:
            nu = line[0]
            S = line[1]
            gamma_air = line[3]
            gamma_self = line[4]
            E = line[5]
            n_air = line[6]
            delta_air = line[7]
            
            # 计算多普勒展宽和压力展宽
            GammaD = self.cGammaD * math.sqrt(T / mass) * nu
            gamma_total = gamma_air * p * (1 - concentration) * ((self.T_ref / T) ** n_air) + \
                         gamma_self * p * concentration * ((self.T_ref / T) ** n_air)
            
            # 确定每条谱线的计算域
            wing_width = omega_wing * (GammaD + gamma_total)
            line_start = nu - wing_width
            line_end = nu + wing_width
            
            # 找到在全局波数范围内的索引
            indices = np.where((wavenumber >= line_start) & (wavenumber <= line_end))[0]
            
            if len(indices) == 0:
                continue
                
            # 计算该谱线在局部范围内的线型
            local_wavenumber = wavenumber[indices]
            profile = self.profile_voigt(nu, gamma_air, gamma_self, n_air, delta_air, 
                                       T, p, concentration, local_wavenumber, mass)
            
            # 计算线强 - 使用插值后的配分函数值
            intensity = S * q_T_ref / q_T * \
                       math.exp(-self.c2 * E / T) / math.exp(-self.c2 * E / self.T_ref) * \
                       (1 - math.exp(-self.c2 * nu / T)) / (1 - math.exp(-self.c2 * nu / self.T_ref))
            
            # 累加到吸收系数
            coef_array[indices] += profile * intensity
        
        return coef_array, wavenumber
    
    def coef_mixture(self, T, p, wavenumber=None, start=None, end=None, resolution=0.001, omega_wing=10):
        """
        计算混合气体的总吸收系数
        
        参数:
        T: 温度 (K)
        p: 压力 (atm)
        wavenumber: 波数数组 (可选)
        start: 起始波数 (可选)
        end: 结束波数 (可选)
        resolution: 分辨率 (cm⁻¹)
        omega_wing: 谱线计算域的倍数
        
        返回:
        total_coef: 总吸收系数数组
        wavenumber: 波数数组
        individual_coefs: 各分子吸收系数字典
        """
        if not self.molecules:
            raise ValueError("没有加载任何分子数据")
        
        # 确定统一的波数范围
        if wavenumber is None:
            if start is None:
                # 使用所有分子的最小起始波数和最大结束波数
                all_starts = [molecule_data['default_start'] for molecule_data in self.molecules.values()]
                all_ends = [molecule_data['default_end'] for molecule_data in self.molecules.values()]
                start = min(all_starts)
                end = max(all_ends)
            wavenumber = np.arange(start, end, resolution)
        
        total_coef = np.zeros(len(wavenumber))
        individual_coefs = {}
        
        # 计算每个分子的吸收系数并累加
        for molecule_name in self.molecule_list:
            print(f"计算分子 {molecule_name} 的吸收系数...")
            coef_array, _ = self.coef_single(molecule_name, T, p, wavenumber, start, end, resolution, omega_wing)
            individual_coefs[molecule_name] = coef_array
            total_coef += coef_array
        
        return total_coef, wavenumber, individual_coefs
    
    def OD_mixture(self, T, p, l, wavenumber=None, start=None, end=None, resolution=0.001, omega_wing=10):
        """
        计算混合气体的光学深度、吸收率和透射率
        
        参数:
        T: 温度 (K)
        p: 压力 (atm)
        l: 吸收路径长度 (cm)
        wavenumber: 波数数组 (可选)
        start: 起始波数 (可选)
        end: 结束波数 (可选)
        resolution: 分辨率 (cm⁻¹)
        omega_wing: 谱线计算域的倍数
        
        返回:
        OD: 总光学深度
        Ab: 总吸收率
        Tr: 总透射率
        wavenumber: 波数数组
        total_coef: 总吸收系数
        individual_ODs: 各分子光学深度字典
        """
        # 计算总吸收系数
        total_coef, wavenumber, individual_coefs = self.coef_mixture(T, p, wavenumber, start, end, resolution, omega_wing)
        
        # 计算总光学深度、透射率和吸收率
        # 注意：这里使用总压力p，因为各分子的浓度已经在吸收系数计算中考虑了
        density = p * self.cP / (self.cBolts) / T  # 总分子数密度
        
        OD = total_coef * density * l
        Tr = np.exp(-OD)
        Ab = 1 - Tr
        
        # 计算各分子的光学深度
        individual_ODs = {}
        for molecule_name, coef_array in individual_coefs.items():
            individual_ODs[molecule_name] = coef_array * density * l
        
        return OD, Ab, Tr, wavenumber, total_coef, individual_ODs
    
    def get_molecule_info(self, molecule_name=None):
        """
        获取分子信息
        
        参数:
        molecule_name: 分子名称，如果为None则返回所有分子信息
        
        返回:
        分子信息字符串
        """
        if molecule_name is None:
            info = f"已加载 {len(self.molecules)} 个分子:\n"
            for name in self.molecule_list:
                molecule_data = self.molecules[name]
                info += f"\n分子: {name}\n"
                info += f"  分子ID: {molecule_data['molecule_info']['molecule_id']}\n"
                info += f"  同位素ID: {molecule_data['molecule_info']['isotope_id']}\n"
                info += f"  体积分数: {molecule_data['concentration']}\n"
                info += f"  分子质量: {molecule_data['molar_mass']:.6f} g/mol\n"
                info += f"  谱线数量: {len(molecule_data['database'])}\n"
                info += f"  波数范围: {molecule_data['default_start']:.4f} - {molecule_data['default_end']:.4f} cm⁻¹\n"
            return info
        else:
            if molecule_name not in self.molecules:
                return f"分子 {molecule_name} 未加载"
            
            molecule_data = self.molecules[molecule_name]
            info = f"分子: {molecule_name}\n"
            info += f"分子ID: {molecule_data['molecule_info']['molecule_id']}\n"
            info += f"同位素ID: {molecule_data['molecule_info']['isotope_id']}\n"
            info += f"体积分数: {molecule_data['concentration']}\n"
            info += f"分子质量: {molecule_data['molar_mass']:.6f} g/mol\n"
            info += f"谱线数量: {len(molecule_data['database'])}\n"
            info += f"波数范围: {molecule_data['default_start']:.4f} - {molecule_data['default_end']:.4f} cm⁻¹\n"
            info += f"配分函数文件: {molecule_data['q_file']}\n"
            info += f"数据库文件: {molecule_data['par_file']}\n"
            
            if molecule_data['partition_function_data'] is not None:
                temps = molecule_data['partition_function_data']['temperatures']
                info += f"配分函数温度范围: {temps[0]} - {temps[-1]} K\n"
            
            if len(molecule_data['database']) > 0:
                info += f"线强范围: {np.min(molecule_data['database'][:,1]):.2e} - {np.max(molecule_data['database'][:,1]):.2e} cm⁻¹/(molecule·cm⁻²)"
            
            return info

def main():
    """
    使用示例 - 多分子混合光谱
    """
    # 初始化HitranSpectrum类
    hitran = HitranSpectrum(q_folder='/home/xliu/prj/210-TDLAS/3-simulation/Q/')
    
    # 添加多个分子，每个分子有不同的体积分数
    hitran.add_molecule('/home/xliu/prj/210-TDLAS/3-simulation/hitran_database/02_CO2/CO2-4650_5410.par', concentration=0.1, molecule_name='CO2')
    hitran.add_molecule('/home/xliu/prj/210-TDLAS/3-simulation/hitran_database/01_H2O/H2O-4650_5410.par', concentration=0.1, molecule_name='H2O')
    
    # 显示所有分子信息
    print(hitran.get_molecule_info())
    
    # 设置计算参数（对所有分子相同）
    T = 600  # 温度 (K)
    p = 1    # 压力 (atm)
    l = 10  # 吸收路径长度 (cm)
    
    # 计算混合光谱
    print("计算混合光谱...")
    OD, Ab, Tr, wavenumber, total_coef, individual_ODs = hitran.OD_mixture(
        T, p, l, start=4650, end=5410, resolution=0.01, omega_wing=10
    )
    
    # 绘图
    plt.figure(figsize=(12, 10))
    
    # 绘制总吸收率
    plt.subplot(3, 1, 1)
    plt.plot(wavenumber, total_coef, 'k-', linewidth=1.5, label='总吸收率')
    plt.ylabel('吸收率')
    plt.legend()
    plt.title(f'混合气体吸收光谱 (T={T}K, p={p}atm, l={l}cm)')
    
    # 绘制各分子吸收率
    plt.subplot(3, 1, 2)
    for molecule_name, od in individual_ODs.items():
        ab_single = 1 - np.exp(-od)
        plt.plot(wavenumber, ab_single, label=f'{molecule_name} (c={hitran.molecules[molecule_name]["concentration"]})')
    plt.ylabel('吸收率')
    plt.legend()
    plt.title('各分子吸收光谱')
    
    # 绘制透射率
    plt.subplot(3, 1, 3)
    plt.plot(wavenumber, Tr, 'b-', linewidth=1.5, label='透射率')
    plt.xlabel('波数 (cm$^{-1}$)')
    plt.ylabel('透射率')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
